import argparse
import os
import warnings

import cv2
from timm.utils import AverageMeter

from training import losses
from training.augmentations import InstanceAugmentations
from training.config import load_config
from training.instance_datasets import SpacenetInstanceDataset
from training.losses import dice_round, EuclideanDistance
from training.utils import create_optimizer, all_gather

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import zoo

from apex.parallel import DistributedDataParallel, convert_syncbn_model
from tensorboardX import SummaryWriter

from apex import amp

import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import DataParallel, MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
warnings.simplefilter("ignore")

torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser("PyTorch Spacenet7 Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
    arg('--workers', type=int, default=8, help='number of cpu threads to use')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='mask_center_')
    arg('--data-dir', type=str, default="/mnt/datasets/spacenet/train/")
    arg('--folds-csv', type=str, default='folds.csv')
    arg('--logdir', type=str, default='logs')
    arg('--zero-score', action='store_true', default=False)
    arg('--from-zero', action='store_true', default=False)
    arg('--distributed', action='store_true', default=False)
    arg("--local_rank", default=0, type=int)
    arg("--opt-level", default='O1', type=str)
    arg("--predictions", default="../oof_preds", type=str)
    arg("--test_every", type=int, default=1)
    arg('--freeze-epochs', type=int, default=0)


    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    conf = load_config(args.config)
    model = zoo.__dict__[conf['network']](seg_classes=conf['num_classes'], backbone_arch=conf['encoder'],
                                          use_last_decoder=conf.get('use_last_decoder', False))

    model = model.cuda()
    if args.distributed:
        model = convert_syncbn_model(model)
    mask_loss_function = losses.__dict__[conf["mask_loss"]["type"]](**conf["mask_loss"]["params"]).cuda()
    loss_functions = {
        "mask_loss": mask_loss_function,
        "center_loss": MSELoss().cuda(),
        "offset_loss": MSELoss().cuda(),
        "offset_e_loss": EuclideanDistance().cuda(),
    }
    optimizer, scheduler = create_optimizer(conf['optimizer'], model)

    dice_best = 0
    start_epoch = 0
    batch_size = conf['optimizer']['batch_size']
    augmentations = InstanceAugmentations()
    data_train = SpacenetInstanceDataset(mode="train",
                                         sigma=conf.get("sigma", 6),
                                         fold=args.fold,
                                         data_path=args.data_dir,
                                         folds_csv=args.folds_csv,
                                         normalize=conf["normalize"],
                                         transforms=augmentations.create_train_transforms(conf["input"]),
                                         multiplier=conf["data_multiplier"])
    data_val = SpacenetInstanceDataset(mode="val",
                                       fold=args.fold,
                                       sigma=conf.get("sigma", 6),
                                       data_path=args.data_dir,
                                       folds_csv=args.folds_csv,
                                       normalize=conf["normalize"],
                                       transforms=augmentations.create_val_transforms()
                                       )
    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(data_val, shuffle=False)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=args.workers,
                                   shuffle=train_sampler is None, sampler=train_sampler, pin_memory=False,
                                   drop_last=True)
    val_batch_size = 1
    val_data_loader = DataLoader(data_val, sampler=val_sampler, batch_size=val_batch_size, num_workers=args.workers,
                                 shuffle=False,
                                 pin_memory=False)
    snapshot_name = "{}{}_{}_{}".format(args.prefix, conf['network'], conf['encoder'], args.fold)

    os.makedirs(args.logdir, exist_ok=True)
    summary_writer = SummaryWriter(args.logdir + '/' + snapshot_name)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            state_dict = checkpoint['state_dict']
            if conf['optimizer'].get('zero_decoder', False):
                for key in state_dict.copy().keys():
                    if key.startswith("module.final"):
                        del state_dict[key]
            state_dict = {k[7:]: w for k, w in state_dict.items()}
            orig_state_dict  = model.state_dict()
            mismatched_keys = []
            for k, v in state_dict.items():
                ori_size = orig_state_dict[k].size() if k in orig_state_dict else None
                if v.size() != ori_size:
                    print("SKIPPING!!! Shape of {} changed from {} to {}".format(k, v.size(), ori_size))
                    mismatched_keys.append(k)
            for k in mismatched_keys:
                del state_dict[k]
            model.load_state_dict(state_dict, strict=False)
            if not args.from_zero:
                start_epoch = checkpoint['epoch']
                if not args.zero_score:
                    dice_best = checkpoint.get('dice_best', 0)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if args.from_zero:
        start_epoch = 0
    current_epoch = start_epoch

    if conf['fp16']:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          loss_scale='dynamic')

    if args.distributed:
        model = DistributedDataParallel(model, delay_allreduce=True)
    else:
        model = DataParallel(model).cuda()
    for epoch in range(start_epoch, conf['optimizer']['schedule']['epochs']):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        if epoch < args.freeze_epochs:
            print("Freezing encoder!!!")
            model.module.encoder_stages.eval()
            for p in model.module.encoder_stages.parameters():
                p.requires_grad = False
        else:
            model.module.encoder_stages.train()
            for p in model.module.encoder_stages.parameters():
                p.requires_grad = True
        model.module.train()
        torch.cuda.empty_cache()
        train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                    args.local_rank)

        model = model.eval()
        if epoch % args.test_every == 0:
            preds_dir = os.path.join(args.predictions, snapshot_name)
            dice = evaluate_val(args, val_data_loader, dice_best,
                                     model,
                                     snapshot_name=snapshot_name,
                                     current_epoch=current_epoch,
                                     summary_writer=summary_writer,
                                     predictions_dir=preds_dir)
            if epoch > 10:
                dice_best = dice
        current_epoch += 1


def evaluate_val(args, data_val, dice_best, model, snapshot_name, current_epoch, summary_writer,
                 predictions_dir):
    print("Test phase")
    model = model.eval()
    dice = validate(model, data_loader=data_val, predictions_dir=predictions_dir, distributed=args.distributed)
    if args.local_rank == 0:
        summary_writer.add_scalar('val/dice', float(dice), global_step=current_epoch)
        if dice > dice_best:
            print("Dice improved from {} to {}".format(dice_best, dice))
            if args.output_dir is not None:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.state_dict(),
                    'dice_best': dice,
                    'dice': dice,
                }, args.output_dir + snapshot_name + "_best_dice")
            dice_best = dice
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'dice_best': dice_best,
            'dice': dice,
        }, args.output_dir + snapshot_name + "_last")
        print("dice: {}, dice_best: {}".format(dice, dice_best))
    return dice_best


def validate(net, data_loader, predictions_dir, distributed):
    os.makedirs(predictions_dir, exist_ok=True)
    preds_dir = predictions_dir + "/predictions"
    os.makedirs(preds_dir, exist_ok=True)
    dices = []
    with torch.no_grad():
        for sample in tqdm(data_loader):
            imgs = sample["image"].cuda().float()
            mask = sample["mask"].cuda().float()

            output, center = net(imgs)
            binary_pred = torch.sigmoid(output)

            for i in range(output.shape[0]):
                d = dice_round(binary_pred[:, 0:1, :], mask[:, 0:1, ...], t=0.5).item()
                dices.append(d)
                cv2.imwrite(os.path.join(preds_dir, sample["img_name"][i] + ".png"),
                            (np.moveaxis(binary_pred[i].cpu().numpy(), 0, -1)[..., :3] * 255))
                cv2.imwrite(os.path.join(preds_dir, sample["img_name"][i] + "_centers.png"),
                            (np.moveaxis(center[i].cpu().numpy(), 0, -1)[..., :1] * 255))

    dices = np.array(dices)

    if distributed:
        dices = all_gather(dices)
        dices = np.concatenate(dices)

    return np.mean(dices)


def train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                local_rank):
    losses = AverageMeter()
    c_losses = AverageMeter()
    d_losses = AverageMeter()
    dices = AverageMeter()
    iterator = tqdm(train_data_loader)
    model.train()
    if conf["optimizer"]["schedule"]["mode"] == "epoch":
        scheduler.step(current_epoch)
    for i, sample in enumerate(iterator):
        imgs = sample["image"].cuda()
        masks = sample["mask"].cuda().float()
        # if torch.sum(masks) < 100:
        #     continue
        centers = sample["center"].cuda().float()

        seg_mask, center_mask = model(imgs)
        with torch.no_grad():
            pred = torch.sigmoid(seg_mask)
            d = dice_round(pred[:, 0:1, ...].cpu(), masks[:, 0:1, ...].cpu(), t=0.5).item()
        dices.update(d, imgs.size(0))

        mask_loss = loss_functions["mask_loss"](seg_mask, masks)
        # if torch.isnan(mask_loss):
        #     print("nan loss, skipping!!!")
        #     optimizer.zero_grad()
        #     continue
        center_loss = loss_functions["center_loss"](center_mask, centers)
        center_loss *= 50
        loss = mask_loss + center_loss

        loss /= 2
        if current_epoch == 0:
            loss /= 10
        losses.update(loss.item(), imgs.size(0))
        d_losses.update(mask_loss.item(), imgs.size(0))

        c_losses.update(center_loss.item(), imgs.size(0))
        iterator.set_postfix({"lr": float(scheduler.get_lr()[-1]),
                              "epoch": current_epoch,
                              "loss": losses.avg,
                              "dice": dices.avg,
                              "d_loss": d_losses.avg,
                              "c_loss": c_losses.avg,
                              })
        optimizer.zero_grad()
        if conf['fp16']:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
        optimizer.step()
        torch.cuda.synchronize()

        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * len(train_data_loader))

    if local_rank == 0:
        for idx, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            summary_writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=current_epoch)
        summary_writer.add_scalar('train/loss', float(losses.avg), global_step=current_epoch)


if __name__ == '__main__':
    main()
