import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed

from torch.optim import SGD

from adamw import AdamW

from losses import dice_round, ComboLoss, LabelSmoothing

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from sklearn.model_selection import train_test_split
from sklearn import metrics

from zoo.models import EfficientNet_Unet_Double

from Dataset import TrainDatasetDouble, ValDataset
from utils import *

from ddp_utils import all_gather, reduce_tensor



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--train_dir", default='/data/SN7_buildings/train/')
args = parser.parse_args()


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


models_folder = '/wdata/weights'

masks_dir = '/wdata/masks_4k_2'

train_dir = args.train_dir


def validate(model, val_data_loader):
    dices = []
    dices2 = []
    dices3 = []

    if args.local_rank == 0:
        iterator = tqdm(val_data_loader)
    else:
        iterator = val_data_loader

    with torch.no_grad():
        for i, sample in enumerate(iterator):
            with torch.cuda.amp.autocast():
                imgs = sample["img"].cuda(non_blocking=True)
                otps = sample["msk"].cpu().numpy()

                res = model(imgs)

                probs = torch.sigmoid(res)
                pred = probs.cpu().numpy() > 0.5
            for j in range(otps.shape[0]):
                dices.append(dice(otps[j, 0], pred[j, 0]))
                dices2.append(dice(otps[j, 1], pred[j, 1]))
                dices3.append(dice(otps[j, 2], pred[j, 2]))

    
    dices = np.asarray(dices)
    dices2 = np.asarray(dices2)
    dices3 = np.asarray(dices3)


    dices = np.concatenate(all_gather(dices))
    dices2 = np.concatenate(all_gather(dices2))
    dices3 = np.concatenate(all_gather(dices3))

    torch.cuda.synchronize()

    d0 = np.mean(dices)
    d2 = np.mean(dices2)
    d3 = np.mean(dices3)

    if args.local_rank == 0:
        print("Val Dice: {} Dice2: {} Dice3: {} Len: {}".format(d0, d2, d3, len(dices)))

    return d0


def evaluate_val(val_data_loader, best_score, model, snapshot_name, current_epoch):
    model.eval()
    _sc = validate(model, val_data_loader=val_data_loader)

    if args.local_rank == 0:
        if _sc > best_score:
            torch.save({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'best_score': _sc,
            }, path.join(models_folder, snapshot_name))

            best_score = _sc
        print("dice: {}\tdice_best: {}".format(_sc, best_score))
    return best_score, _sc


def train_epoch(current_epoch, combo_loss, model, optimizer, scaler, train_data_loader):
    losses = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()
    dices = AverageMeter()
    dices2 = AverageMeter()
    dices3 = AverageMeter()
    
    if args.local_rank == 0:
        iterator = tqdm(train_data_loader)
    else:
        iterator = train_data_loader
    model.train()

    _lr = optimizer.param_groups[0]['lr']

    for i, sample in enumerate(iterator):
        with torch.cuda.amp.autocast():
            imgs = sample["img"].cuda(non_blocking=True)
            otps = sample["msk"].cuda(non_blocking=True)

            res = model(imgs)

            loss1 = combo_loss(res[:, 0, ...], otps[:, 0, ...]) + combo_loss(res[:, 3, ...], otps[:, 3, ...]) + combo_loss(res[:, 6, ...], otps[:, 6, ...])
            loss2 = combo_loss(res[:, 1, ...], otps[:, 1, ...]) + combo_loss(res[:, 4, ...], otps[:, 4, ...])
            loss3 = combo_loss(res[:, 2, ...], otps[:, 2, ...]) + combo_loss(res[:, 5, ...], otps[:, 5, ...])
            
            loss = 0.3 * (loss1  + loss2 * 0.25 + 0.25 * loss3)

            if current_epoch < 1:
                loss = loss * 0.1 # warm-up to try fix explosion

            with torch.no_grad():
                _probs = torch.sigmoid(res[:, 0, ...])
                dice_sc = 1 - dice_round(_probs, otps[:, 0, ...])
                _probs = torch.sigmoid(res[:, 1, ...])
                dice_sc2 = 1 - dice_round(_probs, otps[:, 1, ...])
                _probs = torch.sigmoid(res[:, 2, ...])
                dice_sc3 = 1 - dice_round(_probs, otps[:, 2, ...])

        if i % 5 == 0:
            if args.distributed:
                reduced_loss1 = reduce_tensor(loss1.data)
                reduced_loss2 = reduce_tensor(loss2.data)
                reduced_loss3 = reduce_tensor(loss3.data)
                reduced_dice = reduce_tensor(dice_sc)
                reduced_dice2 = reduce_tensor(dice_sc2)
                reduced_dice3 = reduce_tensor(dice_sc3)
            else:
                reduced_loss1 = loss1.data
                reduced_loss2 = loss2.data
                reduced_loss3 = loss3.data
                reduced_dice = dice_sc
                reduced_dice2 = dice_sc
                reduced_dice3 = dice_sc

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss1), imgs.size(0))
            losses2.update(to_python_float(reduced_loss2), imgs.size(0))
            losses3.update(to_python_float(reduced_loss3), imgs.size(0))
            dices.update(reduced_dice, imgs.size(0)) 
            dices2.update(reduced_dice2, imgs.size(0)) 
            dices3.update(reduced_dice3, imgs.size(0)) 

        if args.local_rank == 0:
            iterator.set_description(
                "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); Loss2 {loss2.val:.4f} ({loss2.avg:.4f}); Loss3 {loss3.val:.4f} ({loss3.avg:.4f}); dice {dices.val:.4f} ({dices.avg:.4f}); dice2 {dices2.val:.4f} ({dices2.avg:.4f}); dice3 {dices3.val:.4f} ({dices3.avg:.4f});".format(
                    current_epoch, _lr, loss=losses, loss2=losses2, loss3=losses3, dices=dices, dices2=dices2, dices3=dices3))


        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999) # clip harder to prevent explosion!
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()


    if args.local_rank == 0:
        print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}; Loss2 {loss2.avg:.4f}; Loss3 {loss3.avg:.4f}; Dice {dices.avg:.4f}; Dice2 {dices2.avg:.4f}; Dice3 {dices3.avg:.4f}".format(
                    current_epoch, _lr, loss=losses, loss2=losses2, loss3=losses3, dices=dices, dices2=dices2, dices3=dices3))

            

if __name__ == '__main__':
    t0 = timeit.default_timer()


    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)

        pg = dist.init_process_group(backend="nccl",
                        rank=args.local_rank) #,world_size=args.world_size

        args.world_size = dist.get_world_size()



    makedirs(models_folder, exist_ok=True)
    
    fold = 1 #int(sys.argv[1])
    
    # train_ids, val_ids = train_test_split(all_ids, test_size=0.05, random_state=fold)

    all_files = []
    val_files = []
    used_aois = set([])
    aoi_list = {}
    for f in sorted(listdir(masks_dir)):
        aoi = f.split('mosaic_')[1].split('.png')[0]
        fn = f.split('.')[0]
        all_files.append(fn) #({'fn': fn, 'aoi': aoi, 'date': 100 * int(f.split('_')[2]) + int(f.split('_')[3])})
        if aoi not in used_aois:
            used_aois.add(aoi)
            val_files.append(fn)
            aoi_list[aoi] = []
        aoi_list[aoi].append(fn)


    train_files = np.asarray(all_files)
    val_files = np.asarray(val_files)

    cudnn.benchmark = True

    batch_size = 2
    val_batch = 1

    best_snapshot_name = 'eff6_4k_{0}_best_double_0'.format(fold)
    last_snapshot_name = 'eff6_4k_{0}_last_double_0'.format(fold)

    np.random.seed(1112)
    random.seed(1112)
    
    steps_per_epoch = len(train_files) // batch_size
    validation_steps = len(val_files) // val_batch

    if args.local_rank == 0:
        print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

    data_train = TrainDatasetDouble(train_files, train_dir, masks_dir, aoi_list, crop_size=704, scale=4, tune=True)
    data_val = ValDataset(val_files, train_dir, masks_dir, scale=4)

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(data_val)

    # collate_fn = lambda b: fast_collate(b, memory_format)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=6, shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler) #, collate_fn=collate_fn
    val_data_loader = DataLoader(data_val, batch_size=val_batch, num_workers=6, shuffle=False, pin_memory=True, sampler=val_sampler)

    model = EfficientNet_Unet_Double(name='efficientnet-b6', pretrained=False)
    
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model, pg)

    model = model.cuda()


    params = model.parameters()

    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0001},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ] 

    optimizer = AdamW(params, lr=1e-4) # SGD(params, lr=0.01, momentum=0.9) #, weight_decay=1e-4) #AdamW(params, lr=5e-4) #SGD(params, lr=0.04, momentum=0.9, weight_decay=1e-4) #params) #, nesterov=True #AdamW(params, lr=1e-4)  #SGD(params, lr=0.001, momentum=0.9, weight_decay=1e-7, nesterov=True) #AdamW(params, lr=1e-3, weight_decay=0.1) #Novograd(params, lr=4e-4, weight_decay=2e-5) #AdamW(params, lr=1e-4, weight_decay=0.15)

    # model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    loss_scaler = torch.cuda.amp.GradScaler()

    
    snap_to_load = 'eff6_4k_{0}_best_tuned_0'.format(fold)
    if args.local_rank == 0:
        print("=> loading checkpoint '{}'".format(snap_to_load))
    checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
    loaded_dict = checkpoint['state_dict']
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
        if k in loaded_dict and sd[k].size() != loaded_dict[k].size(): # load last layer as well to prevent explosion
            if 'bias' not in k:
                sd[k][:3, :48] = 0.5 * loaded_dict[k]
                sd[k][:3, 48:] = 0.5 * loaded_dict[k]
                sd[k][3:6, :48] = 0.5 * loaded_dict[k]
                sd[k][3:6, 48:] = 0.5 * loaded_dict[k]
                sd[k][6:, :48] = 0.5 * loaded_dict[k][:1]
                sd[k][6:, 48:] = 0.5 * loaded_dict[k][:1]
            else:
                sd[k][:3] = loaded_dict[k]
                sd[k][3:6] = loaded_dict[k]
                sd[k][6:] = loaded_dict[k][:1]

            print(k, sd[k].size(), loaded_dict[k].size())
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    if args.local_rank == 0:
        print("loaded checkpoint '{}' (epoch {}, best_score {})"
            .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))


    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=False, threshold=0.00001, threshold_mode='abs', cooldown=0, min_lr=1e-07, eps=1e-07) # MultiStepLR(optimizer, milestones=[4, 8, 12, 16, 25, 30, 65, 90, 100, 110, 130, 150, 170, 180, 190], gamma=0.5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[7, 12, 15, 21, 26, 29, 40, 50], gamma=0.5)

    combo_loss = ComboLoss({'dice': 1.0, 'focal': 2.0}, per_image=True).cuda()

    best_score = 0
    for epoch in range(27):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_epoch(epoch, combo_loss, model, optimizer, loss_scaler, train_data_loader)
        scheduler.step()

        # if epoch % 5 == 0:
        #     torch.cuda.empty_cache()
        #     best_score, _sc = evaluate_val(val_data_loader, best_score, model, best_snapshot_name, epoch)

        if args.local_rank == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_score': best_score,
            }, path.join(models_folder, last_snapshot_name))

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))