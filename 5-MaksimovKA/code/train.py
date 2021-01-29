import sys
import torch
import os.path as osp
import os
from model import make_model
from losses import get_loss
from optimizers import get_optimizer
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import CheckpointCallback, EarlyStoppingCallback
from callbacks import DiceCallback
from dataset import SemSegDataset
from torch.utils.data import DataLoader
from fire import Fire
from importlib import import_module


def get_config(filename):
    module_name = osp.basename(filename)[:-3]

    config_dir = osp.dirname(filename)

    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    }
    return cfg_dict


def main(config_path,
         gpu='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    config = get_config(config_path)
    model_name = config['model_name']
    val_fold = config['val_fold']
    folds_to_use = config['folds_to_use']
    alias = config['alias']
    log_path = osp.join(config['logs_path'],
                        alias + str(val_fold) + '_' + model_name)

    device = torch.device(config['device'])
    weights = config['weights']
    loss_name = config['loss']
    optimizer_name = config['optimizer']
    lr = config['lr']
    decay = config['decay']
    momentum = config['momentum']
    epochs = config['epochs']
    fp16 = config['fp16']
    n_classes = config['n_classes']
    input_channels = config['input_channels']
    main_metric = config['main_metric']

    best_models_count = config['best_models_count']
    minimize_metric = config['minimize_metric']


    folds_file = config['folds_file']
    train_augs = config['train_augs']
    preprocessing_fn = config['preprocessing_fn']
    limit_files = config['limit_files']
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    num_workers = config['num_workers']
    valid_augs = config['valid_augs']
    val_batch_size = config['val_batch_size']
    multiplier = config['multiplier']

    train_dataset = SemSegDataset(
        mode='train',
        n_classes=n_classes,
        folds_file=folds_file,
        val_fold=val_fold,
        folds_to_use=folds_to_use,
        augmentation=train_augs,
        preprocessing=preprocessing_fn,
        limit_files=limit_files,
        multiplier=multiplier)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers)

    valid_dataset = SemSegDataset(
        mode='valid',
        folds_file=folds_file,
        n_classes=n_classes,
        val_fold=val_fold,
        folds_to_use=folds_to_use,
        augmentation=valid_augs,
        preprocessing=preprocessing_fn,
        limit_files=limit_files)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=val_batch_size,
                              shuffle=False,
                              num_workers=num_workers)

    model = make_model(
        model_name=model_name).to(device)

    loss = get_loss(loss_name=loss_name)
    optimizer = get_optimizer(optimizer_name=optimizer_name,
                              model=model,
                              lr=lr,
                              momentum=momentum,
                              decay=decay)

    if config['scheduler'] == 'steps':
        print('steps lr')
        steps = config['steps']
        step_gamma = config['step_gamma']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=steps, gamma=step_gamma)
    callbacks = []

    dice_callback = DiceCallback()
    callbacks.append(dice_callback)
    callbacks.append(CheckpointCallback(save_n_best=best_models_count))


    runner = SupervisedRunner(device=device)
    loaders = {'train': train_loader, 'valid': valid_loader}

    runner.train(model=model,
                 criterion=loss,
                 optimizer=optimizer,
                 loaders=loaders,
                 scheduler=scheduler,
                 callbacks=callbacks,
                 logdir=log_path,
                 num_epochs=epochs,
                 verbose=True,
                 main_metric=main_metric,
                 minimize_metric=minimize_metric,
                 fp16=fp16
                 )


if __name__ == '__main__':
    Fire(main)