import os.path

import torch


def checkpoint_epoch_filename(epoch):
    """[summary]

    Args:
        epoch ([type]): [description]

    Returns:
        [type]: [description]
    """
    assert 0 <= epoch <= 9999
    return f'checkpoint_{epoch:04d}.ckpt'


def checkpoint_latest_filename():
    """[summary]

    Returns:
        [type]: [description]
    """
    return 'checkpoint_latest.ckpt'


def save_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, epoch,
                    best_score):
    """[summary]

    Args:
        checkpoint_path ([type]): [description]
        model ([type]): [description]
        optimizer ([type]): [description]
        lr_scheduler ([type]): [description]
        epoch ([type]): [description]
        best_score ([type]): [description]
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'best_score': best_score
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, epoch,
                    best_score):
    """[summary]

    Args:
        checkpoint_path ([type]): [description]
        model ([type]): [description]
        optimizer ([type]): [description]
        lr_scheduler ([type]): [description]
        epoch ([type]): [description]
        best_score ([type]): [description]

    Returns:
        [type]: [description]
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    epoch = checkpoint['epoch']
    best_score = checkpoint['best_score']

    return model, optimizer, lr_scheduler, epoch, best_score


def checkpoint_exists(checkpoint_dir):
    """[summary]

    Args:
        checkpoint_dir ([type]): [description]

    Returns:
        [type]: [description]
    """
    checkpoint_path = os.path.join(checkpoint_dir,
                                   checkpoint_latest_filename())
    return os.path.exists(checkpoint_path)


def load_latest_checkpoint(checkpoint_dir, model, optimizer, lr_scheduler,
                           epoch, best_score):
    """[summary]

    Args:
        checkpoint_dir ([type]): [description]
        model ([type]): [description]
        optimizer ([type]): [description]
        lr_scheduler ([type]): [description]
        epoch ([type]): [description]
        best_score ([type]): [description]

    Returns:
        [type]: [description]
    """
    if checkpoint_exists(checkpoint_dir):
        print('resume training from the last checkpoint!')
        model, optimizer, lr_scheduler, epoch, best_score = load_checkpoint(
            os.path.join(checkpoint_dir, checkpoint_latest_filename()), model,
            optimizer, lr_scheduler, epoch, best_score)

    return model, optimizer, lr_scheduler, epoch, best_score
