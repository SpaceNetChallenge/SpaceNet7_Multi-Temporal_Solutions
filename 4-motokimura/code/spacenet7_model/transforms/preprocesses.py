import functools

import numpy as np

import albumentations as albu
import torch


def get_spacenet7_preprocess(config, is_test):
    """[summary]

    Args:
        config ([type]): [description]
        is_test (bool): [description]

    Returns:
        [type]: [description]
    """
    in_channels = config.MODEL.IN_CHANNELS
    assert in_channels in [3, 4]

    if in_channels == 3:
        mean = np.array([0.485, 0.456, 0.406
                         ]) * 255  # imagenet mean in RGB order
        std = np.array([0.229, 0.224, 0.225
                        ]) * 255  # imagenet std in RGB order
    else:
        mean = np.array([0.485, 0.456, 0.406, 0.5
                         ]) * 255  # imagenet mean in RGB order + alpha mean
        std = np.array([0.229, 0.224, 0.225, 0.5
                        ]) * 255  # imagenet std in RGB order + aplha std

    # tile mean and std if concat previous/next frames
    N_tile = 1
    if config.INPUT.CONCAT_PREV_FRAME:
        N_tile += 1
    if config.INPUT.CONCAT_NEXT_FRAME:
        N_tile += 1
    mean = np.tile(mean, N_tile)
    std = np.tile(std, N_tile)

    mean = mean[np.newaxis, np.newaxis, :]
    std = std[np.newaxis, np.newaxis, :]

    if is_test:
        to_tensor = albu.Lambda(image=functools.partial(_to_tensor))
    else:
        to_tensor = albu.Lambda(image=functools.partial(_to_tensor),
                                mask=functools.partial(_to_tensor))

    preprocess = [
        albu.Lambda(
            image=functools.partial(_normalize_image, mean=mean, std=std)),
        to_tensor,
    ]
    return albu.Compose(preprocess)


def _normalize_image(image, mean, std, **kwargs):
    """[summary]

    Args:
        image ([type]): [description]
        mean ([type]): [description]
        std ([type]): [description]

    Returns:
        [type]: [description]
    """
    normalized = image.astype('float32')
    normalized = (image - mean) / std
    return normalized


def _to_tensor(x, **kwargs):
    """[summary]

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    x = x.transpose(2, 0, 1).astype('float32')
    return torch.from_numpy(x)
