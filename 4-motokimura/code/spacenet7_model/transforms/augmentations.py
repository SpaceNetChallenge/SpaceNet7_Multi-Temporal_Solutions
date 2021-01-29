import functools
import random

import numpy as np

import albumentations as albu


def get_spacenet7_augmentation(config,
                               is_train,
                               tta_resize_wh=None,
                               tta_hflip=False,
                               tta_vflip=False):
    """[summary]

    Args:
        config ([type]): [description]
        is_train (bool): [description]
        tta_resize_wh ([type], optional): [description]. Defaults to None.
        tta_hflip (bool, optional): [description]. Defaults to False.
        tta_vflip (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    size_scale = config.TRANSFORM.SIZE_SCALE

    if is_train:
        assert tta_resize_wh is None
        assert not tta_hflip
        assert not tta_vflip

        # size after cropping
        base_width = config.TRANSFORM.TRAIN_RANDOM_CROP_SIZE[0]
        base_height = config.TRANSFORM.TRAIN_RANDOM_CROP_SIZE[1]

        augmentation = [
            # random flip
            albu.HorizontalFlip(p=config.TRANSFORM.TRAIN_HORIZONTAL_FLIP_PROB),
            albu.VerticalFlip(p=config.TRANSFORM.TRAIN_VERTICAL_FLIP_PROB),
            # random rotate
            albu.ShiftScaleRotate(
                scale_limit=0.0,
                rotate_limit=config.TRANSFORM.TRAIN_RANDOM_ROTATE_DEG,
                shift_limit=0.0,
                p=config.TRANSFORM.TRAIN_RANDOM_ROTATE_PROB,
                border_mode=0),
            # random crop
            albu.RandomCrop(width=base_width,
                            height=base_height,
                            always_apply=True),
            # random brightness
            albu.Lambda(image=functools.partial(
                _random_brightness,
                brightness_std=config.TRANSFORM.TRAIN_RANDOM_BRIGHTNESS_STD,
                p=config.TRANSFORM.TRAIN_RANDOM_BRIGHTNESS_PROB)),
        ]

        resize_width = int(size_scale * base_width)
        resize_height = int(size_scale * base_height)

    else:
        # size after padding
        base_width = config.TRANSFORM.TEST_SIZE[0]
        base_height = config.TRANSFORM.TEST_SIZE[1]

        augmentation = [
            # padding
            albu.PadIfNeeded(min_width=base_width,
                             min_height=base_height,
                             always_apply=True,
                             border_mode=0),
        ]

        # tta flipping
        if tta_hflip:
            augmentation.append(albu.HorizontalFlip(always_apply=True))
        if tta_vflip:
            augmentation.append(albu.VerticalFlip(always_apply=True))

        # tta size jitter
        if tta_resize_wh is None:
            tta_width, tta_height = base_width, base_height
        else:
            tta_width, tta_height = tta_resize_wh
        resize_width = int(size_scale * tta_width)
        resize_height = int(size_scale * tta_height)

    if (base_width != resize_width) or (base_height != resize_height):
        # append resizing
        augmentation.append(
            albu.Resize(width=resize_width,
                        height=resize_height,
                        always_apply=True))

    return albu.Compose(augmentation)


def _random_brightness(image, brightness_std, p=1.0, **kwargs):
    """[summary]

    Args:
        image ([type]): [description]
        brightness_std ([type]): [description]
        p (float, optional): [description]. Defaults to 1.0.

    Returns:
        [type]: [description]
    """
    if brightness_std <= 0:
        return image

    if random.random() >= p:
        return image

    gauss = np.random.normal(0, brightness_std)
    brightness_noise = gauss * image
    noised = image + brightness_noise

    return noised
