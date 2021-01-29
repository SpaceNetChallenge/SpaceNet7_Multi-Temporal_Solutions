import json
import os.path
from glob import glob

import albumentations as albu
import torch.utils.data

from ..transforms import get_augmentation, get_preprocess
from ..utils import get_subdirs, train_list_filename, val_list_filename
from .spacenet7 import SpaceNet7Dataset, SpaceNet7TestDataset


def get_dataloader(config, is_train):
    """[summary]

    Args:
        config ([type]): [description]
        is_train (bool): [description]

    Returns:
        [type]: [description]
    """
    # get path to train/val json files
    split_id = config.INPUT.TRAIN_VAL_SPLIT_ID
    train_list = os.path.join(config.INPUT.TRAIN_VAL_SPLIT_DIR,
                              train_list_filename(split_id))
    val_list = os.path.join(config.INPUT.TRAIN_VAL_SPLIT_DIR,
                            val_list_filename(split_id))

    preprocessing = get_preprocess(config, is_test=False)
    augmentation = get_augmentation(config, is_train=is_train)

    if is_train:
        data_list_path = train_list
        batch_size = config.DATALOADER.TRAIN_BATCH_SIZE
        num_workers = config.DATALOADER.TRAIN_NUM_WORKERS
        shuffle = config.DATALOADER.TRAIN_SHUFFLE
    else:
        data_list_path = val_list
        batch_size = config.DATALOADER.VAL_BATCH_SIZE
        num_workers = config.DATALOADER.VAL_NUM_WORKERS
        shuffle = False

    with open(data_list_path) as f:
        data_list = json.load(f)

    dataset = SpaceNet7Dataset(config,
                               data_list,
                               augmentation=augmentation,
                               preprocessing=preprocessing)

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers)


def get_test_dataloader(config,
                        tta_resize_wh=None,
                        tta_hflip=False,
                        tta_vflip=False):
    """[summary]

    Args:
        config ([type]): [description]
        tta_resize_wh ([type], optional): [description]. Defaults to None.
        tta_hflip (bool, optional): [description]. Defaults to False.
        tta_vflip (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    preprocessing = get_preprocess(config, is_test=True)
    augmentation = get_augmentation(config,
                                    is_train=False,
                                    tta_resize_wh=tta_resize_wh,
                                    tta_hflip=tta_hflip,
                                    tta_vflip=tta_vflip)

    # get full paths to image files
    if config.TEST_TO_VAL:
        # use val split for test.
        data_list_path = os.path.join(
            config.INPUT.TRAIN_VAL_SPLIT_DIR,
            val_list_filename(config.INPUT.TRAIN_VAL_SPLIT_ID))
        with open(data_list_path) as f:
            data_list = json.load(f)

    else:
        # use test data for test (default).
        data_list = []
        test_dir = config.INPUT.TEST_DIR
        aois = get_subdirs(test_dir)
        for aoi in aois:
            image_paths = glob(
                os.path.join(test_dir, aoi, 'images_masked/*.tif'))
            image_paths.sort()

            N = len(image_paths)
            for i in range(N):
                # current frame
                image_path = image_paths[i]
                # previous frame
                image_prev_path = image_paths[0] if i == 0 \
                    else image_paths[i - 1]
                # next frame
                image_next_path = image_paths[N - 1] if i == N - 1 \
                    else image_paths[i + 1]
                # append them to data list
                data_list.append({
                    'image_masked': image_path,
                    'image_masked_prev': image_prev_path,
                    'image_masked_next': image_next_path
                })

    dataset = SpaceNet7TestDataset(config,
                                   data_list,
                                   augmentation=augmentation,
                                   preprocessing=preprocessing)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.DATALOADER.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATALOADER.TEST_NUM_WORKERS,
    )
