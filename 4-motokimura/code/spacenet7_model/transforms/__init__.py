from .augmentations import get_spacenet7_augmentation
from .preprocesses import get_spacenet7_preprocess


def get_preprocess(config, is_test):
    """[summary]

    Args:
        config ([type]): [description]
        is_test (bool): [description]

    Returns:
        [type]: [description]
    """
    return get_spacenet7_preprocess(config, is_test)


def get_augmentation(config,
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
    return get_spacenet7_augmentation(config,
                                      is_train,
                                      tta_resize_wh=tta_resize_wh,
                                      tta_hflip=tta_hflip,
                                      tta_vflip=tta_vflip)
