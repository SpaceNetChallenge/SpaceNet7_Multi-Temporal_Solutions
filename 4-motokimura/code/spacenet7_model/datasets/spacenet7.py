import json

import numpy as np

from skimage import io
from torch.utils.data import Dataset


class SpaceNet7Dataset(Dataset):
    CLASSES = [
        'building_footprint',  # 1st (R) channel in mask
        'building_boundary',  # 2nd (G) channel in mask
        'building_contact',  # 3rd (B) channel in mask
    ]

    def __init__(self,
                 config,
                 data_list,
                 augmentation=None,
                 preprocessing=None):
        """[summary]

        Args:
            config ([type]): [description]
            data_list ([type]): [description]
            augmentation ([type], optional): [description]. Defaults to None.
            preprocessing ([type], optional): [description]. Defaults to None.
        """
        # generate full path to image/label files
        self.image_paths, self.mask_paths = [], []
        for data in data_list:
            self.image_paths.append(data['image_masked'])
            self.mask_paths.append(data['building_mask'])

        # path to previous frame
        if config.INPUT.CONCAT_PREV_FRAME:
            self.image_prev_paths = []
            for data in data_list:
                self.image_prev_paths.append(data['image_masked_prev'])

        # path to next frame
        if config.INPUT.CONCAT_NEXT_FRAME:
            self.image_next_paths = []
            for data in data_list:
                self.image_next_paths.append(data['image_masked_next'])

        # convert str names to class values on masks
        classes = config.INPUT.CLASSES
        if not classes:
            # if classes is empty, use all classes
            classes = self.CLASSES
        self.class_values = [self.CLASSES.index(c) for c in classes]

        self.device = config.MODEL.DEVICE

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.in_channels = config.MODEL.IN_CHANNELS
        assert self.in_channels in [3, 4]

        self.concat_prev_frame = config.INPUT.CONCAT_PREV_FRAME
        self.concat_next_frame = config.INPUT.CONCAT_NEXT_FRAME

    def __getitem__(self, i):
        """[summary]

        Args:
            i ([type]): [description]

        Returns:
            [type]: [description]
        """
        image = io.imread(self.image_paths[i])
        mask = io.imread(self.mask_paths[i])

        if self.in_channels == 3:
            # remove alpha channel
            image = image[:, :, :3]
        _, _, c = image.shape
        assert c == self.in_channels

        # concat previous frame
        if self.concat_prev_frame:
            image_prev = io.imread(self.image_prev_paths[i])
            if self.in_channels == 3:
                image_prev = image_prev[:, :, :3]
            _, _, c = image_prev.shape
            assert c == self.in_channels
            image = np.concatenate([image_prev, image], axis=2)

        # concat next frame
        if self.concat_next_frame:
            image_next = io.imread(self.image_next_paths[i])
            if self.in_channels == 3:
                image_next = image_next[:, :, :3]
            _, _, c = image_next.shape
            assert c == self.in_channels
            image = np.concatenate([image, image_next], axis=2)

        # extract certain classes from mask
        masks = [(mask[:, :, v] > 0) for v in self.class_values]
        mask = np.stack(masks,
                        axis=-1).astype('float')  # XXX: multi class setting.

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.image_paths)


class SpaceNet7TestDataset(Dataset):
    def __init__(self,
                 config,
                 data_list,
                 augmentation=None,
                 preprocessing=None):
        """[summary]

        Args:
            config ([type]): [description]
            data_list ([type]): [description]
            augmentation ([type], optional): [description]. Defaults to None.
            preprocessing ([type], optional): [description]. Defaults to None.
        """
        # generate full path to image/label files
        self.image_paths = []
        for data in data_list:
            self.image_paths.append(data['image_masked'])

        # path to previous frame
        if config.INPUT.CONCAT_PREV_FRAME:
            self.image_prev_paths = []
            for data in data_list:
                self.image_prev_paths.append(data['image_masked_prev'])

        # path to next frame
        if config.INPUT.CONCAT_NEXT_FRAME:
            self.image_next_paths = []
            for data in data_list:
                self.image_next_paths.append(data['image_masked_next'])

        self.device = config.MODEL.DEVICE

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.in_channels = config.MODEL.IN_CHANNELS
        assert self.in_channels in [3, 4]

        self.concat_prev_frame = config.INPUT.CONCAT_PREV_FRAME
        self.concat_next_frame = config.INPUT.CONCAT_NEXT_FRAME

    def __getitem__(self, i):
        """[summary]

        Args:
            i ([type]): [description]

        Returns:
            [type]: [description]
        """
        image_path = self.image_paths[i]
        image = io.imread(image_path)

        if self.in_channels == 3:
            # remove alpha channel
            image = image[:, :, :3]
        _, _, c = image.shape
        assert c == self.in_channels

        # concat previous frame
        if self.concat_prev_frame:
            image_prev = io.imread(self.image_prev_paths[i])
            if self.in_channels == 3:
                image_prev = image_prev[:, :, :3]
            _, _, c = image_prev.shape
            assert c == self.in_channels
            image = np.concatenate([image_prev, image], axis=2)

        # concat next frame
        if self.concat_next_frame:
            image_next = io.imread(self.image_next_paths[i])
            if self.in_channels == 3:
                image_next = image_next[:, :, :3]
            _, _, c = image_next.shape
            assert c == self.in_channels
            image = np.concatenate([image, image_next], axis=2)

        original_shape = image.shape

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return {
            'image': image,
            'image_path': image_path,
            'original_shape': original_shape,
        }

    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.image_paths)
