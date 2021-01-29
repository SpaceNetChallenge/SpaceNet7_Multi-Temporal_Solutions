import os
import cv2
import pandas as pd
import numpy as np
import skimage.io
import torch
import albumentations as albu
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

class SemSegDataset(Dataset):
    def __init__(
            self,
            mode='train',
            folds_file='/wdata/folds.csv',
            folds_to_use=(2, 3, 4, 5, 6, 7, 8),
            val_fold=1,
            n_classes=3,
            augmentation=None,
            preprocessing=None,
            limit_files=None,
            multiplier=5

    ):
        folds = pd.read_csv(folds_file, dtype={'image_name': object})

        if mode == 'train':
            folds = folds[folds['fold_number'].isin(folds_to_use)]
        elif mode == 'valid':
            folds = folds[folds['fold_number'] == val_fold]
        else:
            folds = folds['image_name'].tolist()
        self.mode = mode
        if limit_files:
            folds = folds[:limit_files]
        images = folds['image_path'].tolist()
        masks = folds['mask_path'].tolist()
        if multiplier and mode == 'train':
            images = images * multiplier
            masks = masks * multiplier

        self.n_classes = n_classes
        self.masks = masks
        self.images = images

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    @staticmethod
    def _read_img(image_path):
        img = skimage.io.imread(image_path, plugin='tifffile')
        return img

    def __getitem__(self, i):
        image = self._read_img(self.images[i])[:, :, :3]
        mask = self._read_img(self.masks[i])[:, :, :self.n_classes]

        fie_id = self.images[i].split('/')[-1]
        aoi = self.images[i].split('/')[-3]
        aoi_images = sorted([el for el in self.images if el.split('/')[-3] == aoi and  el.split('/')[-1] != fie_id])
        aoi_masks = sorted([el for el in self.masks if el.split('/')[-2] == aoi and  el.split('/')[-1] != fie_id])
        if self.mode == 'train':
            index = random.randint(0, len(aoi_images) - 1)
        else:
            index = 0
        image_path = aoi_images[index]
        mask_path = aoi_masks[index]
        image2 = self._read_img(image_path)[:, :, :3]
        mask2 = self._read_img(mask_path)[:, :, :self.n_classes]
        correct_augs = [albu.PadIfNeeded(image.shape[0], image.shape[1], p=1.0)]
        sample = albu.Compose(correct_augs, p=1)(image=image2,
                                                      mask=mask2)
        image2, mask2 = sample['image'], sample['mask']

        # print(image.shape, mask.shape)
        if self.augmentation:
            target = {}
            target['image2'] = 'image'
            target['mask2'] = 'mask'
            sample = albu.Compose(self.augmentation, p=1, additional_targets=target)(image=image,
                                                          mask=mask,
                                                          image2=image2,
                                                          mask2=mask2)

            image, mask, image2, mask2 = sample['image'], sample['mask'], sample['image2'], sample['mask2']
        if self.preprocessing:
            image = self.preprocessing(image)
            image2 = self.preprocessing(image2)

        mask = mask[...] / 255.0
        mask2 = mask2[...] / 255.0
        mask_change = (np.clip(
            ((np.clip(mask - mask2, 0, 1) + np.clip(mask2 - mask, 0, 1)) > 0).astype(np.uint8),
                                0, 1)).astype(np.uint8) * 255
        mask_change = mask_change[...] / 255.0
        mask = np.concatenate((mask, mask_change), axis=-1)
        image = np.concatenate((image, image2), axis=-1)
        # dont use contours
        mask[:, :, 1] = 0.0
        mask[:, :, 4] = 0.0
        image = np.moveaxis(image, -1, 0)
        mask = np.moveaxis(mask, -1, 0)
        image = torch.as_tensor(image, dtype=torch.float)
        mask = torch.as_tensor(mask, dtype=torch.float)
        return image, mask

    def __len__(self):
        return len(self.images)


class TestSemSegDataset(Dataset):
    def __init__(
            self,
            images_dir='/data/test_public/',
            augmentation=None,
            preprocessing=None,
            limit_files=None

    ):
        self.images_dir = images_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        aois = os.listdir(images_dir)
        all_files = []
        sizes = []
        print('getting sizes infos')
        for aoi in tqdm(aois):
            aoi_path = os.path.join(images_dir, aoi, 'images_masked')
            files = os.listdir(aoi_path)
            files = [os.path.join(aoi_path, el) for el in files]
            sizes += [(self._read_img(el)).shape for el in files]
            all_files += files

        ids = all_files
        if limit_files:
            ids = all_files[:limit_files]
            sizes = sizes[:limit_files]

        self.ids = ids
        self.sizes = sizes

    @staticmethod
    def _read_img(image_path):
        img = skimage.io.imread(image_path, plugin='tifffile')
        return img

    def __getitem__(self, i):

        image_path = os.path.join(self.images_dir, self.ids[i])
        image = self._read_img(image_path)[:, :, :3]
        if self.augmentation:
            sample = albu.Compose(self.augmentation, p=1)(image=image)

            image = sample['image']

        if self.preprocessing:
            image = self.preprocessing(image)

        image = np.moveaxis(image, -1, 0)
        image = torch.as_tensor(image, dtype=torch.float)
        return image

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    dataset = SemSegDataset(mode='train',
                            n_classes=3,
                            augmentation=albu.Compose([albu.PadIfNeeded(1024, 1024)], p=1))
    check_loader = DataLoader(dataset=dataset,
                              batch_size=1,
                              shuffle=False)
    for step, (x, y) in enumerate(check_loader):
        if step > 5000:
            break
        print('step is', step, x.shape, y.shape)