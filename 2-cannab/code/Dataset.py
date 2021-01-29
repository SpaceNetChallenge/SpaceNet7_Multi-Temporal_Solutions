import torch
from torch.utils.data import Dataset
# from torch.utils.data._utils.collate import default_collate

import numpy as np
import random

import cv2
from skimage import io

# try:
#     np.random.bit_generator = np.random._bit_generator
# except:
#     pass

from imgaug import augmenters as iaa

from utils import *

from os import path, listdir


# import matplotlib.pyplot as plt
# import seaborn as sns


class TrainDataset(Dataset):
    def __init__(self, train_files, img_dir, msk_dir, crop_size=512, scale=3, tune=False):
        super().__init__()
        self.train_files = train_files
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.crop_size = crop_size
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
        self.scale = scale
        self.tune = tune

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, idx):
        try_again = True
        img = None
        msk = None
        img_id = -1
        while try_again:
            try_again = False
            img_id = self.train_files[idx]

            try:
                # img = cv2.imread(path.join(self.img_dir, '{}.png'.format(img_id)), cv2.IMREAD_COLOR)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #.astype(np.float32)
                
                _aoi = img_id.split('mosaic_')[1]
                img = io.imread(path.join(self.img_dir, _aoi, 'images_masked', '{}.tif'.format(img_id)))

                h, w = (np.asarray(img.shape[:2]) * self.scale).astype('int32')
                img = cv2.resize(img[..., :3], (w, h), interpolation=cv2.INTER_NEAREST) #[..., ::-1]

                msk = cv2.imread(path.join(self.msk_dir, '{}.png'.format(img_id)), cv2.IMREAD_COLOR)
                    
            except Exception as ex:
                try_again = True
                print('Exception occured: {}. File: {}'.format(ex, img_id))
                
            if try_again:
                idx = random.randint(0, len(self.train_files) - 1)

        
        k = random.randrange(4)
        img = np.rot90(img, k=k)
        msk = np.rot90(msk, k=k)
        

        if random.random() > 0.5:
            img = img[::-1, ...]
            msk = msk[::-1, ...]

        if random.random() > 0.5:
            img = img[:, ::-1, ...]
            msk = msk[:, ::-1, ...]

        _p = 0.4
        if self.tune:
            _p = 0.9
        if random.random() > _p:
            shift_pnt = (random.randint(-400, 400), random.randint(-400, 400))
            img = shift_image(img, shift_pnt)
            msk = shift_image(msk, shift_pnt)

        _p = 0.4
        if self.tune:
            _p = 0.8
        if random.random() > _p:
            rot_pnt =  (img.shape[0] // 2 + random.randint(-400, 400), img.shape[1] // 2 + random.randint(-400, 400))
            scale = 1
            if random.random() > 0.7:
                scale = random.normalvariate(1.0, 0.15)
            angle = random.randint(0, 45) - 23
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                msk = rotate_image(msk, angle, scale, rot_pnt)
                

        crop_size = self.crop_size

        x0 = random.randint(0, img.shape[1] - crop_size)
        y0 = random.randint(0, img.shape[0] - crop_size)

        b_sc = 0
        for _try in range(4):
            _x0 = random.randint(0, img.shape[1] - crop_size)
            _y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = msk[_y0:_y0+crop_size, _x0:_x0+crop_size, :].sum()
            if _sc > b_sc:
                b_sc = _sc
                x0 = _x0
                y0 = _y0
                if self.tune:
                    break

        img = img[y0:y0+crop_size, x0:x0+crop_size, :]
        msk = msk[y0:y0+crop_size, x0:x0+crop_size, :]


        _p = 0.95
        if self.tune:
            _p = 0.98
        if random.random() > 0.5:
            if random.random() > _p:
                img = clahe(img)
            elif random.random() > _p:
                img = gauss_noise(img)
            elif random.random() > _p:
                img = cv2.blur(img, (3, 3))
        else:
            if random.random() > _p:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > _p:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > _p:
                img = contrast(img, 0.9 + random.random() * 0.2)


        if random.random() > 0.99:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)


        msk = (msk > 127) * 1

        img = preprocess_inputs_rgb(img)

        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        
        msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).long()

        sample = {'img': img, 'msk': msk, 'img_id': img_id}

        return sample



class ValDataset(Dataset):
    def __init__(self, val_files, img_dir, msk_dir, scale=3):
        super().__init__()
        self.val_files = val_files
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.scale = scale

    def __len__(self):
        if self.scale == 4:
            return len(self.val_files) * 2
        return len(self.val_files)

    def __getitem__(self, idx):
        if self.scale == 4:
            img_id = self.val_files[idx // 2]
        else:
            img_id = self.val_files[idx]

        # img = cv2.imread(path.join(self.img_dir, '{}.png'.format(img_id)), cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #.astype(np.float32)
        
        _aoi = img_id.split('mosaic_')[1]
        img = io.imread(path.join(self.img_dir, _aoi, 'images_masked', '{}.tif'.format(img_id)))

        h, w = (np.asarray(img.shape[:2]) * self.scale).astype('int32')
        img = cv2.resize(img[..., :3], (w, h), interpolation=cv2.INTER_NEAREST) #[..., ::-1]

        msk = cv2.imread(path.join(self.msk_dir, '{}.png'.format(img_id)), cv2.IMREAD_COLOR)
        
        h0, w0 = img.shape[:2]

        _h = 0
        _w = 0
        if h0 % 32 != 0:
            _h = 32 - (h0 % 32)
        if w0 % 32 != 0:
            _w = 32 - (w0 % 32)
        
        if _h != 0 or _w != 0:
            img = np.pad(img, ((0, _h), (0, _w), (0, 0)))
            msk = np.pad(msk, ((0, _h), (0, _w), (0, 0)))

        if self.scale == 4:
            if idx % 2 == 0:
                img = img[:2048, ...]
                msk = msk[:2048, ...]
            else:
                img = img[2048:, ...]
                msk = msk[2048:, ...]

        msk = (msk > 127) * 1

        img = preprocess_inputs_rgb(img)

        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        
        msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).long()

        sample = {'img': img, 'msk': msk, 'img_id': img_id}

        return sample


class TestDataset(Dataset):
    def __init__(self, test_files, scale=3):
        super().__init__()
        self.test_files = test_files
        self.scale = scale

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, idx):
        img_id = self.test_files[idx]

        img = io.imread(img_id)
    
        h, w = (np.asarray(img.shape[:2]) * self.scale).astype('int32')
        
        img = cv2.resize(img[..., :3], (w, h), interpolation=cv2.INTER_NEAREST) #INTER_LANCZOS4 [..., ::-1]

        h0, w0 = img.shape[:2]

        _h = 0
        _w = 0
        if h0 % 32 != 0:
            _h = 32 - (h0 % 32)
        if w0 % 32 != 0:
            _w = 32 - (w0 % 32)
        
        if _h != 0 or _w != 0:
            img = np.pad(img, ((0, _h), (0, _w), (0, 0)))

        img = preprocess_inputs_rgb(img)
        img0 = img.transpose((2, 0, 1)) # torch.from_numpy(.copy()).float()

        # img1 = torch.from_numpy(img[::-1, :, :].transpose((2, 0, 1)).copy()).float()
        # img2 = torch.from_numpy(img[:, ::-1, :].transpose((2, 0, 1)).copy()).float()
        # img3 = torch.from_numpy(img[::-1, ::-1, :].transpose((2, 0, 1)).copy()).float()
        
        sample = {'img': img0, 'img_id': img_id} #, 'img1': img1, 'img2': img2, 'img3': img3,

        return sample




class TrainDatasetDouble(Dataset):
    def __init__(self, train_files, img_dir, msk_dir, aoi_list, crop_size=512, scale=3, tune=False):
        super().__init__()
        self.train_files = train_files
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.crop_size = crop_size
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
        self.scale = scale
        self.tune = tune
        self.aoi_list = aoi_list

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, idx):
        try_again = True
        img = None
        msk = None
        img_id = -1
        while try_again:
            try_again = False
            img_id = self.train_files[idx]

            try:
                # img = cv2.imread(path.join(self.img_dir, '{}.png'.format(img_id)), cv2.IMREAD_COLOR)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #.astype(np.float32)
                
                _aoi = img_id.split('mosaic_')[1]

                img_id2 = img_id
                while img_id2 == img_id:
                    img_id2 = random.choice(self.aoi_list[_aoi])

                fns = sorted([img_id, img_id2])

                img = io.imread(path.join(self.img_dir, _aoi, 'images_masked', '{}.tif'.format(fns[0])))
                h, w = (np.asarray(img.shape[:2]) * self.scale).astype('int32')
                img = cv2.resize(img[..., :3], (w, h), interpolation=cv2.INTER_NEAREST) #[..., ::-1]
                msk = cv2.imread(path.join(self.msk_dir, '{}.png'.format(fns[0])), cv2.IMREAD_COLOR)
                    
                img2 = io.imread(path.join(self.img_dir, _aoi, 'images_masked', '{}.tif'.format(fns[1])))
                h, w = (np.asarray(img2.shape[:2]) * self.scale).astype('int32')
                img2 = cv2.resize(img2[..., :3], (w, h), interpolation=cv2.INTER_NEAREST) #[..., ::-1]
                msk2 = cv2.imread(path.join(self.msk_dir, '{}.png'.format(fns[1])), cv2.IMREAD_COLOR)

            except Exception as ex:
                try_again = True
                print('Exception occured: {}. File: {}'.format(ex, img_id))
                
            if try_again:
                idx = random.randint(0, len(self.train_files) - 1)

        
        k = random.randrange(4)
        img = np.rot90(img, k=k)
        msk = np.rot90(msk, k=k)
        img2 = np.rot90(img2, k=k)
        msk2 = np.rot90(msk2, k=k)
        

        if random.random() > 0.5:
            img = img[::-1, ...]
            msk = msk[::-1, ...]
            img2 = img2[::-1, ...]
            msk2 = msk2[::-1, ...]

        if random.random() > 0.5:
            img = img[:, ::-1, ...]
            msk = msk[:, ::-1, ...]
            img2 = img2[:, ::-1, ...]
            msk2 = msk2[:, ::-1, ...]

        _p = 0.4
        if self.tune:
            _p = 0.9
        if random.random() > _p:
            shift_pnt = (random.randint(-400, 400), random.randint(-400, 400))
            img = shift_image(img, shift_pnt)
            msk = shift_image(msk, shift_pnt)
            img2 = shift_image(img2, shift_pnt)
            msk2 = shift_image(msk2, shift_pnt)

        _p = 0.4
        if self.tune:
            _p = 0.8
        if random.random() > _p:
            rot_pnt =  (img.shape[0] // 2 + random.randint(-400, 400), img.shape[1] // 2 + random.randint(-400, 400))
            scale = 1
            if random.random() > 0.7:
                scale = random.normalvariate(1.0, 0.15)
            angle = random.randint(0, 45) - 23
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                msk = rotate_image(msk, angle, scale, rot_pnt)
                img2 = rotate_image(img2, angle, scale, rot_pnt)
                msk2 = rotate_image(msk2, angle, scale, rot_pnt)
                

        crop_size = self.crop_size

        x0 = random.randint(0, img.shape[1] - crop_size)
        y0 = random.randint(0, img.shape[0] - crop_size)

        b_sc = 0
        b_sc2 = 0
        msk_diff = 1 * (msk2[..., 0] > 127) - 1 * (msk[..., 0] > 127)
        msk_diff[msk_diff < 0] = 0

        for _try in range(6):
            _x0 = random.randint(0, img.shape[1] - crop_size)
            _y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = msk[_y0:_y0+crop_size, _x0:_x0+crop_size, :].sum()
            _sc2 = msk_diff[_y0:_y0+crop_size, _x0:_x0+crop_size].sum()
            if (_sc2 > b_sc2) or ((_sc2 == b_sc2) and (_sc > b_sc)):
                b_sc = _sc
                b_sc2 = _sc2
                x0 = _x0
                y0 = _y0
                # if self.tune:
                #     break

        img = img[y0:y0+crop_size, x0:x0+crop_size, :]
        msk = msk[y0:y0+crop_size, x0:x0+crop_size, :]
        img2 = img2[y0:y0+crop_size, x0:x0+crop_size, :]
        msk2 = msk2[y0:y0+crop_size, x0:x0+crop_size, :]
        msk_diff = msk_diff[y0:y0+crop_size, x0:x0+crop_size, np.newaxis] > 0

        _p = 0.95
        if self.tune:
            _p = 0.98
        if random.random() > 0.5:
            if random.random() > _p:
                img = clahe(img)
                img = clahe(img)
            elif random.random() > _p:
                img = gauss_noise(img)
            elif random.random() > _p:
                img = cv2.blur(img, (3, 3))
        else:
            if random.random() > _p:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > _p:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > _p:
                img = contrast(img, 0.9 + random.random() * 0.2)


        if random.random() > 0.5:
            if random.random() > _p:
                img2 = clahe(img2)
                img2 = clahe(img2)
            elif random.random() > _p:
                img2 = gauss_noise(img2)
            elif random.random() > _p:
                img2 = cv2.blur(img2, (3, 3))
        else:
            if random.random() > _p:
                img2 = saturation(img2, 0.9 + random.random() * 0.2)
            elif random.random() > _p:
                img2 = brightness(img2, 0.9 + random.random() * 0.2)
            elif random.random() > _p:
                img2 = contrast(img2, 0.9 + random.random() * 0.2)


        if random.random() > 0.99:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)
            img2 = el_det.augment_image(img2)


        msk = np.concatenate([msk, msk2], axis=2)
        msk = (msk > 127) * 1
        msk = np.concatenate([msk, msk_diff], axis=2)

        img = preprocess_inputs_rgb(img)
        img2 = preprocess_inputs_rgb(img2)

        img = np.concatenate([img, img2], axis=2)

        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        
        msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).long()

        sample = {'img': img, 'msk': msk, 'img_id': img_id}

        return sample


class TestDatasetDouble(Dataset):
    def __init__(self, test_files, scale=3):
        super().__init__()
        self.test_files = test_files
        self.scale = scale

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, idx):
        img_id, img_id2 = self.test_files[idx]

        img = io.imread(img_id)
        h, w = (np.asarray(img.shape[:2]) * self.scale).astype('int32')
        img = cv2.resize(img[..., :3], (w, h), interpolation=cv2.INTER_NEAREST)
        h0, w0 = img.shape[:2]
        _h = 0
        _w = 0
        if h0 % 32 != 0:
            _h = 32 - (h0 % 32)
        if w0 % 32 != 0:
            _w = 32 - (w0 % 32)
        if _h != 0 or _w != 0:
            img = np.pad(img, ((0, _h), (0, _w), (0, 0)))

        img2 = io.imread(img_id2)
        h, w = (np.asarray(img2.shape[:2]) * self.scale).astype('int32')
        img2 = cv2.resize(img2[..., :3], (w, h), interpolation=cv2.INTER_NEAREST)
        h0, w0 = img2.shape[:2]
        _h = 0
        _w = 0
        if h0 % 32 != 0:
            _h = 32 - (h0 % 32)
        if w0 % 32 != 0:
            _w = 32 - (w0 % 32)
        if _h != 0 or _w != 0:
            img2 = np.pad(img2, ((0, _h), (0, _w), (0, 0)))


        img = preprocess_inputs_rgb(img)
        img2 = preprocess_inputs_rgb(img2)

        img = np.concatenate([img, img2], axis=2)

        img0 = img.transpose((2, 0, 1)) 
        img0 = torch.from_numpy(img0.copy()).float()

        sample = {'img': img0, 'img_id': img_id, 'img_id2': img_id2}

        return sample