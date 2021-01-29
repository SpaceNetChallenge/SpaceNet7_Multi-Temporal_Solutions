import glob
import os
import traceback

import cv2
import numpy as np
import pandas as pd
import skimage.io
import torch
from scipy.ndimage import binary_dilation
from skimage import measure
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize


class SpacenetInstanceDataset(Dataset):
    def __init__(self, data_path, mode, fold=0, folds_csv='folds.csv', transforms=None,
                 normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                 multiplier=1,
                 sigma=6
                 ):
        super().__init__()
        self.data_path = data_path
        self.mode = mode

        df = pd.read_csv(folds_csv)
        self.df = df
        self.normalize = normalize
        self.fold = fold
        if self.mode == "train":
            ids = df[df['fold'] != fold]['id'].tolist()
        else:
            ids = sorted(list(set(df[(df['fold'] == fold)]['id'].tolist())))
        self.transforms = transforms
        self.names = ids
        self.sigma = sigma
        if mode == "train":
            self.names = self.names * multiplier
        print("names ", len(self.names))
        #self.cache = {}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        group = name.split("mosaic_")[-1][:-4]
        img_path = os.path.join(self.data_path, group, "images_masked_split", name + ".tif")

        try:
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            alpha = image[..., 3]
            image = image[..., :3]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask_path = os.path.join(self.data_path, "labels_split", name + ".tif")
            labels = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            labels = np.expand_dims(labels, -1)

            seg_mask_path = os.path.join(self.data_path, "masks_split", name + ".png")
            seg_mask = cv2.imread(seg_mask_path, cv2.IMREAD_COLOR)
            seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2RGB)

            # change_labels_path = os.path.join(self.data_path, "change_labels_split", name + ".tif")
            # change_labels = cv2.imread(change_labels_path, cv2.IMREAD_UNCHANGED)
        except:
            traceback.print_exc()
            print(img_path)
        #rectangles = self.cache.get(self.names[idx], [])


        rectangles = pd.read_csv(os.path.join(self.data_path, "boxes_split", name + ".csv")).values.tolist()

        sample = self.transforms(image=image, mask=seg_mask, labels=labels, rectangles=rectangles)
        if self.mode == "val":
            height, width = labels.shape[:2]
            center = np.zeros((1, height, width), dtype=np.float32)
            offset = np.zeros((2, height, width), dtype=np.float32)
        else:
            center, offset = self.prepare_masks(sample["labels"].copy()[..., 0])
        # change_labels = sample["change_labels"]
        # change_labels = change_labels > 0
        # change_labels = binary_dilation(change_labels, iterations=5)
        mask = sample["mask"]
        # inner_labels = (mask[..., 2] == 0) & (mask[..., 0] > 0)
        # sample['inner_labels'] = torch.from_numpy(inner_labels).float() / 255.

        sample['img_name'] = name
        sample['mask'] = torch.from_numpy(np.moveaxis(sample["mask"], -1, 0)).contiguous().float() / 255.
        image = torch.from_numpy(np.moveaxis(sample["image"], -1, 0)).float() / 255
        #change_labels = torch.from_numpy(change_labels).float()
        image = normalize(image, **self.normalize)
        sample['image'] = image
        sample['offset'] = offset
        sample['center'] = center
        #sample['change_labels'] = torch.stack([change_labels, change_labels, change_labels], 0)
        month = np.zeros((12, 1, 1))
        month[int(name.split("_")[3]) - 1, 0, 0] = 1
        sample['month'] = month
        del sample["labels"]
        del sample["rectangles"]
        return sample

    def add_boxes(self, labels, rectangles):
        for rprop in measure.regionprops(labels):
            y1, x1, y2, x2 = rprop.bbox
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

            rectangles.append((x, y, w, h))

    def prepare_masks(self, mask):
        center_pts = []
        height, width = mask.shape
        center = np.zeros((1, height, width), dtype=np.float32)
        sigma = self.sigma

        y_coord = np.ones_like(mask, dtype=np.float32)
        x_coord = np.ones_like(mask, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        offset = np.zeros((2, height, width), dtype=np.float32)

        for i in np.unique(mask):
            if i == 0:
                continue
            mask_index = np.where(mask == i)
            if len(mask_index[0]) == 0:
                # the instance is completely cropped
                continue

            # Find instance area
            ins_area = len(mask_index[0])
            center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
            center_pts.append([center_y, center_x])
            # generate center heatmap
            y, x = int(center_y), int(center_x)
            # upper left
            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            # bottom right
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

            c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], width)
            aa, bb = max(0, ul[1]), min(br[1], height)
            center[0, aa:bb, cc:dd] = np.maximum(
                center[0, aa:bb, cc:dd], g[a:b, c:d])

            # generate offset (2, h, w) -> (y-dir, x-dir)
            offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
            offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
            offset[offset_y_index] = center_y - y_coord[mask_index]
            offset[offset_x_index] = center_x - x_coord[mask_index]
        return center, offset


class SpacenetTestDataset(Dataset):
    def __init__(self, data_path,
                 size=2048,
                 normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                 ):
        super().__init__()
        self.data_path = data_path
        self.normalize = normalize
        self.names = list(glob.glob(data_path + "*/images_masked/*.tif", recursive=True))
        self.size = size
        print("names ", len(self.names))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        group = name.split("mosaic_")[-1][:-4]

        imag_name = os.path.splitext(os.path.basename(name))[0]
        image = cv2.imread(name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        factor = self.size//h
        image = cv2.resize(image, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)
        corrected_image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        h, w = image.shape[:2]
        corrected_image[:h, :w] = image
        image = corrected_image
        sample = {"image": image}
        sample['img_name'] = imag_name
        sample['group'] = group
        image = torch.from_numpy(np.moveaxis(sample["image"], -1, 0)).float() / 255
        image = normalize(image, **self.normalize)
        sample['image'] = image
        return sample
