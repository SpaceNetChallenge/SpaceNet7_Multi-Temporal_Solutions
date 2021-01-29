import os
import traceback

import cv2
import numpy as np
import pandas as pd
import skimage.io
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize


class SpacenetLocDataset(Dataset):
    def __init__(self, data_path, mode, fold=0, folds_csv='folds.csv', transforms=None,
                 normalize={"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]}, multiplier=1):
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
        if mode == "train":
            self.names = self.names * multiplier
        print("names ", len(self.names))

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

            mask_path = os.path.join(self.data_path, "masks_split", name + ".png")
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask[alpha == 0, :] = 0
        except:
            traceback.print_exc()
            print(img_path)
        sample = self.transforms(image=image, mask=mask)
        sample['img_name'] = name
        sample['mask'] = torch.from_numpy(np.ascontiguousarray(np.moveaxis(sample["mask"], -1, 0))).float() / 255.
        image = torch.from_numpy(np.moveaxis(sample["image"], -1, 0)).float() / 255
        image = normalize(image, **self.normalize)
        sample['image'] = image
        return sample
