import os
import random
from glob import glob
from random import shuffle

import cv2
import pandas as pd
import skimage.io
train_dir = "/mnt/datasets/spacenet/train/"
groups = [g for g in os.listdir(train_dir) if "L15" in g]
random.seed(777)
shuffle(groups)
data = []
for i, group in enumerate(groups):
    for f in os.listdir(os.path.join(train_dir, group, "images_masked")):
        if not os.path.exists(os.path.join(train_dir, "masks", f.replace(".tif", "_Buildings.png"))):
            continue
        if f.endswith("tif"):
            id = f.replace(".tif", "")
            for y in range(2):
                for x in range(2):
                    data.append([id + "_{}_{}".format(y, x), i//6])
pd.DataFrame(data, columns=["id", "fold"]).to_csv("../folds.csv", index=False)