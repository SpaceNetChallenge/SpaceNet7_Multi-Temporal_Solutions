import os
import cv2
import shutil
from tqdm import tqdm
import numpy as np

paths = sorted([os.path.join('/wdata/folds_predicts/', el) for el in sorted(os.listdir('/wdata/folds_predicts/'))])
print(paths)
save_path = '/wdata/mix_predicts/'
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(save_path)

aois = os.listdir(paths[0])

for aoi in tqdm(aois):
    aoi_save_path = os.path.join(save_path, aoi)
    if not os.path.exists(aoi_save_path):
        os.mkdir(aoi_save_path)
    files = os.listdir(os.path.join(paths[0], aoi))
    for _file in files:
        for mask_i, model_path in enumerate(paths):

            mask_path = os.path.join(model_path, aoi, _file)
            #print(mask_path)
            if mask_i == 0:
                pred_data = cv2.imread(mask_path) / 255.0
            else:
                pred_data += cv2.imread(mask_path) / 255.0
        pred_data = pred_data / len(paths)
        data = (pred_data * 255).astype(np.uint8)
        file_name = os.path.join(save_path, aoi, _file)
        cv2.imwrite(file_name, data)