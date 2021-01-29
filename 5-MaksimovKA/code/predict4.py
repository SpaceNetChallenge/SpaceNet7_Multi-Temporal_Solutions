import argparse
import sys
import torch
import os.path as osp
import os

import torch
import os
import shutil
import numpy as np
import albumentations as albu
import cv2

from model import make_model
from pytorch_toolbelt.inference import tta
from tta import flip_image2mask
from dataset import TestSemSegDataset
from fire import Fire
from tqdm import tqdm
import skimage.io
from importlib import import_module


def get_config(filename):
    module_name = osp.basename(filename)[:-3]

    config_dir = osp.dirname(filename)

    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    }
    return cfg_dict

def main(config_path='/SN7/configs/siamse_2.py',
         test_images='/data/SN7_buildings/test_public/',
         test_predict_result='/wdata/folds_predicts/',
         batch_size=1,
         workers=1,
         gpu='3'):

    with torch.no_grad():

        config = get_config(config_path)
        model_name = config['model_name']
        weights_path = config['load_from']
        device = config['device']
        preprocessing_fn = config['preprocessing_fn']
        valid_augs = config['valid_augs']
        limit_files = config['limit_files']

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        if not os.path.exists(test_predict_result):
            os.mkdir(test_predict_result)
        fold_name = weights_path.split('/')[-3]
        folder_to_save = os.path.join(test_predict_result, fold_name)
        if not os.path.exists(folder_to_save):
           # shutil.rmtree(folder_to_save)
           os.mkdir(folder_to_save)

        test_dataset = TestSemSegDataset(images_dir=test_images,
                                         preprocessing=preprocessing_fn,
                                         augmentation=valid_augs,
                                         limit_files=limit_files)

        print('Loading {}'.format(weights_path))
        model = make_model(
            model_name=model_name).to(device)

        model.load_state_dict(torch.load(weights_path)['model_state_dict'])

        model.eval()
        model = tta.TTAWrapper(model, flip_image2mask)

        file_names = sorted(test_dataset.ids)
        aois = sorted(list(set([el.split('/')[-3] for el in file_names])))

        correct_augs = [albu.PadIfNeeded(1024, 1024, p=1.0)]
        target = {}
        target['image2'] = 'image'
        additional_targets = target

        for aoi in aois[15:]:
            print('####')
            print(aoi)
            aoi_files = [el for el in file_names if el.split('/')[-3] == aoi]
            for _file in tqdm(aoi_files[:]):
                other_files = sorted((list(set(aoi_files) - {_file})))
                for other_index, other_file in enumerate(other_files):
                    data1 = skimage.io.imread(_file, plugin='tifffile')[..., :3]
                    data2 = skimage.io.imread(other_file, plugin='tifffile')[..., :3]
                    sample = albu.Compose(correct_augs, p=1, additional_targets=target)(image=data1,
                                                               image2=data2)
                    data1, data2 = sample['image'], sample['image2']
                    data1 = preprocessing_fn(data1)
                    data2 = preprocessing_fn(data2)
                    image = np.concatenate((data1, data2), axis=-1)
                    image = np.moveaxis(image, -1, 0)
                    image = np.expand_dims(image, 0)
                    image = torch.as_tensor(image, dtype=torch.float)
                    if other_index == 0:
                        runner_out = model(image.cuda())
                    else:
                        runner_out += model(image.cuda())
                runner_out = runner_out / len(other_files)
                aoi_path = os.path.join(folder_to_save, aoi)
                if not os.path.exists(aoi_path):
                    os.mkdir(aoi_path)
                res_name = _file.split('/')[-1].split('.')[0] + '.png'
                file_name = os.path.join(aoi_path, res_name)
                image_pred = runner_out.cpu().detach().numpy()
                data = image_pred[0, :3, ...]
                data = np.moveaxis(data, 0, -1)
                data = (data * 255).astype(np.uint8)
                data[:, :, 1] = 0
                cv2.imwrite(file_name, data)


if __name__ == '__main__':
    Fire(main)


