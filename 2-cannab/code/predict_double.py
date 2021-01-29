import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader


import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from zoo.models import EfficientNet_Unet_Double

from Dataset import TestDatasetDouble


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

models_folder = '/wdata/weights'

test_dir = '/data/SN7_buildings/test_public'
if len(sys.argv) > 0:
    test_dir = sys.argv[1]

out_dir = '/wdata/test_pred_4k_double'



if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(out_dir, exist_ok=True)

    cudnn.benchmark = True

    test_batch_size = 4

    all_files = []

    all_pairs = []

    for d in sorted(listdir(test_dir)):
        if not path.isdir(path.join(test_dir, d, 'images_masked')):
            continue
        _prev = ''
        for f in sorted(listdir(path.join(test_dir, d, 'images_masked'))):
            if '.tif' in f:
                if _prev != '':
                    all_pairs.append((_prev, path.join(test_dir, d, 'images_masked', f)))
                _prev = path.join(test_dir, d, 'images_masked', f)
                all_files.append(path.join(test_dir, d, 'images_masked', f))

    test_data = TestDatasetDouble(all_pairs, scale=4)

    test_data_loader = DataLoader(test_data, batch_size=test_batch_size, num_workers=6, shuffle=False)

    models = []

    fold = 0

    model = EfficientNet_Unet_Double(name='efficientnet-b7', pretrained=None).cuda()
    model = nn.DataParallel(model).cuda()

    snap_to_load = 'eff7_4k_{0}_last_double_0'.format(fold)
    print("=> loading checkpoint '{}'".format(snap_to_load))
    checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
    loaded_dict = checkpoint['state_dict']
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict:
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
        checkpoint['epoch'], checkpoint['best_score']))

    model = model.eval()
    models.append(model)

    fold = 1

    model = EfficientNet_Unet_Double(name='efficientnet-b6', pretrained=None).cuda()
    model = nn.DataParallel(model).cuda()

    snap_to_load = 'eff6_4k_{0}_last_double_0'.format(fold)
    print("=> loading checkpoint '{}'".format(snap_to_load))
    checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
    loaded_dict = checkpoint['state_dict']
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict:
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
        checkpoint['epoch'], checkpoint['best_score']))

    model = model.eval()
    models.append(model)


    torch.cuda.empty_cache()
    with torch.no_grad():
        for sample in tqdm(test_data_loader):
            imgs = sample["img"].cpu().numpy()
            img_ids =  sample["img_id"]
            img_ids2 =  sample["img_id2"]

            msk_preds = []
            msk_cnts = []
            ids = []
            ids2 = []
            for i in range(0, len(img_ids), 1):
                img_id = img_ids[i]
                img_id2 = img_ids2[i]
                ids.append(img_id)
                ids2.append(img_id2)
                msk_preds.append(np.zeros((7, imgs.shape[2], imgs.shape[3]), dtype='float32'))
                msk_cnts.append(np.zeros((7, imgs.shape[2], imgs.shape[3]), dtype='uint8'))

            
            with torch.cuda.amp.autocast():
                for _i in range(2):
                    if _i == 0:
                        inp = torch.from_numpy(imgs.copy()).float().cuda(non_blocking=True)
                    if _i == 1:
                        inp = torch.from_numpy(imgs[:, :, ::-1, :].copy()).float().cuda(non_blocking=True)
                    if _i == 2:
                        inp = torch.from_numpy(imgs[:, :, :, ::-1].copy()).float().cuda(non_blocking=True)
                    if _i == 3:
                        inp = torch.from_numpy(imgs[:, :, ::-1, ::-1].copy()).float().cuda(non_blocking=True)
                    for model in models:
                        _msk = np.zeros((inp.shape[0], 7, inp.shape[2], inp.shape[3]), dtype='float32')
                        _cnts = np.zeros((inp.shape[0], 7, inp.shape[2], inp.shape[3]), dtype='uint8')

                        out = model(inp[:, :, :1440, :])
                        msk_pred0 = torch.sigmoid(out).cpu().numpy()
                        _msk[:, :, :1376, :] += msk_pred0[:, :, :1376, :]
                        _cnts[:, :, :1376, :] += 1
                        
                        out = model(inp[:, :, inp.shape[2]-1440:, :])
                        msk_pred0 = torch.sigmoid(out).cpu().numpy()
                        _msk[:, :, _msk.shape[2]-1376:, :] += msk_pred0[:, :, msk_pred0.shape[2]-1376:, :]
                        _cnts[:, :, _msk.shape[2]-1376:, :] += 1

                        out = model(inp[:, :, 1312:1312+1440, :])
                        msk_pred0 = torch.sigmoid(out).cpu().numpy()
                        _msk[:, :, 1344:1312+1440-32, :] += msk_pred0[:, :, 32:msk_pred0.shape[2]-32, :]
                        _cnts[:, :, 1344:1312+1440-32, :] += 1

                        if _i == 1:
                            _msk = _msk[:, :, ::-1, :].copy()
                            _cnts = _cnts[:, :, ::-1, :].copy()
                        if _i == 2:
                            _msk = _msk[:, :, :, ::-1]
                            _cnts = _cnts[:, :, :, ::-1]
                        if _i == 3:
                            _msk = _msk[:, :, ::-1, ::-1]
                            _cnts = _cnts[:, :, ::-1, ::-1]
                        for i in range(len(ids)):
                            msk_preds[i] += _msk[i].copy()
                            msk_cnts[i] += _cnts[i].copy()
                        torch.cuda.empty_cache()
                    inp = inp.cpu()
                    del inp
                    torch.cuda.empty_cache()


            for i in range(len(ids)):
                msk_pred = msk_preds[i] / msk_cnts[i]
                msk_pred = (msk_pred * 255).astype('uint8')
                msk_pred = msk_pred.transpose(1, 2, 0)
                msk_pred = msk_pred.astype('uint8')

                fn = ids[i].split('/')[-1].split('.')[0]
                if path.exists(path.join(out_dir, fn + '.png')):
                    msk0 = cv2.imread(path.join(out_dir, fn + '.png'), cv2.IMREAD_COLOR)
                    msk0 = (msk_pred[..., :3] * 0.5 + 0.5 * msk0).astype('uint8')
                else:
                    msk0 = msk_pred[..., :3]
                cv2.imwrite(path.join(out_dir, fn + '.png'), msk0, [cv2.IMWRITE_PNG_COMPRESSION, 4])

                fn = ids2[i].split('/')[-1].split('.')[0]
                if path.exists(path.join(out_dir, fn + '.png')):
                    msk0 = cv2.imread(path.join(out_dir, fn + '.png'), cv2.IMREAD_COLOR)
                    msk0 = (msk_pred[..., 3:6] * 0.5 + 0.5 * msk0).astype('uint8')
                else:
                    msk0 = msk_pred[..., 3:6]
                cv2.imwrite(path.join(out_dir, fn + '.png'), msk0, [cv2.IMWRITE_PNG_COMPRESSION, 4])


    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))