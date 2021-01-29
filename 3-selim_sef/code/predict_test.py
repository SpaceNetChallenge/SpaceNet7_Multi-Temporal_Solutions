import argparse
import os
import re
import warnings

from apex import amp

import zoo
from training.config import load_config
from training.instance_datasets import SpacenetTestDataset

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.simplefilter("ignore")


def load_model(config_path, weights_path):
    conf = load_config(config_path)
    model = zoo.__dict__[conf['network']](seg_classes=conf["num_classes"], backbone_arch=conf['encoder'])
    print("=> loading checkpoint '{}'".format(weights_path))
    checkpoint = torch.load(weights_path, map_location="cpu")
    print("best_dice", checkpoint['dice_best'])
    print("epoch", checkpoint['epoch'])
    state_dict = {re.sub("^module.", "", k): w for k, w in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model.cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Spacenet Test Predictor")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', default='configs/b7adam.json', help='path to configuration file')
    arg('--data-path', type=str, default='/mnt/datasets/spacenet/test_public/',
        help='Path to test images')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--dir', type=str, default='../test_results/3k/b70')
    arg('--model', type=str, default='weights/3k_mask_center_timm_effnet_dragon_tf_efficientnet_b7_ns_0_best_dice')

    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = load_model(args.config, args.model)
    model = amp.initialize(model)

    model = torch.nn.DataParallel(model).cuda()

    data_val = SpacenetTestDataset(data_path=args.data_path, size=3072)
    val_data_loader = DataLoader(data_val, batch_size=1, num_workers=8,
                                 shuffle=False,
                                 pin_memory=False)
    with torch.no_grad():
        for sample in tqdm(val_data_loader):
            img_name = sample["img_name"][0]
            imgs = sample["image"].cpu().float()
            output, center, *_ = model(imgs)
            output = torch.sigmoid(output)

            o, c, *_ = model(torch.flip(imgs, dims=[2]))
            o = torch.sigmoid(o)
            output += torch.flip(o, dims=[2])
            center += torch.flip(c, dims=[2])

            o, c, *_ = model(torch.flip(imgs, dims=[3]))
            o = torch.sigmoid(o)
            output += torch.flip(o, dims=[3])
            center += torch.flip(c, dims=[3])

            o, c, *_ = model(torch.flip(imgs, dims=[2, 3]))
            o = torch.sigmoid(o)
            output += torch.flip(o, dims=[2, 3])
            center += torch.flip(c, dims=[2, 3])

            output /= 4
            center /= 4
            center = np.clip(center.cpu().numpy()[0][0], 0, 1)
            binary_pred = np.moveaxis(output.cpu().numpy()[0], 0, -1)
            cv2.imwrite(os.path.join(args.dir, sample["img_name"][0] + "_mask.png"), binary_pred * 255)
            cv2.imwrite(os.path.join(args.dir, sample["img_name"][0] + "_centers.png"), center * 255)