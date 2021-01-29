#!/usr/bin/env python3
import os.path
import timeit

import cv2
import numpy as np
from tqdm import tqdm

import _init_path
from spacenet7_model.configs import load_config
from spacenet7_model.datasets import get_test_dataloader
from spacenet7_model.models import get_model
from spacenet7_model.utils import (crop_center, dump_prediction_to_png,
                                   experiment_subdir, get_aoi_from_path)


def main():
    """[summary]
    """
    config = load_config()
    print('successfully loaded config:')
    print(config)

    # prepare dataloaders, flipping flags, and weights for averaging
    test_dataloaders, flags_hflip, flags_vflip, weights = [], [], [], []

    # default dataloader (w/o tta)
    test_dataloaders.append(get_test_dataloader(config))
    weights.append(1.0)
    flags_hflip.append(False)
    flags_vflip.append(False)

    # dataloaders w/ tta size jittering
    for tta_resize_wh, weight in zip(config.TTA.RESIZE,
                                     config.TTA.RESIZE_WEIGHTS):
        test_dataloaders.append(
            get_test_dataloader(config, tta_resize_wh=tta_resize_wh))
        weights.append(weight)
        flags_hflip.append(False)
        flags_vflip.append(False)

    # dataloader w/ tta horizontal flipping
    if config.TTA.HORIZONTAL_FLIP:
        test_dataloaders.append(get_test_dataloader(config, tta_hflip=True))
        weights.append(config.TTA.HORIZONTAL_FLIP_WEIGHT)
        flags_hflip.append(True)
        flags_vflip.append(False)

    # dataloader w/ tta vertical flipping
    if config.TTA.VERTICAL_FLIP:
        test_dataloaders.append(get_test_dataloader(config, tta_vflip=True))
        weights.append(config.TTA.VERTICAL_FLIP_WEIGHT)
        flags_hflip.append(False)
        flags_vflip.append(True)

    # normalize weights
    weights = np.array(weights)
    weights /= weights.sum()

    # prepare model to test
    model = get_model(config)
    model.eval()

    # prepare directory to output predictions
    exp_subdir = experiment_subdir(config.EXP_ID)
    pred_root = os.path.join(config.PREDICTION_ROOT, exp_subdir)
    os.makedirs(pred_root, exist_ok=False)

    test_width, test_height = config.TRANSFORM.TEST_SIZE

    # test loop
    for batches in tqdm(zip(*test_dataloaders),
                        total=len(test_dataloaders[0])):
        # prepare buffers for image file name and predicted array
        batch_size = len(batches[0]['image'])
        output_paths = [None] * batch_size
        orig_image_sizes = [None] * batch_size
        predictions_averaged = np.zeros(shape=[
            batch_size,
            len(config.INPUT.CLASSES), test_height, test_width
        ])

        for dataloader_idx, batch in enumerate(batches):
            images = batch['image'].to(config.MODEL.DEVICE)
            image_paths = batch['image_path']
            original_heights, original_widths, _ = batch['original_shape']

            predictions = model.module.predict(images)
            predictions = predictions.cpu().numpy()

            for batch_idx in range(len(predictions)):
                pred = predictions[batch_idx]
                path = image_paths[batch_idx]
                orig_h = original_heights[batch_idx].item()
                orig_w = original_widths[batch_idx].item()

                # resize (only when resize tta or input resizing is applied)
                _, pred_height, pred_width = pred.shape
                if (pred_width != test_width) or (pred_height != test_height):
                    pred = pred.transpose(1, 2, 0)  # CHW -> HWC
                    pred = cv2.resize(pred, dsize=(test_width, test_height))
                    pred = pred.transpose(2, 0, 1)  # HWC -> CHW

                # flip (only when flipping tta is applied)
                if flags_vflip[dataloader_idx]:
                    pred = pred[:, ::-1, :]
                if flags_hflip[dataloader_idx]:
                    pred = pred[:, :, ::-1]

                # store predictions into the buffer
                predictions_averaged[
                    batch_idx] += pred * weights[dataloader_idx]

                # prepare sub-directory under pred_root
                filename = os.path.basename(path)
                filename, _ = os.path.splitext(filename)
                filename = f'{filename}.png'
                aoi = get_aoi_from_path(path)
                out_dir = os.path.join(pred_root, aoi)
                os.makedirs(out_dir, exist_ok=True)

                # store output paths and original image sizes into the buffers
                output_path = os.path.join(out_dir, filename)
                orig_image_wh = (orig_w, orig_h)
                if dataloader_idx == 0:
                    output_paths[batch_idx] = output_path
                    orig_image_sizes[batch_idx] = orig_image_wh
                else:
                    assert output_paths[batch_idx] == output_path
                    assert orig_image_sizes[batch_idx] == orig_image_wh

        for output_path, orig_image_wh, pred_averaged in zip(
                output_paths, orig_image_sizes, predictions_averaged):
            # remove padded area
            pred_averaged = crop_center(pred_averaged, crop_wh=orig_image_wh)

            # dump to .png file
            dump_prediction_to_png(output_path, pred_averaged)


if __name__ == '__main__':
    t0 = timeit.default_timer()

    main()

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
