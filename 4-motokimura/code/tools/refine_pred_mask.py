#!/usr/bin/env python3

import os
import multiprocessing as mp
import timeit
from glob import glob

import numpy as np
from skimage import io
from tqdm import tqdm

import _init_path
from spacenet7_model.configs import load_config
from spacenet7_model.utils import (dump_prediction_to_png, ensemble_subdir,
                                   get_subdirs, load_prediction_from_png,
                                   map_wrapper)


def mask_array(array, idx, n_behind, n_ahead):
    """[summary]

    Args:
        array ([type]): [description]
        idx ([type]): [description]
        n_behind ([type]): [description]
        n_ahead ([type]): [description]

    Returns:
        [type]: [description]
    """
    first = max(0, idx - n_behind)
    last = min(idx + n_ahead + 1, len(array))
    array_masked = array[first:last].copy()
    return array_masked


def compute_aggregated_prediction(preds, idx, n_behind, n_ahead):
    """[summary]

    Args:
        preds ([type]): [description]
        idx ([type]): [description]
        n_behind ([type]): [description]
        n_ahead ([type]): [description]

    Returns:
        [type]: [description]
    """
    preds_masked = mask_array(preds, idx, n_behind, n_ahead)
    pred_aggregated = np.nanmean(preds_masked, axis=0)
    pred_aggregated[np.isnan(pred_aggregated)] = 0.0
    return pred_aggregated


def refine_masks(aoi, input_root, out_root, config):
    """[summary]

    Args:
        aoi ([type]): [description]
        input_root ([type]): [description]
        out_root ([type]): [description]
        config ([type]): [description]
    """
    pred_paths = glob(os.path.join(input_root, aoi, '*.png'))
    pred_paths.sort()

    out_dir = os.path.join(out_root, aoi)
    os.makedirs(out_dir, exist_ok=False)

    footprint_channel = config.INPUT.CLASSES.index('building_footprint')
    boundary_channel = config.INPUT.CLASSES.index('building_boundary')
    contact_channel = config.INPUT.CLASSES.index('building_contact')

    # store all predicted masks (in the aoi) into a buffer to compute aggregated mask
    preds = np.zeros(shape=[
        len(pred_paths),
        len(config.INPUT.CLASSES), config.TRANSFORM.TEST_SIZE[1],
        config.TRANSFORM.TEST_SIZE[0]
    ])
    roi_masks = []
    for i, pred_path in enumerate(pred_paths):
        # get ROI from the image
        pred_filename = os.path.basename(pred_path)
        image_filename, _ = os.path.splitext(pred_filename)
        image_filename = f'{image_filename}.tif'
        if config.TEST_TO_VAL:
            # only for local valication
            # XXX: SN7 train dir is hard coded...
            image_path = os.path.join('/data/spacenet7/spacenet7/train', aoi,
                                      'images_masked', image_filename)
        else:
            # default
            image_path = os.path.join(config.INPUT.TEST_DIR, aoi,
                                      'images_masked', image_filename)
        image = io.imread(image_path)
        roi_mask = image[:, :, 3] > 0
        roi_masks.append(roi_mask)
        h, w = roi_mask.shape

        # get pred
        pred = load_prediction_from_png(pred_path, len(config.INPUT.CLASSES))
        pred[:, np.logical_not(roi_mask)] = np.NaN
        pad_w = config.TRANSFORM.TEST_SIZE[0] - w
        pad_h = config.TRANSFORM.TEST_SIZE[1] - h
        pred = np.pad(pred, ((0, 0), (0, pad_h), (0, pad_w)),
                      'constant',
                      constant_values=np.NaN)
        preds[i] = pred

    # refine each predicted mask with the aggregated mask
    for i, pred_path in enumerate(pred_paths):
        # compute aggregated mask for footprint
        preds_footprint = preds[:, footprint_channel, :, :]
        pred_aggregated_footprint = compute_aggregated_prediction(
            preds_footprint, i, config.REFINEMENT_FOOTPRINT_NUM_FRAMES_BEHIND,
            config.REFINEMENT_FOOTPRINT_NUM_FRAMES_AHEAD)

        # compute aggregated mask for boundary
        preds_boundary = preds[:, boundary_channel, :, :]
        pred_aggregated_boundary = compute_aggregated_prediction(
            preds_boundary, i, config.REFINEMENT_BOUNDARY_NUM_FRAMES_BEHIND,
            config.REFINEMENT_BOUNDARY_NUM_FRAMES_AHEAD)

        # compute aggregated mask for contact
        preds_contact = preds[:, contact_channel, :, :]
        pred_aggregated_contact = compute_aggregated_prediction(
            preds_contact, i, config.REFINEMENT_CONTACT_NUM_FRAMES_BEHIND,
            config.REFINEMENT_CONTACT_NUM_FRAMES_AHEAD)

        # refine
        pred_refined = preds[i].copy()

        w_footprint = config.REFINEMENT_FOOTPRINT_WEIGHT
        pred_refined[footprint_channel] = (1 - w_footprint) * pred_refined[
            footprint_channel] + w_footprint * pred_aggregated_footprint

        w_boundary = config.REFINEMENT_BOUNDARY_WEIGHT
        pred_refined[boundary_channel] = (1 - w_boundary) * pred_refined[
            boundary_channel] + w_boundary * pred_aggregated_boundary

        w_contact = config.REFINEMENT_CONTACT_WEIGHT
        pred_refined[contact_channel] = (1 - w_contact) * pred_refined[
            contact_channel] + w_contact * pred_aggregated_contact

        # handle padded area and non-ROI area
        roi_mask = roi_masks[i]
        h, w = roi_mask.shape
        pred_refined = pred_refined[:, :h, :w]
        pred_refined[:, np.logical_not(roi_mask)] = 0

        # dump
        pred_filename = os.path.basename(pred_path)
        dump_prediction_to_png(os.path.join(out_dir, pred_filename),
                               pred_refined)


if __name__ == '__main__':
    t0 = timeit.default_timer()

    config = load_config()

    assert len(config.ENSEMBLE_EXP_IDS) >= 1

    subdir = ensemble_subdir(config.ENSEMBLE_EXP_IDS)
    input_root = os.path.join(config.ENSEMBLED_PREDICTION_ROOT, subdir)
    out_root = os.path.join(config.REFINED_PREDICTION_ROOT, subdir)
    aois = get_subdirs(input_root)

    n_thread = config.REFINEMENT_NUM_THREADS
    n_thread = n_thread if n_thread > 0 else mp.cpu_count()
    print(f'N_thread for multiprocessing: {n_thread}')

    print('preparing input args...')
    input_args = []
    for aoi in aois:
        input_args.append([refine_masks, aoi, input_root, out_root, config])

    print('running multiprocessing...')
    with mp.Pool(processes=n_thread) as pool:
        with tqdm(total=len(input_args)) as t:
            for _ in pool.imap_unordered(map_wrapper, input_args):
                t.update(1)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
