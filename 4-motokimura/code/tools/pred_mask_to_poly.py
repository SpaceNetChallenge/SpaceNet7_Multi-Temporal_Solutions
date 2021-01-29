#!/usr/bin/env python3

import os
import multiprocessing as mp
import timeit
from glob import glob

import _init_path
from spacenet7_model.configs import load_config
from spacenet7_model.utils import (
    ensemble_subdir, get_subdirs, load_prediction_from_png,
    compute_building_score, gen_building_polys_using_contours,
    gen_building_polys_using_watershed, gen_building_polys_using_watershed_2,
    map_wrapper)
from tqdm import tqdm


def generate_polys(mask_path, output_path, config):
    """[summary]

    Args:
        mask_path ([type]): [description]
        output_path ([type]): [description]
        config ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    pred_array = load_prediction_from_png(mask_path,
                                          n_channels=len(config.INPUT.CLASSES))

    footprint_channel = config.INPUT.CLASSES.index('building_footprint')
    boundary_channel = config.INPUT.CLASSES.index('building_boundary')
    contact_channel = config.INPUT.CLASSES.index('building_contact')

    footprint_score = pred_array[footprint_channel]
    boundary_score = pred_array[boundary_channel]
    contact_score = pred_array[contact_channel]
    building_score = compute_building_score(
        footprint_score,
        boundary_score,
        contact_score,
        alpha=config.BOUNDARY_SUBTRACT_COEFF,
        beta=config.CONTACT_SUBTRACT_COEFF)

    if config.METHOD_TO_MAKE_POLYGONS == 'contours':
        polys = gen_building_polys_using_contours(
            building_score,
            config.BUILDING_MIM_AREA_PIXEL,
            config.BUILDING_SCORE_THRESH,
            simplify=False,
            output_path=output_path)
    elif config.METHOD_TO_MAKE_POLYGONS == 'watershed':
        polys = gen_building_polys_using_watershed(
            building_score,
            config.WATERSHED_SEED_MIN_AREA_PIXEL,
            config.WATERSHED_MIN_AREA_PIXEL,
            config.WATERSHED_SEED_THRESH,
            config.WATERSHED_MAIN_THRESH,
            output_path=output_path)
    elif config.METHOD_TO_MAKE_POLYGONS == 'watershed2':
        polys = gen_building_polys_using_watershed_2(
            footprint_score,
            boundary_score,
            contact_score,
            config.WATERSHED2_SEED_MIN_AREA_PIXEL,
            config.WATERSHED2_MIN_AREA_PIXEL,
            config.WATERSHED2_SEED_THRESH,
            config.WATERSHED2_MAIN_THRESH,
            config.WATERSHED2_SEED_BOUNDARY_SUBTRACT_COEFF,
            config.WATERSHED2_SEED_CONTACT_SUBTRACT_COEFF,
            config.WATERSHED2_BOUNDARY_SUBTRACT_COEFF,
            config.WATERSHED2_CONTACT_SUBTRACT_COEFF,
            output_path=output_path)
    else:
        raise ValueError()

    return polys


if __name__ == '__main__':
    t0 = timeit.default_timer()

    config = load_config()

    assert len(config.ENSEMBLE_EXP_IDS) >= 1

    subdir = ensemble_subdir(config.ENSEMBLE_EXP_IDS)
    input_root = os.path.join(config.REFINED_PREDICTION_ROOT, subdir)
    aois = get_subdirs(input_root)

    out_root = os.path.join(config.POLY_ROOT, subdir)
    os.makedirs(out_root, exist_ok=False)

    n_thread = config.POLY_NUM_THREADS
    n_thread = n_thread if n_thread > 0 else mp.cpu_count()
    print(f'N_thread for multiprocessing: {n_thread}')

    print('preparing input args...')
    input_args = []
    for i, aoi in enumerate(aois):
        paths = glob(os.path.join(input_root, aoi, '*.png'))
        paths.sort()

        out_dir = os.path.join(out_root, aoi)
        os.makedirs(out_dir, exist_ok=False)

        for path in paths:
            filename = os.path.basename(path)
            filename, _ = os.path.splitext(filename)
            output_path = os.path.join(out_dir, f'{filename}.geojson')

            input_args.append([generate_polys, path, output_path, config])

    print('running multiprocessing...')
    pool = mp.Pool(processes=n_thread)
    with tqdm(total=len(input_args)) as t:
        for _ in pool.imap_unordered(map_wrapper, input_args):
            t.update(1)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
