#!/usr/bin/env python3

import multiprocessing as mp
import os
import timeit

import pandas as pd

import _init_path
from spacenet7_model.configs import load_config
from spacenet7_model.utils import (convert_geojsons_to_csv, ensemble_subdir,
                                   get_subdirs, interpolate_polys, map_wrapper,
                                   remove_polygon_empty_row_if_polygon_exists,
                                   solution_filename,
                                   track_footprint_identifiers)
from tqdm import tqdm

if __name__ == '__main__':
    t0 = timeit.default_timer()

    config = load_config()

    assert len(config.ENSEMBLE_EXP_IDS) >= 1

    subdir = ensemble_subdir(config.ENSEMBLE_EXP_IDS)
    input_root = os.path.join(config.POLY_ROOT, subdir)
    aois = get_subdirs(input_root)

    # prepare json and output directories
    tracked_poly_root = os.path.join(config.TRACKED_POLY_ROOT, subdir)
    os.makedirs(tracked_poly_root, exist_ok=False)

    if config.SOLUTION_OUTPUT_PATH and config.SOLUTION_OUTPUT_PATH != 'none':
        # only for deployment phase
        out_path = config.SOLUTION_OUTPUT_PATH
    else:
        out_path = os.path.join(tracked_poly_root, solution_filename())

    # some parameters
    verbose = True
    super_verbose = False

    n_thread = config.TRACKING_NUM_THREADS
    n_thread = n_thread if n_thread > 0 else mp.cpu_count()
    print(f'N_thread for multiprocessing: {n_thread}')

    # track footprint and save the results as geojson files
    # prepare args and output directories
    input_args = []
    for i, aoi in enumerate(aois):
        json_dir = os.path.join(tracked_poly_root, aoi)
        os.makedirs(json_dir, exist_ok=False)

        input_dir = os.path.join(input_root, aoi)

        input_args.append([
            track_footprint_identifiers, config, input_dir, json_dir, verbose,
            super_verbose
        ])

    # run multiprocessing
    with mp.Pool(processes=n_thread) as pool:
        with tqdm(total=len(input_args)) as t:
            for _ in pool.imap_unordered(map_wrapper, input_args):
                t.update(1)

    # convert the geojson files into a dataframe
    json_dirs = [
        os.path.join(tracked_poly_root, aoi)
        for aoi in get_subdirs(tracked_poly_root)
    ]
    solution_df = convert_geojsons_to_csv(json_dirs,
                                          output_csv_path=None,
                                          population='proposal')
    solution_df = pd.DataFrame(solution_df)  # GeoDataFrame to DataFrame

    # interpolate master polys
    if config.TRACKING_ENABLE_POST_INTERPOLATION:
        print('running post interpolation. this may take ~10 min...')

        # XXX: SN7 train dir is hard coded...
        test_root = '/data/spacenet7/spacenet7/train' if config.TEST_TO_VAL else config.INPUT.TEST_DIR

        # prepare args and output directories
        input_args = []
        for aoi in aois:
            aoi_mask = solution_df.filename.str.endswith(aoi)
            solution_df_aoi = solution_df[aoi_mask]
            input_args.append([
                interpolate_polys, aoi, solution_df_aoi, tracked_poly_root,
                test_root
            ])

        # run multiprocessing
        pool = mp.Pool(processes=n_thread)
        polys_to_interpolate_tmp = pool.map(map_wrapper, input_args)
        pool.close()

        # do interpolation
        polys_to_interpolate = []
        for polys in polys_to_interpolate_tmp:
            polys_to_interpolate.extend(polys)
        polys_to_interpolate = pd.DataFrame(polys_to_interpolate)
        solution_df = solution_df.append(polys_to_interpolate)

        # remove "POLYGON EMPTY" row if needed
        solution_df = remove_polygon_empty_row_if_polygon_exists(solution_df)

    print('saving solution csv file...')
    solution_df.to_csv(out_path, index=False)
    print(f'saved solution csv to {out_path}')

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
