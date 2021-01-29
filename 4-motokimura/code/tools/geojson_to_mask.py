#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import os
import timeit

import numpy as np

import gdal
import skimage
import solaris as sol
from solaris.raster.image import create_multiband_geotiff
from solaris.utils.core import _check_gdf_load
from tqdm import tqdm

import _init_path
from spacenet7_model.utils import map_wrapper


def parse_args():
    """[summary]

    Returns:
        [type]: [description]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir',
                        help='directory containing spacenet7 train dataset',
                        default='/data/spacenet7/spacenet7/train/')
    parser.add_argument('--out_dir',
                        help='directory to output mask images',
                        default='/data/spacenet7/building_masks/')
    parser.add_argument('--boundary_width',
                        help='width of boundary mask in meter',
                        type=int,
                        default=3)
    parser.add_argument('--contact_spacing',
                        help='contact spacing for contact mask in meter',
                        type=int,
                        default=10)
    parser.add_argument('--n_thread',
                        help='number of thread or process of multiprocessing',
                        type=int,
                        default=0)
    return parser.parse_args()


def generate_mask(image_path, json_path, output_path_mask, boundary_width,
                  contact_spacing):
    """[summary]

    Args:
        image_path ([type]): [description]
        json_path ([type]): [description]
        output_path_mask ([type]): [description]

    Returns:
        [type]: [description]
    """
    # filter out null geoms (this is always a worthy check)
    gdf_tmp = _check_gdf_load(json_path)
    if len(gdf_tmp) == 0:
        gdf_nonull = gdf_tmp
    else:
        gdf_nonull = gdf_tmp[gdf_tmp.geometry.notnull()]
        try:
            im_tmp = skimage.io.imread(image_path)
        except:
            print(f'[WARN] failed to load image. Skipping {image_path}')
            return None

    # handle empty geojsons
    if len(gdf_nonull) == 0:
        # create masks
        print(f'[WARN] no building was found in: {json_path}')
        im = gdal.Open(image_path)
        proj = im.GetProjection()
        geo = im.GetGeoTransform()
        im = im.ReadAsArray()
        # set masks to 0 everywhere
        mask_arr = np.zeros((3, im.shape[1], im.shape[2]))
        create_multiband_geotiff(mask_arr, output_path_mask, proj, geo)
        return mask_arr

    # three channel mask (takes awhile)
    # https://github.com/CosmiQ/solaris/blob/master/docs/tutorials/notebooks/api_masks_tutorial.ipynb
    mask = sol.vector.mask.df_to_px_mask(
        df=gdf_nonull,
        out_file=output_path_mask,
        channels=['footprint', 'boundary', 'contact'],
        reference_im=image_path,
        boundary_type='outer',
        boundary_width=boundary_width,
        contact_spacing=contact_spacing,
        meters=True,
        shape=(im_tmp.shape[0], im_tmp.shape[1]))
    return mask


if __name__ == '__main__':
    t0 = timeit.default_timer()

    args = parse_args()

    n_thread = args.n_thread if args.n_thread > 0 else mp.cpu_count()
    print(f'N_thread for multiprocessing: {n_thread}')

    os.makedirs(args.out_dir, exist_ok=False)

    # args to be passed to multiprocessing
    input_args = []

    aois = sorted([
        d for d in os.listdir(args.train_dir)
        if os.path.isdir(os.path.join(args.train_dir, d))
    ])

    # prepare input args for multiprocessing
    for aoi in aois:
        image_dir = os.path.join(args.train_dir, aoi, 'images_masked')
        json_dir = os.path.join(args.train_dir, aoi, 'labels_match')

        json_files = sorted([
            f for f in os.listdir(json_dir) if f.endswith('Buildings.geojson')
        ])

        for json_fname in json_files:
            json_path = os.path.join(json_dir, json_fname)

            name_root = json_fname.split('.')[0]
            image_fname = f'{name_root}.tif'.replace('labels',
                                                     'images').replace(
                                                         '_Buildings', '')
            image_path = os.path.join(image_dir, image_fname)

            output_path_mask = os.path.join(args.out_dir, aoi, image_fname)
            os.makedirs(os.path.dirname(output_path_mask), exist_ok=True)

            input_args.append([
                generate_mask, image_path, json_path, output_path_mask,
                args.boundary_width, args.contact_spacing
            ])

    # run multiprocessing
    with mp.Pool(processes=n_thread) as pool:
        with tqdm(total=len(input_args)) as t:
            for _ in pool.imap_unordered(map_wrapper, input_args):
                t.update(1)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
