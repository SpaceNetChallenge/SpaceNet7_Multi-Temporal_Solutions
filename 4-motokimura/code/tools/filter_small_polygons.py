#!/usr/bin/env python3
import argparse
import os
import timeit
from glob import glob

import geopandas as gpd
from shapely.geometry import Polygon
from tqdm import tqdm

import _init_path
from spacenet7_model.utils import get_subdirs, save_empty_geojson


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
                        default='/data/spacenet7/labels_pix_filtered/')
    parser.add_argument('--min_area',
                        help='min area in pixels',
                        type=float,
                        default=4.0)
    return parser.parse_args()


if __name__ == '__main__':
    t0 = timeit.default_timer()

    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=False)

    aois = get_subdirs(args.train_dir)
    for i, aoi in enumerate(aois):
        print(f'processing {aoi} ({i + 1}/{len(aois)}) ...')

        json_paths = glob(
            os.path.join(args.train_dir, aoi, 'labels_match_pix', '*.geojson'))
        json_paths.sort()

        out_dir = os.path.join(args.out_dir, aoi)
        os.makedirs(out_dir, exist_ok=False)

        for json_path in tqdm(json_paths):
            df = gpd.read_file(json_path)

            # add `area` column
            areas = []
            for index, row in df.iterrows():
                area = Polygon(row['geometry']).area
                areas.append(area)
            df['area'] = areas

            # filter out small polygons
            mask = df['area'] >= args.min_area
            df = df[mask]

            # dump
            output_path = os.path.join(out_dir, os.path.basename(json_path))
            if len(df) > 0:
                df.to_file(output_path, driver='GeoJSON')
            else:
                print(f'warning: {output_path} is an empty geojson.')
                save_empty_geojson(output_path)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
