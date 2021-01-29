#!/usr/bin/env python3

import argparse
import json
import os
import random
import timeit
from glob import glob

import numpy as np

import _init_path
from spacenet7_model.utils import get_image_paths, get_aoi_from_path


def parse_args():
    """[summary]

    Returns:
        [type]: [description]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir',
                        help='directory containing spacenet7 train dataset',
                        default='/data/spacenet7/spacenet7/train/')
    parser.add_argument('--mask_dir',
                        help='directory containing building mask image files',
                        default='/data/spacenet7/building_masks/')
    parser.add_argument('--out_dir',
                        help='output root directory',
                        default='/data/spacenet7/split_random/')
    parser.add_argument('--split_num',
                        help='number of split',
                        type=int,
                        default=5)
    return parser.parse_args()


def dump_file_paths(image_paths, output_path, mask_dir):
    """[summary]

    Args:
        image_paths ([type]): [description]
        output_path ([type]): [description]
        mask_dir ([type]): [description]
    """

    results = []

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        aoi = get_aoi_from_path(image_path)
        mask_path = os.path.join(mask_dir, aoi, filename)
        assert os.path.exists(mask_path)

        result = {}
        result['image_masked'] = image_path
        result['building_mask'] = mask_path
        results.append(result)

    with open(output_path, 'w') as f:
        json.dump(results,
                  f,
                  ensure_ascii=False,
                  indent=4,
                  sort_keys=False,
                  separators=(',', ': '))


if __name__ == '__main__':
    t0 = timeit.default_timer()

    args = parse_args()

    os.makedirs(args.out_dir)

    image_paths = get_image_paths(args.train_dir)
    image_paths.sort()

    random.seed(777)
    random.shuffle(image_paths)

    # split aois into train and val
    n = args.split_num
    image_paths_divided = np.array([image_paths[i::n] for i in range(n)])

    for val_idx in range(n):
        # dump file paths for val split
        val_image_paths = image_paths_divided[val_idx]

        dump_file_paths(val_image_paths,
                        os.path.join(args.out_dir, f'val_{val_idx}.json'),
                        args.mask_dir)

        # dump file paths for train split
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False
        train_image_paths = image_paths_divided[train_mask]
        train_image_paths = np.concatenate(train_image_paths, axis=0).tolist()

        dump_file_paths(train_image_paths,
                        os.path.join(args.out_dir, f'train_{val_idx}.json'),
                        args.mask_dir)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
