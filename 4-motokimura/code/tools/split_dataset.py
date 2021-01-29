#!/usr/bin/env python3

import argparse
import json
import os
import random
import timeit
from glob import glob

import numpy as np


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
                        default='/data/spacenet7/split/')
    parser.add_argument('--split_num',
                        help='number of split',
                        type=int,
                        default=5)
    return parser.parse_args()


def dump_file_paths(aois, output_path, train_dir, mask_dir):
    """[summary]

    Args:
        aois ([type]): [description]
        output_path ([type]): [description]
        train_dir ([type]): [description]
        mask_dir ([type]): [description]
    """

    results = []

    for aoi in aois:
        image_paths = glob(
            os.path.join(train_dir, aoi, 'images_masked', '*.tif'))
        image_paths.sort()

        N = len(image_paths)
        for i in range(N):
            # get path to mask
            image_path = image_paths[i]
            filename = os.path.basename(image_path)
            mask_path = os.path.join(mask_dir, aoi, filename)
            assert os.path.exists(mask_path)

            # previous frame
            image_prev_path = image_paths[0] if i == 0 \
                else image_paths[i - 1]

            # next frame
            image_next_path = image_paths[N - 1] if i == N - 1 \
                else image_paths[i + 1]

            result = {}
            result['image_masked'] = image_path
            result['building_mask'] = mask_path
            result['image_masked_prev'] = image_prev_path
            result['image_masked_next'] = image_next_path
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

    aois = sorted([
        d for d in os.listdir(args.train_dir)
        if os.path.isdir(os.path.join(args.train_dir, d))
    ])

    random.seed(777)
    random.shuffle(aois)

    # split aois into train and val
    n = args.split_num
    aois_divided = np.array([aois[i::n] for i in range(n)])

    for val_idx in range(n):
        # dump file paths for val split
        val_aois = aois_divided[val_idx]

        dump_file_paths(val_aois,
                        os.path.join(args.out_dir, f'val_{val_idx}.json'),
                        args.train_dir, args.mask_dir)

        # dump file paths for train split
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False
        train_aois = aois_divided[train_mask]
        train_aois = np.concatenate(train_aois, axis=0).tolist()

        dump_file_paths(train_aois,
                        os.path.join(args.out_dir, f'train_{val_idx}.json'),
                        args.train_dir, args.mask_dir)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
