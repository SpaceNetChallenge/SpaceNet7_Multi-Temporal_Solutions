import os
import traceback

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from argparse import ArgumentParser
from functools import partial
from glob import glob
from multiprocessing import Pool

import cv2
import numpy as np
from skimage import measure
from tqdm import tqdm
import pandas as pd


def split_img_mask(filename: str, train_dir: str, out_dir: str, tile_size: int):
    os.makedirs(os.path.join(out_dir, "masks_split"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels_split"), exist_ok=True)
    fn = filename.replace("_Buildings.png", "")
    group = fn.split("mosaic_")[-1]
    img_path = os.path.join(train_dir, group, "images_masked", fn + ".tif")
    images_dir = os.path.join(out_dir, group, "images_masked_split")
    os.makedirs(images_dir, exist_ok=True)
    try:
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        h, w = image.shape[:2]
        factor = tile_size // 1024
        image = cv2.resize(image, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)
        corrected_image = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
        h, w = image.shape[:2]
        corrected_image[:h, :w] = image
        image = corrected_image
        mask_path = os.path.join(out_dir, "masks", fn + ".png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path)
        else:
            return
        label_path = os.path.join(out_dir, "labels", fn + ".tif")
        if os.path.exists(label_path):
            label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        else:
            return

        num_tiles = tile_size // 3072
        for i in range(num_tiles):
            for j in range(num_tiles):
                size_subtile = tile_size // num_tiles
                start_i = i * size_subtile
                start_j = j * size_subtile
                mask_i_j = mask[start_i:start_i + size_subtile, start_j:start_j + size_subtile]
                image_i_j = image[start_i:start_i + size_subtile, start_j:start_j + size_subtile]
                label_i_j = label[start_i:start_i + size_subtile, start_j:start_j + size_subtile]
                cv2.imwrite(os.path.join(images_dir, fn + "_{}_{}.tif".format(i, j)), image_i_j)
                cv2.imwrite(os.path.join(out_dir, "masks_split", fn + "_{}_{}.png".format(i, j)), mask_i_j)
                cv2.imwrite(os.path.join(out_dir, "labels_split", fn + "_{}_{}.tif".format(i, j)), label_i_j)
    except Exception as e:
        traceback.print_exc()
        print(filename)


def generate_boxes(label_path: str, boxes_dir: str):
    labels = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    file_id = os.path.splitext(os.path.basename(label_path))[0]
    boxes = []
    for rprop in measure.regionprops(labels):
        y1, x1, y2, x2 = rprop.bbox
        x, y, w, h = x1, y1, x2 - x1, y2 - y1

        boxes.append((x, y, w, h))
    csv_path = os.path.join(boxes_dir, "{}.csv".format(file_id))
    pd.DataFrame(boxes, columns=["x", "y", "w", "h"]).to_csv(csv_path, index=False)


if __name__ == '__main__':
    parser = ArgumentParser("split images and masks")
    parser.add_argument("--train-dir", default="/mnt/datasets/spacenet/train/", type=str)
    parser.add_argument("--out-dir", default="/mnt/datasets/spacenet/train/", type=str)
    parser.add_argument("--workers", default=32, type=int, help="Num workers to process masks")
    parser.add_argument("--size", default=3072, type=int)
    args = parser.parse_args()
    pool = Pool(args.workers)
    geo_files = glob(os.path.join(args.train_dir, "*", "labels", "*Buildings.geojson"))
    mask_files = [g.split("/")[-1].replace("geojson", "png") for g in geo_files]

    with Pool(processes=args.workers) as p:
        with tqdm(total=len(mask_files)) as pbar:
            for i, v in tqdm(enumerate(
                    p.imap_unordered(
                        partial(split_img_mask, train_dir=args.train_dir, out_dir=args.out_dir, tile_size=args.size),
                        mask_files))):
                pbar.update()

    labels_dir = os.path.join(args.out_dir, "labels_split")
    boxes_dir = os.path.join(args.out_dir, "boxes_split")
    os.makedirs(boxes_dir, exist_ok=True)
    label_files = glob(os.path.join(labels_dir, "*.tif"))
    with Pool(processes=args.workers) as p:
        with tqdm(total=len(geo_files), desc="creating building boxes") as pbar:
            for i, v in tqdm(enumerate(
                    p.imap_unordered(partial(generate_boxes, boxes_dir=boxes_dir), label_files))):
                pbar.update()
