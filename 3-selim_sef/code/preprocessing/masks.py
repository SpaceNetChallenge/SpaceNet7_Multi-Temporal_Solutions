import os
import traceback
import warnings
from collections import defaultdict

warnings.simplefilter("ignore")
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
from cv2.cv2 import fillPoly

from argparse import ArgumentParser
from functools import partial
from multiprocessing.pool import Pool

import numpy as np
from PIL import Image

from scipy.ndimage import binary_erosion, binary_dilation
from skimage import measure
from skimage.morphology import dilation, square
from skimage.segmentation import relabel_sequential, watershed
from tqdm import tqdm
import pandas as pd


def create_separation(labels, max_border_size=9):
    """
    Creates mask of touching borders between buildings

    Args:
      labels: mask of labeled connected components
    """
    tmp = dilation(labels > 0, square(11))
    tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    tmp = tmp ^ tmp2
    tmp = dilation(tmp, square(3))

    props = measure.regionprops(labels)
    eroded_labels = np.zeros_like(labels)
    for prop in props:
        y1, x1, y2, x2 = prop.bbox
        try:
            if prop.minor_axis_length < 12:
                iterations = 1
            elif prop.minor_axis_length < 12:
                iterations = 2
            elif prop.minor_axis_length < 20:
                iterations = 3
            elif prop.minor_axis_length < 32:
                iterations = 4
            else:
                iterations = 5
        except:
            iterations = 3
        center = binary_erosion(prop.image, iterations=iterations)
        eroded_labels[y1:y2, x1:x2][center > 0] = 1
    mask = np.zeros_like(labels, dtype='bool')

    for y0 in range(labels.shape[0]):
        for x0 in range(labels.shape[1]):
            if not tmp[y0, x0] and labels[y0, x0] == 0:
                continue
            if labels[y0, x0] == 0:
                sz = 3
            else:
                sz = 3
                lbl = labels[y0, x0] - 1
                if props[lbl].minor_axis_length > 12:
                    sz = 4
                if props[lbl].minor_axis_length > 20:
                    sz = 5
            uniq = np.unique(
                labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz), max(0, x0 - sz):min(labels.shape[1], x0 + sz)])
            if len(uniq[uniq > 0]) > 1:
                mask[y0, x0] = True
    mask[eroded_labels > 0] = False
    return mask


def create_mask(mask, max_contour_size=3, max_border_size=9):
    """
    Produces  3 channel image (building mask, birders, touching borders) from the given binary segmentation mask
    Args:
      mask: input mask to be processed
      max_contour_size: size of image countours
      max_border_size: max size of touching borders that will separate close connected components
    """
    labels = mask

    final_mask = np.zeros((labels.shape[0], labels.shape[1], 3))
    building_num = np.max(labels)

    if building_num > 0:
        for i, rprops in enumerate(measure.regionprops(labels)):
            i = i + 1
            y1, x1, y2, x2 = rprops.bbox
            padding = max(16, min((y2 - y1), (x2 - x1)) // 2)
            y1 = max(y1 - padding, 0)
            x1 = max(x1 - padding, 0)
            y2 = min(y2 + padding, labels.shape[0])
            x2 = min(x2 + padding, labels.shape[1])
            # print(i, building_num)
            labels_rprop = labels[y1:y2, x1:x2]
            building_mask = np.zeros_like(labels_rprop, dtype='bool')
            building_mask[labels_rprop == i] = 1
            area = np.sum(building_mask)
            if area < 500:
                contour_size = max_contour_size - 2
            elif area < 1000:
                contour_size = max_contour_size - 1
            else:
                contour_size = max_contour_size
            eroded = binary_erosion(building_mask, iterations=contour_size)
            countour_mask = building_mask ^ eroded
            # plt.imshow(building_mask)
            # plt.show()
            final_mask[..., 0][y1:y2, x1:x2] += building_mask
            final_mask[..., 1][y1:y2, x1:x2] += countour_mask
        final_mask[..., 2] = create_separation(labels, max_border_size=max_border_size)
    return np.clip(final_mask * 255, 0, 255).astype(np.uint8)


def generate_labels(data, labels_dir, masks_dir):
    tile_id, polygons = data
    out_mask_path = os.path.join(masks_dir, tile_id + ".png")
    # if os.path.exists(out_mask_path):
    #     return
    try:
        labels = np.zeros((3072, 3072), np.int16)
        label = 1

        for feat in polygons:
            if feat == "LINESTRING EMPTY" or feat == "POLYGON EMPTY":
                continue
            feat = feat.replace("POLYGON ((", "").replace("POLYGON Z ((", "").replace("), (", "|").replace("),(", "|").replace("(", "").replace(")", "")
            feat_polygons = feat.split("|")
            for i, polygon in enumerate(feat_polygons):
                polygon_coords = []
                for xy in polygon.split(","):
                    xy = xy.strip()
                    x, y, *_ = xy.split(" ")
                    x = float(x)
                    y = float(y)
                    polygon_coords.append([x, y])

                coords = np.round(np.array(polygon_coords) * 3).astype(np.int32)
                fillPoly(labels, [coords], label if i == 0 else 0)
                label += 1
        labels, _, _ = relabel_sequential(labels)
        small_labels = np.zeros_like(labels)
        eroded_labels = np.zeros_like(labels)
        big_labels = np.zeros_like(labels)
        very_big_labels = np.zeros_like(labels)
        for prop in measure.regionprops(labels):
            y1, x1, y2, x2 = prop.bbox
            if prop.minor_axis_length > 8:
                very_big_labels[y1:y2, x1:x2][prop.image > 0] = 1
            elif prop.minor_axis_length > 4:
                big_labels[y1:y2, x1:x2][prop.image > 0] = 1
            else:
                small_labels[y1:y2, x1:x2][prop.image > 0] = 1

            eroded_lbl = binary_erosion(prop.image, iterations=1)

            eroded_labels[y1:y2, x1:x2][eroded_lbl > 0] = prop.label
        cv2.imwrite(os.path.join(labels_dir, tile_id + ".tif"), labels.astype(np.uint16))
        binary_image = binary_dilation((labels > 0), iterations=2).astype(np.uint8)
        tmp = watershed(binary_image, mask=binary_image, markers=eroded_labels, watershed_line=True)
        ws_line = (binary_image ^ (tmp > 0)) * (1 - small_labels)
        fat_ws_line = (binary_dilation(ws_line, iterations=1) > 0) * (1 - big_labels) * (1 - small_labels)
        labels[ws_line > 0] = 0
        labels[fat_ws_line > 0] = 0
        labels, _, _ = relabel_sequential(labels)
        out = create_mask(labels).astype(np.uint8)
        out = Image.fromarray(out)

        out.save(out_mask_path, optimize=True)
    except Exception as e:
        traceback.print_exc()
        print(tile_id)

if __name__ == '__main__':
    parser = ArgumentParser("rasterize masks")
    parser.add_argument("--train-dir", default="/mnt/datasets/spacenet/train/", type=str)
    parser.add_argument("--out-dir", default="/mnt/datasets/spacenet/train/masks/", type=str,
                        help="directory to write masks")
    parser.add_argument("--workers", default=32, type=int, help="Num workers to process masks")
    parser.add_argument("--size", default=3072, type=int)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    labels_dir = args.out_dir.replace("masks", "labels")
    os.makedirs(labels_dir, exist_ok=True)
    change_masks_dir = args.out_dir.replace("masks", "change_masks")
    os.makedirs(change_masks_dir, exist_ok=True)
    change_labels_dir = args.out_dir.replace("masks", "change_labels")
    os.makedirs(change_labels_dir, exist_ok=True)
    csv_path = os.path.join(args.train_dir, "sn7_train_ground_truth_pix.csv")
    df = pd.read_csv(csv_path)
    polygons = df[["filename", "geometry"]].values
    groups = defaultdict(list)
    for row in polygons:
        tile_id, polygon = row
        groups[tile_id].append(polygon)

    with Pool(args.workers) as pool:
        with tqdm(total=len(groups)) as pbar:
            for _ in pool.imap_unordered(partial(generate_labels, labels_dir=labels_dir, masks_dir=args.out_dir),
                                         groups.items()):
                pbar.update()