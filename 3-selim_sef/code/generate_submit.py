import argparse
import os
from typing import Dict, List

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from glob import glob
from shapely import wkt
from skimage.color import label2rgb



from postprocessing.polygonize import mask_to_poly
from postprocessing.tracking import track_footprints_simple

import geojson
from skimage import measure

from functools import partial
from multiprocessing.pool import Pool

import torch
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential
from tqdm import tqdm

from postprocessing.labeling import label_mask

from postprocessing.instance import find_instance_center

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import numpy as np
import pandas as pd


def extract_geo_features(file_id: str, preds_dir: str):
    mask = cv2.imread(os.path.join(preds_dir, file_id + "_mask.png"))
    centers = cv2.imread(os.path.join(preds_dir, file_id + "_centers.png"), cv2.IMREAD_GRAYSCALE)
    centers = torch.from_numpy(np.expand_dims(np.expand_dims(centers, 0), 0).astype(np.float32) / 255)
    center_idx = find_instance_center(ctr_hmp=centers.float(), threshold=0.1, nms_kernel=3)
    seeds = np.zeros(mask.shape[:2], dtype="int16")
    for ci, center in enumerate(center_idx.cpu().numpy()):
        cv2.circle(seeds, (center[1], center[0]), radius=1, thickness=2, color=ci + 1)
    # make separators empty where seeds exist
    mask[seeds > 0, 1] = 0
    mask[seeds > 0, 2] = 0
    ws_instances = label_mask(mask / 255, main_threshold=0.4,
                              seed_threshold=0.6,
                              w_pixel_t=10, pixel_t=45)
    final_instances = remove_small_objects(ws_instances, min_size=55)
    final_instances_ori, _, _ = relabel_sequential(final_instances)

    return file_id, generate_polygon_dicts(final_instances_ori, mask)


def generate_polygon_dicts(final_instances_ori: np.ndarray,
                           mask: np.ndarray,
                           original_size: int = 1024) -> List[Dict]:
    """Generates polygons from a labeled mask
    Parameters
    ----------
    final_instances_ori : ndarray (u)int16
        Contains instance labels
    mask : ndarray uint8
        Binary segmentation mask used to generate these labels should have (H, W, C) format.
        Only first channel is required as it represents object bodies.
    original_size int:
        h/w of the image before rescaling

    Returns
    -------
    features : dict
        List of geojson features as dict
    """
    final_instances = cv2.resize(final_instances_ori, (4096, 4096), interpolation=cv2.INTER_NEAREST)
    features = []
    scale = original_size / 4096
    mask_prob = mask[:, :, 0] / 255
    props_ori = measure.regionprops(final_instances_ori)
    for i, rprop in enumerate(measure.regionprops(final_instances)):
        cropped_mask = mask_prob[props_ori[i].slice] * (props_ori[i].image > 0)
        y1, x1, y2, x2 = rprop.bbox
        geo = mask_to_poly(np.pad(rprop.image, (16, 16)))

        x_t = lambda x: (x1 + x - 16) * scale
        y_t = lambda y: (y1 + y - 16) * scale
        if geo["type"] == "Polygon":
            geo["coordinates"] = [[(x_t(c[0]), y_t(c[1])) for c in geo["coordinates"][0]]]
        else:
            geo["coordinates"] = [[[(x_t(c[0]), y_t(c[1])) for c in coord[0]]] for coord in geo["coordinates"]]
        features.append(
            {"type": "Feature",
             "geometry": geo,
             "properties": {
                 "mean": np.mean(cropped_mask),
                 "median": np.median(cropped_mask),
                 "std": np.std(cropped_mask),
                 "eccentricity": props_ori[i].eccentricity,
                 "minor_axis_length": props_ori[i].minor_axis_length,
                 "major_axis_length": props_ori[i].major_axis_length,
             }
             }
        )
    return features


def track_footprints(in_dir: str, out_csv_dir: str, image_dir: str):
    group = in_dir.split("/")[-1]
    track_footprints_simple(in_dir, os.path.join(out_csv_dir, group + ".csv"), image_dir=image_dir)


def generate_geojson_polygons(ids: List[str], workers: int, preds_dir: str, json_dir: str):
    with Pool(processes=workers) as p:
        with tqdm(total=len(ids), desc="Generating polygons") as pbar:
            for i, v in tqdm(enumerate(
                    p.imap_unordered(partial(extract_geo_features, preds_dir=preds_dir),
                                     ids))):
                pbar.update()
                id, labels = v
                collection = geojson.FeatureCollection(labels)
                group = id.replace("_0_0", "").split("mosaic_")[-1]
                out_dir = os.path.join(json_dir, group)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, id.replace("_0_0", "") + ".geojson"), "w") as f:
                    geojson.dump(collection, f)


def generate_building_tracks(out_json_dirs: List[str], workers: int, out_csv_dir: str, image_dir: str):
    os.makedirs(out_csv_dir, exist_ok=True)
    with Pool(processes=workers) as p:
        with tqdm(total=len(out_json_dirs), desc="Tracking") as pbar:
            for i, v in tqdm(enumerate(
                    p.imap_unordered(
                        partial(track_footprints, out_csv_dir=out_csv_dir, image_dir=image_dir),
                        out_json_dirs))):
                pbar.update()

def simplify_polygon(row):
    filename, id, geometry = row
    p = wkt.loads(geometry)
    p = p.simplify(0.25)
    return filename, id, wkt.dumps(p, rounding_precision=1).replace(".0 ", " ").replace(".0,", ",")

def concat_and_simplify_polygons(workers: int, out_csv_dir: str, out_file: str):
    df = None
    for csv_file in glob(os.path.join(out_csv_dir, "*.csv")):
        print("Reading {} ".format(csv_file))
        df_tile = pd.read_csv(csv_file)
        if df is None:
            df = df_tile
        else:
            df = pd.concat([df, df_tile])
    rows = df.values
    data = []
    with Pool(processes=workers) as p:
        with tqdm(total=len(rows), desc="simplifying polygons boxes") as pbar:
            for i, v in tqdm(enumerate(
                    p.imap_unordered(simplify_polygon, rows))):
                pbar.update()
                data.append(v)

    print("Writing {} polygons to {} ".format(len(data), out_file))
    pd.DataFrame(data, columns=["filename", "id", "geometry"]).to_csv(out_file, index=False)


def parse_args():
    parser = argparse.ArgumentParser("Spacenet 7 solution processor")
    arg = parser.add_argument
    arg('--preds-dir', type=str, default='../test_results/ensemble')
    arg('--workers', type=int, default=28)
    arg('--json-dir', type=str, default="../test/jsons")
    arg('--out-csv-dir', type=str, default="../test/csvs")
    arg('--out-file', type=str, default="../test/solution/solution.csv")
    arg('--image-dir', type=str, default="/mnt/datasets/spacenet/test_public")

    return parser.parse_args()


def main():
    args = parse_args()
    ids = sorted([f.split("_mask")[0] for f in os.listdir(args.preds_dir) if "_mask" in f])
    generate_geojson_polygons(ids=ids,
                              workers=args.workers,
                              preds_dir=args.preds_dir,
                              json_dir=args.json_dir)
    out_dirs = sorted(os.path.join(args.json_dir, f) for f in os.listdir(args.json_dir))

    generate_building_tracks(out_json_dirs=out_dirs,
                             workers=args.workers,
                             out_csv_dir=args.out_csv_dir,
                             image_dir=args.image_dir)
    concat_and_simplify_polygons(workers=args.workers, out_csv_dir=args.out_csv_dir, out_file=args.out_file)


if __name__ == '__main__':
    main()
