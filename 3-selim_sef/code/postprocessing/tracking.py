import glob
import math
import os
import traceback
from collections import defaultdict
from typing import List, Dict

import cv2
import geopandas as gpd
import pyclipper
from cached_property import cached_property
from scipy.spatial import KDTree
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from skimage import measure
from tqdm import tqdm
import numpy as np

from postprocessing.polygonize import mask_to_shapely_polygon


def euclidean(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def simplify(polygon: Polygon, tol=0.2):
    try:
        polygon = polygon.buffer(0)
        return polygon.simplify(tolerance=tol, preserve_topology=True)
    except:
        return polygon


def to_coords(p: Polygon):
    exterior = list(zip(*p.exterior.coords.xy))
    if p.interiors:
        coord = [exterior]
        for interior in p.interiors:
            coord.append(list(zip(*interior.coords.xy)))
        return (np.asarray(coord) * 16).astype(int)
    else:
        return (np.asarray(exterior) * 16).astype(int)


def iou_clipper(p1, p2):
    pc = pyclipper.Pyclipper()
    pc.AddPath(to_coords(p1), pyclipper.PT_CLIP, True)
    pc.AddPath(to_coords(p2), pyclipper.PT_SUBJECT, True)
    I = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
    if len(I) > 0:
        U = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
        Ia = pyclipper.Area(I[0])
        Ua = pyclipper.Area(U[0])
        IoU = Ia / Ua
    else:
        IoU = 0.0
    return IoU


class Building:
    def __init__(self, polygon: Polygon, id: str, meta={}, ) -> None:
        super().__init__()
        self._polygon = polygon
        self._files = [id]
        self._polygons = [polygon]
        self._meta = meta
        self._metas = [meta]

    @property
    def polygon(self):
        return self._polygon

    @property
    def meta(self):
        return self._meta

    @property
    def metas(self):
        return self._metas

    @property
    def polygons(self):
        return self._polygons

    @property
    def files(self):
        return self._files

    def iou(self, poly_pred, polygon=None):
        if polygon is None:
            polygon = self.polygon
        int_area = poly_pred.polygon.intersection(polygon).area
        polygons = [poly_pred.polygon, polygon]
        u = cascaded_union(polygons)
        return float(int_area / u.area)

    def best_polygons(self):
        if hasattr(self, "_best"):
            return self._best
        N = len(self.polygons)
        ious_per_index = np.ones((N, N))
        for i in range(0, len(self.polygons)):
            for j in range(i + 1, len(self.polygons)):
                p1 = self.polygons[i]
                p2 = self.polygons[j]
                int_area = p1.intersection(p2).area
                polygons = [p1, p2]
                u = cascaded_union(polygons)
                ious_per_index[i, j] = float(int_area / u.area)
                ious_per_index[j, i] = float(int_area / u.area)

        polygons = [(self.polygons[i], np.mean(ious_per_index[i])) for i in range(N)]
        polygons.sort(key=lambda x: -x[1])
        self._best = [p for p, _ in polygons]
        return self._best

    def iou_all_polygons(self, poly_pred) -> float:
        ious = []
        for p in self.polygons[:7]:
            try:
                if p.intersects(poly_pred.polygon):
                    ious.append(self.iou(poly_pred, p))
            except:
                pass
        ious = [i for i in ious if not math.isnan(i)]
        return ious

    def intersects(self, poly_pred):
        return poly_pred.polygon.intersects(self.polygon)

    def distance(self, poly_pred):
        other_centroid = poly_pred.centroid.coords[0]
        centroid = self.polygon.centroid.coords[0]
        return euclidean(other_centroid, centroid)

    def append(self, poly_pred):
        self.polygons.append(poly_pred.polygon)
        assert len(poly_pred.files) == 1
        self.files.append(poly_pred.files[0])
        self.metas.append(poly_pred.metas[0])

    @cached_property
    def center(self):
        return self.polygon.centroid.coords[0]

    @cached_property
    def area(self):
        return self.polygon.area


def fill_gaps(all_buildings: List[Building],
              input_files: List[str],
              cropped_polygons: Dict[str, List[Polygon]]) -> List[Building]:
    print("filling gaps")

    all_buildings = [m for m in all_buildings]
    file_to_index = {f: i for i, f in enumerate(input_files)}
    polygons_per_tile = defaultdict(list)
    kdtrees_per_file = {}
    master_polygons = []
    for b in all_buildings:
        for i, f in enumerate(b.files):
            polygons_per_tile[f].append(b.polygons[i])

    counts_added = defaultdict(int)

    for file, polygons in polygons_per_tile.items():
        tree = KDTree([p.centroid.coords[0] for p in polygons])
        kdtrees_per_file[file] = tree
    for b in tqdm(all_buildings):
        fixed_building = Building(b.polygon, b.files[0], b.meta)
        for i in range(1, len(b.files)):
            fixed_building.files.append(b.files[i])
            fixed_building.polygons.append(b.polygons[i])
            fixed_building.metas.append(b.metas[i])
        master_polygons.append(fixed_building)
        if len(b.files) != len(set(b.files)):
            print(b.files)

        max_gap = 8
        min_amount = 7
        if len(b.files) < min_amount:
            continue
        if has_big_gaps(b, input_files, max_gap):
            continue
        start_idx = file_to_index[b.files[0]]
        for input_file in input_files[start_idx + 1:]:
            if input_file in b.files:
                continue

            tree = kdtrees_per_file.get(input_file, None)
            buildings = polygons_per_tile.get(input_file, [])
            distances, indices = [], []
            if tree is not None:
                distances, indices = tree.query(b.polygon.centroid, k=5, distance_upper_bound=64)

            for p in b.best_polygons():
                intersects = False
                for j, index in enumerate(indices):
                    if math.isfinite(distances[j]):
                        b2 = buildings[index]
                        if p.intersects(b2):
                            intersects = True
                            break
                if not intersects:
                    # add to tile if not in empy alpha
                    intersects_empty = False
                    if input_file in cropped_polygons:
                        for cropped in cropped_polygons[input_file]:
                            if cropped.intersects(p):
                                intersects_empty = True
                                break
                    if not intersects_empty:
                        fixed_building.append(Building(p, input_file))
                        counts_added[input_file] += 1
                        break
    print(counts_added)
    return master_polygons


def has_big_gaps(b, input_files, max_gap):
    big_gaps = False
    current_gap = 0
    for f in input_files:
        if f in b.files:
            current_gap = 0
        else:
            current_gap += 1
        if current_gap > max_gap:
            big_gaps = True
            break
    return big_gaps


def building_from_row(row, file_id) -> Building:
    meta = {
        "mean": row["mean"],
        "median": row["median"],
        "std": row["std"],
        "major_axis_length": row.major_axis_length,
        "minor_axis_length": row.minor_axis_length,
        "eccentricity": row.eccentricity,

    }
    return Building(simplify(row.geometry), file_id, meta)


def track_footprints_simple(json_dir: str, out_file: str, image_dir: str = "/mnt/datasets/spacenet/train/"):
    input_files = sorted([input_file for input_file in os.listdir(json_dir)])
    master_polygons = []  # type: List[Building]

    group = input_files[0].split("mosaic_")[-1]
    tif_tiles = glob.glob(os.path.join(image_dir, group.replace(".geojson", ""), "images_masked", "*.tif"))
    assert len(tif_tiles) == len(input_files)
    cropped_polygons = defaultdict(list)
    for img_path in tif_tiles:
        tile = os.path.splitext(os.path.basename(img_path))[0]
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        alpha = image[..., 3]
        if np.sum(alpha == 0) < 64:
            continue
        labels = measure.label(alpha == 0)
        for l in range(1, np.max(labels) + 1):
            cropped_polygons[tile + ".geojson"].append(mask_to_shapely_polygon(labels == l))

    for input_file in tqdm(input_files):
        file_id = input_file
        df = gpd.read_file(os.path.join(json_dir, input_file))
        if not master_polygons:
            for i, row in df.iterrows():
                building = building_from_row(row, file_id, )
                master_polygons.append(building)
        else:
            tree = KDTree([p.center for p in master_polygons])
            proposal_polygons = [building_from_row(row, file_id) for i, row in df.iterrows()]
            for proposal in proposal_polygons:
                indices = tree.query(proposal.center, k=3)[1]
                best_idx = -1
                best_iou = 0
                for i in indices:
                    if input_files in master_polygons[i].files:
                        # already used, skipping
                        continue
                    try:
                        ious = master_polygons[i].iou_all_polygons(proposal)
                        iou = 0
                        if ious:
                            iou = np.max(ious)
                        if iou > 0.2 and iou > best_iou:
                            best_idx = i
                            best_iou = iou
                    except:
                        traceback.print_exc()
                        pass

                if best_idx > -1:
                    master_poly = master_polygons[best_idx]
                    if file_id not in master_poly.files:
                        master_poly.append(proposal)
                else:
                    master_polygons.append(proposal)
    data = []
    min_buildings = len(input_files) // 4
    all_polygons = []
    for i, building in enumerate(tqdm(master_polygons)):
        if len(building.files) >= min_buildings:
            all_polygons.append(building)
        else:
            if is_correct_late_change(building, input_files):
                all_polygons.append(building)
    all_polygons = fill_gaps(all_polygons, input_files, cropped_polygons)

    current_id = 0
    not_empty_files = set()
    for building in all_polygons:
        for j, f in enumerate(building.files):
            not_empty_files.add(f)
            f = f.replace(".geojson", "")
            data.append(
                [
                    f,
                    current_id,
                    building.polygons[j].wkt
                ]
            )
        current_id += 1
    for f in input_files:
        if f not in not_empty_files:
            data.append([f.replace(".geojson", ""), 0, "POLYGON EMPTY"])

    import pandas as pd
    pd.DataFrame(data, columns=["filename", "id", "geometry"]).to_csv(out_file, index=False)


def is_correct_late_change(building, input_files):
    position = len(building.files) - input_files.index(building.files[0])
    if position == 3:
        min_amount = 3
        median_prob = 0.92
    elif position == 4:
        min_amount = 3
        median_prob = 0.9
    elif position == 5:
        min_amount = 4
        median_prob = 0.85
    else:
        return False
    median = building.meta["median"]
    correct_late_change = median > median_prob and len(building.files) >= min_amount
    return correct_late_change
