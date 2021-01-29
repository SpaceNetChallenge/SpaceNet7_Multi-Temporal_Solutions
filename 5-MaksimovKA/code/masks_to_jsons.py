import os
import numpy as np
import rasterio.features
import cv2
from scipy import ndimage as ndi
from skimage.morphology import watershed
from tqdm import tqdm
import shutil
from shapely.geometry import shape
import geopandas as gpd

from multiprocessing.pool import Pool
from multiprocessing import cpu_count

path = '/wdata/mix_predicts/'
aois = os.listdir(path)
prob_trs = 0.3
shift = 0.4
min_lolygon_area = 4

json_main_path = '/wdata/jsons_predicts/'
if os.path.exists(json_main_path):
    shutil.rmtree(json_main_path)
os.mkdir(json_main_path)

crs = rasterio.crs.CRS()
params = []
for aoi in aois:
    params.append(aoi)

def save_one_aoi(aoi):
    aoi_path = os.path.join(path, aoi)
    files = [os.path.join(aoi_path, el) for el in os.listdir(aoi_path)]
    for file_i, _file in enumerate(files):
        pred = cv2.imread(_file) / 255.0
        # plt.figure(figsize=(48, 48))
        # plt.imshow(pred[..., 0])
        pred_data = pred[...]
        img_copy = np.copy(pred_data[:, :, 0])
        m = pred_data[:, :, 0] * (1 - pred_data[:, :, 2])
        # m = pred_data[:, :, 0]
        # plt.figure(figsize=(48, 48))
        # plt.imshow(m > prob_trs + shift)
        img_copy[m <= prob_trs + shift] = 0
        img_copy[m > prob_trs + shift] = 1
        img_copy = img_copy.astype(np.bool)
        markers = ndi.label(img_copy, output=np.uint32)[0]
        # print(len(np.unique(markers)))
        if file_i == 0:
            total_seeds = (np.copy(img_copy)).astype(np.uint8)
            # total_seeds = m[...]
        else:
            total_seeds += (np.copy(img_copy)).astype(np.uint8)
            # total_seeds += m[...]
        # plt.figure(figsize=(48, 48))
        # plt.imshow(img_copy)

    # plt.figure(figsize=(48, 48))
    # print(np.unique(total_seeds))
    # total_seeds = total_seeds / len(files)
    total_seeds = total_seeds >= int((4 / 25) * len(files))
    # total_seeds = total_seeds >= 4
    # total_seeds = total_seeds > 0.5
    # plt.imshow(total_seeds)
    # markers = ndi.label(total_seeds, output=np.uint32)[0]
    # print('####')
    # print(len(np.unique(markers)))
    for file_index, _file in enumerate(files[:]):
        # print(_file)
        pred = cv2.imread(_file) / 255.0
        # plt.figure(figsize=(48, 48))
        # plt.imshow(pred[..., 0])
        pred_data = pred[...]

        img_copy = np.copy(pred_data[:, :, 0])
        # m = pred_data[:, :, 0] * (1 - pred_data[:, :, 2])
        m = pred_data[:, :, 0]
        # plt.figure(figsize=(48, 48))
        # plt.imshow(m > prob_trs + shift)
        img_copy[m <= prob_trs + shift] = 0
        img_copy[m > prob_trs + shift] = 1
        img_copy = img_copy.astype(np.bool)

        mask_img = np.copy(pred_data[:, :, 0])
        mask_img[mask_img <= prob_trs] = 0
        mask_img[mask_img > prob_trs] = 1
        mask_img = mask_img.astype(np.bool)
        markers = ndi.label(total_seeds, output=np.uint32)[0]
        # markers = ndi.label(img_copy, output=np.uint32)[0]
        pred_labels = watershed(mask_img, markers, mask=mask_img, watershed_line=True)
        if np.sum(pred_labels) == 0:
            pred_labels[:4, :4] = 1
        # print(len(np.unique(pred_labels)))
        # plt.figure()
        # plt.imshow(pred_labels)
        # print(len(np.unique(labels)))
        polygons = []
        values = []
        dets = []
        aoi = _file.split('/')[-2]
        aoi_json_path = os.path.join(json_main_path, aoi)
        if not os.path.exists(aoi_json_path):
            os.mkdir(aoi_json_path)

        polygon_generator = rasterio.features.shapes(pred_labels,
                                                     pred_labels > 0  # ,
                                                     # transform=transform
                                                     )
        i = 0
        for polygon, value in polygon_generator:
            p = shape(polygon).buffer(0.0)
            if p.area >= min_lolygon_area:
                polygons.append(shape(polygon).buffer(0.0))
                values.append(value)
        polygon_gdf = gpd.GeoDataFrame({'geometry': polygons},
                                       crs=crs.to_wkt())
        polygon_gdf['geometry'] = polygon_gdf['geometry'].apply(
            lambda x: x.simplify(tolerance=0.5)
        )
        # print(len(polygon_gdf))
        # print(len(polygon_gdf))
        _id = _file.split('/')[-1].split('.')[0]
        save_path = os.path.join(aoi_json_path, _id + '.geojson')
        # print(save_path)
        polygon_gdf.to_file(save_path, driver='GeoJSON')

n_cpus = cpu_count()
# make_jsons(params[0])
pool = Pool(n_cpus)
for _ in tqdm(pool.imap_unordered(save_one_aoi, params), total=len(params)):
    pass