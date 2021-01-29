import shutil
import os
import fire
import numpy as np
import cv2
import geopandas as gpd
import skimage.io
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage import measure
from skimage.morphology import dilation, square, watershed
from scipy.ndimage import binary_erosion
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from rasterio import features
from scipy.ndimage import binary_dilation


def create_separation(labels):
    tmp = dilation(labels > 0, square(3))
    tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    tmp = tmp ^ tmp2
    tmp = dilation(tmp, square(3))

    msk1 = np.zeros_like(labels, dtype='bool')

    for y0 in range(labels.shape[0]):
        for x0 in range(labels.shape[1]):
            if not tmp[y0, x0]:
                continue
            sz = 1
            uniq = np.unique(labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1),
                             max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)])
            if len(uniq[uniq > 0]) > 1:
                msk1[y0, x0] = True
    return msk1


def mask_fro_id(param):
    _id, labels_path, rasters_path, result_path = param
    label_path = os.path.join(labels_path, _id + '_Buildings.geojson')
    raster_path = os.path.join(rasters_path, _id + '.tif')
    geoms = gpd.read_file(label_path)['geometry'].tolist()
    image = cv2.imread(raster_path)
    h, w, c = image.shape

    buildings = np.zeros((h, w), dtype=np.int64)
    outer_contur = np.zeros((h, w), dtype=np.int64)
    inter_contur = np.zeros((h, w), dtype=np.int64)
    contour_size = 1

    for i in range(len(geoms)):
        mask = features.rasterize([(geoms[i], 1)], out_shape=(h, w))
        buildings += mask
        dilated = binary_dilation(mask, iterations=contour_size)
        countour_mask = dilated ^ mask
        outer_contur += countour_mask

        eroded = binary_erosion(mask, iterations=contour_size)
        countour_mask = eroded ^ mask
        inter_contur += countour_mask

    outer_contur = (outer_contur > 0).astype(np.uint8)
    inter_contur = (inter_contur > 0).astype(np.uint8)
    buildings = (buildings > 0).astype(np.uint8)
    buildings[outer_contur == 1] = 0

    labels = ndi.label(buildings, output=np.uint32)[0]
    separation = create_separation(labels)
    separation = separation > 0
    separation = separation.astype(np.uint8)

    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[:, :, 0] = buildings * 255
    result[:, :, 1] = inter_contur * 255
    result[:, :, 2] = separation * 255
    out_path = os.path.join(result_path, _id + '.tif')
    skimage.io.imsave(out_path, result, plugin='tifffile')


def create_masks(data_root_path='/data/SN7_buildings/train/',
                 result_path='/wdata/train_masks/'):

    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)
    ids = os.listdir(data_root_path)
    all_params = []
    for _id in tqdm(ids[:]):
        id_path = os.path.join(data_root_path, _id)
        if not os.path.isdir(id_path):
            continue
        sub_res_path = os.path.join(result_path, _id)
        os.mkdir(sub_res_path)
        labels_path = os.path.join(id_path, 'labels_match_pix')
        rasters_path = os.path.join(id_path, 'images')

        files = sorted(os.listdir(labels_path))
        files = [el for el in files if 'UDM' not in el]
        files = ['_'.join(el.split('.')[0] .split('_')[:-1])  for el in files]
        params = [(el, labels_path, rasters_path, sub_res_path) for el in files]
        all_params += params

    n_cpus = cpu_count()
    pool = Pool(n_cpus)
    for _ in tqdm(pool.imap_unordered(mask_fro_id, all_params), total=len(all_params)):
      pass


if __name__ == '__main__':
    fire.Fire(create_masks)
