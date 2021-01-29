# -*- coding: utf-8 -*-
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

from os import path, mkdir, listdir, makedirs
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import timeit
import cv2
from tqdm import tqdm
from skimage import io
from skimage import measure
from skimage.morphology import square, erosion, dilation
from skimage.morphology import remove_small_objects, watershed, remove_small_holes
from skimage.color import label2rgb
from scipy import ndimage
import pandas as pd
from sklearn.model_selection import KFold
from shapely.wkt import dumps
from shapely.geometry import shape, Polygon
from collections import defaultdict
from multiprocessing import Pool

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import rasterio
from rasterio import features
import shapely
import shapely.ops
import geopandas as gpd

import json

threshold = 140
sep_thr = 0.6

out_dir = '/wdata/test_pred_4k_double'

imgs_dir = '/data/SN7_buildings/test_public'
if len(sys.argv) > 0:
    imgs_dir = sys.argv[1]

scale = 4

def process_image(fid):

    try:
        aoi = fid.split('mosaic_')[1]
        img = io.imread(path.join(imgs_dir, aoi, 'images_masked', '{0}.tif'.format(fid)))

        msk = cv2.imread(path.join(out_dir, '{}.png'.format(fid)), cv2.IMREAD_UNCHANGED)

        msk = msk[:img.shape[0] * scale, :img.shape[1] * scale, :]

        msk0 = msk / 255.
        msk0 = msk0[..., 0] * (1 - 0.5 * msk0[..., 1]) * (1 - 0.5 * msk0[..., 2])
        msk0 = 1 * (msk0 > sep_thr)
        msk0 = msk0.astype(np.uint8)

        y_pred = measure.label(msk0, connectivity=2, background=0)
        props = measure.regionprops(y_pred)
        y_pred = measure.label(y_pred, connectivity=2, background=0)

        shp_msk = (255 - msk[..., 0])
        shp_msk = shp_msk.astype('uint8')
        y_pred = watershed(shp_msk, y_pred, mask=((msk[..., 0] > threshold)), watershed_line=False)

        props = measure.regionprops(y_pred)

        for i in range(len(props)):
            if props[i].area < 40:
                y_pred[y_pred == i+1] = 0
        y_pred = measure.label(y_pred, connectivity=1, background=0).astype('int32')

        crs = rasterio.crs.CRS()
        polygon_generator = rasterio.features.shapes(y_pred, y_pred > 0)
        polygons = []
        values = []  # pixel values for the polygon in mask_arr 
        for polygon, value in polygon_generator:
            p = shape(polygon).buffer(0.0)

            if p.area >= 0:
                polygons.append(shape(polygon).buffer(0.0))
                values.append(value)

        polygon_gdf = gpd.GeoDataFrame({'geometry': polygons, 'value': values},
                                    crs=crs.to_wkt())

        # save output files
        if len(polygon_gdf) > 0:
            output_path = path.join(out_dir, 'grouped', aoi, '{}.geojson'.format(fid))
            makedirs(path.join(out_dir, 'grouped', aoi), exist_ok=True)
            polygon_gdf.to_file(output_path, driver='GeoJSON')


    except Exception as ex:
        print('Exception occured: {}. File: {}'.format(ex, fid))



if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(path.join(out_dir, 'grouped'), exist_ok=True)
    val_files = []

    for f in listdir(out_dir):
        if '.png' in f:
            val_files.append(f.split('.png')[0])

    val_files = np.asarray(val_files)

    total_tp = 0
    total_fn = 0
    total_fp = 0

    with Pool() as pool:
        results = pool.map(process_image, val_files)


    print("OK!")

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))