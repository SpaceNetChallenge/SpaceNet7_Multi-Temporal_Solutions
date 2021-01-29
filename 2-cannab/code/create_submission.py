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
from shapely.wkt import dumps, loads
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

import glob

import json

import math

import fiona

import tqdm

pred_dir = '/wdata/test_pred_4k_double'

imgs_dir = '/data/SN7_buildings/test_public'
if len(sys.argv) > 0:
    imgs_dir = sys.argv[1]

out_file = 'solution.csv'
if len(sys.argv) > 1:
    out_file = sys.argv[2] 

scale = 4

def sn7_convert_geojsons_to_csv(json_dirs, population='proposal'):
    '''
    Convert jsons to csv
    Population is either "ground" or "proposal" 
    '''
    
    first_file = True  # switch that will be turned off once we process the first file
    for json_dir in tqdm.tqdm(json_dirs):
        json_files = sorted(glob.glob(os.path.join(json_dir, '*.geojson')))
        for json_file in tqdm.tqdm(json_files):
            try:
                df = gpd.read_file(json_file)
            except (fiona.errors.DriverError):
                message = '! Invalid dataframe for %s' % json_file
                print(message)
                continue
                #raise Exception(message)
            if population == 'ground':
                file_name_col = df.image_fname.apply(lambda x: os.path.splitext(x)[0])
            elif population == 'proposal':
                file_name_col = os.path.splitext(os.path.basename(json_file))[0]
            else:
                raise Exception('! Invalid population')

            all_geom = []
            for g in df.geometry.scale(xfact=1/scale, yfact=1/scale, origin=(0, 0)):
                g0 = g.simplify(0.25)
                g0 = loads(dumps(g0, rounding_precision=2)) 
                all_geom.append(g0)
            df = gpd.GeoDataFrame({
                'filename': file_name_col,
                'id': df.Id.astype(int),
                'geometry': all_geom,
            })
            if len(df) == 0:
                message = '! Empty dataframe for %s' % json_file
                print(message)
                #raise Exception(message)

            if first_file:
                net_df = df
                first_file = False
            else:
                net_df = net_df.append(df)
                
    
    return net_df


if __name__ == '__main__':
    t0 = timeit.default_timer()

    out_dir_csv = os.path.join(pred_dir, 'csvs')
    os.makedirs(out_dir_csv, exist_ok=True)
    prop_file = os.path.join(out_dir_csv, out_file)

    aoi_dirs = sorted([os.path.join(pred_dir, 'tracked', aoi) \
                    for aoi in os.listdir(os.path.join(pred_dir, 'tracked')) \
                    if os.path.isdir(os.path.join(pred_dir, 'tracked', aoi))])
    print("aoi_dirs:", aoi_dirs)

    net_df = sn7_convert_geojsons_to_csv(aoi_dirs, 'proposal')


    all_files = []
    for f in sorted(listdir(pred_dir)):
        if '.png' in f:
            all_files.append(f.split('.png')[0])

    used_files = net_df['filename'].unique()
    df_add = []
    for f in all_files:
        if f not in used_files:
            df_add.append({'filename': f, 'id': 0, 'geometry': 'POLYGON EMPTY'})

    if len(df_add) > 0:
        df_add = gpd.GeoDataFrame(df_add, columns=['filename', 'id', 'geometry'])
        net_df = net_df.append(df_add)
        print('added', len(df_add), 'files')

    net_df.to_csv(prop_file, index=False, float_format='%.3f')

    print("prop_file:", prop_file)

    print("OK!")

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))