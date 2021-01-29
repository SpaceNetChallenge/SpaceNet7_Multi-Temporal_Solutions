# -*- coding: utf-8 -*-
import os

from numpy.lib.function_base import append
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
from shapely.strtree import STRtree
import geopandas as gpd

import json

import math

pred_dir = '/wdata/test_pred_4k_double'


def calculate_iou(pred_poly, test_data_GDF):
    """Get the best intersection over union for a predicted polygon.
    Adapted from: https://github.com/CosmiQ/solaris/blob/master/solaris/eval/iou.py, but
    keeps index of test_data_GDF
    
    Arguments
    ---------
    pred_poly : :py:class:`shapely.Polygon`
        Prediction polygon to test.
    test_data_GDF : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame of ground truth polygons to test ``pred_poly`` against.
    Returns
    -------
    iou_GDF : :py:class:`geopandas.GeoDataFrame`
        A subset of ``test_data_GDF`` that overlaps ``pred_poly`` with an added
        column ``iou_score`` which indicates the intersection over union value.
    """

    # Fix bowties and self-intersections
    if not pred_poly.is_valid:
        pred_poly = pred_poly.buffer(0.0)

    precise_matches = test_data_GDF[test_data_GDF.intersects(pred_poly)]

    iou_row_list = []
    for idx, row in precise_matches.iterrows():
        # Load ground truth polygon and check exact iou
        test_poly = row.geometry
        # Ignore invalid polygons for now
        if pred_poly.is_valid and test_poly.is_valid:
            intersection = pred_poly.intersection(test_poly).area
            union = pred_poly.union(test_poly).area
            # Calculate iou
            iou_score = intersection / float(union)
            gt_idx = idx
        else:
            iou_score = 0
            gt_idx = -1
        row['iou_score'] = iou_score
        row['gt_idx'] = gt_idx
        iou_row_list.append(row)

    iou_GDF = gpd.GeoDataFrame(iou_row_list)
    return iou_GDF



def process_image(aoi):

    try:

        '''
        Track footprint identifiers in the deep time stack.
        We need to track the global gdf instead of just the gdf of t-1.
        '''
        id_field = 'Id'
        iou_field = 'iou_score'
        verbose = False #True
        super_verbose = False

        min_iou = 0.15 
        json_dir = path.join(pred_dir, 'grouped', aoi)
        out_dir = path.join(pred_dir, 'tracked', aoi)

        os.makedirs(out_dir, exist_ok=True)
        
        # set columns for master gdf
        gdf_master_columns = [id_field, iou_field, 'area', 'geometry']

        json_files = sorted([f
                    for f in os.listdir(os.path.join(json_dir))
                    if f.endswith('.geojson') and os.path.exists(os.path.join(json_dir, f))])


        # # check if only partical matching has been done (this will cause errors)
        # out_files_tmp = sorted([z for z in os.listdir(out_dir) if z.endswith('.geojson')])
        # if len(out_files_tmp) > 0:
        #     if len(out_files_tmp) != len(json_files):
        #         raise Exception("\nError in:", out_dir, "with N =", len(out_files_tmp), 
        #                         "files, need to purge this folder and restart matching!\n")
        #         # return
        #     elif len(out_files_tmp) == len(json_files):
        #         print("\nDir:", os.path.basename(out_dir), "N files:", len(json_files), 
        #             "directory matching completed, skipping...")
        #         # return
        # else:
        #     print("\nMatching json_dir: ", os.path.basename(json_dir), "N json:", len(json_files))
            

        gdf_dict = {}
        cnts_dict = {}

        id_areas = {}

        for j, f in enumerate(json_files):
            
            name_root = f.split('.')[0]
            json_path = os.path.join(json_dir, f)
            
            if verbose and ((j % 1) == 0):
                print("  ", j, "/", len(json_files), "for", os.path.basename(json_dir), "=", name_root)

            # gdf
            gdf_now = gpd.read_file(json_path)
            # drop value if it exists
            gdf_now = gdf_now.drop(columns=['value'])
            # get area
            gdf_now['area'] = gdf_now['geometry'].area
            # initialize iou, id
            gdf_now[iou_field] = -1
            gdf_now[id_field] = -1
            # sort by reverse area
            gdf_now.sort_values(by=['area'], ascending=False, inplace=True)
            gdf_now = gdf_now.reset_index(drop=True)
            # reorder columns (if needed)
            gdf_now = gdf_now[gdf_master_columns]    
            id_set = set([])
                 
            if verbose:
                print("\n")
                print("", j, "file_name:", f)
                print("  ", "gdf_now.columns:", gdf_now.columns)
            
            if j == 0:
                # Establish initial footprints at Epoch0
                # set id
                gdf_now[id_field] = gdf_now.index.values
                gdf_now[iou_field] = 0
                n_new = len(gdf_now)
                n_matched = 0
                id_set = set(gdf_now[id_field].values)
                gdf_master_Out = gdf_now.copy(deep=True)
                # gdf_dict[f] = gdf_now

                for _i, _r in gdf_now.iterrows():
                    id_areas[_r[id_field]] = [_r['area']]

            else:
                # match buildings in epochT to epochT-1
                # see: https://github.com/CosmiQ/solaris/blob/master/solaris/eval/base.py
                # print("gdf_master;", gdf_dict['master']) #gdf_master)

                
                gdf_master_Out = gdf_dict['master'].copy(deep=True)
                gdf_master_Edit = gdf_dict['master'].copy(deep=True)
                
                _used_ids = set([])

                tree = STRtree(gdf_master_Edit.geometry.values)
                
                all_polys = []
                for _p in gdf_master_Edit.geometry.values:
                    all_polys.append(id(_p))
                all_ids = gdf_master_Edit.index.values
                
                id_map = dict(zip(all_polys, all_ids))
                
                if verbose:
                    print("   len gdf_now:", len(gdf_now), "len(gdf_master):", len(gdf_master_Out),
                        "max master id:", np.max(gdf_master_Out[id_field]))
                    print("   gdf_master_Edit.columns:", gdf_master_Edit.columns)
            
                new_id = np.max(gdf_master_Edit[id_field]) + 1
                # if verbose:
                #    print("new_id:", new_id)
                idx = 0
                n_new = 0
                n_matched = 0
                for pred_idx, pred_row in gdf_now.iterrows():
                    if verbose:
                        if (idx % 1000) == 0:
                            print("    ", name_root, idx, "/", len(gdf_now))
                    if super_verbose:
                        # print("    ", i, j, idx, "/", len(gdf_now))
                        print("    ", idx, "/", len(gdf_now))
                    idx += 1
                    pred_poly = pred_row.geometry
                    # if super_verbose:
                    #     print("     pred_poly.exterior.coords:", list(pred_poly.exterior.coords))
                        
                    # get iou overlap
                    _idxs = [id_map[id(_p)] for _p in tree.query(pred_row.geometry)]
                    _idxs = [x for x in _idxs if x not in _used_ids]

                    iou_GDF = calculate_iou(pred_poly, gdf_master_Edit.loc[_idxs])
                    # iou_GDF = iou.calculate_iou(pred_poly, gdf_master_Edit)
                    # print("iou_GDF:", iou_GDF)
                        
                    # Get max iou
                    if not iou_GDF.empty:
                        max_iou_row = iou_GDF.loc[iou_GDF['iou_score'].idxmax(axis=0, skipna=True)]
                        # sometimes we are get an erroneous id of 0, caused by nan area,
                        #   so check for this
                        max_area = max_iou_row.geometry.area
                        if max_area == 0 or math.isnan(max_area):
                            # print("nan area!", max_iou_row, "returning...")
                            raise Exception("\n Nan area!:", max_iou_row, "returning...")
                            # return
                        
                        id_match = max_iou_row[id_field]
                        if id_match in id_set:
                            print("Already seen id! returning...")
                            raise Exception("\n Already seen id!", id_match, "returning...")
                            # return
                        
                        # print("iou_GDF:", iou_GDF)
                        if max_iou_row['iou_score'] >= min_iou:
                            if super_verbose:
                                print("    pred_idx:", pred_idx, "match_id:", max_iou_row[id_field],
                                    "max iou:", max_iou_row['iou_score'])
                            # we have a successful match, so set iou, and id
                            gdf_now.loc[pred_row.name, iou_field] = max_iou_row['iou_score']
                            gdf_now.loc[pred_row.name, id_field] = id_match
                            # drop  matched polygon in ground truth
                            
                            # gdf_master_Edit = gdf_master_Edit.drop(max_iou_row.name, axis=0) 
                            _used_ids.add(max_iou_row.name)

                            n_matched += 1

                            id_areas[id_match].append(pred_poly.area)
                            # # update gdf_master geometry?
                            # # Actually let's leave the geometry the same so it doesn't move around...

                            _m0 = np.median(id_areas[id_match])
                            if abs(_m0 - pred_poly.area) < abs(_m0 - gdf_master_Out.at[max_iou_row['gt_idx'], 'area']):
                                gdf_master_Out.at[max_iou_row['gt_idx'], 'geometry'] = pred_poly
                                gdf_master_Out.at[max_iou_row['gt_idx'], 'area'] = pred_poly.area
                                gdf_master_Out.at[max_iou_row['gt_idx'], iou_field] = max_iou_row['iou_score']
                        
                        else:
                            # no match, 
                            if super_verbose:
                                print("    Minimal match! - pred_idx:", pred_idx, "match_id:",
                                    max_iou_row[id_field], "max iou:", max_iou_row['iou_score'])
                                print("      Using new id:", new_id)
                            if (new_id in id_set) or (new_id == 0):
                                raise Exception("trying to add an id that already exists, returning!")
                                # return
                            gdf_now.loc[pred_row.name, iou_field] = 0
                            gdf_now.loc[pred_row.name, id_field] = new_id
                            id_set.add(new_id)
                            # update master, cols = [id_field, iou_field, 'area', 'geometry']
                            gdf_master_Out.loc[new_id] = [new_id, 0, pred_poly.area, pred_poly]
                            id_areas[new_id] = [pred_poly.area]
                            new_id += 1
                            n_new += 1
                        
                    else:
                        # no match (same exact code as right above)
                        if super_verbose:
                            print("    pred_idx:", pred_idx, "no overlap, new_id:", new_id)
                        if (new_id in id_set) or (new_id == 0):
                            raise Exception("trying to add an id that already exists, returning!")
                            # return
                        gdf_now.loc[pred_row.name, iou_field] = 0
                        gdf_now.loc[pred_row.name, id_field] = new_id
                        id_set.add(new_id)
                        # update master, cols = [id_field, iou_field, 'area', 'geometry']
                        gdf_master_Out.loc[new_id] = [new_id, 0, pred_poly.area, pred_poly]
                        id_areas[new_id] = [pred_poly.area]
                        new_id += 1
                        n_new += 1
                        
            # print("gdf_now:", gdf_now)
            gdf_dict[f] = gdf_now
            gdf_dict['master'] = gdf_master_Out
            cnts_dict[f] = (n_new, n_matched)


        cnts = {}
        start_idx = {}
        n = len(json_files)
        for idx, f in enumerate(json_files):
            for i in gdf_dict[f][id_field].values:
                if i not in cnts:
                    cnts[i] = 0
                    start_idx[i] = idx
                cnts[i] = cnts[i] + 1
        
        for f in json_files:
            rmv = []
            for i in gdf_dict[f][id_field].values:
                _m0 = np.median(id_areas[i])

                if _m0 < 220:
                    min_cnt = 5
                    min_ratio = 0.8
                elif _m0 < 1000:
                    min_cnt = 2
                    min_ratio = 0.5
                else:
                    min_cnt = 2
                    min_ratio = 0.5

                if start_idx[i] > 0:
                    if (cnts[i] < min_cnt) or (cnts[i] / (n - start_idx[i]) < min_ratio):
                        rmv.append(i)

            if len(rmv) > 0:
                gdf_dict[f] = gdf_dict[f].drop(gdf_dict[f][gdf_dict[f][id_field].isin(rmv)].index) 

            output_path = os.path.join(out_dir, f)

            # save!
            if len(gdf_dict[f]) > 0:
                gdf_dict[f].to_file(output_path, driver="GeoJSON")
            else:
                print("Empty dataframe, writing empty gdf", output_path)
                open(output_path, 'a').close()

            # if verbose:
            n_new, n_matched = cnts_dict[f]
            print("  ", aoi, "  ", "N_new, N_matched, N_removed:", n_new, n_matched, len(rmv))


    except Exception as ex:
        print('Exception occured: {}. aoi: {}'.format(ex, aoi))



if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(path.join(pred_dir, 'tracked'), exist_ok=True)

    aois = []

    for d in listdir(path.join(pred_dir, 'grouped')):
        if path.isdir(path.join(pred_dir, 'grouped', d)):
            aois.append(d)

    aois = np.asarray(aois)


    with Pool() as pool:
        results = pool.map(process_image, aois)


    print("OK!")

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))