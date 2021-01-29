import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import sys

import numpy as np
np.random.seed(1)
import random
random.seed(1)
import pandas as pd
import cv2
import timeit
from os import path, makedirs, listdir
import sys
sys.setrecursionlimit(10000)
from multiprocessing import Pool
from skimage.morphology import square, dilation, watershed, erosion
from skimage import io
from skimage import measure

from shapely.wkt import loads

from tqdm import tqdm



labels_dir = '/wdata/labels_4k_2'
masks_dir = '/wdata/masks_4k_2'

train_dir = '/data/SN7_buildings/train/'
if len(sys.argv) > 0:
    train_dir = sys.argv[1]

df = pd.read_csv(path.join(train_dir, 'sn7_train_ground_truth_pix.csv'))


all_files = df['filename'].unique()


def mask_for_polygon(poly, im_size, scale):
    img_mask = np.zeros(im_size, np.uint8)
    int_coords = lambda x: (np.array(x) * scale).round().astype('int32')
    if poly.exterior is not None and len(poly.exterior.coords) > 1:
        try:
            exteriors = np.int32([int_coords(poly.exterior.coords)[:, :2]])
            interiors = [int_coords(pi.coords)[:, :2] for pi in poly.interiors]
            cv2.fillPoly(img_mask, exteriors, 1)
            cv2.fillPoly(img_mask, interiors, 0)
        except Exception as e:
            print(e)
            print(train_dir)
            print(exteriors)
            print(interiors)
            raise
            
    return img_mask


def draw_polygon(poly, img, borders_img, scale, lbl):
    int_coords = lambda x: (np.array(x) * scale).round().astype('int32')
    if poly.exterior is not None and len(poly.exterior.coords) > 1:
        try:
            exteriors = np.int32([int_coords(poly.exterior.coords)[:, :2]])
            interiors = [int_coords(pi.coords)[:, :2] for pi in poly.interiors]
            cv2.fillPoly(img, exteriors, lbl)
            cv2.polylines(borders_img, exteriors, True, 1, 1, lineType=cv2.LINE_4)
            cv2.fillPoly(img, interiors, 0)
        except Exception as e:
            print(e)
            print(train_dir)
            print(exteriors)
            print(interiors)
            raise
#     return img


def process_image(fn):
    scale = 4 #3
    
    d = fn.split('mosaic_')[1]
    
    vals = df[(df['filename'] == fn)][['id', 'geometry']].values
    
    img = io.imread(path.join(train_dir, d, 'images_masked', fn + '.tif'))
    
    h, w = (np.asarray(img.shape[:2]) * scale).astype('int32')
    
    labels = np.zeros((h, w), dtype='uint16')
    border_msk = np.zeros_like(labels, dtype='uint8')
    tmp = np.zeros((h, w), dtype='bool')

    cur_lbl = 0
    
    all_polys = []
    all_areas = []
    for i in range(vals.shape[0]):
        if vals[i, 0] >= 0:
            _p = loads(vals[i, 1])
            all_polys.append(_p)
            all_areas.append(-1 * _p.area)
    
    all_areas, all_polys = zip(*sorted(zip(all_areas, all_polys), key=lambda pair: pair[0]))

    for p in all_polys:
        cur_lbl += 1    
        draw_polygon(p, labels, border_msk, scale, cur_lbl)
        
    
    labels = measure.label(labels, connectivity=2, background=0)

    cv2.imwrite(path.join(labels_dir, fn + '.tif'), labels)

    props = measure.regionprops(labels)
    
    msk = np.zeros((h, w), dtype='uint8')
    
    if cur_lbl > 0:
        border_msk = border_msk > 0

        tmp = dilation(labels > 0, square(5))
        tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
        tmp = tmp ^ tmp2
        tmp = tmp | border_msk
        tmp = dilation(tmp, square(3))
        
    
        msk0 = labels > 0

        msk1 = np.zeros_like(labels, dtype='bool')
        
        border_add = np.zeros_like(labels, dtype='bool')

        for y0 in range(labels.shape[0]):
            for x0 in range(labels.shape[1]):
                if not tmp[y0, x0]:
                    continue
                
                can_rmv = False
                if labels[y0, x0] == 0:
                    sz = 2
                else:
                    sz = 1
                    can_rmv = True
                    minor_axis_length = props[labels[y0, x0] - 1].minor_axis_length
                    area = props[labels[y0, x0] - 1].area
                    if minor_axis_length < 6 or area < 20:
                        can_rmv = False
                    if minor_axis_length > 25 and area > 150: #area
                        sz = 2
                    if minor_axis_length > 35 and area > 300:
                        sz = 3

                uniq = np.unique(labels[max(0, y0-sz):min(labels.shape[0], y0+sz+1), max(0, x0-sz):min(labels.shape[1], x0+sz+1)])
#                 can_rmv = False
                if len(uniq[uniq > 0]) > 1:
                    msk1[y0, x0] = True
                    if labels[y0, x0] > 0:
                        if can_rmv:
                            msk0[y0, x0] = False
                        else:
                            border_add[y0, x0] = True
                            
                if msk0[y0, x0] and sz > 1 and (0 in uniq):
                    border_add[y0, x0] = True
    
        msk1 = 255 * msk1
        msk1 = msk1.astype('uint8')

        new_border_msk = (erosion(msk0, square(3)) ^ msk0) | border_add
        
        msk0 = 255 * msk0
        msk0 = msk0.astype('uint8')

        msk2 = 255 * new_border_msk
        msk2 = msk2.astype('uint8')
        msk = np.stack((msk0, msk1, msk2))
        msk = np.rollaxis(msk, 0, 3)

        cv2.imwrite(path.join(masks_dir, fn + '.png'), msk, [cv2.IMWRITE_PNG_COMPRESSION, 5])
        


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(labels_dir, exist_ok=True)
    makedirs(masks_dir, exist_ok=True)

    with Pool() as pool:
        _ = pool.map(process_image, all_files)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))