#!/bin/bash

source activate solaris

TRAIN_ROOT=$1  # path/to/SN7_buildings/train/

source /work/settings.sh

echo ''
echo 'generating building masks...'
/work/tools/geojson_to_mask.py \
    --train_dir ${TRAIN_ROOT} \
    --out_dir ${BUILDING_MASK_ROOT}

echo ''
echo 'splitting dataset...'
/work/tools/split_dataset.py \
    --train_dir ${TRAIN_ROOT} \
    --mask_dir ${BUILDING_MASK_ROOT} \
    --out_dir ${TRAIN_VAL_SPLIT_ROOT}
