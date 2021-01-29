#!/bin/bash

source activate solaris

TEST_ROOT=$1  # path/to/SN7_buildings/test/
OUTPUT_CSV_PATH=$2  # path/to/solution.csv

source /work/settings.sh

echo ''
echo 'ensembling model predictions...'
/work/tools/ensemble_models.py \
    INPUT.TEST_DIR ${TEST_ROOT} \
    PREDICTION_ROOT ${PREDICTION_ROOT} \
    ENSEMBLED_PREDICTION_ROOT ${ENSEMBLED_PREDICTION_ROOT} \
    ENSEMBLE_EXP_IDS "[200, 201, 202, 203, 204, 500, 501, 502, 503, 504]"

echo ''
echo 'refining predicted masks...'
/work/tools/refine_pred_mask.py \
    INPUT.TEST_DIR ${TEST_ROOT} \
    ENSEMBLED_PREDICTION_ROOT ${ENSEMBLED_PREDICTION_ROOT} \
    REFINED_PREDICTION_ROOT ${REFINED_PREDICTION_ROOT} \
    ENSEMBLE_EXP_IDS "[200, 201, 202, 203, 204, 500, 501, 502, 503, 504]"

echo ''
echo 'generating polygons...'
/work/tools/pred_mask_to_poly.py \
    REFINED_PREDICTION_ROOT ${REFINED_PREDICTION_ROOT} \
    POLY_ROOT ${POLY_ROOT} \
    ENSEMBLE_EXP_IDS "[200, 201, 202, 203, 204, 500, 501, 502, 503, 504]"

echo ''
echo 'tracking polygons...'
/work/tools/track_polys.py \
    INPUT.TEST_DIR ${TEST_ROOT} \
    POLY_ROOT ${POLY_ROOT} \
    TRACKED_POLY_ROOT ${TRACKED_POLY_ROOT} \
    SOLUTION_OUTPUT_PATH ${OUTPUT_CSV_PATH} \
    ENSEMBLE_EXP_IDS "[200, 201, 202, 203, 204, 500, 501, 502, 503, 504]"
