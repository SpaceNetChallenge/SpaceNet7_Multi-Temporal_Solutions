#!/bin/bash

START_TIME=$SECONDS

source activate solaris

TEST_ROOT=$1  # path/to/SN7_buildings/test/
OUTPUT_CSV_PATH=$2  # path/to/solution.csv

source /work/settings.sh

rm -rf ${PREDICTION_ROOT} ${ENSEMBLED_PREDICTION_ROOT} ${REFINED_PREDICTION_ROOT} ${POLY_ROOT} ${TRACKED_POLY_ROOT} ${TEST_STDOUT_ROOT}

# predict with trained models
/work/scripts/test/test_cnns.sh ${TEST_ROOT}

# postprocess predictions
/work/scripts/test/postprocess.sh ${TEST_ROOT} ${OUTPUT_CSV_PATH}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo 'Total time for testing: ' $(($ELAPSED_TIME / 60 + 1)) '[min]'