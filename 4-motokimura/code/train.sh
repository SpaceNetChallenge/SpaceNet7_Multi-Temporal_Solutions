#!/bin/bash

START_TIME=$SECONDS

source activate solaris

TRAIN_ROOT=$1  # path/to/SN7_buildings/train/

source /work/settings.sh

# removing motokimura's home build models here!
rm -rf ${MODEL_ROOT}/*

# preprocess dataset
/work/scripts/train/preprocess.sh ${TRAIN_ROOT}

# train CNN models
/work/scripts/train/train_cnns.sh

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo 'Total time for testing: ' $(($ELAPSED_TIME / 60 + 1)) '[min]'
