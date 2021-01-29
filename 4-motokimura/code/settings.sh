#!/bin/bash

# paths to model related files
MODEL_ROOT=/work/models

LOG_ROOT=${MODEL_ROOT}/logs
WEIGHT_ROOT=${MODEL_ROOT}/weights

# path to artifact files
ARTIFACT_ROOT=/wdata

BUILDING_MASK_ROOT=${ARTIFACT_ROOT}/building_masks
TRAIN_VAL_SPLIT_ROOT=${ARTIFACT_ROOT}/split

PREDICTION_ROOT=${ARTIFACT_ROOT}/predictions
ENSEMBLED_PREDICTION_ROOT=${ARTIFACT_ROOT}/ensembled_predictions
REFINED_PREDICTION_ROOT=${ARTIFACT_ROOT}/refined_predictions
POLY_ROOT=${ARTIFACT_ROOT}/polygons
TRACKED_POLY_ROOT=${ARTIFACT_ROOT}/tracked_polygons

# path to stdout files
TRAIN_STDOUT_ROOT=${ARTIFACT_ROOT}/stdout/train
TEST_STDOUT_ROOT=${ARTIFACT_ROOT}/stdout/test
