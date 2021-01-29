#!/bin/bash

ENV=desktop1  # "desktop1", "desktop2", "mac", or "aws"
if [ $# -eq 1 ]; then
    ENV=$1
fi

# set environment specific parameters
if [ $ENV = desktop1 ]; then
	RUNTIME="--runtime nvidia"
	FEATURE_ROOT=/mnt/sdb1/spacenet7/
elif [ $ENV = desktop2 ]; then
	RUNTIME="--runtime nvidia"
	FEATURE_ROOT=${HOME}/data/spacenet7/
elif [ $ENV = mac ]; then
	RUNTIME=""
	FEATURE_ROOT=${HOME}/data/spacenet7/
elif [ $ENV = aws ]; then
	RUNTIME="--runtime nvidia"
	FEATURE_ROOT=/mnt/nfs/kimura/spacenet7/
else
	echo 'Usage: ./run.sh $ENV'
	echo '(ENV must be "desktop1", "desktop2", "mac" or "aws")'
	exit 1
fi

# set jupyter port
JUPYTER_PORT=8889
echo "mapping port docker:${JUPYTER_PORT} --> host:${JUPYTER_PORT}"

# set image name
IMAGE="spacenet7:dev"

# set project root dicrectory to map to docker
THIS_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=`dirname ${THIS_DIR}`

# set path to directories to map to docker
DATA_DIR=${HOME}/data
WEIGHTS_DIR=${FEATURE_ROOT}/weights
LOG_DIR=${FEATURE_ROOT}/logs
CHECKPOINT_DIR=${FEATURE_ROOT}/checkpoints
PREDICTION_DIR=${FEATURE_ROOT}/predictions
ENSEMBLED_PREDICTION_DIR=${FEATURE_ROOT}/ensembled_predictions
REFINED_PREDICTION_DIR=${FEATURE_ROOT}/refined_predictions
POLY_DIR=${FEATURE_ROOT}/polygons
TRACKED_POLY_DIR=${FEATURE_ROOT}/tracked_polygons
SOLUTION_DIR=${FEATURE_ROOT}/solutions
VAL_PREDICTION_DIR=${FEATURE_ROOT}/val_predictions
VAL_ENSEMBLED_PREDICTION_DIR=${FEATURE_ROOT}/val_ensembled_predictions
VAL_REFINED_PREDICTION_DIR=${FEATURE_ROOT}/val_refined_predictions
VAL_POLY_DIR=${FEATURE_ROOT}/val_polygons
VAL_TRACKED_POLY_DIR=${FEATURE_ROOT}/val_tracked_polygons

# run container
CONTAINER="spacenet7_dev"

docker run ${RUNTIME} -it --rm --ipc=host \
	-p ${JUPYTER_PORT}:${JUPYTER_PORT} \
	-p 6006:6006 \
	-v ${PROJ_DIR}:/work \
	-v ${DATA_DIR}:/data \
	-v ${WEIGHTS_DIR}:/weights \
	-v ${LOG_DIR}:/logs \
	-v ${CHECKPOINT_DIR}:/checkpoints \
	-v ${PREDICTION_DIR}:/predictions \
	-v ${ENSEMBLED_PREDICTION_DIR}:/ensembled_predictions \
	-v ${REFINED_PREDICTION_DIR}:/refined_predictions \
	-v ${POLY_DIR}:/polygons \
	-v ${TRACKED_POLY_DIR}:/tracked_polygons \
	-v ${VAL_PREDICTION_DIR}:/val_predictions \
	-v ${VAL_ENSEMBLED_PREDICTION_DIR}:/val_ensembled_predictions \
	-v ${VAL_REFINED_PREDICTION_DIR}:/val_refined_predictions \
	-v ${VAL_POLY_DIR}:/val_polygons \
	-v ${VAL_TRACKED_POLY_DIR}:/val_tracked_polygons \
	--name ${CONTAINER} \
	${IMAGE} /bin/bash
