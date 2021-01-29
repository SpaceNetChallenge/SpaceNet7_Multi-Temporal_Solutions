#!/bin/bash

source activate solaris

source /work/settings.sh

TRAIN_ARGS="\
    INPUT.TRAIN_VAL_SPLIT_DIR ${TRAIN_VAL_SPLIT_ROOT} \
    LOG_ROOT ${LOG_ROOT} \
    WEIGHT_ROOT ${WEIGHT_ROOT} \
    SAVE_CHECKPOINTS False \
    DUMP_GIT_INFO False \
    EVAL.VAL_INTERVAL_EPOCH 3 \
"
# comment out the line below for debugging
#TRAIN_ARGS=${TRAIN_ARGS}" SOLVER.EPOCHS 10"

mkdir -p ${TRAIN_STDOUT_ROOT}

CONFIG_020x='/work/configs/unet_timm-efficientnet-b3_scale-3.0_v_02.yml'  # exp_200~204 (effnet-b3, scale=3, focal)
CONFIG_050x='/work/configs/unet_timm-efficientnet-b3_scale-4.0_v_01.yml'  # exp_500~504 (effnet-b3, scale=4, bce)

echo ''
echo 'training... (1/3)'
echo 'this will take ~8 hours'
echo 'you can check progress from '${TRAIN_STDOUT_ROOT}'/*.out'

## 200
nohup env CUDA_VISIBLE_DEVICES=0 /work/tools/train_spacenet7_model.py \
    --config ${CONFIG_020x} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 200 \
    > ${TRAIN_STDOUT_ROOT}/exp_0200.out 2>&1 &

## 201
nohup env CUDA_VISIBLE_DEVICES=1 /work/tools/train_spacenet7_model.py \
    --config ${CONFIG_020x} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 1 \
    EXP_ID 201 \
    > ${TRAIN_STDOUT_ROOT}/exp_0201.out 2>&1 &

## 202
nohup env CUDA_VISIBLE_DEVICES=2 /work/tools/train_spacenet7_model.py \
    --config ${CONFIG_020x} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 2 \
    EXP_ID 202 \
    > ${TRAIN_STDOUT_ROOT}/exp_0202.out 2>&1 &

## 203
nohup env CUDA_VISIBLE_DEVICES=3 /work/tools/train_spacenet7_model.py \
    --config ${CONFIG_020x} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 3 \
    EXP_ID 203 \
    > ${TRAIN_STDOUT_ROOT}/exp_0203.out 2>&1 &

wait

echo ''
echo 'training... (2/3)'
echo 'this will take ~11 hours'
echo 'you can check progress from '${TRAIN_STDOUT_ROOT}'/*.out'

## 204
nohup env CUDA_VISIBLE_DEVICES=0 /work/tools/train_spacenet7_model.py \
    --config ${CONFIG_020x} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 4 \
    EXP_ID 204 \
    > ${TRAIN_STDOUT_ROOT}/exp_0204.out 2>&1 &

## 500
nohup env CUDA_VISIBLE_DEVICES=1 /work/tools/train_spacenet7_model.py \
    --config ${CONFIG_050x} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 0 \
    EXP_ID 500 \
    > ${TRAIN_STDOUT_ROOT}/exp_0500.out 2>&1 &

## 501
nohup env CUDA_VISIBLE_DEVICES=2 /work/tools/train_spacenet7_model.py \
    --config ${CONFIG_050x} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 1 \
    EXP_ID 501 \
    > ${TRAIN_STDOUT_ROOT}/exp_0501.out 2>&1 &

## 502
nohup env CUDA_VISIBLE_DEVICES=3 /work/tools/train_spacenet7_model.py \
    --config ${CONFIG_050x} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 2 \
    EXP_ID 502 \
    > ${TRAIN_STDOUT_ROOT}/exp_0502.out 2>&1 &

wait

echo ''
echo 'training... (3/3)'
echo 'this will take ~11 hours'
echo 'you can check progress from '${TRAIN_STDOUT_ROOT}'/*.out'

## 503
nohup env CUDA_VISIBLE_DEVICES=0 /work/tools/train_spacenet7_model.py \
    --config ${CONFIG_050x} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 3 \
    EXP_ID 503 \
    > ${TRAIN_STDOUT_ROOT}/exp_0503.out 2>&1 &

## 504
nohup env CUDA_VISIBLE_DEVICES=1 /work/tools/train_spacenet7_model.py \
    --config ${CONFIG_050x} \
    ${TRAIN_ARGS} \
    INPUT.TRAIN_VAL_SPLIT_ID 4 \
    EXP_ID 504 \
    > ${TRAIN_STDOUT_ROOT}/exp_0504.out 2>&1 &

wait

echo 'done training all models!'
