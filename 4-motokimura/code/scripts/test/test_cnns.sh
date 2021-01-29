#!/bin/bash

source activate solaris

TEST_ROOT=$1  # path/to/SN7_buildings/test/

source /work/settings.sh

# predict with trained models
TEST_ARGS="\
    --exp_log_dir ${LOG_ROOT} \
    --model_weight_dir ${WEIGHT_ROOT} \
    PREDICTION_ROOT ${PREDICTION_ROOT} \
    INPUT.TEST_DIR ${TEST_ROOT} \
"

mkdir -p ${TEST_STDOUT_ROOT}

echo ''
echo 'predicting... (1/3)'
echo 'this will take ~10 min'
echo 'you can check progress from '${TEST_STDOUT_ROOT}'/*.out'

nohup env CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet7_model.py \
    --exp_id 200 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_ROOT}/exp_0200.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 /work/tools/test_spacenet7_model.py \
    --exp_id 201 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_ROOT}/exp_0201.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 /work/tools/test_spacenet7_model.py \
    --exp_id 202 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_ROOT}/exp_0202.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 /work/tools/test_spacenet7_model.py \
    --exp_id 203 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_ROOT}/exp_0203.out 2>&1 &

wait

echo ''
echo 'predicting... (2/3)'
echo 'this will take ~15 min'
echo 'you can check progress from '${TEST_STDOUT_ROOT}'/*.out'

nohup env CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet7_model.py \
    --exp_id 204 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_ROOT}/exp_0204.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 /work/tools/test_spacenet7_model.py \
    --exp_id 500 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_ROOT}/exp_0500.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 /work/tools/test_spacenet7_model.py \
    --exp_id 501 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_ROOT}/exp_0501.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 /work/tools/test_spacenet7_model.py \
    --exp_id 502 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_ROOT}/exp_0502.out 2>&1 &

wait

echo ''
echo 'predicting... (3/3)'
echo 'this will take ~15 min'
echo 'you can check progress from '${TEST_STDOUT_ROOT}'/*.out'

nohup env CUDA_VISIBLE_DEVICES=0 /work/tools/test_spacenet7_model.py \
    --exp_id 503 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_ROOT}/exp_0503.out 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 /work/tools/test_spacenet7_model.py \
    --exp_id 504 \
    ${TEST_ARGS} \
    > ${TEST_STDOUT_ROOT}/exp_0504.out 2>&1 &

wait

echo 'done predicting all models!'
