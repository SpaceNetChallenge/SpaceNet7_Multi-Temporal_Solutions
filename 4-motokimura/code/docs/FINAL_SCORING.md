# Instructions for Final Scoring

This document provides instructions for the final testing/scoring of motokimura's solution.

## Prepare SpaceNet-7 data

```
DATA_ROOT=${HOME}/data
DATA_DIR=${DATA_ROOT}/SN7_buildings  # path to download SpaceNet-7 dataset
mkdir -p ${DATA_DIR}

# download and extract train data
cd ${DATA_DIR}
aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz .
tar -xvf SN7_buildings_train.tar.gz

# download and extract train csv data
cd ${DATA_DIR}
aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train_csvs.tar.gz .
tar -xvf SN7_buildings_train_csvs.tar.gz
cp ${DATA_DIR}/csvs/sn7_train_ground_truth_pix.csv ${DATA_DIR}/train

# download and extract test data
cd ${DATA_DIR}
aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_test_public.tar.gz .
tar -xvf SN7_buildings_test_public.tar.gz

# please prepare private test data under `DATA_DIR`!
```

## Prepare temporal directory

```
WDATA_ROOT=${HOME}/wdata  # path to directory for temporal files (train logs, prediction results, etc.)
mkdir -p ${WDATA_ROOT}
```

## Build image

```
cd ${CODE_DIR}  # `code` directory containing `Dockerfile`, `train.sh`, `test.sh`, and etc. 
nvidia-docker build -t motokimura .
```

During the build, my home built models are downloaded
so that `test.sh` can run without re-training the models.

## Prepare container

```
# launch container
nvidia-docker run --ipc=host -v ${DATA_ROOT}:/data:ro -v ${WDATA_ROOT}:/wdata -it motokimura
```

It's necessary to add `--ipc=host` option when run docker (as written in [flags.txt](flags.txt)).
Otherwise multi-threaded PyTorch dataloader will crash.

## Train

**WARNINGS: `train.sh` updates my home built models downloaded during docker build.**

```
# start training!
(in container) ./train.sh /data/SN7_buildings/train

# if you need logs:
(in container) ./train.sh /data/SN7_buildings/train 2>&1 | tee /wdata/train.log
```

Note that this is a sample call of `train.sh`. 
i.e., you need to specify the correct path to training data folder.

## Test

```
# start testing!
(in container) ./test.sh /data/SN7_buildings/test_public /wdata/solution.csv

# if you need logs:
(in container) ./test.sh /data/SN7_buildings/test_public /wdata/solution.csv 2>&1 | tee /wdata/test.log
```

Note that this is a sample call of `test.sh`. 
i.e., you need to specify the correct paths to testing image folder and output csv file.
