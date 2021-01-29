#!/usr/bin/env bash
ARG1=${1:-/data/SN7_buildings/train/}
rm -rf /wdata/*
# make directories
mkdir -p /wdata/train_masks/  /wdata/pretrained_models/checkpoints/ /wdata/segmentation_logs/ /wdata/final_models/

# create masks and folds
echo "creating masks"
python3 /project/create_masks.py --data_root_path $ARG1
echo "creating folds"
python3 /project/create_folds.py --images_path $ARG1

# load pretrained
echo "loading pretrained weights"
gdown https://drive.google.com/uc?id=1WVQWAwCFgkwmSwkxp-kLiS89IdFrh-T8 -O /wdata/pretrained_models/checkpoints/senet154-c7b49a05.pth
echo "training"

python3 /project/train.py --data_path $ARG1 --config_path /project/config.py --gpu '"0"'