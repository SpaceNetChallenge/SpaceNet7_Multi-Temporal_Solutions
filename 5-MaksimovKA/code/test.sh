#!/usr/bin/env bash
ARG1=${1:-/data/SN7_buildings/test_public/}
ARG2=${2:-/wdata/solution.csv}

mkdir -p /wdata/segmentation_logs/ /wdata/folds_predicts/ /wdata/pretrained_models/checkpoints/

if [ "$(ls -A /wdata/segmentation_logs/)" ]; then
     echo "trained weights available"
else
    echo "loading pretrained weights"

    mkdir -p /wdata/segmentation_logs/fold_1_siamse-senet154/checkpoints/
    gdown https://drive.google.com/uc?id=1WXnq7biVcRoYbbKKj2o0huti3rJ6JlKs -O /wdata/segmentation_logs/fold_1_siamse-senet154/checkpoints/best.pth
fi

gdown https://drive.google.com/uc?id=1WVQWAwCFgkwmSwkxp-kLiS89IdFrh-T8 -O /wdata/pretrained_models/checkpoints/senet154-c7b49a05.pth

python3 /project/predict.py --config_path /project/config.py --gpu '"0"' --test_images $ARG1 --workers 1 --batch_size 1 \
& python3 /project/predict2.py --config_path /project/config.py --gpu '"1"' --test_images $ARG1 --workers 1 --batch_size 1 \
& python3 /project/predict3.py --config_path /project/config.py --gpu '"2"' --test_images $ARG1 --workers 1 --batch_size 1 \
& python3 /project/predict4.py --config_path /project/config.py --gpu '"3"' --test_images $ARG1 --workers 1 --batch_size 1 & wait

python3 /project/mean_folds.py
python3 /project/masks_to_jsons.py
python3 /project/map_jsons.py
python3 /project/submit.py --out_file $ARG2