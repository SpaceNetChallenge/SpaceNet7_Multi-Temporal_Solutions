#!/bin/bash

DATA=$1
OUT_FILE=$2

echo "Predicting masks..."
python predict_test.py --gpu 0 --data-path $DATA --dir /wdata/predictions/1 --config configs/b7adam.json \
  --model weights/run1_mask_center_timm_effnet_dragon_tf_efficientnet_b7_ns_0_best_dice &

python predict_test.py --gpu 1 --data-path $DATA --dir /wdata/predictions/2 --config configs/b7adam.json \
  --model weights/run2_mask_center_timm_effnet_dragon_tf_efficientnet_b7_ns_0_best_dice &

python predict_test.py --gpu 2 --data-path $DATA --dir /wdata/predictions/3 --config configs/b7adam.json \
  --model weights/run3_mask_center_timm_effnet_dragon_tf_efficientnet_b7_ns_0_best_dice &

python predict_test.py --gpu 3 --data-path $DATA --dir /wdata/predictions/b61 --config configs/b6adam.json \
  --model weights/run1_mask_center_timm_effnet_dragon_tf_efficientnet_b6_ns_0_best_dice &

wait
echo "Ensembling masks..."
python ensemble.py --folds_dir /wdata/predictions/ --ensembling_dir /wdata/ensemble
echo "Generating solution..."

python generate_submit.py --preds-dir /wdata/ensemble/ \
  --preds-dir /wdata/ensemble \
  --workers 24 \
  --json-dir /wdata/geojsons \
  --out-csv-dir /wdata/csvs \
  --image-dir $DATA \
  --out-file $OUT_FILE

echo "Done!!!"