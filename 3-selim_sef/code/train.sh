#!/bin/bash

DATA=$1
GPUS=4
mkdir -p /wdata/train_data/oof_preds
LOG_DIR=/wdata/logs/
mkdir -p $LOG_DIR


echo "Preparing masks..."
python preprocessing/masks.py --train-dir $DATA --out-dir /wdata/train_data/masks --workers 24

echo "Resize train images and split if needed..."
python preprocessing/split_images.py --train-dir $DATA --out-dir /wdata/train_data/ --workers 24

echo "Training B7 run 1..."
PYTHONPATH=. python -u -m torch.distributed.launch --nproc_per_node=$GPUS \
  --master_port 9901 training/train_instance_no_offset.py --distributed \
  --data-dir /wdata/train_data/ --config configs/b7adam.json --predictions /wdata/train_data/oof_preds \
  --workers 8 --opt-level O1 --freeze-epochs 2 --prefix run1_mask_center_ --fold 0 > $LOG_DIR/run1 &
wait
echo "Training B7 run 2..."
PYTHONPATH=. python -u -m torch.distributed.launch --nproc_per_node=$GPUS \
  --master_port 9901 training/train_instance_no_offset.py --distributed \
  --data-dir /wdata/train_data/ --config configs/b7adam.json --predictions /wdata/train_data/oof_preds \
  --workers 8 --opt-level O1 --freeze-epochs 2 --prefix run2_mask_center_ --fold 0 > $LOG_DIR/run2 &
wait

echo "Training B7 run 3..."
PYTHONPATH=. python -u -m torch.distributed.launch --nproc_per_node=$GPUS \
  --master_port 9901 training/train_instance_no_offset.py --distributed \
  --data-dir /wdata/train_data/ --config configs/b7adam.json --predictions /wdata/train_data/oof_preds \
  --workers 8 --opt-level O1 --freeze-epochs 2 --prefix run3_mask_center_ --fold 0 > $LOG_DIR/run3 &
wait

echo "Training B6 run 1..."
PYTHONPATH=. python -u -m torch.distributed.launch --nproc_per_node=$GPUS \
  --master_port 9901 training/train_instance_no_offset.py --distributed \
  --data-dir /wdata/train_data/ --config configs/b6adam.json --predictions /wdata/train_data/oof_preds \
  --workers 8 --opt-level O1 --freeze-epochs 2 --prefix run1_mask_center_ --fold 0 > $LOG_DIR/run1b6 &
wait
echo "Completed!!!"
