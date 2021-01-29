mkdir -p foo /wdata/logs

echo "Creating masks..."
python create_masks.py "$@" > /wdata/logs/create_masks.out 2>&1

echo "Training b7..."
python -m torch.distributed.launch --nproc_per_node=4 train_b7_full2.py --train_dir "$@" > /wdata/logs/train_b7_full2.out 2>&1
python -m torch.distributed.launch --nproc_per_node=4 tune_b7_full.py --train_dir "$@" > /wdata/logs/tune_b7_full.out 2>&1
python -m torch.distributed.launch --nproc_per_node=4 train_b7_double_full.py --train_dir "$@" > /wdata/logs/train_b7_double_full.out 2>&1

echo "Training b6..."
python -m torch.distributed.launch --nproc_per_node=4 train_b6_full2.py --train_dir "$@" > /wdata/logs/train_b6_full2.out 2>&1
python -m torch.distributed.launch --nproc_per_node=4 tune_b6_full2.py --train_dir "$@" > /wdata/logs/tune_b6_full2.out 2>&1
python -m torch.distributed.launch --nproc_per_node=4 train_b6_double_full.py --train_dir "$@" > /wdata/logs/train_b6_double_full.out 2>&1

echo "All models trained!"