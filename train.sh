export OMP_NUM_THREADS=8
torchrun --nproc_per_node=4 dinov2/train/train.py --config-file=vitb14.yaml --output-dir=/workspace/output_model