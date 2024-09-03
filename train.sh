export OMP_NUM_THREADS=8

#HUGE HACK
unset "${!SLURM@}"
#HACK

torchrun --nproc_per_node=8 dinov2/train/train.py --config-file=vitb14.yaml --output-dir=/workspace/output_model