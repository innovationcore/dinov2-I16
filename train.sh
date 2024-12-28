export OMP_NUM_THREADS=1

#HUGE HACK
unset "${!SLURM@}"
#HACK

torchrun --nproc_per_node=1 dinov2/train/train.py --config-file=$1.yaml --output-dir=/workspace/dinov2-I16/$1