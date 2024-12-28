# your_training_script.py
import torch
import torch.distributed as dist
import os

def main():
    # Initialize the distributed environment
    dist.init_process_group(backend="nccl")

    # Get the rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {rank} of {world_size} running on {os.uname()[1]}")

    # Example: simple distributed operation
    tensor = torch.ones(1).to(rank)
    dist.all_reduce(tensor)
    print(f"Rank {rank}: Tensor after all_reduce: {tensor}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()