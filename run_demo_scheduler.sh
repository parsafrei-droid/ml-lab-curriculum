#!/bin/bash
#SBATCH --job-name=nanotab_gpu_test    # Name of your job
#SBATCH --partition=gpu_a100_short     # The short testing queue
#SBATCH --gres=gpu:1                   # Request exactly 1 GPU
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=4              # Request 4 CPU cores
#SBATCH --mem=20G                      # Request 20 GB of RAM
#SBATCH --time=00:10:00                # Time limit (HH:MM:SS)

python scripts/demo_scheduler.py