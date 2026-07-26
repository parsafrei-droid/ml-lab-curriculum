#!/bin/bash
#SBATCH --job-name=sw_eval
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:29:00
#SBATCH --output=slurm-%x-%j.out
set -e
cd "$(dirname "$0")"
source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source ../.venv/bin/activate
python evaluate.py --checkpoint "results/$1/checkpoint.pth" --max-n-samples 5000
