#!/bin/bash
# Evaluate one Phase-A checkpoint on TabArena. Kept separate from training so each job
# stays inside the 30-minute dev_gpu_a100_il limit.
#   usage: sbatch eval_ours_a100.sh baseline_a100_s42
#SBATCH --job-name=tw_eval
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -e
name="$1"
REPO=/pfs/data6/home/fr/fr_fr/fr_or51/projects/ml-lab-curriculum
cd "$REPO/experiments/second_wave"
source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source "$REPO/.venv/bin/activate"

python -c "import torch; print('GPU:', torch.cuda.get_device_name())"
python evaluate.py --checkpoint "results/$name/checkpoint.pth" --max-n-samples 5000
