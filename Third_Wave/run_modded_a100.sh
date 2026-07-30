#!/bin/bash
# Phase B: modded-nanoTabPFN on the same A100 as Phase A.
#   run 3 (baseline):   sbatch run_modded_a100.sh baseline
#   run 4 (curriculum): sbatch run_modded_a100.sh curriculum
# Run 4 differs from run 3 by --feature-curriculum only: the dump's datasets are fed
# in order of increasing feature count. Model, optimizer, batch size, seed: identical.
#SBATCH --job-name=tw_modded
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -e
mode="$1"
REPO=/pfs/data6/home/fr/fr_fr/fr_or51/projects/ml-lab-curriculum
cd "$REPO/Third_Wave/modded-nanotabpfn"
source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source "$REPO/.venv/bin/activate"

# The eval loop calls openml for all 38 TabArena tasks after every epoch. The cache is
# pre-warmed on the login node so this reads from disk instead of hammering OpenML
# (an uncached run is what got the earlier attempt cancelled).
export OPENML_CACHE_DIR="$HOME/.cache/openml"

# modded evaluates all 38 TabArena tasks after *every* epoch (~9s), which dominates the
# job far more than training does (~1s/epoch after the one-off compile). The reference
# record reaches the target in 57 epochs; 120 keeps both runs inside the partition's
# 30-minute limit (~21 min baseline, ~23 min curriculum) with headroom to spare.
SEED="${SEED:-11}"
EPOCHS="${EPOCHS:-120}"
if [ "$mode" = "curriculum" ]; then
  python train_nano.py --name tw_curriculum --seed "$SEED" --epochs "$EPOCHS" --feature-curriculum
else
  python train_nano.py --name tw_baseline --seed "$SEED" --epochs "$EPOCHS"
fi
