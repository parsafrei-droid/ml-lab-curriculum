#!/bin/bash
# SLURM template for one scenario on a GPU node (bwUniCluster / uni GPU).
# Usage:   sbatch scripts/submit_job.sh curriculum_combined
#
# It trains the scenario, then evaluates the checkpoint on TabArena. Adjust the
# partition / module / time lines to your cluster - those names are site-specific.

#SBATCH --job-name=curriculum
#SBATCH --partition=gpu_4          # <-- your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out

set -e
SCENARIO=$1
if [ -z "$SCENARIO" ]; then echo "usage: sbatch submit_job.sh <scenario_name>"; exit 1; fi

# module load devel/cuda/12.2     # <-- if your cluster needs it
source .venv/bin/activate

echo "=== training $SCENARIO ==="
python scripts/run.py --config experiments/configs/$SCENARIO.yaml

echo "=== evaluating $SCENARIO on TabArena ==="
python scripts/eval_tabarena.py --checkpoint results/$SCENARIO/checkpoint.pth --tasks tabarena

echo "=== done: commit results/$SCENARIO/ (checkpoint stays local) ==="
