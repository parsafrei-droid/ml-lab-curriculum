#!/bin/bash
# SLURM template for one scenario on a CPU node (bwUniCluster).
# Usage:   sbatch scripts/submit_job_cpu.sh curriculum_features
#
# Same as submit_job.sh but requests no GPU - use this for scenarios you're
# happy to run on CPU so you don't tie up scarce GPU allocation. Adjust the
# partition / time lines to your cluster - those names are site-specific.

#SBATCH --job-name=curriculum_cpu
#SBATCH --partition=single         # <-- your CPU partition (check `sinfo`)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out

set -e
SCENARIO=$1
if [ -z "$SCENARIO" ]; then echo "usage: sbatch submit_job_cpu.sh <scenario_name>"; exit 1; fi

source .venv/bin/activate

echo "=== training $SCENARIO (CPU) ==="
python scripts/run.py --config experiments/configs/$SCENARIO.yaml

echo "=== evaluating $SCENARIO on TabArena ==="
# --max-n-samples caps rows per OpenML task: datapoint attention is O(n^2) in
# rows and large tasks OOM on CPU/GPU alike (see curriculum_reverse job 5682708)
python scripts/eval_tabarena.py --checkpoint results/$SCENARIO/checkpoint.pth --tasks tabarena --max-n-samples 5000

echo "=== done: commit results/$SCENARIO/ (checkpoint stays local) ==="
