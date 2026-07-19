#!/bin/bash
# SLURM template for one scenario on a GPU node (bwUniCluster / uni GPU),
# using scripts/run_with_toy_probe.py instead of scripts/run.py - trains the
# same way but also scores TOY_TASKS_CLASSIFICATION (real OpenML data) every
# checkpoint, writing results/<name>/toy_tabarena.csv + toy_tabarena_curve.png
# alongside the usual loss.csv/val.csv. See run_with_toy_probe.py / toy_tabarena_probe.py.
#
# Usage:   sbatch scripts/submit_job_toy_probe.sh curriculum_combined
#          STEPS=10000 sbatch scripts/submit_job_toy_probe.sh curriculum_combined
#
# It trains the scenario, then evaluates the checkpoint on TabArena. Adjust the
# partition / module / time lines to your cluster - those names are site-specific.

#SBATCH --job-name=curriculum_toy_probe
#SBATCH --partition=gpu_a100_il          # A100, smallest/least-contended GPU tier this cluster has for a >30min single job
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out

set -e
SCENARIO=$1
if [ -z "$SCENARIO" ]; then echo "usage: sbatch submit_job_toy_probe.sh <scenario_name>"; exit 1; fi
NAME="${SCENARIO}${STEPS:+_e$STEPS}"
STEPS_ARGS=""
if [ -n "$STEPS" ]; then STEPS_ARGS="--steps $STEPS --name $NAME"; fi

source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source .venv/bin/activate

echo "=== training $NAME (with per-checkpoint toy TabArena probe) ==="
python scripts/run_with_toy_probe.py --config experiments/configs/$SCENARIO.yaml $STEPS_ARGS

echo "=== evaluating $NAME on TabArena ==="
# --max-n-samples caps rows per OpenML task: datapoint attention is O(n^2) in
# rows and large tasks OOM the GPU (see curriculum_reverse job 5682708)
python scripts/eval_tabarena.py --checkpoint results/$NAME/checkpoint.pth --tasks tabarena --max-n-samples 5000

echo "=== done: commit results/$NAME/ (checkpoint stays local) ==="
