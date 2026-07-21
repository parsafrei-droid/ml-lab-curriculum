#!/bin/bash
# One CHUNK of a long pool-mode run, on the 30-min short partition. Same idea
# as scripts/run_chunk_toy_probe.sh but trains via scripts/run_pool.py (the
# fixed-pool-and-order curriculum mechanism - see curriculum/pool.py). Chained
# with --dependency=afterok by scripts/submit_chunked_pool.sh.
#
#   sbatch scripts/run_chunk_pool.sh <config> <name> <seed> <steps> <stop_after> [resume]
#
# Pool-mode does batch_size separate forward/backward passes per optimizer step
# (see run_pool.py's docstring for why), so its steps/min is not the same as
# scheduler-mode's - check the first chunk's slurm log / loss.csv interval_time_s
# before assuming submit_chunked_pool.sh's default PER_CHUNK still fits 29 min.
#
#SBATCH --job-name=chunk_pool
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:29:00
#SBATCH --output=slurm-%x-%j.out

set -e
CONFIG="$1"; NAME="$2"; SEED="$3"; STEPS="$4"; STOP="$5"; RESUME="$6"

source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source .venv/bin/activate

echo "===== chunk (pool): $NAME  up to step $STOP / $STEPS  (resume=${RESUME:-no}) ====="
if [ "$RESUME" = "resume" ]; then
  python scripts/run_pool.py --config "$CONFIG" --name "$NAME" --seed "$SEED" --steps "$STEPS" --stop-after-step "$STOP" --resume
else
  python scripts/run_pool.py --config "$CONFIG" --name "$NAME" --seed "$SEED" --steps "$STEPS" --stop-after-step "$STOP"
fi
echo "===== chunk done ====="
