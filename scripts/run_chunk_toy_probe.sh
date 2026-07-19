#!/bin/bash
# One CHUNK of a long run, on the 30-min short partition - toy-probe variant.
# Same as scripts/run_chunk.sh but trains via scripts/run_with_toy_probe.py, so
# each chunk also appends to results/<name>/toy_tabarena.csv (real-data ROC-AUC
# on TOY_TASKS_CLASSIFICATION every checkpoint). Trains from wherever the last
# chunk left off (--resume) up to --stop-after-step. Chained with
# --dependency=afterok by scripts/submit_chunked_toy_probe.sh.
#
#   sbatch scripts/run_chunk_toy_probe.sh <config> <name> <seed> <steps> <stop_after> [resume]
#
#SBATCH --job-name=chunk_toy_probe
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

echo "===== chunk (toy probe): $NAME  up to step $STOP / $STEPS  (resume=${RESUME:-no}) ====="
if [ "$RESUME" = "resume" ]; then
  python scripts/run_with_toy_probe.py --config "$CONFIG" --name "$NAME" --seed "$SEED" --steps "$STEPS" --stop-after-step "$STOP" --resume
else
  python scripts/run_with_toy_probe.py --config "$CONFIG" --name "$NAME" --seed "$SEED" --steps "$STEPS" --stop-after-step "$STOP"
fi
echo "===== chunk done ====="
