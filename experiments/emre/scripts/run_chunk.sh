#!/bin/bash
# One CHUNK of a long run, on the 30-min short partition. Trains from wherever the
# last chunk left off (via run.py --resume) up to --stop-after-step. Chained with
# --dependency=afterok by scripts/submit_chunked.sh so a full run completes as
# several of these back to back. Used when gpu_a100_il (the only long partition) is
# unavailable and batch-32 is too slow to fit one short job.
#
#   sbatch scripts/run_chunk.sh <config> <name> <seed> <steps> <stop_after> [resume]
#
#SBATCH --job-name=chunk
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

echo "===== chunk: $NAME  up to step $STOP / $STEPS  (resume=${RESUME:-no}) ====="
if [ "$RESUME" = "resume" ]; then
  python scripts/run.py --config "$CONFIG" --name "$NAME" --seed "$SEED" --steps "$STEPS" --stop-after-step "$STOP" --resume
else
  python scripts/run.py --config "$CONFIG" --name "$NAME" --seed "$SEED" --steps "$STEPS" --stop-after-step "$STOP"
fi
echo "===== chunk done ====="
