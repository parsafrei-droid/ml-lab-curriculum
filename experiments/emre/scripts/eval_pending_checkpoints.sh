#!/bin/bash
# One CHUNK-style job: evaluate a single already-trained checkpoint on TabArena.
# For checkpoints that finished training but never got scored (originally: for
# the six e10000 runs - see git blame on this comment for the OOM story).
#
#   sbatch scripts/eval_pending_checkpoints.sh <name>
#
# Uses scripts/eval_tabarena.py's own defaults (max-n-features=120,
# max-n-samples=5000) rather than repeating them here - see that script's
# docstring for why 120 (recovers the legacy 16-dataset set while still
# excluding the pathologically wide tasks that OOM the big architecture).
# Writes results/<name>/tabarena_scores.json with mean_roc_auc_all/binary/16.
#
#SBATCH --job-name=eval_pending
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:29:00
#SBATCH --output=slurm-%x-%j.out

set -e
NAME="$1"
if [ -z "$NAME" ]; then echo "usage: sbatch eval_pending_checkpoints.sh <name>"; exit 1; fi

source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source .venv/bin/activate

python scripts/eval_tabarena.py --checkpoint "results/$NAME/checkpoint.pth" --tasks tabarena
echo "===== eval done for $NAME ====="
