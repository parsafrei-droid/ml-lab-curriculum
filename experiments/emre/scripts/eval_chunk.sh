#!/bin/bash
# Final step of a chunked run: evaluate the finished checkpoint on TabArena.
# Chained after the last training chunk with --dependency=afterok.
#
#   sbatch scripts/eval_chunk.sh <name>
#
#SBATCH --job-name=evalchunk
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:29:00
#SBATCH --output=slurm-%x-%j.out

set -e
NAME="$1"
source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source .venv/bin/activate

python scripts/eval_tabarena.py --checkpoint "results/$NAME/checkpoint.pth" --tasks tabarena --max-n-samples 5000
echo "===== eval done for $NAME ====="
