#!/bin/bash
# Same as eval_chunk.sh, but restricted to binary-classification TabArena
# tasks (--binary-only), to check the model specifically against binary
# datasets rather than the full mixed binary/multiclass suite.
#
#   sbatch scripts/eval_chunk_binary.sh <name>
#
#SBATCH --job-name=evalchunkbin
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

python scripts/eval_tabarena.py --checkpoint "results/$NAME/checkpoint.pth" --tasks tabarena --max-n-samples 5000 --binary-only --output-name tabarena_scores_binary.json
echo "===== binary-only eval done for $NAME ====="
