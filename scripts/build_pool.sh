#!/bin/bash
# One CHUNK of a pool build, on the 30-min GPU short partition. Chained with
# --dependency=afterok by scripts/submit_chunked_pool_build.sh, each chunk
# appending to the pool file via --append/--stop-after-count.
#
#   sbatch scripts/build_pool.sh <config> <count_this_chunk>
#
#SBATCH --job-name=build_pool
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:29:00
#SBATCH --output=slurm-%x-%j.out

set -e
CONFIG="$1"; COUNT="$2"
if [ -z "$CONFIG" ] || [ -z "$COUNT" ]; then
  echo "usage: sbatch build_pool.sh <config.yaml> <count_this_chunk>"; exit 1
fi

source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source .venv/bin/activate

echo "===== building pool from $CONFIG (+${COUNT} items this chunk) ====="
python scripts/build_pool.py --config "$CONFIG" --append --stop-after-count "$COUNT"
echo "===== chunk done ====="
