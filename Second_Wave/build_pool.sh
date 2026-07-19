#!/bin/bash
#SBATCH --job-name=sw_pool
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:29:00
#SBATCH --output=slurm-%x-%j.out
# usage: sbatch build_pool.sh <out.pt> <size> [seed]
set -e
cd /pfs/data6/home/fr/fr_fr/fr_or51/projects/ml-lab-curriculum/Second_Wave
source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source ../.venv/bin/activate
OUT="$1"; SIZE="$2"; SEED="${3:-0}"
python pool.py --out "$OUT" --size "$SIZE" --min-features 2 --max-features 60 \
  --max-classes 10 --num-datapoints 200 --seed "$SEED"
echo "===== pool $OUT ($SIZE, seed $SEED) done ====="
