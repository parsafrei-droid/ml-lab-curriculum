#!/bin/bash
#SBATCH --job-name=sw_pool_cpu
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
# usage: sbatch build_pool_cpu.sh <out.pt> <size>
set -e
cd /pfs/data6/home/fr/fr_fr/fr_or51/projects/ml-lab-curriculum/Second_Wave
source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
source ../.venv/bin/activate
OUT="$1"; SIZE="$2"
python pool.py --out "$OUT" --size "$SIZE" --min-features 2 --max-features 60 \
  --max-classes 10 --num-datapoints 200 --seed 0
echo "===== pool $OUT ($SIZE) done ====="
