#!/bin/bash
#SBATCH --job-name=sw_run
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:29:00
#SBATCH --output=slurm-%x-%j.out
# usage: sbatch run_one.sh <config> <name> <seed>
set -e
cd /pfs/data6/home/fr/fr_fr/fr_or51/projects/ml-lab-curriculum/Second_Wave
source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source ../.venv/bin/activate
CONFIG="$1"; NAME="$2"; SEED="$3"
python train.py --config "configs/$CONFIG.yaml" --name "$NAME" --seed "$SEED"
python evaluate.py --checkpoint "results/$NAME/checkpoint.pth" --max-n-samples 5000
echo "===== $NAME done ====="
