#!/bin/bash
# Train ONE config on the A100 dev partition.
# The partition caps jobs at 30 min; a single training run is ~12 min, so we submit
# one job per config instead of looping over both (the two-in-one loop in
# run_ours_a100.sh does not fit, and evaluation is submitted separately).
#   usage: SEED=42 sbatch run_ours_a100_one.sh baseline
#SBATCH --job-name=tw_ours
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -e
cfg="$1"
# Slurm copies the batch script into /var/spool/slurmd, so $0 does not point at the
# repo here; use an absolute path instead of "$(dirname "$0")".
REPO=/pfs/data6/home/fr/fr_fr/fr_or51/projects/ml-lab-curriculum
cd "$REPO/experiments/second_wave"
source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source "$REPO/.venv/bin/activate"

python -c "import torch; print('GPU:', torch.cuda.get_device_name())"

SEED="${SEED:-42}"
name="${cfg}_a100_s${SEED}"
python train.py --config configs/$cfg.yaml --name $name --seed $SEED
