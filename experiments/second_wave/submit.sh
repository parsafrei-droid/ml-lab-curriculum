#!/bin/bash
#SBATCH --job-name=second_wave
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -e
cd "$(dirname "$0")"

source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source ../../.venv/bin/activate

POOL_SIZE="${POOL_SIZE:-80000}"
SEEDS="${SEEDS:-42 1 2}"
CONFIGS="${CONFIGS:-baseline curriculum_features}"

if [ ! -f pools/main.pt ]; then
  python pool.py --out pools/main.pt --size $POOL_SIZE --min-features 2 --max-features 60 \
    --max-classes 10 --num-datapoints 200 --seed 0
fi

for cfg in $CONFIGS; do
  for seed in $SEEDS; do
    name="${cfg}_s${seed}"
    python train.py --config configs/$cfg.yaml --name $name --seed $seed
    python evaluate.py --checkpoint results/$name/checkpoint.pth --max-n-samples 5000
  done
done

python plot.py
