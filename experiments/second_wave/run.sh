#!/bin/bash
set -e
cd "$(dirname "$0")"

POOL_SIZE="${POOL_SIZE:-80000}"
SEEDS="${SEEDS:-42 1 2}"
CONFIGS="${CONFIGS:-baseline curriculum_features}"

python pool.py --out pools/main.pt --size $POOL_SIZE --min-features 2 --max-features 60 \
  --max-classes 10 --num-datapoints 200 --seed 0

for cfg in $CONFIGS; do
  for seed in $SEEDS; do
    name="${cfg}_s${seed}"
    python train.py --config configs/$cfg.yaml --name $name --seed $seed
    python evaluate.py --checkpoint results/$name/checkpoint.pth --max-n-samples 5000
  done
done

python plot.py
