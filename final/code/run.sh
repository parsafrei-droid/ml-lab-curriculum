#!/bin/bash
set -e
cd "$(dirname "$0")"

# Build the dump once. 80,000 tables = 2500 steps x 32 effective batch, so a run
# sees each table exactly once.
if [ ! -f pools/main.pt ]; then
  python pool.py --out pools/main.pt --size 80000 --min-features 2 --max-features 60 \
    --max-classes 10 --num-datapoints 200 --seed 0
fi

# Then every ordering reads that same dump. Three seeds each.
for seed in 42 1 2; do
  for cfg in baseline curriculum_features curriculum_context \
             curriculum_features_reverse curriculum_context_reverse \
             curriculum_combined curriculum_restart; do
    python train.py --config configs/$cfg.yaml --name ${cfg}_s${seed} --seed $seed
    python evaluate.py --checkpoint results/${cfg}_s${seed}/checkpoint.pth
  done
done

python plot.py
