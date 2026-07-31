#!/bin/bash
# Random-search lr screen for `baseline` ONLY - paper-style (Appendix A: 200
# random configs, short training, cheap synthetic proxy) but scoped to just
# the lr axis, over a WIDER range than both scripts/lr_screen.sh's grid and
# the paper's own [1e-4, 5e-2] (their search space never had to account for
# our bigger model/prior, and our current default of 1e-4 sits right at
# their lower bound - so this extends further in both directions).
#
# Same proxy-only philosophy as lr_screen.sh: each trial is a short (default
# 500-step) run_with_toy_probe.py training run, NO eval_tabarena.py call - the
# primary signal is still results/<name>/meta.json's final_val_loss (the
# shared *synthetic* validation set FixedValidationCallback already scores for
# free), analogous to the paper's 1600-synthetic-dataset proxy; the toy-probe
# entry point also writes results/<name>/toy_tabarena.csv per trial (real
# OpenML AUC every checkpoint), same reasoning as lr_screen.sh - though with
# 200 trials that's 200x the real-OpenML-cache load, worth knowing given this
# project's cache-race history. This only screens candidates; confirm the
# winner with a real run_with_toy_probe.py + eval_tabarena.py pass at the
# actual target step budget (10000) afterward.
#
# Reuses lr_screen.sh's naming (<scenario>_screen_lr<value>_s<seed>), so
# scripts/rank_lr_screen.py ranks these too, no separate reader needed.
#
#   N_TRIALS=200 STEPS=500 sbatch scripts/lr_random_search.sh
#
# ~200 trials x (500 steps x ~0.037s/step + ~15s process/import overhead)
# is roughly 2 hours - that's why this runs on the long partition, not the
# 30-min gpu_a100_short lr_screen.sh/lr_sweep_one.sh use.
#
#SBATCH --job-name=lr_random_search
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

set -e
SCENARIO="${SCENARIO:-baseline}"
N_TRIALS="${N_TRIALS:-200}"
STEPS="${STEPS:-500}"           # must be a multiple of CHECKPOINT_STEPS (100)
SEED="${SEED:-42}"              # training seed - fixed across trials, only lr varies
SAMPLE_SEED="${SAMPLE_SEED:-0}" # RNG seed for the lr sampling itself (reproducible search)
LR_LOW="${LR_LOW:-1e-6}"
LR_HIGH="${LR_HIGH:-3e-1}"

source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source .venv/bin/activate

# log-uniform samples (paper's search space was also log-scale for lr), one per line
LRS=$(python3 -c "
import numpy as np
rng = np.random.default_rng($SAMPLE_SEED)
lo, hi = np.log10($LR_LOW), np.log10($LR_HIGH)
for v in 10 ** rng.uniform(lo, hi, size=$N_TRIALS):
    print(f'{v:.6g}')
")

set +e
i=0
for lr in $LRS; do
  i=$((i + 1))
  name="${SCENARIO}_screen_lr${lr}_s${SEED}"
  echo ""
  echo "############################## [$i/$N_TRIALS] $name (steps=$STEPS) ##############################"
  python scripts/run_with_toy_probe.py --config experiments/configs/$SCENARIO.yaml \
    --lr $lr --name $name --seed $SEED --steps $STEPS
done

echo ""
echo "=== done ($N_TRIALS trials). now: python scripts/rank_lr_screen.py ==="
