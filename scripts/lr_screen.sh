#!/bin/bash
# Cheap lr screen: mirrors the nanoTabPFN paper's HPO philosophy (many candidates,
# short training, a cheap synthetic proxy metric - see paper Appendix A) rather
# than scripts/lr_sweep.sh's approach (few candidates, full step budget, full
# real-TabArena eval per candidate - expensive per point, so only 4 lr values).
#
# Each candidate here trains for --steps only (short, default 500) via
# run_with_toy_probe.py and stops - NO eval_tabarena.py call. The primary
# proxy signal is still results/<name>/meta.json's final_val_loss/final_val_acc
# (FixedValidationCallback, scored on one shared *synthetic* validation set -
# exactly analogous to the paper's 1600 synthetic held-out datasets); using
# the toy-probe entry point on top also gets you results/<name>/toy_tabarena.csv
# (real OpenML AUC every checkpoint) for free per trial, for the same reason
# every training script in this project defaults to it now. Note this does
# add real OpenML calls to every trial - with many trials (e.g.
# lr_random_search.sh's 200) that's real added load on the shared
# ~/.cache/openml directory (see this project's cache-race history); rank_lr_
# screen.py itself still only reads final_val_loss/final_val_acc from meta.json.
#
# This only screens candidates - it does NOT replace a real-TabArena
# confirmation run of the winner at the target step budget (e.g. 10000).
# After this finishes: python scripts/rank_lr_screen.py, then run the top
# pick(s) through the normal run_with_toy_probe.py + eval_tabarena.py path at
# full steps.
#
#   LRS="..." SCENARIOS="..." STEPS=500 SEEDS="42" sbatch scripts/lr_screen.sh
#
#SBATCH --job-name=lr_screen
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:29:00
#SBATCH --output=slurm-%x-%j.out

set -e
# log-spaced, wider than lr_sweep.sh's 4 points since each trial is now cheap
# (no TabArena eval, short training) - extends a bit below/above the paper's
# own [1e-4, 5e-2] range since our model/batch differ from theirs.
LRS="${LRS:-0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03}"
SCENARIOS="${SCENARIOS:-baseline curriculum_combined}"
SEEDS="${SEEDS:-42}"
STEPS="${STEPS:-500}"   # must be a multiple of CHECKPOINT_STEPS (100)

source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source .venv/bin/activate

set +e
for lr in $LRS; do
  for scenario in $SCENARIOS; do
    for seed in $SEEDS; do
      name="${scenario}_screen_lr${lr}_s${seed}"
      echo ""
      echo "############################## $name (steps=$STEPS) ##############################"
      python scripts/run_with_toy_probe.py --config experiments/configs/$scenario.yaml \
        --lr $lr --name $name --seed $seed --steps $STEPS
    done
  done
done

echo ""
echo "=== done. now: python scripts/rank_lr_screen.py ==="
