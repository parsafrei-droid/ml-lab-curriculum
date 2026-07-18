#!/bin/bash
# BwUniCluster job: Dominika's baseline = pretrain nanoTabPFN on the UNCHANGED
# TabICLv2 prior, then evaluate on TabArena. One H100 node, train + eval in one go.
#
#   sbatch phase2/job_baseline.sh
#
# Runs the phase2/configs/baseline_default_prior.yaml config as-is (batch 32,
# max_features=100 = the prior's own default) at 5000 steps (--epochs 50). The
# H100 has 80 GB, so the batch-32 config that OOM'd the T4/K80 fits comfortably.
#
# Outputs land in results/baseline_default_prior_e50_s42/:
#   loss.csv / loss_curve.png       - training curve
#   val.csv  / val_loss_curve.png   - validation curve (shared fixed val set)
#   checkpoint.pth                  - the trained model
#   tabarena_scores.json            - per-dataset ROC-AUC + mean

#SBATCH --job-name=baseline_default_prior
#SBATCH --partition=gpu_h100        # production H100 (3-day max); train+eval > 30 min
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=02:00:00             # ~5 min train + up to ~1 h TabArena eval, w/ margin
#SBATCH --output=slurm-%x-%j.out

set -e

NAME=baseline_default_prior_e50_s42
CONFIG=phase2/configs/baseline_default_prior.yaml

# --- environment: same modules the venv was built against, plus CUDA ---
source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8

# --- build the venv on first run, then reuse it ---
# setup_env.py clones the two upstream deps, applies the tabicl import patch, and
# installs everything into the active environment (cross-platform, cluster-aware).
if [ ! -d .venv ]; then
  echo "=== .venv not found: creating it and installing deps (first run only) ==="
  python -m venv .venv
  source .venv/bin/activate
  # setup_env.py does `pip install -e TFM-Playground -e tabicl <DEPS>` WITH full
  # dependency resolution, so schedulefree / einops / openml come in via the two
  # editable packages' own requirements. No stubs needed on the cluster.
  python scripts/setup_env.py
else
  source .venv/bin/activate
fi

# Safety net: ensure the few names run.py/eval import that aren't guaranteed by the
# editable installs are present (no-op if already installed).
python - <<'PY' || pip install schedulefree einops openml
import importlib
for m in ("schedulefree", "einops", "openml"):
    importlib.import_module(m)
print("run/eval deps present")
PY

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== [1/2] pretrain the unchanged-prior baseline: $NAME (5000 steps) ==="
python scripts/run.py --config "$CONFIG" --epochs 50 --name "$NAME"

echo "=== [2/2] evaluate $NAME on TabArena ==="
# --max-n-samples caps rows per OpenML task: datapoint attention is O(n^2) in rows,
# so very large tasks would OOM even the H100.
python scripts/eval_tabarena.py --checkpoint "results/$NAME/checkpoint.pth" \
    --tasks tabarena --max-n-samples 5000

echo "=== done. results in results/$NAME/ ==="
echo "    curves: loss_curve.png, val_loss_curve.png"
echo "    score : tabarena_scores.json"
