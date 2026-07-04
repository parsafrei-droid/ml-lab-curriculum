#!/bin/bash
# Paper-reference runs: nanoTabPFN's published small recipe (2500 steps), 3 seeds,
# scored on our TabArena eval. A fixed context bar for the budget plots — not part
# of the curriculum sweep, so it lives in its own tiny job.
#
#   sbatch scripts/run_paper_small.sh
#
#SBATCH --job-name=paper_small
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source .venv/bin/activate

set +e
FAILED=""
for s in 42 43 44; do
  echo ""
  echo "############################## paper_small_s$s ##############################"
  if ! python scripts/run.py --config experiments/configs/paper_small.yaml --name paper_small_s$s --seed $s; then
    echo "!!! TRAIN FAILED for paper_small_s$s — skipping eval, continuing" >&2
    FAILED="$FAILED train:paper_small_s$s"; continue
  fi
  if ! python scripts/eval_tabarena.py --checkpoint results/paper_small_s$s/checkpoint.pth --tasks tabarena --max-n-samples 5000; then
    echo "!!! EVAL FAILED for paper_small_s$s — continuing" >&2
    FAILED="$FAILED eval:paper_small_s$s"
  fi
done

echo ""
[ -n "$FAILED" ] && echo "=== done, but FAILED:$FAILED ===" || echo "=== paper_small done ==="
