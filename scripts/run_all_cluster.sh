#!/bin/bash
# Run EVERY scenario at full scale, over several seeds, then evaluate each on
# TabArena. This is the full experiment for the poster.
#
#   sbatch scripts/run_all_cluster.sh          # submit as one GPU job
#   bash   scripts/run_all_cluster.sh          # or run directly on a GPU node
#
# Each run writes results/<scenario>_s<seed>/ (checkpoint stays local, the small
# csv/json/png get committed). ~6 min per run x 18 runs ~= 2 h.

#SBATCH --job-name=curriculum_all
#SBATCH --partition=gpu_h100          # bwUniCluster 3.0 H100 partition (3-day max; the _short one caps at 30 min)
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

set -e

SCENARIOS="baseline curriculum_combined curriculum_reverse curriculum_noise curriculum_features curriculum_classes curriculum_rows curriculum_combined_slow"
SEEDS="42 43 44"

# bwUniCluster 3.0: load the same Python the venv was built against, plus CUDA.
source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source .venv/bin/activate

# A crash in one run must not abort the other 17 (task requirement), so per-run
# commands are non-fatal: failures are logged and the loop moves on. set -e stays
# off for the loop body; the env setup above already succeeded or we'd have exited.
set +e
FAILED=""

for scenario in $SCENARIOS; do
  for seed in $SEEDS; do
    name="${scenario}_s${seed}"
    echo ""
    echo "############################## $name ##############################"
    if ! python scripts/run.py --config experiments/configs/$scenario.yaml --seed $seed --name $name; then
      echo "!!! TRAIN FAILED for $name (rc=$?) — skipping eval, continuing" >&2
      FAILED="$FAILED train:$name"
      continue
    fi
    if ! python scripts/eval_tabarena.py --checkpoint results/$name/checkpoint.pth --tasks tabarena --max-n-samples 5000; then
      echo "!!! EVAL FAILED for $name (rc=$?) — continuing" >&2
      FAILED="$FAILED eval:$name"
    fi
  done
done

echo ""
if [ -n "$FAILED" ]; then
  echo "=== done, but these steps FAILED:$FAILED ==="
else
  echo "=== all runs done. now: python scripts/compare_results.py, then commit results/*_s*/ ==="
fi
