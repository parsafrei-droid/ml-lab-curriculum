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
#SBATCH --partition=gpu_a100_short    # <-- your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

set -e

SCENARIOS="baseline curriculum_combined curriculum_reverse curriculum_noise curriculum_features curriculum_combined_slow"
SEEDS="42 43 44"

# module load devel/cuda/12.2     # <-- if your cluster needs it
source .venv/bin/activate

for scenario in $SCENARIOS; do
  for seed in $SEEDS; do
    name="${scenario}_s${seed}"
    echo ""
    echo "############################## $name ##############################"
    python scripts/run.py --config experiments/configs/$scenario.yaml --seed $seed --name $name
    python scripts/eval_tabarena.py --checkpoint results/$name/checkpoint.pth --tasks tabarena --max-n-samples 5000
  done
done

echo ""
echo "=== all runs done. now: python scripts/compare_results.py, then commit results/*_s*/ ==="
