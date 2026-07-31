#!/bin/bash
# Run ONE (scenario, seed) as a short <30-min job on the free H100 short partition,
# used when gpu_h100 (production) is reserved/drained. Splitting the sweep into
# one job per run keeps each well under the 30-min cap and lets them run in
# parallel across the free nodes.
#
#   sbatch scripts/run_one_short.sh <config> <name> <seed> [epochs]
#     <config> path to the scenario YAML
#     <name>   result folder under results/
#     <seed>   RNG seed
#     [epochs] optional --epochs override (omit to use the config's own epochs)
#
#SBATCH --job-name=run1
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:29:00
#SBATCH --output=slurm-%x-%j.out

set -e
CONFIG="$1"; NAME="$2"; SEED="$3"; EPOCHS="$4"

source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source .venv/bin/activate

echo "############################## $NAME ##############################"
if [ -n "$EPOCHS" ]; then
  python scripts/run.py --config "$CONFIG" --name "$NAME" --seed "$SEED" --epochs "$EPOCHS"
else
  python scripts/run.py --config "$CONFIG" --name "$NAME" --seed "$SEED"
fi
python scripts/eval_tabarena.py --checkpoint "results/$NAME/checkpoint.pth" --tasks tabarena --max-n-samples 5000
echo "=== $NAME done ==="
