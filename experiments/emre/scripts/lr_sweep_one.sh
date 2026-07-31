#!/bin/bash
# One (scenario, lr, seed) run of the lr sweep, as a short job on the free
# short partition (gpu_a100_il is reserved). Body matches scripts/lr_sweep.sh's
# inner loop exactly; the sweep is just split into per-run jobs so it fits the
# 30-min cap and runs in parallel. Each 5k-step run is ~6 min.
#
#   sbatch scripts/lr_sweep_one.sh <scenario> <lr> <seed> <steps>
#
#SBATCH --job-name=lr1
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:29:00
#SBATCH --output=slurm-%x-%j.out

set -e
SCENARIO="$1"; LR="$2"; SEED="$3"; STEPS="$4"
name="${SCENARIO}_lr${LR}_s${SEED}"

source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source .venv/bin/activate

echo "############################## $name ##############################"
python scripts/run.py --config experiments/configs/$SCENARIO.yaml \
  --lr $LR --name $name --seed $SEED --steps $STEPS
python scripts/eval_tabarena.py --checkpoint results/$name/checkpoint.pth \
  --tasks tabarena --max-n-samples 5000
echo "===== $name done ====="
