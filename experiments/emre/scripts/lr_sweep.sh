#!/bin/bash
# Learning-rate sweep on OUR (big) model, for baseline AND curriculum_combined.
# Answers two things at once:
#   1. what's the best-tuned baseline in our setting (we only ever used lr 1e-4,
#      which was never tuned)? -> the fair big-model comparison.
#   2. is the curriculum less sensitive to lr (a flatter AUC-vs-lr curve)? -> the
#      "curriculum reduces the need to tune" hypothesis.
#
#   sbatch scripts/lr_sweep.sh
#
# Writes results/<scenario>_lr<value>_s<seed>/; then: python scripts/plot_lr_sweep.py

#SBATCH --job-name=lr_sweep
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -e
LRS="${LRS:-0.0001 0.0003 0.001 0.003}"
SCENARIOS="${SCENARIOS:-baseline curriculum_combined}"
STEPS="${STEPS:-5000}"
SEEDS="${SEEDS:-42}"

source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source .venv/bin/activate

set +e
for lr in $LRS; do
  for scenario in $SCENARIOS; do
    for seed in $SEEDS; do
      name="${scenario}_lr${lr}_s${seed}"
      echo ""
      echo "############################## $name ##############################"
      python scripts/run.py --config experiments/configs/$scenario.yaml \
        --lr $lr --name $name --seed $seed --steps $STEPS || continue
      python scripts/eval_tabarena.py --checkpoint results/$name/checkpoint.pth \
        --tasks tabarena --max-n-samples 5000
    done
  done
done

echo ""
echo "=== done. now: python scripts/plot_lr_sweep.py ==="
