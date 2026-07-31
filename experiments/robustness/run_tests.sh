#!/bin/bash
#SBATCH --job-name=robustness
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -e

# One config per job so nothing hits the 30 min cap. Call as:
#   sbatch run_tests.sh pool_seed 1
#   sbatch run_tests.sh pool_seed 2
#   sbatch run_tests.sh noise noise_baseline
#   sbatch run_tests.sh noise curriculum_noise
#   sbatch run_tests.sh noise curriculum_noise_reverse
TEST="$1"
ARG="$2"

REPO="${REPO:-$HOME/ml-lab-curriculum}"
cd "$REPO"
source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source "$REPO/.venv/bin/activate"
python -c "import torch; print('GPU:', torch.cuda.get_device_name())"

CODE="$REPO/final/code"
ROB="$REPO/experiments/robustness"

if [ "$TEST" = "pool_seed" ]; then
  # Test A: rebuild the dump with a different generator seed, then rerun the two
  # runs that carry the headline. Everything else is untouched.
  POOL="$ROB/pools/main_p${ARG}.pt"
  if [ ! -f "$POOL" ]; then
    python "$CODE/pool.py" --out "$POOL" --size 80000 --min-features 2 --max-features 60 \
      --max-classes 10 --num-datapoints 200 --seed "$ARG"
  fi
  for cfg in baseline curriculum_features; do
    sed "s|^pool: .*|pool: $POOL|" "$CODE/configs/$cfg.yaml" > "$ROB/configs/${cfg}_p${ARG}.yaml"
    python "$CODE/train.py" --config "$ROB/configs/${cfg}_p${ARG}.yaml" \
      --name "${cfg}_pool${ARG}_s42" --seed 42
    python "$CODE/evaluate.py" \
      --checkpoint "$CODE/results/${cfg}_pool${ARG}_s42/checkpoint.pth"
  done

elif [ "$TEST" = "noise" ]; then
  # Test B: same design, new axis. Build the noise-labelled dump once, then run
  # shuffle / low-to-high / high-to-low over it.
  POOL="$ROB/pools/noise.pt"
  if [ ! -f "$POOL" ]; then
    python "$ROB/noise_pool.py" --out "$POOL" --size 80000 --seed 0
  fi
  python "$CODE/train.py" --config "$ROB/configs/${ARG}.yaml" --name "${ARG}_s42" --seed 42
  python "$CODE/evaluate.py" --checkpoint "$CODE/results/${ARG}_s42/checkpoint.pth"

else
  echo "usage: sbatch run_tests.sh {pool_seed <n> | noise <config>}" >&2
  exit 1
fi
