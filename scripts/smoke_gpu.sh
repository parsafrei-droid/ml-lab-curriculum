#!/bin/bash
#SBATCH --job-name=curriculum_smoke
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:25:00
#SBATCH --output=slurm-%x-%j.out

set -e
source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source .venv/bin/activate

echo "=== torch / cuda ==="
python -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "=== quick train (baseline, seed 42, real config = full 2000 steps) ==="
python scripts/run.py --config experiments/configs/baseline.yaml --seed 42 --name _smoke_baseline

echo "=== eval on TabArena (checks OpenML network + GPU eval) ==="
python scripts/eval_tabarena.py --checkpoint results/_smoke_baseline/checkpoint.pth --tasks tabarena --max-n-samples 5000

echo "=== SMOKE OK ==="
