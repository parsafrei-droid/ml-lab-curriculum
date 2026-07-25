#!/bin/bash
# One-off CPU job: run scripts/plot_hp_sweep.py (no GPU needed, just
# matplotlib/numpy - which segfault on this cluster's login node, so this
# runs on a real compute node instead).
#
#   sbatch scripts/submit_plot_hp_sweep.sh

#SBATCH --job-name=plot_hp_sweep
#SBATCH --partition=dev_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out

set -e
source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
source .venv/bin/activate

python scripts/plot_hp_sweep.py
