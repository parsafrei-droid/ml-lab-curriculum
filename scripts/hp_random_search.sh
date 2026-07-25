#!/bin/bash
# Tests: "does curriculum pretraining help more the further a config drifts
# from the paper's tuned optimum?" Samples N_TRIALS random (lr, num_datapoints)
# configs from the nanoTabPFN paper's own search space (Table 1, Appendix A)
# via scripts/sample_hp_configs.py, then trains BOTH arms of every trial:
#   paper_binary_hp<i>_s42   - no curriculum, at that trial's lr/num_datapoints
#   early_ramp_hp<i>_s42     - curriculum,    at the SAME lr/num_datapoints
# Everything else (architecture, batch_size, steps, seed, feature/class
# ceiling) is held fixed at paper_binary.yaml's exact values in both arms, so
# curriculum is the only thing that differs within a pair - see
# scripts/sample_hp_configs.py's docstring for why batch_size/weight_decay
# aren't also swept here.
#
# Each trial trains the FULL 2500 steps (paper_binary.yaml's own budget, not
# shortened) via run_with_toy_probe.py - no eval_tabarena.py call, same
# cheap-first philosophy as lr_screen.sh/lr_random_search.sh: the comparable
# signal for all 2*N_TRIALS runs is meta.json's final_val_loss/final_val_acc
# (FixedValidationCallback, one shared synthetic set) plus the per-checkpoint
# toy_tabarena.csv (real OpenML AUC) run_with_toy_probe.py adds for free.
# After this finishes: python scripts/rank_hp_sweep.py, then confirm any
# standout trial with a real eval_tabarena.py pass at the same checkpoint.
#
#   N_TRIALS=10 SAMPLE_SEED=42 sbatch scripts/hp_random_search.sh
#
# Timing: this repo's own paper-scale e2500 runs took ~1000-1200s each at
# num_datapoints=154 (see results/paper_binary_e2500_s42/meta.json). Datapoint
# attention is O(n^2) in rows, and this sweep's num_datapoints range goes up
# to 300 (~3.8x the reference cost per step), so budget well above the naive
# 2*N_TRIALS*20min estimate. Runs on the long partition (2-day max), not the
# 30-min short one lr_screen.sh uses for its much shorter 500-step trials.
#
#SBATCH --job-name=hp_random_search
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%x-%j.out

set -e
N_TRIALS="${N_TRIALS:-10}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"  # RNG seed for sampling (lr, num_datapoints) itself

source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source .venv/bin/activate

echo "=== sampling $N_TRIALS configs (sample-seed=$SAMPLE_SEED) ==="
python scripts/sample_hp_configs.py --n "$N_TRIALS" --sample-seed "$SAMPLE_SEED"

set +e
i=0
for trial in $(seq 0 $((N_TRIALS - 1))); do
  for prefix in paper_binary early_ramp; do
    i=$((i + 1))
    name="${prefix}_hp${trial}_s42"
    echo ""
    echo "############################## [$i/$((N_TRIALS * 2))] $name ##############################"
    python scripts/run_with_toy_probe.py --config experiments/configs/hp_sweep/${name}.yaml
  done
done

echo ""
echo "=== done ($((N_TRIALS * 2)) runs). now: python scripts/rank_hp_sweep.py ==="
