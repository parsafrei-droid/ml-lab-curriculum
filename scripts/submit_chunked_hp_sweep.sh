#!/bin/bash
# Chunked version of scripts/hp_random_search.sh: trains the same 2*N_TRIALS
# runs (paper_binary_hp<i>_s42 / early_ramp_hp<i>_s42) as chains of short jobs
# on the 30-min short partition instead of one long job on a long partition
# (gpu_a100_il/gpu_h100/gpu_h100_il) - use this when the long partitions are
# stuck in a priority queue with no visibility into why. Same reasoning as
# this project's own scripts/submit_chunked_toy_probe.sh, which exists for the
# same reason; this reuses scripts/run_chunk_toy_probe.sh UNMODIFIED (it's
# already generic over config path / name / seed / steps).
#
# No final eval_chunk.sh call: unlike submit_chunked_toy_probe.sh's scenarios,
# this sweep deliberately stays proxy-only (meta.json's final_val_loss/
# final_val_acc + the free toy_tabarena.csv column) for all 20 runs - see
# scripts/rank_hp_sweep.py. Confirm any standout trial with a real
# eval_tabarena.py pass afterward, by hand.
#
# PER_CHUNK defaults to 500 steps, not submit_chunked_toy_probe.sh's 1200:
# this sweep's num_datapoints ranges up to 300 (~3.8x the ~154-row reference
# cost per step - datapoint attention is O(n^2) in rows), so a chunk sized for
# the reference speed could blow the short partition's 29-min cap on the
# largest sampled num_datapoints. 500 steps stays safely under it even then.
#
#   python scripts/sample_hp_configs.py --train-seed 42   # generate experiments/configs/hp_sweep/*.yaml first
#   scripts/submit_chunked_hp_sweep.sh [n_trials] [per_chunk]
#   SEED=43 scripts/submit_chunked_hp_sweep.sh             # repeat with a different training seed
#                                                           # (same 10 sampled configs - see sample_hp_configs.py)

set -e
N_TRIALS="${1:-10}"
PER_CHUNK="${2:-500}"
SEED="${SEED:-42}"
STEPS=2500   # fixed - paper_binary.yaml's own budget, not swept

CONFIG_DIR="experiments/configs/hp_sweep"
if [ ! -d "$CONFIG_DIR" ] || [ -z "$(ls -A "$CONFIG_DIR"/*.yaml 2>/dev/null)" ]; then
  echo "no configs in $CONFIG_DIR - run: python scripts/sample_hp_configs.py"
  exit 1
fi

for trial in $(seq 0 $((N_TRIALS - 1))); do
  for prefix in paper_binary early_ramp; do
    name="${prefix}_hp${trial}_s${SEED}"
    config="${CONFIG_DIR}/${name}.yaml"
    if [ ! -f "$config" ]; then
      echo "missing $config - skipping (did you run sample_hp_configs.py --train-seed $SEED with a matching --n?)"
      continue
    fi
    echo "### $name : chaining chunks of $PER_CHUNK steps up to $STEPS ###"

    dep=""            # dependency on the previous chunk
    stop=0
    first=1
    while [ "$stop" -lt "$STEPS" ]; do
      stop=$(( stop + PER_CHUNK ))
      [ "$stop" -gt "$STEPS" ] && stop=$STEPS
      if [ "$first" -eq 1 ]; then
        resume=""; first=0
      else
        resume="resume"
      fi
      jid=$(sbatch $dep --job-name="c_${name}" \
            scripts/run_chunk_toy_probe.sh "$config" "$name" "$SEED" "$STEPS" "$stop" "$resume" \
            | awk '{print $NF}')
      echo "   chunk ->$stop : job $jid ${dep:+(after ${dep##*:})}"
      dep="--dependency=afterok:$jid"
    done
  done
done
echo "all chains submitted. watch: squeue -u \$USER"
