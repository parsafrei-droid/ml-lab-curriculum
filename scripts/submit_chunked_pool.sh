#!/bin/bash
# Submit a full chunked run for each (scenario x seed) using scripts/run_pool.py
# (the fixed-pool-and-order curriculum mechanism) - a dependency-chained sequence
# of short training jobs plus a final eval, same idiom as
# scripts/submit_chunked_toy_probe.sh. Requires the pool file each scenario's
# config points at (`pool:` key) to already exist - build it first with
# scripts/build_pool.sh / scripts/build_pool.py.
#
#   scripts/submit_chunked_pool.sh "<scenarios>" "<seeds>" <steps> [steps_per_chunk] [after_jobid]
#
# e.g.:
#   scripts/submit_chunked_pool.sh "pool_curriculum_noise_binary pool_baseline_shuffle_binary" "42" 10000
#
# [after_jobid], if given, makes every scenario/seed's FIRST chunk depend on
# that job (afterok) - so a pool build that hasn't finished yet can still be
# submitted against right now: the training chunks queue immediately but only
# actually start once the pool build job completes.
#
# Result folders are tagged _e<steps> (matching run_all_cluster.sh/submit_chunked_toy_probe.sh),
# e.g. pool_curriculum_noise_binary_e10000_s42.
#
# PER_CHUNK default is deliberately conservative relative to
# submit_chunked_toy_probe.sh's 1200: pool-mode does batch_size separate
# forward/backward passes per optimizer step (see run_pool.py's docstring),
# so its steps/min is not the same as scheduler-mode's stacked-batch approach.
# Check the first chunk's slurm-c_*.out / results/<name>/loss.csv
# interval_time_s and raise PER_CHUNK on the next scenario/seed if it finished
# well under 29 min.

set -e
SCENARIOS="$1"
SEEDS="$2"
STEPS="$3"
PER_CHUNK="${4:-400}"
AFTER_JOBID="${5:-}"

if [ "$STEPS" -eq 2000 ]; then TAG=""; else TAG="_e${STEPS}"; fi

for scenario in $SCENARIOS; do
  for seed in $SEEDS; do
    name="${scenario}${TAG}_s${seed}"
    config="experiments/configs/${scenario}.yaml"
    echo "### $name : chaining chunks of $PER_CHUNK steps up to $STEPS (pool) ###"

    dep="${AFTER_JOBID:+--dependency=afterok:$AFTER_JOBID}"
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
            scripts/run_chunk_pool.sh "$config" "$name" "$seed" "$STEPS" "$stop" "$resume" \
            | awk '{print $NF}')
      echo "   chunk ->$stop : job $jid ${dep:+(after ${dep##*:})}"
      dep="--dependency=afterok:$jid"
    done

    ejid=$(sbatch $dep --job-name="e_${name}" scripts/eval_chunk.sh "$name" | awk '{print $NF}')
    echo "   eval : job $ejid (after ${dep##*:})"
  done
done
echo "all chains submitted. watch: squeue -u \$USER"
