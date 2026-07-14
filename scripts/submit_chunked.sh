#!/bin/bash
# Submit a full chunked run for each (scenario x seed): a dependency-chained
# sequence of short jobs (train chunks + a final eval), so a long batch-32 run
# completes on the 30-min short partition instead of waiting for gpu_h100.
#
#   scripts/submit_chunked.sh "<scenarios>" "<seeds>" <steps> [steps_per_chunk]
#
# e.g. the tuned 5k smoke (1 seed):
#   scripts/submit_chunked.sh "baseline_tuned curriculum_combined_tuned" "42" 5000
#
# Result folders are tagged _e<steps> (matching run_all_cluster.sh), e.g.
# baseline_tuned_e5000_s42. Verified numerically equivalent to a single-shot run
# (see the resume path in scripts/run.py).

set -e
SCENARIOS="$1"
SEEDS="$2"
STEPS="$3"
PER_CHUNK="${4:-1200}"        # ~1200 steps ~= 19 min at batch-32 (safe under 30-min cap)

if [ "$STEPS" -eq 2000 ]; then TAG=""; else TAG="_e${STEPS}"; fi

for scenario in $SCENARIOS; do
  for seed in $SEEDS; do
    name="${scenario}${TAG}_s${seed}"
    config="experiments/configs/${scenario}.yaml"
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
            scripts/run_chunk.sh "$config" "$name" "$seed" "$STEPS" "$stop" "$resume" \
            | awk '{print $NF}')
      echo "   chunk ->$stop : job $jid ${dep:+(after ${dep##*:})}"
      dep="--dependency=afterok:$jid"
    done

    # final eval, after the last training chunk
    ejid=$(sbatch $dep --job-name="e_${name}" scripts/eval_chunk.sh "$name" | awk '{print $NF}')
    echo "   eval : job $ejid (after ${dep##*:})"
  done
done
echo "all chains submitted. watch: squeue -u \$USER"
