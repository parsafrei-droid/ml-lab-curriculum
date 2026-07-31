#!/bin/bash
# Submit one eval_pending_checkpoints.sh job per checkpoint, CHAINED
# sequentially (--dependency=afterany) rather than in parallel. All 6 jobs
# hit the same shared OpenML task-split cache; running them in parallel last
# time corrupted 5/6 jobs' reads of that cache mid-write from another job
# (arff.BadDataFormat/BadLayout). One at a time avoids that entirely - each
# eval only takes ~2 min anyway.
#
#   scripts/submit_eval_pending.sh
#   scripts/submit_eval_pending.sh "name1 name2 ..."   # custom list

set -e
NAMES="${1:-baseline_e10000_s42 baseline_e10000_s43 baseline_e10000_s44 curriculum_combined_e10000_s42 curriculum_combined_e10000_s43 curriculum_combined_e10000_s44}"

dep=""
for name in $NAMES; do
  jid=$(sbatch $dep --job-name="eval_${name}" scripts/eval_pending_checkpoints.sh "$name" | awk '{print $NF}')
  echo "submitted eval for $name : job $jid ${dep:+(after ${dep##*:})}"
  dep="--dependency=afterany:$jid"
done
echo "watch: squeue -u \$USER"
