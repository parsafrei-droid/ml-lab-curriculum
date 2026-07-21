#!/bin/bash
# Build a pool across dependency-chained 29-min GPU chunks (scripts/build_pool.sh),
# each appending via --append/--stop-after-count - same idiom as
# scripts/submit_chunked_toy_probe.sh, applied to pool generation instead of training.
#
#   scripts/submit_chunked_pool_build.sh <config> <total_items> [count_per_chunk]

set -e
CONFIG="$1"
TOTAL="$2"
PER_CHUNK="${3:-15000}"

if [ -z "$CONFIG" ] || [ -z "$TOTAL" ]; then
  echo "usage: scripts/submit_chunked_pool_build.sh <config.yaml> <total_items> [count_per_chunk]"
  exit 1
fi

echo "### building $CONFIG : chaining chunks of $PER_CHUNK items up to $TOTAL ###"
dep=""
done=0
first=1
while [ "$done" -lt "$TOTAL" ]; do
  done=$(( done + PER_CHUNK ))
  [ "$done" -gt "$TOTAL" ] && done=$TOTAL
  jid=$(sbatch $dep --job-name="pool_$(basename "$CONFIG" .yaml)" \
        scripts/build_pool.sh "$CONFIG" "$PER_CHUNK" \
        | awk '{print $NF}')
  echo "   chunk (+$PER_CHUNK, cum target $done) : job $jid ${dep:+(after ${dep##*:})}"
  dep="--dependency=afterok:$jid"
  first=0
done
echo "all chunks submitted. watch: squeue -u \$USER"
