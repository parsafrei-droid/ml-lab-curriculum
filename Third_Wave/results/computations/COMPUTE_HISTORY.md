## 1. Summary

| | |
|---|---|
| Total runs | 305 (299 Slurm jobs + 6 Kaggle notebook runs) |
| Total GPU run time | 56.67 h |
| Total time waiting in the queue | 268.73 h |
| First run | 2026-06-25 |
| Last run | 2026-07-30 |
| Calendar span | 36 days (17 days with activity) |
| Peak jobs running at the same time | 15 |

Queueing only applies to cluster jobs, since Kaggle notebooks start on demand. Compared
like for like, we waited 268.73 h for 48.71 h of Slurm compute, so waiting took about
5.5x longer than computing.

## 2. GPUs used

| GPU | Where | Runs | Run time | Share | Time queued |
|---|---|---:|---:|---:|---:|
| NVIDIA H100 94 GB | BwUniCluster 3.0 (`uc3n*`) | 230 | 42.62 h | 75.2 % | 267.93 h |
| NVIDIA Tesla T4 | Kaggle notebooks | 6 | 7.96 h | 14.0 % | none |
| NVIDIA A100 80 GB | BwUniCluster 2.0 (`uc2n*`) | 65 | 6.07 h | 10.7 % | 0.79 h |
| AMD MI300 | BwUniCluster 3.0 (`uc3n083`) | 3 | 0.01 h | < 0.1 % | 0.01 h |
| CPU only | BwUniCluster 2.0 | 1 | 0.00 h | 0 % | 0.00 h |
| **Total** | | **305** | **56.67 h** | 100 % | **268.73 h** |

The H100 did most of the work: it is the newest GPU generation on the cluster and has
the most nodes available to us (12 in `gpu_h100`, 7 in `gpu_h100_short`). The A100 was
used early for setup and the first sweeps, then again for the Third Wave, where we
wanted a fixed uncontended GPU so wall-clock comparisons between two models were fair.
The MI300 was tried in 3 short probe jobs (~30 s total); our stack is CUDA-only, so we
did not pursue it.

### By partition

| Partition | Max wall-clock | Jobs | Run time | Queue time | Mean wait |
|---|---|---:|---:|---:|---:|
| `gpu_h100_short` | 30 min | 139 | 20.90 h | 40.53 h | 19.5 min |
| `gpu_h100` | 3 days | 52 | 17.23 h | 222.65 h | 477 min |
| `gpu_a100_short` | 30 min | 45 | 4.79 h | 0.21 h | 0.3 min |
| `dev_gpu_h100` | 30 min | 36 | 4.49 h | 4.75 h | 9.8 min |
| `dev_gpu_a100_il` | 30 min | 16 | 1.28 h | 0.58 h | 2.3 min |
| `gpu_mi300` | 3 days | 3 | 0.01 h | 0.01 h | 0.4 min |
| other (`cpu`, `*_il`) | — | 8 | 0.00 h | 0.00 h | 0.0 min |

Same hardware, very different waits: 19.5 min on `gpu_h100_short` against 477 min on
`gpu_h100`. This is the most useful thing we learned about the cluster.

## 3. Compute per phase

| Phase | Dates | Jobs | Run time | Queue time |
|---|---|---:|---:|---:|
| Setup & GPU probes | Jun 25 – Jun 27 | 5 | 0.10 h | 0.03 h |
| First Wave (TabICL / TFM-Playground) | Jul 01 – Jul 06 | 170 | 32.39 h | 41.60 h |
| Second Wave (nanoTabPFN feature curriculum) | Jul 18 – Jul 28 | 109 | 14.94 h | 226.53 h |
| ↳ Second Wave on Kaggle | Jul 25 – Jul 27 | 6 | 7.96 h | none |
| Third Wave (modded-nanoTabPFN 2×2) | Jul 30 | 15 | 1.28 h | 0.58 h |
| **Total** | Jun 25 – Jul 30 | **305** | **56.67 h** | **268.73 h** |

The First Wave used the most compute: many short jobs across a wide sweep of curriculum
shapes and seeds. The Second Wave used less compute but waited five times longer,
because its runs no longer fitted in the 30-minute partitions.

## 4. Why the last runs moved to Kaggle

The final Second Wave experiment compared an on-the-fly feature ramp against a full
baseline (3 seeds × 2 configs). These runs generate data during training instead of
using a pre-built pool, so one run needed more than 30 minutes. That locked us out of
every `*_short` partition and forced us onto `gpu_h100`.

| Job | Partition | Waited | Ran |
|---|---|---:|---:|
| `otf_cur_s2` (6065894) | `gpu_h100` | 77.07 h | 0.72 h |
| `otf_cur_s42` (6065892) | `gpu_h100` | 68.75 h | 0.78 h |
| `otf_cur_s1` (6065893) | `gpu_h100` | 68.75 h | 0.77 h |

Roughly three days of waiting per job for 45 minutes of compute. Before these ran we
had already submitted and cancelled the same experiment three times (job groups
`6036397–6036402`, `6036477–6036480`, `6036590–6036595`) trying to fit it into short and
dev partitions. We then moved it to Kaggle, which gives a free Tesla T4 session that
starts immediately.

| | BwUniCluster H100 | Kaggle T4 |
|---|---|---|
| Time to start | up to 77 h | seconds |
| One curriculum run | 45 min (3 jobs) | 1.20 h (3 runs) |
| One baseline run | ~55 min (implied) | 1.45 h (3 runs) |
| All 6 runs | ~5 h compute, spread over days of queueing | ~10 h, same day |

Comparing the same configuration on both, the T4 was 1.59x slower (1.20 h against
45 min). The six runs used 7.96 h of measured training time. Including setup, the
periodic validation passes and the TabArena evaluation over 38 datasets, the notebook
sessions took about 10 h in total. We accepted a 1.6x slower GPU to avoid a three-day
queue.

The 7.96 h is the sum of `cum_time_s` from the last row of each of the 6 Kaggle logs.
That counter only accumulates training steps, so it excludes setup, evaluation and idle
time. The ~10 h is the full session wall-clock, which Kaggle does not log to file.

An earlier attempt used Google Colab (commits `ad13687` … `7e00fa4`), but dependency
conflicts in the TabICL pretraining stack made it unreliable and the notebook was
dropped.

## 5. Running on BwUniCluster

### Environment

The project virtualenv segfaults on any real import unless the cluster modules are
loaded first, so every script starts with:

```bash
source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source /path/to/ml-lab-curriculum/.venv/bin/activate
```

Without this, `.venv/bin/python -V` still prints a version, but `import torch` dies with
exit code 139 and no error message.

### Job script

A Slurm job is a shell script with `#SBATCH` directives on top. This is
`Second_Wave/train_short.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=sw_train              # name shown in squeue / sacct
#SBATCH --partition=gpu_h100_short       # which queue, so which hardware + time limit
#SBATCH --gres=gpu:1                     # 1 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                # 8 CPU cores for the data loader
#SBATCH --mem=32G
#SBATCH --time=00:29:00                  # must be under the partition limit
#SBATCH --output=slurm-%x-%j.out         # %x = job name, %j = job id
set -e

source /usr/share/lmod/lmod/init/bash
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
source ../.venv/bin/activate

python train.py --config "configs/$1.yaml" --name "$2" --seed "$3"
```

The script takes arguments (`$1` config, `$2` run name, `$3` seed), so one script covers
a whole family of jobs.

### Commands

```bash
sbatch train_short.sh curriculum_features curriculum_features_s42 42   # submit
squeue -u $USER                                                        # pending / running
scontrol show job <jobid>                                              # why still pending
sacct -X -j <jobid> --format=JobID,JobName,Elapsed,State               # what happened
scancel <jobid>                                                        # cancel
```

`sinfo` is permission-denied for our account; use `scontrol show partition <name>` to
check a partition's limits.

### Two mistakes that cost us jobs

1. `cd "$(dirname "$0")"` does not work. Slurm copies the script to
   `/var/spool/slurmd/job<ID>/slurm_script`, so `$0` does not point at the repository.
   The `cd` fails and the job dies after about 9 seconds. Use an absolute path.
2. Exceeding the partition time limit kills the job rather than extending it.
   `dev_gpu_a100_il` caps at 30 minutes, so a script that loops over two configs and
   evaluates will hit the wall and lose everything. 7 jobs died this way, destroying
   3.43 h of compute, which is 7 % of all the compute we ran on the cluster.

## 6. Why we split jobs and ran them in parallel

We started with one big job looping over every config and seed
(`Second_Wave/submit.sh`: `gpu_h100`, `--time=12:00:00`, 2 configs × 3 seeds) and
replaced it with many small jobs. 236 of our 299 jobs (78.9 %) went to a ≤ 30-minute
partition. We did this in two ways:

| | Fan-out | Chaining |
|---|---|---|
| What | independent runs submitted as separate jobs | one long run cut into ≤ 30-min chunks that resume from a checkpoint |
| How | plain `sbatch` per run | `sbatch --dependency=afterok:<previous job>` |
| Order | simultaneously, on different nodes | strictly one after another |
| Jobs | 181 | 63 (in 15 chains) |

The 15 chains averaged 44 minutes of compute each (18 to 83 min), so every one exceeded
the 30-minute cap. Three reasons for working this way:

1. **Short partitions start much sooner.** Slurm backfills small short jobs into gaps
   between large reservations, so asking for less gets scheduled earlier: 19.5 min mean
   wait on `gpu_h100_short` against 477 min on `gpu_h100`, for the same GPUs.
2. **Real parallelism.** Different seeds and curriculum shapes share nothing, so Slurm
   can place them on different nodes at once. Our peak was 15 jobs running
   simultaneously (2026-07-01, the 5-curricula × 3-seeds sweep) and our largest single
   submission was 60 jobs (2026-07-05).
3. **Failure isolation.** In one big job, a crash in seed 3 destroys the finished work
   of seeds 1 and 2, and a timeout destroys everything. With one job per run a failure
   costs one run. Since 26 % of our jobs did not finish cleanly, this mattered often.

The cost is bookkeeping: 299 jobs and 299 log files instead of a handful, which is why
this document is generated from `sacct`.

## 7. Job outcomes

| Outcome | Jobs | Share | Run time | Meaning |
|---|---:|---:|---:|---|
| `COMPLETED` | 222 | 74 % | 44.11 h | Finished cleanly |
| `CANCELLED` | 61 | 20 % | 1.02 h | We cancelled it (wrong config, re-scoped, resubmitted elsewhere) |
| `FAILED` | 9 | 3 % | 0.15 h | Crashed (bad path, OOM, import error) |
| `TIMEOUT` | 7 | 2 % | 3.43 h | Hit the partition time limit |
| **Total** | **299** | 100 % | **48.71 h** | |

26 % of jobs produced no result, which is normal for experimental work. The
cancellations were cheap (1.02 h total): 55 of the 61 never started, and 16 of them were
the on-the-fly experiment being pulled back from queues it could not fit into. The
timeouts were expensive (3.43 h, averaging 29 minutes each, so they ran to the wall and
lost everything). All 7 are the same experiment: six `tr__bas` / `tr__cur` jobs at
00:29:29 against the 30-minute `gpu_h100_short` limit, and one `otf__bas_s42` at
00:29:03 on `dev_gpu_h100`.

## 8. Reproducing the numbers

```bash
# every job this project submitted
sacct -X -S 2026-06-01 -E now -P \
  --format=JobID,JobName%60,Partition,State,Elapsed,ElapsedRaw,Start,End,Submit,NodeList

# total run time in hours
sacct -X -S 2026-06-01 -E now -P --format=ElapsedRaw \
  | awk 'NR>1{s+=$1} END{printf "%.2f h\n", s/3600}'

# Kaggle run time: last cum_time_s of each notebook log
python3 -c "
import csv,glob
print(sum(float(list(csv.DictReader(open(f)))[-1]['cum_time_s'])
          for f in glob.glob('Second_Wave/results/kaggle_run/*log.csv'))/3600, 'h')"
```

`analyze.py` writes `compute_summary.json`; `plot_compute.py` draws the figure from it.

Definitions:

- **Run time** is job wall-clock (`Elapsed`), from start to end.
- **Queue time** is `Start − Submit`. Jobs cancelled before starting have none and are
  excluded from queue averages (244 of 299 jobs have a start time).
- Kaggle runs have no queue time.
