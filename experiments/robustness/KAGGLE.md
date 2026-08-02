# Running the robustness tests on Kaggle

Fallback for when BwUniCluster is unavailable. Same pipeline, same configs, same
seeds as `run_tests.sh` — only the scheduler changes.

## Notebook settings

* **Accelerator: GPU P100** (single GPU; the code uses one device). T4 x2 also works,
  only the first GPU is used.
* **Internet: ON** — required. `evaluate.py` downloads 51 TabArena tasks from OpenML,
  and setup clones the two upstream repos.
* **Persistence: Files only** — lets a killed session resume instead of rebuilding pools.

## Run it detached, not interactively

An interactive session dies when you disconnect, so closing the tab or sleeping the laptop
kills the run. Put the cells in the notebook, then use **Save Version → Save & Run All
(Commit)**. That executes the whole notebook on Kaggle's servers with no browser attached;
you can close everything and collect the result from the *Versions* tab later.

A committed run starts in a **fresh container** and inherits nothing from your interactive
session, so the clone + setup cell must be in the notebook rather than something you ran by
hand first. Put one test in one committed notebook and let it build its own pool — that is
why each session below repeats the setup cell.

## Cell 1 — clone + setup (~5 min, once per session)

```python
!git clone --branch second-wave --single-branch \
    https://github.com/parsafrei-droid/ml-lab-curriculum.git /kaggle/working/ml-lab-curriculum
%cd /kaggle/working/ml-lab-curriculum
!python experiments/robustness/kaggle_setup.py
```

Wait for `SETUP OK`. If it prints a CUDA warning, fix the accelerator before going on —
a CPU run will not finish in a session.

## Cell 2 — Test A, pool seed 1

```python
!python experiments/robustness/kaggle_run.py --test pool_seed --arg 1
```

## Cell 3 — Test A, pool seed 2

```python
!python experiments/robustness/kaggle_run.py --test pool_seed --arg 2
```

## Cells 4-6 — Test B

```python
!python experiments/robustness/kaggle_run.py --test noise --arg noise_baseline
!python experiments/robustness/kaggle_run.py --test noise --arg curriculum_noise
!python experiments/robustness/kaggle_run.py --test noise --arg curriculum_noise_reverse
```

The three Test B runs share one dump, so the first cell pays the build cost and the
other two reuse it. That shared dump is the point of the design — same data for all
three, only the order differs.

## Cell 7 — summary

```python
!python experiments/robustness/kaggle_run.py --test summarise
```

Check `ordering_profile` before trusting any score:

| run | expected `noise` / `features` profile |
| --- | --- |
| `baseline_pool{1,2}_s42` | features ~ 0.0 (shuffle) |
| `curriculum_features_pool{1,2}_s42` | features ~ +1.0 |
| `noise_baseline_s42` | noise ~ 0.0 |
| `curriculum_noise_s42` | noise ~ +1.0 |
| `curriculum_noise_reverse_s42` | noise ~ -1.0 |

A profile that is not near its expected value means the ordering did not apply and the
run should be discarded, not reported.

## Time budget

Measured locally at steady state (1000-table sample, warmup and import cost excluded):
**77 ms/table, so ~1.7 h per 80k pool.** Kaggle's 4-core CPU is in the same class.

| stage | measured / estimated cost | count |
| --- | --- | --- |
| pool build (80k tables, CPU-bound) | ~1.7 h each | 3 |
| training run (2500 steps x 32) | ~15-25 min on P100 | 5 |
| evaluation (51 OpenML tasks) | ~10-20 min | 5 |

Each test fits comfortably in one 12 h session:

| session | command | rough cost |
| --- | --- | --- |
| A1 | `--test pool_seed --arg 1` | ~2.5 h (1.7 h pool + 2 train/eval) |
| A2 | `--test pool_seed --arg 2` | ~2.5 h |
| B | the three `noise` cells | ~3 h (1 shared pool + 3 train/eval) |

Total ~8 h. `--test all` would be ~8 h in one go, which fits the 12 h limit, but keep the
tests in separate committed runs so a failure in one does not cost you the others.

If you want the ~1.7 h build off your GPU quota, run it in a **CPU-only** session first
(Accelerator: None):

```python
!python experiments/robustness/kaggle_run.py --test build_pools --arg 1
```

Worth doing if quota is tight, but at 1.7 h it is not the emergency the first draft of this
file claimed. Running build+train together in one GPU session is simpler and costs ~8 h of a
~30 h/week quota.

Every stage is resumable, so a killed session loses at most the stage in flight: re-run the
same cell and it skips whatever already finished. Keep **Persistence: Files only** on, or a
dead session means rebuilding a 5 h pool.

### Why the build is not parallelised

Tempting, but wrong here. `pool.py` draws tables one at a time from a single seeded RNG
stream (`batch_size=1`), so the dump is a deterministic function of that draw order. Splitting
generation across processes would produce a *different* dump for the same `--seed`, which
destroys the one thing Test A is measuring. The cost is inherent to the design.

## Getting results out

`experiments/robustness/results/<run>/` holds `meta.json`, `tabarena_scores.json`,
`log.csv`, `config.yaml`. Download `/kaggle/working/ml-lab-curriculum/experiments/robustness/results`
from the notebook's Output tab, drop it into a local clone, and commit:

```bash
git add experiments/robustness/results
git commit -m "Robustness test results (Kaggle)"
git push
```

Checkpoints are deliberately not copied — they are large and gitignored.

## Fidelity notes

Identical to the cluster path: pinned dep commits (TFM-Playground `98e33be`, tabicl
`8f665ed`), pool geometry (80k, 2-60 features, 10 classes, 200 datapoints), `--seed 42`,
and all config values. `kaggle_run.py` shells out to the same `pool.py` / `noise_pool.py` /
`train.py` / `evaluate.py` with the same arguments; the pool-path rewrite reproduces the
sbatch `sed` line exactly.

The one genuine difference is **GPU model**: the spec assumes H100, Kaggle gives P100/T4.
That changes wall-clock and `peak_gpu_gb` in `meta.json`, and floating-point
non-associativity means scores will not be bit-identical to an H100 run. It does not change
the comparison being made — every run within a test uses the same GPU, the same dump and the
same seed, so the baseline-vs-curriculum *gap* is measured under identical conditions. Do not
mix a Kaggle number and a cluster number in the same comparison; report the gap, and note the
hardware.
