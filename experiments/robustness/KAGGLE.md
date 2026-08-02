# Running the robustness tests on Kaggle

Fallback for when BwUniCluster is unavailable. Same pipeline, same configs, same
seeds as `run_tests.sh` — only the scheduler changes.

## Notebook settings

* **Accelerator: GPU P100** (single GPU; the code uses one device). T4 x2 also works,
  only the first GPU is used.
* **Internet: ON** — required. `evaluate.py` downloads 51 TabArena tasks from OpenML,
  and setup clones the two upstream repos.
* **Persistence: Files only** — lets a killed session resume instead of rebuilding pools.

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

## Time budget — read this before starting

Pool building dominates, and it is worse than "slow". Measured on a local CPU by timing a
200-table build and extrapolating: **~235 ms/table, so ~5 h for one 80k pool.** Kaggle's
4-core CPU is in the same class, possibly slower. Three pools are needed (`main_p1`,
`main_p2`, `noise`), so pool building alone is **~15 h of CPU**, before any training.

| stage | measured / estimated cost | count |
| --- | --- | --- |
| pool build (80k tables, CPU-bound) | **~5 h each** | 3 |
| training run (2500 steps x 32) | ~15-25 min on P100 | 5 |
| evaluation (51 OpenML tasks) | ~10-20 min | 5 |

This does **not** fit in one 12 h session, and `--test all` will not finish. It also runs
into Kaggle's ~30 h/week GPU quota, most of which would be spent with the GPU idle while
the CPU generates tables.

**Use one session per stage:**

| session | cells | rough cost |
| --- | --- | --- |
| 1 | setup + `pool_seed --arg 1` | ~6 h (5 h pool + 2 runs) |
| 2 | setup + `pool_seed --arg 2` | ~6 h |
| 3 | setup + all three `noise` cells | ~6 h (1 shared pool + 3 runs) |

The pool build is pure CPU, so build it in a **CPU-only** session (Accelerator: None) and burn
no GPU quota:

```python
!python experiments/robustness/kaggle_run.py --test build_pools --arg 1
# or, to build all three back to back (long — needs multiple sessions):
!python experiments/robustness/kaggle_run.py --test build_pools
```

Then start a GPU session with persistence on; the runner detects the existing pool and skips
straight to training.

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
