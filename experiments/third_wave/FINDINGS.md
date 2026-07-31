# The ordering effect does not survive a compute-bound model

**Third Wave — feature curriculum vs random order, on one A100**

Second Wave found that ordering pretraining data by feature count (few → many) beats random order
on our small nanoTabPFN. The obvious objection was that our model is too small to be compute-bound,
so a FLOP saving would not show up in wall-clock. This wave tested the ordering on
[modded-nanoTabPFN](https://github.com/borawhocodess/modded-nanotabpfn), a speedrun that *is*
compute-bound (Muon, bf16, `torch.compile`), to see whether the effect converts into real time.

It does not. On the speedrun model the curriculum is **slower and worse** at equal time.

---

## The 2x2

All four runs on the same **NVIDIA A100 80GB PCIe** (node `uc2n902`); runs 3 and 4 were on separate
physical GPUs of that node (bus `19:00.0` and `1B:00.0`), so neither contended for the other.

| # | model | order | metric | result | time |
| - | - | - | - | - | - |
| 1 | ours (80k pool) | random | synthetic val ROC-AUC | 0.5975 final | 935.7s |
| 2 | ours (80k pool) | feature curriculum | synthetic val ROC-AUC | 0.5989 final / **0.6029 best @721s** | 909.7s |
| 3 | modded (own dump) | dump order | TabArena ROC-AUC | **0.8066 best @e57** | 119.0s train |
| 4 | modded (own dump) | feature curriculum | TabArena ROC-AUC | 0.7823 best @e48 | 119.5s train |

The two models are on **separate y-axes** in `results/third_wave.png`: run 1/2 report a synthetic
validation metric (~0.59), run 3/4 report TabArena average ROC-AUC (~0.80). They are not comparable
quantities. Compare within a model, on time.

## 1 vs 2 — ours: the curriculum wins, but not on total wall-clock

Total wall-clock is essentially unchanged (909.7s vs 935.7s, 2.8% faster) and `cum_flops` is
*identical* (2.767e14) because both runs take the same number of steps over the same pool. So there
is no FLOP saving to convert at fixed step count — as expected for a model whose time is dominated
by fixed overheads.

What does move is quality per unit time. The curriculum reaches **0.6029 at 721s**; the baseline
never gets there, peaking at 0.5980 at 899s. On held-out TabArena (16 of 51 tasks, the subset with
≤5000 samples — same subset for both):

| | all (n=16) | binary (n=10) | multiclass (n=6) |
| - | - | - | - |
| run 1 baseline | 0.7686 | 0.7719 | 0.7630 |
| run 2 curriculum | **0.7983** | **0.8015** | **0.7929** |

+0.030 overall, consistent across both subsets. This reproduces Second Wave on A100.

## 3 vs 4 — modded: the curriculum loses on time

Neither run reached the 0.8068 target inside modded's own `max_train_mins = 2` budget; the baseline
came within 0.0002 (0.80664 at epoch 57, matching the reference's 57-epoch record almost exactly).
So "time-to-target" is undefined for both, and the honest comparison is **iso-quality**: the highest
AUC *both* runs reached is 0.7823.

| | training time to 0.7823 |
| - | - |
| run 3 baseline | **70.6s** |
| run 4 curriculum | 119.5s |

The curriculum is **1.69x slower** to the same quality. Two mechanisms, both working against it:

1. **Per-epoch cost is higher, not lower.** Steady-state epoch time (median, epoch ≥3) is **1.52s**
   for the curriculum vs **1.03s** for the baseline — ~48% *more* time per epoch despite batches
   being 21.6% narrower. The one-off `torch.compile` is not the culprit: it is 46.7s for the
   curriculum against 50.4s for the baseline, i.e. slightly cheaper. So narrow batches simply do not
   make this model faster — at `batch_size=2` with 1000 rows the feature dimension is not the
   bottleneck, while the shape churn a ramp introduces costs real time in the compiled path.
2. **Equal *time* is unequal *epochs*.** The 2-minute training cap is a time budget, so the
   curriculum's costlier epochs buy fewer of them — it stopped at epoch 48 where the baseline
   reached 69.

## The honest caveat about run 4

A feature curriculum under a fixed time budget necessarily spends that budget on the narrow end of
the distribution. Run 4 stopped at epoch 48 of a 120-epoch ramp, so it had only seen `n_features`
2–9 (mean 5.6), while the baseline saw the full 1–20 (mean 11.0) from the first epoch. Part of the
0.024 AUC gap is therefore **coverage**, not ordering.

This is not a fixable confound — it is what a few→many curriculum *is* when the run is short. But it
does mean run 4 should not be read as "curriculum ordering is harmful"; it should be read as
"under a fixed time budget on this model, a few→many ramp does not pay for itself."

We also note what we did **not** measure: whether the curriculum would win given enough epochs to
complete its ramp. Both runs were cut off by the training-time cap, and the curves in
`results/third_wave.png` were still rising for run 4 when it stopped.

## What had to change to run this at all (and why it is not a confound)

Two deviations from the original plan, both documented in `README.md`:

- **A strict `argsort` of the dump is not an ordering change.** A run consumes `steps*batch_size` = 64
  datasets per epoch and reaches the target near epoch 57 — 3,648 of 256,000 datasets. Sorted, those
  3,648 are *all* `n_features` 1–2, and the first 20-feature dataset would not appear until epoch
  3,887 of 4,000. Run 4 would have trained on a truncated distribution and never converged. Run 4
  therefore sorts within a **budget-sized subset** sampled across the full width distribution, so it
  sees the same data as run 3 in a few→many order. Mean per-batch feature width: 11.0 (curriculum)
  vs 14.0 (baseline) — 21.6% narrower.
- **`torch._dynamo.config.assume_static_by_default = False`.** dynamo compiles the first shape it
  sees as static and only promotes a dim to dynamic after seeing it change. The dump's default order
  varies the feature width immediately; the curriculum's early batches share one width, so the
  feature dim got baked in as static and a stride guard failed once the width grew. The flag is set
  for **both** runs. Verified non-perturbing: run 3's epoch-1 AUC is **0.46138** against upstream's
  **0.46134**.

## Reproducing

`modded-nanotabpfn/` is an upstream clone (gitignored, and its 12GB dump lives in its own
gitignored `workdir/`), so our changes to it are kept here as `modded_feature_curriculum.patch`:

```bash
git clone https://github.com/borawhocodess/modded-nanotabpfn experiments/third_wave/modded-nanotabpfn
cd experiments/third_wave/modded-nanotabpfn && git apply ../modded_feature_curriculum.patch
mkdir -p workdir/dumps && wget -c -O workdir/dumps/dump-d256000b1r1000c20-8.h5 \
  https://salihboraozturk.com/other/ufr/dump-d256000b1r1000c20-8.h5
```

```bash
# Phase A (runs 1 & 2), one job per config - two trainings do not fit one 30-min job
SEED=42 sbatch run_ours_a100_one.sh baseline
SEED=42 sbatch run_ours_a100_one.sh curriculum_features
sbatch eval_ours_a100.sh baseline_a100_s42
sbatch eval_ours_a100.sh curriculum_features_a100_s42

# Phase B (runs 3 & 4) - warm ~/.cache/openml on a login node FIRST; the eval hits all
# 38 TabArena tasks after every epoch and an uncached run floods OpenML
sbatch run_modded_a100.sh baseline
sbatch run_modded_a100.sh curriculum

python collect_results.py && python plot_results.py
```
