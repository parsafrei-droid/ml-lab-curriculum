# Curriculum Pretraining for nanoTabPFN — Progress Update

**ML Lab 2026**

---

## 1 — Curriculum learning performance

**Setup**
- Exact paper baseline (Table 1 recipe), 10,000 steps
- Binary-only datasets, evaluated on TabArena
- Tested ramps of increasing `num_layers`, `hidden_dim`, `noise`
- Several ramp shapes tried; **early ramp** and **late ramp** performed best
- TabArena eval file changed to a **hard-coded set of 26 datasets** (fixed task list, no silent skipping)
- Cross-checked in a labmate's (Parsa) evaluation setup — results were similar

### How the configs are created

Both curriculum configs pin architecture, learning rate, and batch size to the
paper's exact published values. The only thing that changes over training is
a **schedule**: a set of checkpoints at which three prior knobs step up —
`num_layers`, `hidden_dim`, `noise_std`. The baseline holds all three at their
hardest setting for the whole run; the two curricula differ only in *when*
that ramp completes.

![How the curriculum schedule is built](experiments/binary_10k_schedule.png)

| | Late ramp | Early ramp |
|---|---|---|
| Ramp completes at | ~68% of training | ~44% of training |
| Time spent fully at ceiling | ~12% of run | ~56% of run |
| Noise steps per layer-block | 3 levels (0.001 → 0.1 → 0.3) | 4 levels, log-spaced (0.001 → 0.01 → 0.1 → 0.3) |

> How "easy" vs. "hard" was defined for this schedule is discussed under
> **General questions** below.

### Results

![Per seed results](experiments/binary_10k_seeds.png)

| Setup | TabArena AUC (mean ± seed spread) | vs. baseline | Training time |
|---|---|---|---|
| **Baseline** (exact paper recipe) | 0.7719 ± 0.0038 | — | ~77.6 min |
| **Late ramp** | **0.7826 ± 0.0043** | **+0.0107** | ~55.1 min |
| **Early ramp** | 0.7802 ± 0.0042 | +0.0083 | ~55.4 min |

> **Result:** Curriculum pretraining slightly improves over the exact
> baseline performance, with lower training time.

---

## 2 — Random learning rate

**Setup**
- Exact paper baseline, **2,500 steps**
- Binary-only datasets, evaluated on TabArena
- Curriculum ramp: **early ramp only**
- 10 random learning rates sampled from the nanoTabPFN paper's own search space
- Each trained twice: one curriculum config, one baseline config

![Curriculum's edge vs lr mistuning](experiments/hp_sweep_lr_trend.png)

![Per-configuration learning curves, 3 seeds](experiments/hp_sweep_curves.png)

> **Result:** for lower learning rates, curriculum pretraining performed
> better so far. For higher learning rates, results are similar.

**On replication —** the single most dramatic single-seed result did not
hold up under two more seeds:

| | Single seed | 3 seeds (30 config/seed pairs) |
|---|---|---|
| Curriculum wins | 8 / 10 | 17 / 30 |
| Mean Δ AUC | +0.046 | +0.009 |
| Standout config | +0.266 (rescue) | ranges +0.266 → −0.248 across seeds |

**Further work:** further improvements can include adding configs that
consider different difficulty measures as well, since the current schedule's
notion of "easy" vs. "hard" is intuition-based (see below).

---

## General questions

> **How was hardness determined?** Based on intuition. No concrete academic
> work exists to measure the difficulty of a prior — it is hard to formalize,
> but intuitively straightforward to define (fewer layers, less noise = easier).

## Research question ideas

1. Curriculum pretraining slightly improves over the baseline
2. Curriculum pretraining recovers a badly tuned learning rate

---

*All configs, results, and figures referenced above are in the repository
(`experiments/`, `results/`).*
