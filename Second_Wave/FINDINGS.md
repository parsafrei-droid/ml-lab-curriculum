# Where you finish is what you get

**Curriculum pretraining for nanoTabPFN — Second Wave**

We asked whether showing a tabular foundation model easy synthetic datasets before hard ones helps
it pretrain. The answer turned out to be more interesting than yes or no: what matters is not the
difficulty *direction* of the curriculum, but **which regime the curriculum ends in**.

---

## 1. The question

nanoTabPFN is pretrained on a stream of synthetic tables drawn from the TabICLv2 prior, in random
order. Curriculum learning suggests presenting them easy to hard. Does that train a better model?

## 2. What went wrong the first time

Our first attempt ramped many prior knobs at once and used a cheap kNN probe to decide which knobs
mattered. Reading the TabICLv2 source showed that most of what we were "tuning" did nothing:

- `noise_std`, `num_layers`, `hidden_dim` are sampled from a **log-scaled** distribution
  (`log_mean ~ U(log min_mean, log max_mean)`). We only raised `max_mean`, leaving `min_mean` tiny,
  so the sampled values barely moved.
- `num_causes` is **overwritten** with `num_features` whenever `is_causal=False`, which we never
  froze — so that knob did nothing at all.
- `mix_probs = (0.7, 0.3)` means ~30% of datasets are tree-based and ignore every MLP knob.
- A kNN probe measures difficulty for a distance-based model, not for a transformer.

So our earlier conclusion "no single knob controls difficulty" was an artefact of broken knobs, not
a finding. We restarted.

## 3. A design with no confounds

We generate **one fixed pool** of 80,000 datasets from the unchanged TabICLv2 prior. Every run reads
**that same pool** and differs **only in the order** it walks through it. Each dataset is used exactly
once, so data and compute are identical by construction — confirmed: every run reports
`cum_flops = 5.562e12` and the same peak memory.

Difficulty is not a proxy score; it is the measured property we sort by. Ordering axes and their
weights are declared in the config, and each run records an `ordering_profile` (Spearman correlation
of training position against every axis) so the ordering we actually got is documented, not assumed.

Training follows the published nanoTabPFN recipe: 3 layers / embedding 96 / 4 heads / MLP hidden 192,
schedule-free AdamW at lr 0.003892, effective batch 32 via gradient accumulation, 2500 steps,
step-based. Validation is a fixed shared set, split into easy/medium/hard feature bands. Evaluation is
TabArena ROC-AUC (one-vs-rest); after the size filters, 16 of the 51 tasks are actually scored.

## 4. Seven orderings, one pattern

Mean TabArena ROC-AUC over 3 seeds. Baseline is the same pool in random order.

| ordering | ends training on | score | vs baseline |
|---|---|---|---|
| features, few to many | many features | **0.8107** | **+0.020** |
| in-context examples, few to many | many examples | **0.8069** | **+0.016** |
| baseline (shuffle) | mixed | 0.7911 | — |
| restart sawtooth (x3) | many features, briefly | 0.7895 | −0.002 |
| features, many to few | few features | 0.7623 | −0.029 |
| features + in-context combined | example-poor end | 0.7506 | −0.041 |
| in-context examples, many to few | few examples | 0.7424 | −0.049 |

The ranking is not explained by difficulty direction. It is explained by where each run *finishes*.
(`figures/tabarena_roc_auc.png` plots this table.)

**Does the result depend on which datasets are scored?** We split the 16 tasks into the 10 binary and
the 6 multiclass ones (`figures/tabarena_binary_vs_all.png`):

| ordering | all (16) | binary (10) | multiclass (6) |
|---|---|---|---|
| features, few to many | 0.811 | 0.799 | 0.830 |
| in-context examples reversed | 0.807 | 0.799 | 0.819 |
| baseline | 0.791 | 0.789 | 0.795 |
| restart | 0.789 | 0.791 | 0.787 |
| features, many to few | 0.762 | 0.768 | 0.752 |
| combined | 0.751 | 0.734 | 0.778 |
| in-context examples, many to few | 0.742 | 0.730 | 0.764 |

The **ranking is stable** across subsets: the two winners beat the baseline on binary, on multiclass,
and overall; the losers fall below it on all three. So the effect is not an artefact of which datasets
are scored. The **size** of the effect does differ, though: for the feature curriculum the gain is
+0.010 on the binary subset (the one comparable to the paper) but +0.035 on the multiclass subset, so
the +0.020 overall is driven more by the multiclass tasks.

## 5. The decisive test

The two axes have **opposite** difficulty directions: more features is harder, but *fewer* in-context
examples is harder. So "end on the hard stuff" makes opposite predictions for them, which lets us
separate the hypotheses.

We recorded two predictions before running:

- reversing the in-context axis (ending on many examples) should score **above** baseline
- reversing the feature axis (ending on few features) should score **below** baseline

Both were confirmed. Reversing an axis flips the sign of the effect:

| axis | forward | reversed | swing |
|---|---|---|---|
| features | +0.020 | −0.029 | 0.049 |
| in-context examples | −0.049 | +0.016 | 0.065 |

Same data, same compute, same number of steps — only the order reversed.

## 6. Why: the regime gap

We measured the regime the model is actually evaluated in and compared it to the regime it is
pretrained in.

| | pretraining pool | TabArena evaluation |
|---|---|---|
| features | 2 – 60 | median 20, q75 37, max 112 |
| in-context examples | **20 – 180** | **673 – 4500** (median 1439) |

Two things follow:

1. **There is a large coverage gap in in-context examples.** The model never sees more than 180
   in-context examples during pretraining but is evaluated with 673–4500. On an axis this badly
   uncovered, finishing nearer the evaluation regime matters, which is exactly what the reversal
   showed.
2. **Features do not fit a simple "finish near the evaluation median" rule** — the evaluation median
   is 20 features, so ending at 2 is numerically closer than ending at 60, yet ending at 60 wins.
   The better reading is that feature count is a *capacity* axis where the harder regime subsumes the
   easier one: a model that has just been trained on 60 features handles 9 features fine, but not the
   reverse.

So the honest statement is not one rule but two regularities:

- axes where pretraining does not cover evaluation → finish close to evaluation
- axes where harder subsumes easier → finish hard

## 7. Compute efficiency

The feature curriculum is not only slightly more accurate — it reaches a given quality using less
compute. Because the datasets are ordered by feature count, the curriculum spends its early steps on
small (few-feature) tables, which are cheap, and only reaches the large expensive tables late.

To see this we plot validation ROC-AUC against **estimated FLOPs** (floating-point operations — a
hardware-independent measure of how much arithmetic the model did), instead of against steps. FLOPs
are recomputed per step from the actual feature counts the model processed, using the model's real
architecture (two attentions per layer: one across rows, one across columns, so cost grows with both
the number of rows and the number of features).

`figures/compute_efficiency.png` (baseline vs feature curriculum) and
`figures/val_auc_vs_flops.png` (all orderings) show the result: the feature curriculum sits up and to
the left — it reaches, for example, ~0.58 validation ROC-AUC at roughly a third of the FLOPs the
random-order baseline needs for the same score. Both use the same 80,000 datasets and the same total
compute by the end; the curriculum just front-loads the cheap work.

**Important scope of this claim.** This is measured on the fixed pool: the same data, ordered cheap-
first vs random. It is a real, measured efficiency result. It is *not* a claim about an "on-the-fly"
setup where a baseline trains on full-size tables throughout — we did not complete that run (on-the-
fly data generation is CPU-bound and timed out on the short queue). Note also that an on-the-fly
feature ramp is a *different* schedule from the pool ordering (a widening `[2, cap]` window that always
includes easy tables and ends on a mix, versus the pool's narrow bands that end on pure hard tables),
so its accuracy cannot be assumed equal to the pool result without running it.

## 8. What we are not claiming

- With 3 seeds the between-seed spread is about 0.01. The **negative** effects (−0.03 to −0.05) are
  outside that. The **positive** effects (+0.020, +0.016) are around twice the spread:
  consistent and directional, but not conclusive. More seeds would settle it.
- Only 16 datasets are scored, which is a small evaluation set and contributes to that spread.
- Everything here is at one model size, one step budget, and one prior. We have not tested whether
  the effect survives scale.
- The compute-efficiency figure is the fixed-pool result; the on-the-fly variant is unrun (section 7).

## 9. What follows

- Raise `num_datapoints` in the pool so pretraining covers the evaluation range of in-context
  examples. Prediction: the in-context axis effect shrinks. This is a direct test of the account in
  section 6.
- More seeds on the two positive comparisons.
- Optionally, run the on-the-fly feature ramp on a long queue slot to confirm the efficiency story in
  the generate-as-you-go setting.
- If the account holds, the practical advice is simple and slightly counterintuitive: **do not design
  a curriculum around difficulty. Design it around the regime you intend to deploy in, and end
  there.**

---

*All code, configs, logs and figures are in this folder. Every number above comes from runs on one
shared fixed pool at identical compute.*
