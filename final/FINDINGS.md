# Where you finish is what you get

**Curriculum pretraining for nanoTabPFN, ML Lab 2026**

We asked whether showing a tabular foundation model easy synthetic datasets before hard
ones helps it pretrain. The answer turned out to be more interesting than yes or no: what
matters is not the difficulty *direction* of the curriculum, but **which regime the
curriculum ends in**.

---

## 1. The question

nanoTabPFN is pretrained on a stream of synthetic tables from the TabICLv2 prior, in random
order. Curriculum learning suggests presenting them easy to hard. Does that train a better
model?

## 2. What went wrong the first time

Our first attempt ramped many prior knobs at once and used a cheap kNN probe to decide which
knobs mattered. Reading the TabICLv2 source showed that most of what we were "tuning" did
nothing:

- `noise_std`, `num_layers` and `hidden_dim` are sampled from a **log-scaled** distribution.
  We only raised `max_mean` and left `min_mean` tiny, so the sampled values barely moved.
- `num_causes` is **overwritten** with `num_features` whenever `is_causal=False`, which we
  never froze, so that knob did nothing at all.
- `mix_probs = (0.7, 0.3)` means about 30% of datasets are tree-based and ignore every MLP
  knob.
- A kNN probe measures difficulty for a distance-based model, not for a transformer.

So our earlier conclusion that "no single knob controls difficulty" was an artefact of
broken knobs, not a finding. We restarted with a design that could not have this problem.

## 3. A design with no confounds

We generate **one fixed pool** of 80,000 datasets from the unchanged prior. Every run reads
**that same pool** and differs **only in the order** it walks through it. Each dataset is
used exactly once, so data and compute are identical by construction. Confirmed: every run
reports the same `pool_size` and the same `cum_flops`.

Difficulty is not a proxy score, it is the measured property we sort by. Ordering axes and
weights are declared in the config, and each run records an `ordering_profile` (Spearman
correlation of training position against every axis), so the ordering we actually got is
documented rather than assumed.

Training follows the published nanoTabPFN recipe: 3 layers, embedding 96, 4 heads, MLP
hidden 192, schedule-free AdamW at lr 0.003892, effective batch 32 via gradient
accumulation, 2500 steps. Validation is a fixed shared set split into easy, medium and hard
feature bands. Evaluation is TabArena ROC-AUC; after the size filters, 16 of the 51 tasks
are scored.

## 4. Seven orderings, one pattern

Mean TabArena ROC-AUC over 3 seeds. Baseline is the same pool in random order.

| ordering | ends training on | score | vs baseline |
|---|---|---|---|
| features, few to many | many features | **0.8107** | **+0.020** |
| in-context examples, few to many | many examples | **0.8069** | **+0.016** |
| baseline (shuffle) | mixed | 0.7911 | - |
| restart sawtooth (x3) | many features, briefly | 0.7895 | -0.002 |
| features, many to few | few features | 0.7623 | -0.029 |
| features + in-context combined | example-poor end | 0.7506 | -0.041 |
| in-context examples, many to few | few examples | 0.7424 | -0.049 |

The ranking is not explained by difficulty direction. It is explained by where each run
*finishes*.

Splitting the 16 tasks into the 10 binary and 6 multiclass ones, the **ranking is stable**:
the two winners beat the baseline on binary, on multiclass and overall, and the losers fall
below it on all three. The **size** of the effect does differ. For the feature curriculum
the gain is +0.010 on the binary subset but +0.035 on multiclass, so the +0.020 overall is
driven more by the multiclass tasks.

## 5. The decisive test

The two axes have **opposite** difficulty directions: more features is harder, but *fewer*
in-context examples is harder. So "end on the hard stuff" makes opposite predictions for
them, which lets us separate the hypotheses.

We recorded two predictions before running:

- reversing the in-context axis (ending on many examples) should score **above** baseline
- reversing the feature axis (ending on few features) should score **below** baseline

Both were confirmed. Reversing an axis flips the sign of the effect:

| axis | forward | reversed | swing |
|---|---|---|---|
| features | +0.020 | -0.029 | 0.049 |
| in-context examples | -0.049 | +0.016 | 0.065 |

Same data, same compute, same number of steps. Only the order reversed.

## 6. Why: the regime gap

| | pretraining pool | TabArena evaluation |
|---|---|---|
| features | 2 - 60 | median 20, q75 37, max 112 |
| in-context examples | **20 - 180** | **673 - 4500** (median 1439) |

Two things follow:

1. **There is a large coverage gap in in-context examples.** The model never sees more than
   180 during pretraining but is evaluated with 673 to 4500. On an axis this badly covered,
   finishing nearer the evaluation regime matters, which is what the reversal showed.
2. **Features do not fit a simple "finish near the evaluation median" rule.** The evaluation
   median is 20 features, so ending at 2 is numerically closer than ending at 60, yet ending
   at 60 wins. The better reading is that feature count is a *capacity* axis where the harder
   regime subsumes the easier one: a model just trained on 60 features handles 9 fine, but
   not the reverse.

So the honest statement is not one rule but two regularities:

- axes where pretraining does not cover evaluation, finish close to evaluation
- axes where harder subsumes easier, finish hard

## 7. Compute efficiency, and its limit

Because the datasets are ordered by feature count, the curriculum spends its early steps on
small cheap tables and only reaches the large expensive ones late. Plotting validation
ROC-AUC against estimated FLOPs instead of steps, the curriculum reaches a given quality at
roughly a third of the compute the baseline needs. Both use the same total compute by the
end; the curriculum just front-loads the cheap work.

Repeating the two runs on one A100 with wall-clock logging reproduced the quality result
(+0.030 TabArena) and showed the timing story clearly. **Total** run time is essentially
unchanged, 910 s against 936 s, because both take the same number of steps over the same
pool. What moves is quality per unit time: the curriculum reaches the baseline's best score
at 395 s where the baseline needs 899 s, so 2.3x sooner at equal quality.

Then we tested whether that converts on a model that is genuinely compute bound.
modded-nanoTabPFN is a speedrun of the same architecture (Muon, bf16, `torch.compile`). On
it the curriculum is **1.69x slower** to the same quality, 119.5 s against 70.6 s. Its
steady-state epochs cost more, 1.52 s against 1.03 s, despite batches being 21.6% narrower,
because at batch size 2 with 1000 rows the feature dimension is not the bottleneck and the
shape variation a ramp introduces costs real time in the compiled path.

The reading: the curriculum's benefit is in **sample efficiency**, better quality per step,
not in compute per step. So "does the curriculum make training faster" now has a measured
answer rather than an assumed one, and on a compute-bound model the answer is no.

Two caveats on that run. It is a single seed per cell. And the curriculum run stopped at
epoch 48 of a 120-epoch ramp, having seen only 2 to 9 features while the baseline saw the
full range from the start, so part of the gap is coverage rather than order. A few-to-many
ramp under a fixed time budget necessarily spends that budget on the narrow end, so it
should be read as "does not pay for itself at this budget", not as "ordering is harmful".

## 8. A second approach: ramping the prior instead of the data

In parallel we tried changing the prior's own knobs during training (`num_layers`,
`hidden_dim`, `noise_std`) instead of ordering a fixed pool. The baseline holds all three at
their hardest setting for the whole run; the curriculum climbs to that ceiling and then
holds. On binary TabArena at 10,000 steps over 3 seeds:

| setup | TabArena AUC | vs baseline | training time |
|---|---|---|---|
| baseline (exact paper recipe) | 0.7719 ± 0.0038 | - | ~77.6 min |
| late ramp | **0.7826 ± 0.0043** | **+0.0107** | ~55.1 min |
| early ramp | 0.7802 ± 0.0042 | +0.0083 | ~55.4 min |

Same direction as the pool result, and cheaper because generating tables from a small SCM is
cheaper than from a large one. Two honest notes. Difficulty here is defined by intuition
(fewer layers and less noise is easier), not measured, which is exactly the weakness the
fixed-pool design was built to avoid. And a learning-rate sweep over 10 sampled rates showed
the curriculum mostly helps when the learning rate is badly tuned low; at a well-tuned rate
it converges faster but finishes in the same place. The single most dramatic single-seed
result there did not survive two more seeds (8/10 wins became 17/30, mean delta +0.046
became +0.009).

## 9. A methodological note worth keeping

"Sort the data by difficulty" is ill-defined when a run consumes only a small fraction of
the pool. On a 256,000-dataset dump where a run reaches its target after 3,648 datasets, a
strict global sort is a **data filter wearing a curriculum's clothes**: those 3,648 would
all have 1 or 2 features and the run would never see the rest of the distribution. Any
curriculum over a large pool has to reason about budget against pool size, or it measures
the wrong thing.

## 10. What follows

- Raise `num_datapoints` in the pool so pretraining covers the evaluation range of in-context
  examples. Prediction: the in-context axis effect shrinks. This is a direct test of section 6.
- More seeds on the two positive comparisons, and an equal-epoch rerun of the compute-bound
  test to separate ordering from coverage.
- If the account holds, the practical advice is simple and slightly counterintuitive: **do
  not design a curriculum around difficulty. Design it around the regime you intend to
  deploy in, and end there.**
