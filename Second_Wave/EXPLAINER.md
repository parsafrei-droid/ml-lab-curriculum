# Explainer — the project from the ground up

A plain-language walkthrough of what we built, how it works, and how to read every figure.

## 1. The model and how it learns

Our model is nanoTabPFN, a small tabular foundation model. It does not learn the way ordinary models
do. An ordinary model is trained on your dataset with gradient descent. This one is not.

nanoTabPFN is pretrained once, on millions of synthetic tables. After that, to use it on a new
dataset you hand it the labelled training rows `(X_train, y_train)` **and** the unlabelled test rows
`X_test` all together as a single input, and it predicts the test labels in one forward pass — with no
gradient steps on your data at all.

The labelled rows you put in the input are called **in-context examples**. During pretraining the
model learned *how to learn from a handful of in-context examples*. So more in-context examples means
more information, which usually makes the task easier; fewer examples makes it harder. This "learning
at inference time, from examples in the input" is the whole idea behind TabPFN-style models.

## 2. What "curriculum" means here

We do not change the model. We ask whether the **order** in which the synthetic tables are shown
during pretraining matters. Curriculum learning says: show easy tables before hard ones.

To test this without confounds we use one fixed pool of tables, and change only the order:

- baseline: the tables in **random** order
- curriculum: the same tables **sorted** by a difficulty axis (e.g. few features to many features)

Because it is the same tables in a different order, the data and the total compute are identical; the
only thing that changes is the sequence.

## 3. The pool (the "dump")

We generate **80,000** synthetic tables once and save them to a file. Each table is drawn from the
TabICLv2 prior with:

- number of features: sampled uniformly in **[2, 60]** (lowest 2, highest 60, mean ~31)
- number of rows: fixed at **200**
- number of classes: up to **10** (randomly 2–10)

The tables are stored as a **flat list in generation order** — they are *not* sorted in the file.
The ordering (shuffle, or sorted by an axis) is applied at training time, when the training loop reads
the pool. That is what guarantees "same data, only the order differs".

Why exactly 80,000? Because training runs for **2500 steps** with an effective **batch of 32**
(via gradient accumulation), so the model sees `2500 × 32 = 80,000` tables over the whole run. Sizing
the pool to exactly that means the model sees each table **once**, in a single easy-to-hard sweep with
no repeats — which is also the closest thing to an "on-the-fly" stream of fresh tables. A smaller pool
would be cycled through several times, repeating the ramp.

Each training run takes about **12 minutes** on an H100 GPU (2500 steps). Building the pool is a
separate one-time cost.

## 4. How we tested

1. Build one fixed pool of 80,000 tables.
2. Train the model under several **orderings** of that pool: random (baseline), features low→high,
   features high→low, in-context examples high→low and low→high, a combined ordering, and a
   sawtooth "restart" ordering.
3. Run each ordering with **3 random seeds** to check the effect is not luck.
4. Score during training on a fixed validation set, and after training on the real TabArena benchmark.
5. Every run reports its `pool_size` and estimated `cum_flops`; these are **identical across all runs**
   (80,000 and the same FLOP total), which is the proof that only the order differed.
6. Decisive test: **reverse** an axis and check whether the effect flips sign. It did, for both axes,
   matching predictions we recorded before running — so we have a mechanism, not just a correlation.

## 5. How we evaluate

There are two evaluation stages.

**During training — a fixed synthetic validation set.** Every 100 steps we score the model on the
same 256 synthetic tables (identical for every run, split into easy/medium/hard feature bands). This
gives the convergence curve. It is an internal signal only.

**After training — TabArena, real data.** We run the trained checkpoint on 16 real OpenML datasets.
For each, the model is given the training rows in-context and predicts the test rows. This is the
ground truth, because it is real data with no synthetic randomness.

We report three metrics. They measure different things, so we use all three and lead with ROC-AUC.

- **ROC-AUC** (Receiver Operating Characteristic — Area Under the Curve). It measures **ranking**: the
  probability that a randomly chosen positive example is given a higher predicted score than a
  randomly chosen negative one. 0.5 is random, 1.0 is perfect. It ignores the decision threshold and
  is fairly robust to class imbalance, but it only cares about the *order* of the scores, not whether
  the probabilities themselves are well-calibrated. For multiclass we average one-vs-rest. This is the
  primary metric because the paper and TabArena use it.
- **Log-loss** (cross-entropy). For each row it takes `-log(probability the model assigned to the true
  class)` and averages. It punishes **confident wrong** predictions very hard, and rewards
  well-calibrated probabilities. Lower is better, 0 is perfect, and it is unbounded above. It catches
  something ROC-AUC cannot: a model can rank correctly (good AUC) but still be badly calibrated (bad
  log-loss).
- **Balanced accuracy.** Take the fraction correct within each class (per-class recall) and average
  over classes. This uses the model's hard prediction (the top class), so it depends on the decision
  threshold, and averaging over classes makes it robust to imbalance. 0.5 is random for binary, 1.0
  is perfect.

Because some TabArena datasets are binary and some are multiclass, we report each metric split three
ways: over **all** datasets, over the **binary** subset (comparable to the paper), and over the
**multiclass** subset — so we can see whether a result depends on which datasets are scored.

## 6. What a FLOP is

A FLOP is one floating-point operation (an add or a multiply) — a unit of **how much arithmetic** the
model did. We use it instead of wall-clock time because time depends on the hardware (an H100 differs
from a laptop), whereas FLOPs do not, so it is a fair, hardware-independent measure of compute.

A bigger table costs more FLOPs. nanoTabPFN runs two attentions per layer — one across rows and one
across columns — so the cost grows with both the number of rows and the number of features. A
few-feature table is therefore genuinely cheaper, which is why the feature curriculum, by doing the
small tables first, front-loads cheap compute.

## 7. How to read each figure

- **feature_ramp.png / context_ramp.png** — x is the training step, y is the average feature count (or
  in-context example count) at that step. The curriculum line should slope up (2→60); the baseline
  should be flat (~31). This is a sanity check that the ordering was actually applied.
- **val_auc.png / val_loss.png** — convergence curves on the fixed validation set, one line per
  ordering. Higher (or lower loss) and earlier is better. The curriculum rises earlier; by the end
  they converge.
- **regime_gap.png** — two panels. Blue bars are the 16 real TabArena datasets; the orange band is the
  range our pretraining pool covers. Features overlap well; **in-context examples do not** — the pool
  tops out at 180 while the benchmark uses 673–4500. That gap is the key to why the in-context axis is
  so sensitive.
- **compute_efficiency.png** — the headline. x is estimated cumulative FLOPs (log scale), y is
  validation ROC-AUC, for baseline vs feature curriculum, with a spread band over seeds. Up-and-left
  is better. The curriculum reaches a given quality at roughly a third of the compute the baseline
  needs, because it front-loads the cheap small tables. Both use the same total compute by the end.
- **val_auc_vs_flops.png** — the same axes for all seven orderings. The feature curriculum sits
  up-and-left (best); the reversed feature ordering sits down-and-right (worst — it front-loads the
  *expensive* big tables and also ends in the wrong regime).

## 8. The one-line takeaway

Ordering the pretraining data helps, but not because "easy first" is magic. What matters is the regime
the curriculum **ends** in: the model specialises to it, and that helps when it matches the evaluation
condition (and hurts when it does not). As a bonus, ordering cheap tables first reaches good quality
with less compute.
