# Second Wave — curriculum pretraining for nanoTabPFN

A clean restart after the first round of feedback. The first version tried to
control difficulty through many prior knobs at once and used a kNN proxy to guess
which knobs mattered. Several of those knobs turned out not to do what we thought,
so here we narrow the scope to one axis we fully understand and compare orderings
on a single fixed pool of data.

## Research question

Does presenting the datasets nanoTabPFN sees during pretraining in order of
increasing number of features (easy to hard) lead to faster convergence and/or
better TabArena performance than a random order, on the same data at the same
compute?

## Design

We generate one fixed pool of datasets from the TabICLv2 prior, then train two
models on that exact pool:

- baseline: the pool in random order (`order: shuffle`)
- curriculum: the pool sorted by number of features, few to many (`order: curriculum`)

Because both read the same file, the data and the total compute are identical.
The only difference is the order. This is the setup Dominika suggested for
removing the distribution and compute confounds.

Difficulty here is simply the number of features in a dataset, which is set
directly on the prior and read at generation time. We do not touch the sampled
internal hyperparameters (noise, MLP depth/width, number of causes): those are
log-scaled and partly ignored by the generator, which is why our earlier attempts
to move them did nothing.

## Baseline settings (paper recipe)

Matches the published nanoTabPFN recipe:

- model: 3 layers, embedding 96, 4 heads, MLP hidden 192
- optimizer: schedule-free AdamW, learning rate 0.003892, gradient clip 1.0
- effective batch size 32 (gradient accumulation)
- 2500 steps

Intentionally ours, not the paper's:

- prior: TabICLv2 (the paper used v1); features 2..60 instead of the paper's 3..4,
  so there is a feature axis to study; 200 rows per dataset
- multiclass head (10 outputs) so the model can be scored on the multiclass
  TabArena tasks

## Metrics

Logged every `eval_every` steps to `results/<name>/log.csv`:

- train loss, validation loss, validation accuracy, validation ROC-AUC
- validation ROC-AUC split by feature band (easy 2..10, medium 10..30, hard 30..60)
- cumulative wall-clock, cumulative FLOP estimate, peak GPU memory

The validation set is fixed and shared across all runs, built once with a fixed
seed. The FLOP estimate is an approximation dominated by attention
(rows^2 * embedding per layer) plus the feature-encoding term, times three for the
backward pass; wall-clock is the direct measurement.

## How to run

Build the pool once, then train and evaluate:

```
python pool.py --out pools/main.pt --size 80000
python train.py --config configs/baseline.yaml --name baseline_s42 --seed 42
python evaluate.py --checkpoint results/baseline_s42/checkpoint.pth
```

Or the whole sweep (pool, both orders, seeds 42/1/2, evaluation, plots):

```
bash run.sh
```

`train.py` writes `curve.png` for each run. `plot.py` writes the comparison
figures to `figures/`: validation ROC-AUC and loss against steps (mean and std
band across seeds), and the feature-per-step ramp.

Pool size note: a single monotonic ramp needs the pool as large as
`total_steps * grad_accum` (80000 here). A smaller pool is cycled through several
times, so the ramp repeats each pass.

## Files

- `pool.py` — generate / load the fixed pool; order it (shuffle or curriculum).
- `prior.py` — build the TabICLv2 loader; build the fixed banded validation set.
- `train.py` — step-based training on the pool, with FLOP/memory logging.
- `evaluate.py` — TabArena ROC-AUC for a checkpoint.
- `plot.py` — per-run and comparison curves.
- `configs/` — baseline and curriculum settings (identical apart from `order`).
