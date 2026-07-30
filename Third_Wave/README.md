# Third Wave — does the curriculum help a compute-bound speedrun model?

Our Second-Wave finding is that ordering the pretraining data (feature curriculum) beats random order
and reaches a given quality with fewer FLOPs. But on our small model the FLOP saving does not show up
in wall-clock, because the model is too small to be compute-bound — the time is dominated by fixed
overheads (data movement / synthetic generation).

modded-nanoTabPFN (https://github.com/borawhocodess/modded-nanotabpfn) is a speedrun of nanoTabPFN
training: Muon optimizer, bfloat16, torch.compile, a larger embedding, and a small pre-generated dump.
It is deliberately optimised until the model *is* compute-bound. That makes it the right setting to
ask whether our ordering effect converts into a real wall-clock saving.

## The 2x2 design

Two models, two data orders — four runs, all on the same GPU type, all logging wall-clock:

| # | model | data order | what it tests |
|---|---|---|---|
| 1 | ours (nanoTabPFN, our 80k pool) | random (shuffle) | our baseline |
| 2 | ours (nanoTabPFN, our 80k pool) | feature curriculum (few→many) | ordering effect on our model |
| 3 | modded-nanoTabPFN (its own dump) | its default order | modded baseline |
| 4 | modded-nanoTabPFN (its own dump) | feature curriculum (few→many) | ordering effect on modded |

The comparisons this enables:

- **1 vs 2** and **3 vs 4** — does the curriculum help *within* each model?
- **(1,2) vs (3,4)** — how much the speedrun optimisations buy on their own.
- The headline plot is **validation ROC-AUC vs wall-clock time**, four lines, same GPU. We also record
  time-to-reach modded's target (~0.807 ROC-AUC) for each run.

## Rules for a fair time comparison

- All four runs on the **same GPU type** (e.g. an A100). Record `torch.cuda.get_device_name()` for each.
- Log wall-clock for every run (our `train.py` already writes `cum_time_s`; modded logs its own time).
- **Runs 3 and 4 use modded's own dump**, not our pool — our pool made modded's evaluation flood OpenML
  and the job was cancelled. Keep each model on the data it was built for.
- Everything else in a pair is identical; only the data order changes.

## What "feature curriculum" means for modded (run 4)

modded trains from a small pre-generated dump. For run 4 we do not change the model or recipe — we only
feed the datasets in order of increasing feature count instead of the default order.

**Why it is not a strict sort of the whole dump.** The dump holds 256,000 datasets, but a run consumes
only `steps * batch_size` = 64 per epoch, and the reference record reaches the target in 57 epochs —
3,648 datasets. A strict `argsort` over all 256k would hand those 3,648 slots entirely to `n_features`
1–2 (the first 20-feature dataset would not appear until epoch 3,887 of 4,000). Run 4 would then train
on a truncated slice of the distribution and never converge, so it would differ from run 3 in *which*
data it sees — not just the order, which is the one thing we want to vary.

Instead run 4 draws a budget-sized subset (`curriculum_epochs * steps * batch_size` datasets, sampled
without replacement across the full width distribution) and sorts **that** ascending. The result is a
smooth ramp from `n_features` 2 → 20 over the run, spanning the same distribution run 3 sees, with
mean per-batch feature width 11.0 vs 14.0 for the default order (**21.6% narrower**). Same data, same
count, only the order differs.

## Honest expectation

- 1 vs 2: probably close in wall-clock (our model is overhead-bound), but the FLOP/step advantage is
  still there.
- 3 vs 4: modded is compute-bound, so if the ordering effect is real it should show up as a wall-clock
  (time-to-target) saving here. That is the point of this wave.
