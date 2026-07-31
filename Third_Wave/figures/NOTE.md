# The dump-idea plot (ROC-AUC vs time)

This is the plot from Dominika's sketch, built from our real A100 runs (seed 42, same 80k
pool, only the order differs).

- `time_curve.png` — x is wall-clock pretraining time, y is validation ROC-AUC.
- `flops_curve.png` — same thing on a FLOPs x-axis.

**What it shows:** to reach the baseline's best quality (ROC-AUC 0.598), the curriculum needs
**395 s** and the baseline needs **899 s** — so the curriculum gets there **2.3x sooner** in
wall-clock (**4.6x** fewer FLOPs). That gap is the "x" arrow in the plot.

**One thing to keep straight when we present it:** the *total* time for a full 2500-step run is
about the same for both (~910 vs ~936 s) — the saving is at *equal quality*, not at the end. So
the honest framing is "the curriculum reaches a given quality sooner", not "the whole run is
faster".

Note: the dump numbers in the sketch (200k / 500 datapoints) were placeholders; the figure uses
our actual pool (80k datasets, 2-60 features, 200 datapoints).

## The noise-curriculum plot

`noise_schedule.png` — Emre's noise ramp (the early-ramp panel), redrawn in the same layout.
The baseline trains at full noise (noise_std 0.3) from step 0; the curriculum climbs to it,
stepping 0.001 -> 0.01 -> 0.1 -> 0.3 inside each num_layers block and holding at the ceiling for
the last ~56% of the run. On binary TabArena (10k steps, 3 seeds) this beat the exact paper
baseline by +0.008 AUC at ~29% less training time.

## The learning-rate panel

`lr_curve.png` — the bottom-right panel of Emre's 10-config lr sweep, redrawn in the same layout.
It is hp5, lr=0.003924, the sampled rate closest to the paper's optimum (0.003892). At a well-tuned
lr the curriculum reaches ~0.97 AUC by step 200 and holds, while the baseline is erratic early and
only catches up near the end; both finish ~0.99 (final delta +0.003). So at this lr the curriculum
buys faster, steadier convergence, not a higher ceiling — the large AUC rescues happen at badly-tuned
low learning rates instead. Bands are the seed min-max over 3 seeds.
