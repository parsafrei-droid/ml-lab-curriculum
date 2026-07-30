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
