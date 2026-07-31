# Prior-knob ramps and the learning-rate sweep

Emre's line of work. Where the second wave orders a fixed pool of datasets, this one leaves
the data stream alone and ramps the **prior's own knobs** during training: `num_layers`,
`hidden_dim` and `noise_std`. The baseline holds all three at their hardest setting for the
whole run; the curriculum climbs to that ceiling and then holds there.

Because the knobs change what gets generated, the runs here do not share a fixed dump. Each
run generates its own tables on the fly, which is inherent to the method rather than a flaw
in it: changing the generator *is* the intervention. Model architecture, step count and
batch size are identical between baseline and curriculum, so the comparison is same model,
same number of gradient steps, different data.

## Results

Binary TabArena, 10,000 steps, 3 seeds:

| setup | TabArena AUC | vs baseline | training time |
|---|---|---|---|
| baseline (exact paper recipe) | 0.7719 ± 0.0038 | - | ~77.6 min |
| late ramp | **0.7826 ± 0.0043** | **+0.0107** | ~55.1 min |
| early ramp | 0.7802 ± 0.0042 | +0.0083 | ~55.4 min |

Same direction as the fixed-pool result, and cheaper, because generating tables from a small
SCM costs less than from a large one.

A second experiment sampled 10 learning rates from the paper's own search space and trained
each twice, baseline against early-ramp curriculum. The curriculum helps most when the
learning rate is badly tuned low; at a well-tuned rate it converges faster but finishes in
the same place.

## Two things to know before quoting these numbers

**Difficulty is defined by intuition here**, not measured: fewer layers and less noise is
assumed easier. That is exactly the weakness the fixed-pool design was built to avoid, so
this is supporting evidence rather than the primary result.

**The `toy_tabarena` metric is not TabArena.** It is a cheap per-checkpoint probe over three
small OpenML datasets (iris, wine, breast_cancer). For a binary model the two three-class
ones are skipped, so it reduces to breast_cancer alone, which is why those curves sit near
0.99. It tracks convergence speed within a run and nothing more. Every number in the table
above comes from the real TabArena evaluation in `results/*/tabarena_scores.json`, not from
that probe.

The single most dramatic single-seed result in the sweep did not survive two more seeds:
8/10 curriculum wins became 17/30, and a mean delta of +0.046 became +0.009.

## Contents

| path | what |
|---|---|
| `scripts/` | Training, evaluation, the sweeps and the cluster submission scripts. |
| `configs/` | Schedules for each ramp variant, plus the generated hp-sweep configs. |
| `figures/` | Summary plots and the aggregated sweep data (`hp_sweep_data_multiseed.json`). |
| `results/` | Per-run configs, loss and validation CSVs, and TabArena scores. |
| `curriculum/` | This branch's version of the scheduler and prior wrapper. |
| `presentation.md` | The update written from these runs. |

Per-run diagnostic PNGs are not kept here; they are regenerable from the CSVs and were
dropped to keep the folder readable.
