# Curriculum pretraining for nanoTabPFN

ML Lab 2026, University of Freiburg. Parsa Rasouli, Emre Ozturk, Omid Frei.
Supervisors: Alexander Pfefferle, Dominika Wozniak.

This folder is the finished product: the figures we stand behind, the code needed to
reproduce the main result, and the write-up. Everything we tried on the way there,
including the parts that did not work, is in [`../experiments/`](../experiments/).

## The question

nanoTabPFN is pretrained on a stream of synthetic tables drawn in random order. Curriculum
learning suggests showing easy tables before hard ones. Does the order matter, and if it
does, why?

## The answer

Order matters, but not because "easy first" is magic. What decides the outcome is **the
regime the curriculum ends in**. The model specialises to whatever it saw last, which helps
when that matches the evaluation condition and hurts when it does not.

| ordering | ends training on | TabArena ROC-AUC | vs baseline |
|---|---|---|---|
| features, few to many | many features | **0.811** | **+0.020** |
| in-context examples, few to many | many examples | **0.807** | **+0.016** |
| baseline (shuffle) | mixed | 0.791 | - |
| restart sawtooth | many features, briefly | 0.789 | -0.002 |
| features, many to few | few features | 0.762 | -0.029 |
| features + in-context combined | example-poor end | 0.751 | -0.041 |
| in-context examples, many to few | few examples | 0.742 | -0.049 |

Mean over 3 seeds. Every run reads the same 80,000-table dump and differs only in order,
so data and compute are identical by construction.

The decisive test: the two axes have **opposite** difficulty directions (more features is
harder, but *fewer* in-context examples is harder). We wrote down two predictions before
running, then reversed each axis. Both flipped sign as predicted, which turns a correlation
into a mechanism.

| axis | forward | reversed | swing |
|---|---|---|---|
| features | +0.020 | -0.029 | 0.049 |
| in-context examples | -0.049 | +0.016 | 0.065 |

## What we are not claiming

- The two positive effects hold on 3/3 seeds and 14/16 datasets, with a Wilcoxon signed-rank
  p of 0.0017 on the paired per-dataset differences (`code/stats.py`). But **all runs read one
  pool**, so the seeds vary model initialisation and shuffle, not the data. Whether the effect
  survives a differently sampled pool is untested; that run is specified in
  [`../experiments/robustness/`](../experiments/robustness/).
- Only 16 of the 51 TabArena tasks pass the size filters, which is a small evaluation set.
- One model size, one step budget, one prior. We did not test whether this survives scale.
- The compute saving is real in FLOPs but does **not** convert to wall-clock on a model that
  is already compute bound. We measured that directly and it went the other way.

## Figures

| file | what it shows |
|---|---|
| `figures/time_curve.png` | **Main figure.** Same dump, same total compute, but the curriculum reaches the baseline's best quality 2.3x sooner in wall-clock. |
| `figures/flops_curve.png` | The same comparison on a FLOPs axis, where the gap is 4.6x. |
| `figures/compute_efficiency.png` | Validation ROC-AUC against cumulative compute, averaged over 3 seeds. |
| `figures/tabarena_roc_auc.png` | Where all seven orderings landed on TabArena. |
| `figures/tabarena_binary_vs_all.png` | The same ranking split into binary and multiclass tasks, to check it is not an artefact of which datasets got scored. |
| `figures/regime_gap.png` | Why it happens: the pool covers TabArena's feature range but not its in-context range. |
| `figures/noise_schedule.png` | The second approach: ramping the prior's own noise knob instead of ordering the data. |
| `figures/compute_dashboard.png` | What the project cost: 305 runs, 56.7 h of GPU time. |

`FINDINGS.md` is the full write-up. `COMPUTE.md` is the compute history.
`poster_text.md` is the poster copy.

## Reproducing the main result

`code/` is the fixed-pool pipeline, trimmed to just what the headline result needs.

```bash
cd code
bash run.sh
```

That builds the dump once, then trains and evaluates all seven orderings at three seeds and
draws the figures. On an H100 one run is about 12 minutes, so the full sweep is roughly
7 hours of GPU time plus the one-off dump build.

| file | what it does |
|---|---|
| `code/pool.py` | Builds the dump and defines the orderings. |
| `code/prior.py` | Wraps the TabICLv2 prior and builds the shared validation set. |
| `code/train.py` | The training loop and the FLOP accounting. |
| `code/evaluate.py` | Scores a checkpoint on TabArena. |
| `code/plot.py` | Draws the figures. |
| `code/configs/` | One config per ordering. They are identical except for `order`, `axes` and `weights`. |

To redraw the figures from our runs instead of your own:

```bash
python code/plot.py --results ../experiments/second_wave/results --out figures \
  --classes ../experiments/second_wave/figures/regime_gap.json
```

## Where everything else lives

- [`../experiments/first_attempt/`](../experiments/first_attempt/) - the first version, before we found the prior knobs were mostly no-ops
- [`../experiments/second_wave/`](../experiments/second_wave/) - the fixed-pool design, including the on-the-fly variant we did not put in `final`
- [`../experiments/third_wave/`](../experiments/third_wave/) - the wall-clock test against modded-nanoTabPFN
- [`../experiments/emre/`](../experiments/emre/) - the prior-knob ramps and the learning-rate sweep
