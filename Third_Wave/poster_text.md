# Poster text (ready to paste into the Google Slides template)

Tone: plain and student friendly, B2 English, no em dashes. Fill the parts in
[square brackets] yourself.

---

## Header

**Title:** Curriculum Pretraining for nanoTabPFN: Does the Order of the Synthetic Data Matter?

**Names:** Parsa Rasouli, Emre [surname], Omid [surname] - University of Freiburg

**Supervisors:** Alexander Pfefferle, Dominika [surname]

**Github:** [your repo link]

---

## Introduction

nanoTabPFN is a small tabular foundation model. It is pretrained once on many
synthetic tables. After that, to solve a new dataset, we give it the labeled
training rows and the unlabeled test rows together, and it predicts in a single
forward pass. There are no gradient steps on the user's data. The labeled rows
in the input are called in-context examples.

Curriculum learning says a model can learn better if it sees easy examples
before hard ones. In this project we ask one simple question: during
pretraining, does the order of the synthetic tables matter, and if it does, why?

---

## Method

- **One fixed pool.** We generate a single pool of 80,000 synthetic tables from
  the TabICLv2 prior. Every run reads the same pool and only the order changes:
  a random shuffle for the baseline, or sorted by a difficulty axis for the
  curriculum. The data and the total compute stay identical, so the only
  variable is the order.
- **Difficulty axes.** We sort by number of features or by number of in-context
  examples. Difficulty is measured from the data, not guessed.
- **A second view.** In parallel we also ramp the prior's own knobs during
  training (number of layers, hidden size, noise) instead of ordering the data.
- **Does it save real time?** We repeat the best ordering on modded-nanoTabPFN,
  a fast speedrun version that is compute bound, to see if a saving in FLOPs
  turns into a saving in wall clock.
- **Evaluation.** Real datasets from TabArena, ROC-AUC, averaged over 3 seeds.

---

## Quantitative Results

TabArena ROC-AUC, mean of 3 seeds. The baseline is the same pool in random order.

| Ordering | Ends training on | ROC-AUC | vs baseline |
|---|---|---|---|
| Features, few to many | many features | 0.811 | +0.020 |
| In-context, few to many | many examples | 0.807 | +0.016 |
| Baseline (shuffle) | mixed | 0.791 | - |
| Features, many to few | few features | 0.762 | -0.029 |
| In-context, many to few | few examples | 0.742 | -0.049 |

- The ranking is not about easy versus hard. It is about the regime where
  training ends. The two axes have opposite difficulty directions, so before
  running we predicted that reversing each axis would flip the sign of the
  effect. Both reversals did exactly that.
- Reproduced on an A100 GPU: the feature curriculum beats the baseline by +0.030.
- Prior-knob ramps (second view): +0.008 to +0.011 over the exact paper
  baseline, and about 29 percent less training time.
- Compute-bound test: on the speedrun model the curriculum is 1.69 times slower
  to reach the same quality. A saving in FLOPs does not become a saving in wall
  clock once the model is already compute bound.
- Honest limits: the positive effects are about twice the seed spread, and the
  evaluation set is small (16 tasks).

---

## Qualitative Results

Put the figures here, main one on top.

1. **Main figure: `figures/time_curve.png`.**
   Caption: Same data, same total compute. The curriculum reaches the baseline's
   best quality about 2.3 times sooner in time (4.6 times fewer FLOPs), because
   it does the cheap small tables first.

2. **`figures/noise_schedule.png`.**
   Caption: The prior-knob view. The baseline trains at full noise from step 0,
   the curriculum climbs up to it and then holds.

3. **`figures/lr_curve.png`.**
   Caption: At the paper's learning rate the curriculum converges almost at once,
   while the baseline catches up only late.

(If there is room, `Second_Wave/figures/tabarena_roc_auc.png` shows the full
ordering ranking as a dot plot.)

---

## References

- [1] Hollmann et al. TabPFN: A Transformer That Solves Small Tabular
  Classification Problems in a Second. ICLR 2023.
- [2] nanoTabPFN (Pfefferle et al.), the base model we build on. [add exact cite]
- [3] TabArena: a benchmark of real tabular datasets. [add exact cite]
- [4] modded-nanoTabPFN. github.com/borawhocodess/modded-nanotabpfn

---

## Acknowledgement

We thank our supervisors Alexander Pfefferle and Dominika [surname] for their
guidance during the ML Lab 2026 at the University of Freiburg.
