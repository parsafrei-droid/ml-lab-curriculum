# Poster text

For the DLL26 UFR template (A0, PDF, 300 DPI). Plain B2 English. Fill the parts in
[square brackets] yourself.

---

## Header

**Title:** Curriculum Pretraining for nanoTabPFN: Does the Order of the Synthetic Data Matter?

**Names:** Parsa Rasouli, Emre Ozturk, Omid Frei - University of Freiburg

**Supervisors:** Alexander Pfefferle, Dominika Wozniak

**Github:** [repo link]

---

## Introduction

nanoTabPFN is a small tabular foundation model. It is pretrained once on many synthetic
tables. After that, to solve a new dataset, we give it the labeled training rows and the
unlabeled test rows together, and it predicts in a single forward pass. There are no
gradient steps on the user's data. The labeled rows in the input are called in-context
examples.

Curriculum learning says a model can learn better if it sees easy examples before hard ones.
In this project we ask one simple question: during pretraining, does the order of the
synthetic tables matter, and if it does, why?

---

## Method

- **One fixed pool.** We generate a single pool of 80,000 synthetic tables from the TabICLv2
  prior. Every run reads the same pool and only the order changes: a random shuffle for the
  baseline, or sorted by a difficulty axis for the curriculum. The data and the total compute
  stay identical, so the only variable is the order.
- **Difficulty axes.** We sort by number of features or by number of in-context examples.
  Difficulty is measured from the data, not guessed.
- **A reversal test.** The two axes have opposite difficulty directions, so we wrote down our
  predictions first and then reversed each axis to see if the effect flips.
- **A second approach.** In parallel we ramp the prior's own knobs during training (layers,
  hidden size, noise) instead of ordering the data.
- **Does it save real time?** We repeat the best ordering on modded-nanoTabPFN, a fast
  speedrun version that is compute bound, to see if a saving in FLOPs turns into a saving in
  wall clock.
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

Reversing an axis flips the sign of the effect, exactly as we predicted before running:
features +0.020 becomes -0.029, in-context -0.049 becomes +0.016.

- The ranking is not about easy versus hard. It is about the regime where training ends.
- Reproduced on an A100: the feature curriculum beats the baseline by +0.030.
- Ramping the prior's knobs instead: +0.008 to +0.011 over the exact paper baseline, at about
  29 percent less training time.
- Compute-bound test: on the speedrun model the curriculum is 1.69 times slower to reach the
  same quality. A saving in FLOPs does not become a saving in wall clock once the model is
  already compute bound.
- Honest limits: the positive effects are about twice the seed spread, and only 16 of the 51
  TabArena tasks pass the size filters.

---

## Qualitative Results

1. **Main figure: `time_curve.png`.**
   Caption: Same data, same total compute. The curriculum reaches the baseline's best quality
   about 2.3 times sooner in wall clock (4.6 times fewer FLOPs), because it does the cheap
   small tables first.

2. **`regime_gap.png`.**
   Caption: Why it happens. Our pool covers TabArena's feature range but tops out at 180
   in-context examples where the benchmark uses 673 to 4500.

3. **`noise_schedule.png`.**
   Caption: The second approach. The baseline trains at full noise from step 0, the curriculum
   climbs up to it and then holds.

(If there is room, `tabarena_roc_auc.png` shows the full ordering ranking as a dot plot.)

---

## References

- [1] Hollmann et al. TabPFN: A Transformer That Solves Small Tabular Classification Problems
  in a Second. ICLR 2023.
- [2] Pfefferle et al. nanoTabPFN. [add exact cite]
- [3] TabArena: a living benchmark for tabular machine learning. [add exact cite]
- [4] modded-nanoTabPFN. github.com/borawhocodess/modded-nanotabpfn

---

## Acknowledgement

We thank our supervisors Alexander Pfefferle and Dominika Wozniak for their guidance during
the ML Lab 2026 at the University of Freiburg. Compute was provided by BwUniCluster.
