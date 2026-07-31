# Robustness tests

Two tests that close the biggest holes a reviewer would find in the main result. Both reuse
`final/code` unchanged, so nothing about the headline pipeline is altered.

**Predictions are written down here before the runs, the same way we did the reversal test.
Whatever comes back gets recorded, including if it goes against us.**

---

## Test A: does the result survive a different dump?

Every run behind the main result reads **one** pool, built with generator seed 0. The three
"seeds" we report vary model initialisation and the baseline's shuffle, but **not the data**.
So we have not measured data-sampling variance at all, and a reviewer can fairly ask whether
+0.020 is a property of the curriculum or of that particular pool.

We rebuild the dump with generator seeds 1 and 2 and rerun only the two runs that carry the
headline, baseline and feature curriculum.

```bash
sbatch run_tests.sh pool_seed 1
sbatch run_tests.sh pool_seed 2
```

Four training runs, about 12 minutes each on an H100, plus a one-off dump build per pool.

**Prediction:** the feature curriculum beats the baseline on both new pools, by somewhere
between +0.010 and +0.030 TabArena ROC-AUC. If the gap holds across three independently
generated pools, the claim moves from "consistent across seeds" to "robust to the data
sample". If it collapses on either pool, we say so and the headline becomes a much weaker
claim about one dump.

---

## Test B: noise as a measured axis

This does two jobs at once.

It tests Emre's finding with our confound-free design. His prior-knob ramps changed
`noise_std` during training, which works, but difficulty there is defined by intuition and
the runs cannot share a fixed dump. Here noise becomes a **measured property of each table**:
`noise_pool.py` pins TabICL's `noise_std` range to one value per dataset and records it, so we
can sort by it exactly like we sort by feature count. Same dump for every run, only the order
changes.

It also gives the "where you finish" account a **new** prediction to make. Right now that
account explains two results with two regularities, which is fair to call post-hoc. A third
axis it has never seen is a real test.

```bash
sbatch run_tests.sh noise noise_baseline
sbatch run_tests.sh noise curriculum_noise
sbatch run_tests.sh noise curriculum_noise_reverse
```

**Prediction:** noise is a capacity axis, like features and unlike in-context examples. A
model that has just trained on noisy tables should handle clean ones, but not the reverse. So:

- `curriculum_noise` (low to high, ending noisy) scores **above** baseline
- `curriculum_noise_reverse` (high to low, ending clean) scores **below** baseline

If both hold, the account has predicted a third axis it was not built on. If the signs come
out flipped, then noise behaves like a coverage axis instead and the account needs the
narrower statement that only holds for features. Either outcome is worth reporting.

---

## What to send back

For each run: the `meta.json` (check `ordering_profile` shows the axis actually applied, near
1.0 for a forward curriculum and near -1.0 for a reversed one) and the `tabarena_scores.json`.

Results land in `final/code/results/`. Move them here before committing:

```bash
mkdir -p experiments/robustness/results
mv final/code/results/* experiments/robustness/results/
git add experiments/robustness/results && git commit -m "Robustness test results" && git push
```

Pools are large and stay out of git.
