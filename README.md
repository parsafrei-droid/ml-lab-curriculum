# Curriculum Pretraining for nanoTabPFN

ML Lab 2026, University of Freiburg.
Parsa Rasouli, Omid Rasouli, Emre Çamlica.
Supervisors: Alexander Pfefferle, Dominika Matus.

This is the complete record of the project: every stage, what it found, what failed, and
how to reproduce the results. All numbers below were re-verified against the raw run
outputs (`tabarena_scores.json`, `log.csv`, `meta.json`) in August 2026.

---

## 1. The question

Tabular foundation models (TabPFN [1], TabICL [5]) are pretrained once on millions of
synthetic tables drawn from a hand-designed prior; afterwards they solve a new dataset in
a single forward pass, taking the labelled training rows (in-context examples) and the
unlabelled test rows together, with no gradient steps. nanoTabPFN [2] is a compact
reimplementation for teaching and research that streams synthetic tables to the model in
random order.

Curriculum learning [3] proposes that showing easy examples before hard ones can help.
Because the pretraining distribution here is fully synthetic, we can hold the training
data completely fixed and change nothing but presentation order — a cleaner setup than
language-model curricula [4]. We asked:

1. Does presentation order alone change the final model?
2. If so, what property of an ordering drives the effect?
3. Can an ordering, or a cheaper prior, reduce pretraining cost?

**Answers:** (1) yes, up to ±0.05 TabArena ROC-AUC at fixed data and compute; (2) the
ascent, not the destination — a controlled test (section 5) shows that deleting the
ending costs nothing while deleting the climb returns the model to baseline; (3) yes,
the same quality arrives 2.3× sooner in wall-clock time (4.6× in FLOPs), and a cheaper
ramped prior beats the paper recipe while training ~25% faster.

On an independently generated pool the ordering effect reproduces at **+0.011**; the
+0.020 in section 4 is the original pool's figure and should not be quoted alone.

---

## 2. Repository layout

```
experiments/first_attempt/   stage 1: scheduling the prior's own hyperparameters
experiments/emre/            stage 1 continuation + prior-ramp (noise/capacity) runs + HP sweep
experiments/second_wave/     stage 2: the fixed-pool ordering experiments (main result)
experiments/third_wave/      stage 3: efficiency plots + modded-nanoTabPFN 2x2 + compute accounting
experiments/robustness/      stage 2b: the path-vs-endpoint runs (runs_endpoint/, runs_path/)
final/                       poster code + figures (start here to reproduce the headline result)
```

Two upstream repos are expected next to the experiment code, plus a shared virtualenv:

```bash
git clone <TFM-Playground> TFM-Playground
git clone <tabicl> tabicl
python -m venv .venv && source .venv/bin/activate
pip install -e TFM-Playground -e tabicl
```

---

## 3. Stage 1 — scheduling the prior's knobs (mostly a negative result)

**Idea:** make the prior gradually "harder" during training by scheduling its
hyperparameters (noise_std, num_layers, hidden_dim, num_causes, features, classes, rows).

**What we learned from the TabICLv2 source** (`tabicl/src/tabicl/prior/`):

- `noise_std`, `num_layers` and `hidden_dim` are sampled from
  `meta_trunc_norm_log_scaled` distributions (`_prior_config.py`), so raising only an
  upper bound barely moves the sampled values — scheduling those bounds is close to a
  no-op.
- `num_causes` is silently overwritten with `num_features` whenever the causal flag is
  off (`_mlp_scm.py`), so scheduling it does nothing for half the sampled SCMs.
- External knobs (feature count, class count) do work.

**Lesson:** stop scheduling generator knobs; treat difficulty as a *measured property of
each generated table* instead. This motivated the fixed-pool design of stage 2.

---

## 4. Stage 2 — ordering a fixed pool (the main experiment)

### Design

- **Pool:** 80,000 tables generated once from the unmodified TabICLv2 prior, generation
  seed 0; feature count uniform over 2–60, 200 rows per table, up to 10 classes, stored
  in generation order (`second_wave/pool.py`).
- **Single pass, identical compute:** 2,500 optimizer steps × effective batch 32 =
  exactly 80,000 tables; every run consumes the same pool once, so order is the only
  free variable.
- **Orderings:** tables sorted by a weighted min-max-normalised score over two axes —
  feature count and in-context example count; a negative weight reverses an axis. Every
  run records the Spearman correlation between training position and each axis
  (`ordering_profile` in `meta.json`).
- **Training recipe (published nanoTabPFN):** 3 layers, embedding 96, 4 heads, MLP
  hidden 192, schedule-free AdamW lr 0.003892, gradient clipping 1.0, grad accumulation
  to effective batch 32.
- **Validation:** 96 held-out synthetic tables (fixed seed 12345), identical across
  runs, stratified into three feature bands (easy 2–10, medium 10–30, hard 30–60).
- **Final scoring:** one-vs-rest ROC-AUC on TabArena [6], seeds 42/1/2; the 5,000-sample
  filter keeps 16 of the 51 tasks (10 binary, 6 multiclass; 5–112 features; 673–4,500
  training rows).

### Results (TabArena ROC-AUC, mean ± sd over 3 seeds; paired deltas)

| Ordering | ROC-AUC | Δ vs baseline |
|---|---|---|
| Features, few → many | **0.811 ± 0.012** | **+0.020 ± 0.005** |
| In-context, few → many | **0.807 ± 0.011** | **+0.016 ± 0.001** |
| Random order (baseline) | 0.791 ± 0.012 | reference |
| Sawtooth, 3 ascending cycles | 0.789 ± 0.008 | −0.002 ± 0.016 |
| Features, many → few | 0.762 ± 0.038 | −0.029 ± 0.027 |
| Combined features + context | 0.751 ± 0.003 | −0.041 ± 0.013 |
| In-context, many → few | 0.742 ± 0.018 | −0.049 ± 0.011 |

Both positive orderings win on 3/3 seeds and improve 14 of 16 tasks (Wilcoxon on
per-dataset deltas: p = 0.0017 for each; `final/code/stats.py`). By task type, the
feature curriculum gains 0.010 on binary and 0.035 on multiclass tasks.

### Interpretation

Two framings were tried and discarded before the controlled test in section 5 settled it.
Both are recorded here so the reasoning is auditable.

**Discarded 1: "the model specialises to whatever regime the curriculum ends in."**
Falsified by band-wise analysis. The reversed feature ordering (ends on few-feature
tables) is not a small-table specialist: it is worse *everywhere*, including the regime
it ends on (easy-band validation 0.585 vs 0.597 for the baseline; −0.008 on the five
low-feature benchmark tasks).

**Discarded 2: "what the final sustained phase trains is what is retained."**
Falsified by the endpoint experiment in section 5. Holding the top of the feature range
for 1,300 steps captures none of the gain, and deleting the ending altogether costs
nothing.

**What survives.**

1. **The gain lives in the ascent, not the destination.** See section 5. Every ordering
   that begins on the smallest tables and works upward scores ~0.801; every ordering
   that begins on a shuffled mixture and only sorts its tail scores ~0.795, however long
   it holds the top.
2. **Transfer is one-way.** Competence on wide tables carries down to narrow ones, not
   the reverse: the ascending curriculum's gain is concentrated on the eleven
   high-feature tasks (+0.028, vs +0.002 on the low-feature ones), while the descending
   ordering loses on both.
3. **The in-context effect is a coverage artifact.** The pool never exceeds 180
   in-context examples while the benchmark provides 673–4,500, so ending on long
   contexts helps for benchmark-coverage reasons; a wider pool should remove it. Do not
   use this axis as evidence about difficulty ordering.

### Efficiency (the headline plot)

A forward pass costs more on a wide table, so the ascending order front-loads cheap ones
and reaches a given quality with far less arithmetic. On the mean of three seeds,
reaching 0.58 on the synthetic validation set takes **5.3× fewer FLOPs and 1.9× fewer
steps** than random order, and the curriculum is ahead at 24 of 25 checkpoints.

Three views of the same runs, all starting from a measured untrained score rather than an
assumed one (`final/code/step0_eval.py`):

| Figure | Script | What it shows |
|---|---|---|
| `flops_curve_mean.png` | `final/code/flops_curve.py` | ROC-AUC vs FLOPs, 3 seeds, ±1 sd bands. The poster figure. |
| `step_curve.png` | `final/code/step_curve.py` | Same runs against training steps. |
| `time_curve.png` | `experiments/third_wave/plot_time_curve.py` | Wall clock, single matched pair (seed 42, A100). |

Two cautions. Wall-clock and FLOPs speedups are not interchangeable here: per-step wall
clock barely moves while per-step FLOPs vary 16×, so the run is overhead-bound and the
FLOPs saving is real but does not show up in time at this model size. And the logged
`cum_flops` column is inconsistent across run sets — the timed pair used the current
`approx_flops`, the three-seed runs an older version about 50× off — so
`final/code/flops_curve.py` recomputes it from the logged feature counts, which
reproduces the correctly logged totals to within 2–4%.

---

## 5. Stage 2b — is it the path or the endpoint? (the controlled test)

The seven orderings of section 4 cannot separate two explanations, because every ordering
that climbs also ends high. Two experiments break the tie: 35 runs, **5 seeds each**, on
a rebuilt pool (79,999 tables, one NaN dropped) at identical compute.
Code: `experiments/robustness/kaggle_endpoint_vs_duration.py` and
`kaggle_path_vs_endpoint.py`. Data: `experiments/robustness/runs_endpoint/` and
`runs_path/`.

**Endpoint vs duration (20 runs, 4 arms).** All arms shuffle the first part of training
and sort only the tail, so they finish at the top of the feature range and hold it for
300 / 700 / 1300 steps:

| Arm | ROC-AUC | realised feature Spearman |
|---|---|---|
| Top held 300 steps | 0.795 ± 0.008 | 0.315 |
| Top held 700 steps | 0.796 ± 0.006 | 0.625 |
| Top held 700, pure | 0.796 ± 0.011 | 0.628 |
| Top held 1300 steps | 0.796 ± 0.007 | 0.888 |

Flat, and all at the shuffled baseline (0.791) rather than the curriculum's 0.811. Even
the 1300-step arm, whose ordering is 0.888-monotone, captures none of the gain. **Both
the endpoint and how long it is held are ruled out.**

**Path vs endpoint (15 runs, 3 arms).**

| Arm | ROC-AUC | vs full sort (paired) |
|---|---|---|
| Full sort (climb + ending) | 0.802 ± 0.005 | reference |
| Coarse staircase (4 blocks) | 0.801 ± 0.015 | −0.001 ± 0.011 |
| Climb, ending removed (last 700 steps shuffled) | 0.801 ± 0.009 | −0.001 ± 0.006 |

Deleting the ending costs nothing; deleting the climb returns to baseline. A coarse
four-block staircase is as good as a perfect sort, so fine-grained order is not the
mechanism either.

**A caveat on wording.** "The climb drives the gain" is nearly right but not literal: the
1300-step arm *does* contain a climb (32 → 59 features over its last 1300 steps) and
still gains nothing. What separates the two groups is that the winning arms begin at the
**bottom** of the range (4, 9, 4 mean features at the first checkpoint) and progress
upward across the whole run, while the losing arms begin on a shuffled mixture (15–27)
and their ascent covers only the upper half. Read the trajectories in
`final/figures/path_vs_endpoint.png` before restating this claim.

**Replication and honest effect size.** These runs use an independently generated pool,
which closes the pool-sample risk: the effect reproduces. But its size is smaller than
section 4 reports — full sort gives 0.802 at 5 seeds here, and 0.801 on the same three
seeds section 4 used, against 0.811 there. **Treat +0.011 as the defensible effect size,
not +0.020.**

Figure: `final/figures/path_vs_endpoint.png` (overlaid trajectories plus a score bar
chart), built by `final/code/path_vs_endpoint_figure.py`.

---

## 6. Stage 3 — reshaping the prior during training (ramps)

Instead of reordering a fixed pool, ramp the generator itself at paper scale (binary
head, 4 features, evaluated on the 26 binary TabArena tasks that survive the filter):

- **Late ramp** (`curriculum_noise_layers_binary`): nested schedule — for each SCM depth
  (2 → 4 → 6 layers, with matching hidden_dim 16 → 64 → 130), cycle noise_std
  0.001 → 0.1 → 0.3; ramp completes at ~68% of the run.
- **Early ramp** (`..._early_ramp`): same ceiling, 4 noise levels per block, ramp
  completes at ~44% and then holds the hardest setting.
- **Paper recipe**: the published optimum, fixed the whole run. (Close but not exact:
  weight_decay 1e-7 is not honored by the training code, num_datapoints is fixed at 154
  rather than sampled from [50, 300], and the layer count is not in the paper's Table 1.)

**At 10,000 steps (3 seeds, mean ± sd):** late ramp 0.783 ± 0.005 (+0.011 ± 0.002),
early ramp 0.780 ± 0.005 (+0.008 ± 0.007), paper recipe 0.772 ± 0.005. The late ramp
also cuts training time by ~27% on the matched GPU (small generating networks are
cheaper to sample from); the early ramp saves only ~11% because it holds the expensive
ceiling longer.

**Budget dependence (honest negative):** at 2,500 steps both ramps trail the paper
recipe (early 0.759, late 0.747 vs 0.764; seed 42). The ramp gains need the longer
budget.

**Robustness (HP sweep):** early ramp vs paper recipe across 10 hyperparameter configs ×
3 seeds at 2,500 steps: mean TabArena delta +0.009, 17/30 pairs won
(`experiments/emre/figures/hp_sweep_data_multiseed.json`).

**Modded-nanoTabPFN 2×2 (negative result):** applying the ascending feature curriculum
to the speedrun codebase's 256k-table dump (`third_wave/modded_feature_curriculum.patch`)
did not help: best TabArena ROC-AUC 0.782 (curriculum) vs 0.807 (baseline), and the
curriculum run was slower per epoch. Recorded in `third_wave/results/summary.json`.

---

## 7. Compute

From `third_wave/results/computations/compute_summary.json` (sacct + Kaggle logs):

- bwUniCluster: **48.7 h of GPU time across 299 Slurm jobs** (230 on H100 94GB, 65 on
  A100 80GB, 3 MI300, 1 CPU-only), 2026-06-25 → 2026-07-30.
- Kaggle: 6 on-the-fly generation runs on T4, ~8.0 h.
- Project total: 56.7 h across 305 runs. Queue time on bwUniCluster totalled ~269 h,
  dominated by three jobs that waited 68–77 h each.

The stage 2b runs (section 5) came later and on Kaggle, so they are **not** in that
accounting: 35 further runs on T4. The bwUniCluster figure quoted in the poster
acknowledgement (48.7 h / 299 jobs) is unaffected.

---

## 8. Reproducing the main result

```bash
cd experiments/second_wave
./run.sh                      # builds the 80k pool (seed 0), trains baseline +
                              # curriculum_features on seeds 42/1/2, evaluates on TabArena
python ../../final/code/stats.py           # paired per-dataset statistics
```

Then the poster figures, from the repository root:

```bash
python final/code/step0_eval.py       # measured untrained score, feeds both curves
python final/code/flops_curve.py      # Figure 1
python final/code/step_curve.py       # same runs against training steps
python final/code/results_table.py    # Table 1
python final/code/auc_curve_table.py  # Table 2
python final/code/pipeline_figure.py  # method schematic
python final/code/intro_figure.py     # introduction schematic
```

Configs for every other ordering are in `experiments/second_wave/configs/`. The ramp
experiments live in `experiments/emre/` (`*_binary_*` configs), evaluated with that
folder's TabArena scripts.

The stage 2b test runs on Kaggle:

```bash
python experiments/robustness/kaggle_endpoint_vs_duration.py   # 20 runs, 4 arms x 5 seeds
python experiments/robustness/kaggle_path_vs_endpoint.py       # 15 runs, 3 arms x 5 seeds
python final/code/path_vs_endpoint_figure.py                   # overlaid-trajectory figure
```

---

## 9. References

1. N. Hollmann, S. Müller, K. Eggensperger, F. Hutter. TabPFN: A transformer that solves
   small tabular classification problems in a second. ICLR 2023.
2. A. Pfefferle, J. Hog, L. Purucker, F. Hutter. nanoTabPFN: A lightweight and
   educational reimplementation of TabPFN. arXiv:2511.03634, 2025.
3. Y. Bengio, J. Louradour, R. Collobert, J. Weston. Curriculum learning. ICML 2009.
4. Y. Zhang, A. Mohamed, H. Abdine, G. Shang, M. Vazirgiannis. Beyond random sampling:
   Efficient language model pretraining via curriculum learning. EACL 2026.
5. J. Qu, D. Holzmüller, G. Varoquaux, M. Le Morvan. TabICL: A tabular foundation model
   for in-context learning on large data. ICML 2025.
6. N. Erickson et al. TabArena: A living benchmark for machine learning on tabular data.
   NeurIPS Datasets and Benchmarks Track, 2025.
