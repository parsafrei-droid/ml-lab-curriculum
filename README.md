# Curriculum Pretraining for nanoTabPFN

ML Lab 2026, University of Freiburg — Parsa, Emre, Omid.

**Question:** nanoTabPFN is normally trained on synthetic datasets shown in
random order. If we instead order them **easy → hard** (a curriculum), does it
train faster or reach a better model for the same compute?

This repo is the code to answer that. Read this whole file before running — it's
the roadmap: how the code works, who runs what, and what to produce for the poster.

---

## What we found so far (before any training)

We can measure how "hard" a generated dataset is *without* training, using a
cheap probe (`curriculum/difficulty.py`: how badly a small kNN learns the task).
Running the sweep gave two results that shaped the whole plan:

1. **No single knob controls difficulty.** Sweeping `max_features`, `max_classes`,
   `noise_std`, `num_layers`, `hidden_dim`, `num_causes` one at a time → every
   line is flat (`experiments/difficulty_sweep.png`). Changing one knob while the
   prior's ~16 other hyper-parameters keep sampling randomly buries the signal.

2. **The whole regime does.** A "narrow" prior (every knob turned down together)
   vs the full default prior shows a clear gap — easy ≈ 0.42, hard ≈ 0.55
   difficulty (`experiments/regime_contrast.png`).

**So our curriculum ramps the entire regime narrow → full, not one knob.** The
single-knob curricula are kept as ablations we *expect* to be weak — that
contrast is itself a result for the poster.

---

## Parts of the code

```
curriculum/                     our code (small, on purpose)
  difficulty.py    difficulty scores: learnability (kNN error) + a geometric one
  prior.py         make_prior() / sample_dataset() around TabICL's generator
  scheduler.py     CurriculumScheduler + apply_knobs() — moves the prior easy→hard
scripts/
  setup_env.py     one-time: clone upstream repos, patch, install
  sweep_difficulty.py  which knobs move difficulty (the finding above)
  visualize_prior.py   easy→hard datasets in 2D (PCA, coloured by class)
  demo_scheduler.py    proof the scheduler actually changes the prior
  run.py           TRAIN one scenario from a YAML config
  eval_tabarena.py EVALUATE a checkpoint on TabArena
  compare_results.py   gather everyone's results into the poster figures
  submit_job.sh    SLURM template for the GPU cluster
experiments/configs/  one YAML per scenario (baseline + 5 curricula)
results/<name>/       what a run produces (see below)
```

The curriculum plugs into TFM-Playground's own `train()` loop untouched: the
scheduler advances itself as the loop pulls batches. The only upstream change is
a one-line import fix, re-applied automatically by `setup_env.py`.

---

## Setup (each machine, once)

```bash
git clone <this-repo-url> ml-lab-curriculum
cd ml-lab-curriculum
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python scripts/setup_env.py        # clones TFM-Playground + tabicl, patches, installs
python scripts/demo_scheduler.py   # quick check it all works
```

---

## The workflow

```bash
# 1) EXPLORE (cheap, no training) — how we picked the difficulty definition
python scripts/sweep_difficulty.py      # -> experiments/difficulty_sweep.png + regime_contrast.png
python scripts/visualize_prior.py       # -> experiments/prior_visualization.png

# 2) TRAIN your scenario  (GPU!) — writes results/<name>/
python scripts/run.py --config experiments/configs/curriculum_combined.yaml
#   on the cluster instead:  sbatch scripts/submit_job.sh curriculum_combined

# 3) EVALUATE your checkpoint on TabArena — writes results/<name>/tabarena_scores.json
python scripts/eval_tabarena.py --checkpoint results/curriculum_combined/checkpoint.pth --tasks tabarena

# 4) COMPARE (one person, after everyone commits their results/)
python scripts/compare_results.py       # -> experiments/comparison_*.png + summary.csv
```

### Who runs what (2 scenarios each)

| person | scenarios | why |
|---|---|---|
| **Parsa** | `baseline`, `curriculum_combined` | the control + the primary hypothesis |
| **Emre**  | `curriculum_reverse`, `curriculum_noise` | sanity check (hard→easy) + noise ablation |
| **Omid**  | `curriculum_features`, `curriculum_combined_slow` | feature ablation + gentler ramp |

Each person runs steps 2–3 for both their scenarios, then commits their
`results/<name>/` folders (checkpoints are git-ignored — only the small
`loss.csv`, `meta.json`, `tabarena_scores.json`, `*.png`, `config.yaml` go in).

---

## What a run produces (`results/<name>/`)

| file | what |
|---|---|
| `checkpoint.pth` | trained model (git-ignored, stays local — 44 MB) |
| `loss.csv` | per epoch training loss (NOT comparable across scenarios — see below) |
| `val.csv` | per epoch **val_loss, val_acc on a shared fixed set** — the comparable metric |
| `loss_curve.png` / `val_loss_curve.png` | those curves, plotted |
| `meta.json` | seed, total_steps, **elapsed_s, sec_per_step, peak_gpu_gb, final_val_loss, final_val_acc** |
| `tabarena_scores.json` | per-dataset + mean ROC-AUC on TabArena |
| `config.yaml` | the exact config used |

`meta.json` carries the **compute-resource** numbers so we can compare "same
compute" fairly, not just final accuracy — that's the actual research question.

### Comparability: use val_loss, not train_loss

**Do not compare `loss.csv` (training loss) across scenarios.** Each scenario
ends on different-difficulty data, so a run that finishes on easy data has a low
final train loss *for free* — it hasn't learned more, it's just being tested on
easier batches. Every scenario is instead scored each epoch on **one shared,
fixed validation set** (`val.csv`), which is identical for all runs. Compare
**`val_loss` / `val_acc`** (comparable) and **TabArena ROC-AUC** (ground truth).

> All results pushed before this were train-loss only — **please rerun your
> scenarios** so they log `val.csv`, and run `baseline` (it was missed the first
> time; without the control we can't conclude anything).

---

## Poster deliverables (what comes out of `compare_results.py`)

- `experiments/comparison_loss.png` — all scenarios' loss curves on one axis
- `experiments/comparison_tabarena.png` — mean TabArena ROC-AUC per scenario
- `experiments/comparison_summary.csv` — the table: steps, time, sec/step, peak
  GPU, final loss, ROC-AUC per scenario

Plus the "why" figures already generated: `difficulty_sweep.png`,
`regime_contrast.png`, `prior_visualization.png`.

The story the poster tells: *(1) how we defined difficulty and why single knobs
don't work, (2) baseline vs curriculum_combined on loss + TabArena, (3) does
order matter (reverse), (4) do single-knob curricula help (ablations).*

---

## Notes

- Configs are set to **2000 steps** (20 epochs × 100) for a first comparison.
  For the final runs bump `epochs` (e.g. 50 → 5000 steps). Thresholds in the
  schedule are absolute global steps — scale them if you change the totals.
- Laptops (CPU) are fine for the explore step and smoke tests; do the real
  training on GPU. On CPU a hard step is ~2–3 s and big datasets can crash.
- `noise_std` etc. are *ranges* the prior samples from — the scheduler sets the
  top of the range (low = easy, high = hard). See `curriculum/scheduler.py`.
