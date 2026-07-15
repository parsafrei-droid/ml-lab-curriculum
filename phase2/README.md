# Phase 2 — the clean, single-knob rebuild

Phase 1 (everything in `experiments/configs/` and `results/`) is kept as-is for
reference, but its central finding was not defensible:

- Difficulty was measured with a **kNN**, then used to make claims about
  **nanoTabPFN** — two very different models. "Hard for kNN" ≠ "hard for the TFM".
- The single-knob sweep was **confounded** (all other prior HPs sampled randomly)
  and several knobs are **near-inert** (`noise_std` etc. only move a log-scale
  ceiling; `num_causes` is ignored on most datasets; ~30% of datasets are
  tree-SCM and ignore the MLP knobs).
- `curriculum_combined` bundles all 7 knobs, so even the (small, consistent)
  TabArena gain we saw **can't be attributed** to any one knob.
- Reality check across seeds 42/43/44: the gain is small (~+0.015 TabArena),
  consistent (6/6 tuned seed-runs), but **contradicted by val_loss** — a lead,
  not a proven result.

## The Phase-2 research question

> Does ramping **`max_features`** low→high during pretraining let nanoTabPFN reach
> a higher TabArena ROC-AUC than training at full `max_features` throughout — at
> the same compute?

One knob (`max_features` — undiluted, always takes effect, the brief's original
"feature curriculum" idea, and a compute lever). One model (nanoTabPFN, measured
by its own performance — **no kNN anywhere**). One outcome (TabArena AUC + val_loss).

## The two arms (differ ONLY in whether `max_features` ramps)

| arm | `max_features` | everything else |
|---|---|---|
| `baseline_features`   | pinned at 60 the whole run | full regime, paper-tuned recipe |
| `curriculum_features` | ramps 4 → 20 → 60          | **identical** — same seeds/steps/lr/model, all other knobs pinned at full |

Recipe = the paper-tuned setup (3-layer/96/4-head/192, lr 0.003892, batch 32),
which is where the Phase-1 lead actually appeared (model trained to ~0.80 AUC,
above chance). Run at ~5000 steps (`--epochs 50`), seeds 42/43/44 → 6 runs.

## How to run

```bash
# smoke test one run first (short, 1 seed) to confirm the configs work:
python scripts/run.py --config phase2/configs/curriculum_features.yaml --epochs 5

# the real comparison: 2 arms x 3 seeds at ~5000 steps
for cfg in baseline_features curriculum_features; do
  for s in 42 43 44; do
    python scripts/run.py --config phase2/configs/$cfg.yaml \
        --epochs 50 --seed $s --name ${cfg}_e50_s$s
    python scripts/eval_tabarena.py \
        --checkpoint results/${cfg}_e50_s$s/checkpoint.pth --tasks tabarena
  done
done
```

## Success bar (decided up front — no moving the goalposts)

The curriculum "works" only if it beats baseline on **TabArena**, averaged over
the 3 seeds, **by more than the seed-to-seed spread** (~±0.01 from Phase 1), AND
does **not** get worse on **val_loss**. Report BOTH metrics, mean ± std. If it
wins TabArena but loses val_loss (as `curriculum_combined` did), that's *not* a
win — it's inconclusive, and we say so.
