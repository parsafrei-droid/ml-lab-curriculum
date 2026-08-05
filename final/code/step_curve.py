"""Validation ROC-AUC against training steps, three orderings, mean of three seeds.

Steps rather than wall clock: hardware independent like FLOPs, but unlike FLOPs it does
not overstate the win, since per-step wall clock is nearly flat while per-step FLOPs vary
by a factor of sixteen. It also puts every run on one axis, which wall clock could not do
because the timed pair ran on a different node type from the three-seed sets.

The shaded band is the gap the run-averaged metric integrates.
"""

import csv
import json
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1f2328"
MUTED = "#6b7280"
GRID = "#e6e8ec"
# Validated three-slot palette: worst adjacent CVD dE 21.7. Orange is below 3:1 on
# white, which the legend and the annotation text cover.
CURR, BASE, REV = "#e8862e", "#3b6fd4", "#8c2f4a"

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "second_wave" / "results"
SEEDS = ("s42", "s1", "s2")

SERIES = [
    ("curriculum_features", "Features, few to many", CURR, 2.4),
    ("baseline", "Random order", BASE, 2.4),
    ("curriculum_features_reverse", "Features, many to few", REV, 2.0),
]

TARGET = 0.58   # a level inside the rising phase, clear of the plateau noise


STEP0 = json.loads((pathlib.Path(__file__).parent.parent / "figures" /
                    "step0_val_auc.json").read_text())


def mean_curve(cfg):
    """Mean and sample sd across seeds at every logged checkpoint.

    Step 0 is the measured score of the untrained model (see step0_eval.py). No ordering
    has been applied at that point, so every ordering of a seed shares it.
    """
    per_seed = []
    for s in SEEDS:
        rows = list(csv.DictReader(open(RESULTS / f"{cfg}_{s}" / "log.csv")))
        per_seed.append([STEP0[s]] + [float(r["val_auc"]) for r in rows])
    steps = [0] + [int(r["step"]) for r in
                   csv.DictReader(open(RESULTS / f"{cfg}_{SEEDS[0]}" / "log.csv"))]
    mean = [statistics.mean(v) for v in zip(*per_seed)]
    sd = [statistics.stdev(v) for v in zip(*per_seed)]
    return steps, mean, sd


def first_reach(steps, ys, target):
    for s, y in zip(steps, ys):
        if y >= target:
            return s
    return None


def build(out_path):
    curves = {cfg: mean_curve(cfg) for cfg, _, _, _ in SERIES}

    fig = plt.figure(figsize=(8.6, 4.9), facecolor="white")
    ax = fig.add_axes([0.10, 0.20, 0.87, 0.76])

    for cfg, label, color, lw in SERIES:
        steps, ys, sd = curves[cfg]
        ax.fill_between(steps, [m - s for m, s in zip(ys, sd)],
                        [m + s for m, s in zip(ys, sd)],
                        color=color, alpha=0.15, linewidth=0, zorder=1)
        ax.plot(steps, ys, color=color, lw=lw, label=label, zorder=3)

    # how many steps the curriculum saves at a representative level
    cs, cy, _ = curves["curriculum_features"]
    bs, by, _ = curves["baseline"]
    s_curr = first_reach(cs, cy, TARGET)
    s_base = first_reach(bs, by, TARGET)
    ax.axhline(TARGET, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate("", xy=(s_base, TARGET), xytext=(s_curr, TARGET),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.5,
                                shrinkA=0, shrinkB=0), zorder=4)
    # caption below the arrow, so it does not sit on top of the curves
    ax.text((s_curr + s_base) / 2, TARGET - 0.004,
            f"{s_base - s_curr} steps earlier  ({s_base / s_curr:.1f}x)",
            ha="center", va="top", fontsize=10, color=INK)

    ax.set_xlabel("Training steps", fontsize=11, color=INK)
    ax.set_ylabel("ROC-AUC", fontsize=11, color=INK)
    ax.set_xlim(0, 2600)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c8cdd6")
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)

    fig.text(0.10, 0.045,
             "Figure 1. Held-out synthetic validation, mean of 3 seeds, bands are ± 1 sd.",
             fontsize=9.5, color=MUTED, ha="left", va="center")

    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}  (target {TARGET}: curriculum {s_curr}, baseline {s_base})")


if __name__ == "__main__":
    build(pathlib.Path(__file__).parent.parent / "figures" / "step_curve.png")
