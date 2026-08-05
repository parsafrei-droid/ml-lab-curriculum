"""Validation ROC-AUC against arithmetic, averaged over seeds.

The logged cum_flops column is not usable across all runs: the timed pair was written
with the current approx_flops and the three-seed runs with an older version, about 50x
off. FLOPs itself is hardware-agnostic, so the fix is to recompute rather than to give
up. We reconstruct it from mean features per step, which reproduces the correctly logged
totals to within 2 to 4 percent because the cost is close to linear in feature count.

Same three orderings and the same style as step_curve.py, so the two are comparable.
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
CURR, BASE, REV = "#e8862e", "#3b6fd4", "#8c2f4a"

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "second_wave" / "results"
SEEDS = ("s42", "s1", "s2")
STEP0 = json.loads((pathlib.Path(__file__).parent.parent / "figures" /
                    "step0_val_auc.json").read_text())

SERIES = [
    ("curriculum_features", "Features, few to many", CURR, 2.4),
    ("baseline", "Random order", BASE, 2.4),
    ("curriculum_features_reverse", "Features, many to few", REV, 2.0),
]

TARGET = 0.58
ROWS_PER_TABLE, ACCUM, EVAL_EVERY = 200, 32, 100
SCALE = 1e12          # plot in TFLOP


def approx_flops(rows, features, e=96, mlp=192, layers=3):
    """Same expression as the training loop's counter, for one table."""
    cols = features + 1
    return 3.0 * layers * (cols * rows * rows * e
                           + rows * cols * cols * e
                           + 2 * rows * cols * e * mlp)


def run_curve(run):
    """Cumulative recomputed FLOPs and validation score at each checkpoint."""
    rows = list(csv.DictReader(open(RESULTS / run / "log.csv")))
    feats = [float(r["mean_features"]) for r in rows]
    aucs = [float(r["val_auc"]) for r in rows]

    cum, xs, prev = 0.0, [0.0], feats[0]
    for f in feats:
        # trapezoid over the 100 steps since the previous checkpoint
        cum += approx_flops(ROWS_PER_TABLE, (prev + f) / 2) * ACCUM * EVAL_EVERY
        xs.append(cum)
        prev = f
    return xs, [STEP0[run.rsplit("_", 1)[1]]] + aucs


def mean_curve(cfg):
    per_seed = [run_curve(f"{cfg}_{s}") for s in SEEDS]
    xs = [statistics.mean(v) for v in zip(*(x for x, _ in per_seed))]
    ys = [statistics.mean(v) for v in zip(*(y for _, y in per_seed))]
    sd = [statistics.stdev(v) for v in zip(*(y for _, y in per_seed))]
    return xs, ys, sd


def first_reach(xs, ys, target):
    for x, y in zip(xs, ys):
        if y >= target:
            return x
    return None


def build(out_path):
    curves = {cfg: mean_curve(cfg) for cfg, _, _, _ in SERIES}

    fig = plt.figure(figsize=(8.6, 4.9), facecolor="white")
    ax = fig.add_axes([0.10, 0.20, 0.87, 0.76])

    for cfg, label, color, lw in SERIES:
        xs, ys, sd = curves[cfg]
        xs = [x / SCALE for x in xs]
        ax.fill_between(xs, [m - s for m, s in zip(ys, sd)],
                        [m + s for m, s in zip(ys, sd)],
                        color=color, alpha=0.15, linewidth=0, zorder=1)
        ax.plot(xs, ys, color=color, lw=lw, label=label, zorder=3)

    cx, cy, _ = curves["curriculum_features"]
    bx, by, _ = curves["baseline"]
    f_curr = first_reach(cx, cy, TARGET) / SCALE
    f_base = first_reach(bx, by, TARGET) / SCALE
    ax.axhline(TARGET, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate("", xy=(f_base, TARGET), xytext=(f_curr, TARGET),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.5,
                                shrinkA=0, shrinkB=0), zorder=4)
    # caption below the arrow, so it does not sit on top of the curves
    ax.text((f_curr + f_base) / 2, TARGET - 0.004,
            f"{f_base / f_curr:.1f}x fewer FLOPs", ha="center", va="top",
            fontsize=10, color=INK)

    ax.set_xlabel("Pretraining FLOPs (TFLOP)", fontsize=11, color=INK)
    ax.set_ylabel("ROC-AUC", fontsize=11, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c8cdd6")
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)

    fig.text(0.10, 0.045,
             "Held-out synthetic validation, mean of 3 seeds, bands are ± 1 sd. "
             "FLOPs recomputed from feature counts.",
             fontsize=9.5, color=MUTED, ha="left", va="center")

    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}  (at {TARGET}: {f_curr:.1f} vs {f_base:.1f} TFLOP, "
          f"{f_base / f_curr:.2f}x)")


if __name__ == "__main__":
    build(pathlib.Path(__file__).parent.parent / "figures" / "flops_curve_mean.png")
