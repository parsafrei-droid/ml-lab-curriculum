"""Validation ROC-AUC averaged over the run, feature axis only.

The curve is integrated over training steps and normalised, starting from chance at step
0, so a run that is good early scores higher than one that only catches up at the end.
The final column shows that the curriculum and the baseline finish at the same quality:
the difference is when that quality arrives.
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
RULE = "#c8cdd6"
HAIR = "#e6e8ec"
BETTER = "#c2621a"
WORSE = "#8c2f4a"

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "second_wave" / "results"
SEEDS = ("s42", "s1", "s2")

ROWS = [
    ("Features, few to many", "curriculum_features"),
    ("Random order (baseline)", "baseline"),
    ("Features, many to few", "curriculum_features_reverse"),
]


STEP0 = json.loads((pathlib.Path(__file__).parent.parent / "figures" /
                    "step0_val_auc.json").read_text())


def curve(run):
    """Step 0 is the measured untrained score for that seed, see step0_eval.py."""
    seed = run.rsplit("_", 1)[1]
    rows = list(csv.DictReader(open(RESULTS / run / "log.csv")))
    xs = [0.0] + [float(r["step"]) for r in rows]
    ys = [STEP0[seed]] + [float(r["val_auc"]) for r in rows]
    area = sum((xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) / 2 for i in range(len(xs) - 1))
    return area / (xs[-1] - xs[0]), ys[-1]


def build(out_path):
    base_area = [curve(f"baseline_{s}")[0] for s in SEEDS]
    data = []
    for label, cfg in ROWS:
        areas, finals = zip(*(curve(f"{cfg}_{s}") for s in SEEDS))
        delta = None if cfg == "baseline" else [a - b for a, b in zip(areas, base_area)]
        data.append((label,
                     statistics.mean(areas), statistics.stdev(areas),
                     None if delta is None else statistics.mean(delta),
                     None if delta is None else statistics.stdev(delta),
                     statistics.mean(finals), statistics.stdev(finals)))

    fig_w, row_h = 9.4, 0.50
    fig_h = row_h * len(data) + 1.75
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    x_l, x_r = 0.30, fig_w - 0.30
    col_name = x_l + 0.06
    col_area, col_delta, col_final = 4.55, 6.55, 8.90   # right edges

    y = fig_h - 0.40
    ax.text(x_l, y, "Table 2 — Validation ROC-AUC averaged over the run", fontsize=14.5,
            fontweight="bold", color=INK, va="center")
    y -= 0.42

    # header
    ax.text(col_name, y, "Ordering", fontsize=10.5, color=MUTED, va="center")
    ax.text(col_area, y, "Mean over run", fontsize=10.5, color=MUTED, va="center",
            ha="right")
    ax.text(col_delta, y, "Δ vs baseline", fontsize=10.5, color=MUTED, va="center",
            ha="right")
    ax.text(col_final, y, "Final", fontsize=10.5, color=MUTED, va="center",
            ha="right")
    y -= 0.20
    ax.plot([x_l, x_r], [y, y], color=INK, lw=1.3)

    for i, (name, area, sd, d, dsd, fin, fsd) in enumerate(data):
        y -= row_h
        bold = "bold" if d is not None and d > 0 else "normal"
        ax.text(col_name, y, name, fontsize=12, color=INK, va="center", fontweight=bold)
        ax.text(col_area, y, f"{area:.3f} ± {sd:.3f}", fontsize=12, color=INK,
                va="center", ha="right", fontweight=bold)
        if d is None:
            ax.text(col_delta, y, "reference", fontsize=11.5, color=MUTED, va="center",
                    ha="right")
        else:
            ax.text(col_delta, y, f"{d:+.3f} ± {dsd:.3f}", fontsize=12,
                    color=BETTER if d > 0 else WORSE, va="center", ha="right",
                    fontweight="bold")
        ax.text(col_final, y, f"{fin:.3f} ± {fsd:.3f}", fontsize=12, color=INK,
                va="center", ha="right")
        if i < len(data) - 1:
            ax.plot([x_l, x_r], [y - row_h / 2, y - row_h / 2], color=HAIR, lw=0.9)

    y -= row_h / 2
    ax.plot([x_l, x_r], [y, y], color=RULE, lw=1.1)
    y -= 0.30
    ax.text(x_l, y, "Averaged over training steps. Mean ± sd, 3 seeds, 96 held-out "
            "synthetic tables.", fontsize=10.5, color=MUTED, va="center")

    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    build(pathlib.Path(__file__).parent.parent / "figures" / "auc_curve_table.png")
