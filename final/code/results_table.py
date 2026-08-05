"""Final TabArena score per ordering.

A plain grid with a delta bar column: from poster distance the pattern of bars is read
before any number, and the whisker on the sawtooth row shows at a glance that its effect
is not separable from zero.
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1f2328"
MUTED = "#6b7280"
RULE = "#c8cdd6"
HAIR = "#e6e8ec"
BETTER = "#c2621a"
WORSE = "#8c2f4a"

TITLE = "Table 1 — TabArena subset, 16 of 51 tasks"
SUB = ("3 seeds  ·  2,500 steps  ·  10 binary and 6 multiclass, one-vs-rest  ·  "
       "identical data and compute")
FOOT = "Mean ± sd over seeds. Deltas are paired against the shuffled baseline."

# label, ROC-AUC mean, sd, paired delta, sd of delta
ROWS = [
    ("Features, few to many", 0.811, 0.012, +0.020, 0.005),
    ("In-context, few to many", 0.807, 0.011, +0.016, 0.001),
    ("Random order (baseline)", 0.791, 0.012, None, None),
    ("Sawtooth (easy to hard, restarted 3x)", 0.789, 0.008, -0.002, 0.016),
    ("Features, many to few", 0.762, 0.038, -0.029, 0.027),
    ("Combined features + context", 0.751, 0.003, -0.041, 0.013),
    ("In-context, many to few", 0.742, 0.018, -0.049, 0.011),
]

FIG_W, ROW_H = 9.4, 0.46
SPAN = 0.062        # the delta axis runs from -SPAN to +SPAN


def build(out_path, bars=True):
    fig_h = ROW_H * len(ROWS) + 1.95
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    x_l, x_r = 0.30, FIG_W - 0.30
    col_name = x_l + 0.06
    if bars:
        # the score column has to end left of x_bar - bar_w or a long negative bar
        # runs over it
        col_score, x_bar, bar_w, col_delta = 5.30, 6.55, 0.95, 9.05
    else:
        col_score, col_delta = 6.45, 8.90

    y = fig_h - 0.40
    ax.text(x_l, y, TITLE, fontsize=14.5, fontweight="bold", color=INK, va="center")
    y -= 0.30
    ax.text(x_l, y, SUB, fontsize=10.5, color=MUTED, va="center")
    y -= 0.34

    ax.text(col_name, y, "Ordering", fontsize=10.5, color=MUTED, va="center")
    ax.text(col_score, y, "ROC-AUC", fontsize=10.5, color=MUTED, va="right" and "center",
            ha="right")
    ax.text(col_delta, y, "Δ vs baseline", fontsize=10.5, color=MUTED, va="center",
            ha="right")
    y -= 0.18
    ax.plot([x_l, x_r], [y, y], color=INK, lw=1.3)

    for i, (name, score, sd, d, dsd) in enumerate(ROWS):
        y -= ROW_H
        bold = "bold" if d is not None and d > 0 else "normal"
        ax.text(col_name, y, name, fontsize=12, color=INK, va="center", fontweight=bold)
        ax.text(col_score, y, f"{score:.3f} ± {sd:.3f}", fontsize=12, color=INK,
                va="center", ha="right", fontweight=bold)

        if bars:
            ax.plot([x_bar, x_bar], [y - 0.15, y + 0.15], color=HAIR, lw=1.0)
            if d is not None:
                w = bar_w * abs(d) / SPAN
                c = BETTER if d > 0 else WORSE
                x0 = x_bar if d > 0 else x_bar - w
                ax.add_patch(plt.Rectangle((x0, y - 0.10), w, 0.20, color=c, lw=0))
                lo = x_bar + bar_w * (d - dsd) / SPAN
                hi = x_bar + bar_w * (d + dsd) / SPAN
                ax.plot([lo, hi], [y, y], color=INK, lw=1.0)
                for tip in (lo, hi):
                    ax.plot([tip, tip], [y - 0.05, y + 0.05], color=INK, lw=1.0)

        if d is None:
            ax.text(col_delta, y, "reference", fontsize=11.5, color=MUTED, va="center",
                    ha="right")
        else:
            ax.text(col_delta, y, f"{d:+.3f} ± {dsd:.3f}", fontsize=12,
                    color=BETTER if d > 0 else WORSE, va="center", ha="right",
                    fontweight="bold")
        if i < len(ROWS) - 1:
            ax.plot([x_l, x_r], [y - ROW_H / 2, y - ROW_H / 2], color=HAIR, lw=0.9)

    y -= ROW_H / 2
    ax.plot([x_l, x_r], [y, y], color=RULE, lw=1.1)
    y -= 0.28
    ax.text(x_l, y, FOOT, fontsize=10.5, color=MUTED, va="center")

    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    build(pathlib.Path(__file__).parent.parent / "figures" / "results_table.png")
