import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Diverging pair validated with the palette checker (all checks pass). Sign is also
# carried by bar direction and the +/- text, so polarity is never colour-alone.
BETTER = "#c2621a"
WORSE = "#8c2f4a"
INK = "#1f2328"
MUTED = "#6b7280"
RULE = "#d9dce1"

# Compact layout: one line per run, mean +/- sample sd over 3 seeds, paired deltas
# against each block's own baseline (same pool, same seed).
BLOCKS = [
    ("Table 1 — Ordering a fixed pool of 80,000 tables",
     "3 seeds  ·  16 TabArena tasks  ·  2,500 steps",
     [("Features, few → many", 0.811, 0.012, +0.020, 0.005),
      ("In-context, few → many", 0.807, 0.011, +0.016, 0.001),
      ("Random order (baseline)", 0.791, 0.012, None, None),
      ("Sawtooth, 3 ascending cycles", 0.789, 0.008, -0.002, 0.016),
      ("Features, many → few", 0.762, 0.038, -0.029, 0.027),
      ("Combined features + context", 0.751, 0.003, -0.041, 0.013),
      ("In-context, many → few", 0.742, 0.018, -0.049, 0.011)]),
    ("Table 2 — Reshaping the prior during training",
     "3 seeds  ·  26 binary TabArena tasks  ·  10,000 steps",
     [("Late ramp", 0.783, 0.005, +0.011, 0.002),
      ("Early ramp", 0.780, 0.005, +0.008, 0.007),
      ("Paper recipe (baseline)", 0.772, 0.005, None, None)]),
]

FOOT = "Mean ± sd. Positive orderings: 3/3 seeds, 14/16 tasks improved, Wilcoxon p = 0.0017."


def build(out_path):
    rows = sum(len(b[2]) for b in BLOCKS)
    row_h = 0.40
    fig_h = row_h * rows + 1.35 * len(BLOCKS) + 0.55
    fig = plt.figure(figsize=(11.0, fig_h), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # score column ends left of the bar zone; span covers the largest |delta| + sd
    x_name, x_score, x_bar, bar_w = 0.03, 0.44, 0.60, 0.12
    span = 0.062

    y = fig_h - 0.40
    for title, subtitle, entries in BLOCKS:
        ax.text(x_name, y, title, fontsize=14.5, fontweight="bold", color=INK, va="center")
        y -= 0.34
        ax.text(x_name, y, subtitle, fontsize=11, color=MUTED, va="center")
        y -= 0.22
        ax.plot([x_name, 0.97], [y, y], color=INK, lw=1.2)
        y -= 0.12

        ax.text(x_score, y, "ROC-AUC", fontsize=10.5, color=MUTED, va="center", ha="right")
        ax.text(x_bar + bar_w + 0.02, y, "Δ vs baseline", fontsize=10.5, color=MUTED,
                va="center", ha="left")
        y -= 0.30

        for name, score, sd, delta, dsd in entries:
            bold = "bold" if delta is not None and delta > 0 else "normal"
            ax.text(x_name, y, name, fontsize=12.5, color=INK, va="center", fontweight=bold)
            ax.text(x_score, y, f"{score:.3f} ± {sd:.3f}", fontsize=12.5, color=INK,
                    va="center", ha="right", fontweight=bold)

            ax.plot([x_bar, x_bar], [y - 0.15, y + 0.15], color=RULE, lw=1.0)
            if delta is None:
                ax.text(x_bar + bar_w + 0.02, y, "reference", fontsize=11, color=MUTED,
                        va="center", ha="left")
            else:
                w = bar_w * abs(delta) / span
                c = BETTER if delta > 0 else WORSE
                x0 = x_bar if delta > 0 else x_bar - w
                ax.add_patch(plt.Rectangle((x0, y - 0.10), w, 0.20, color=c, lw=0))
                lo = x_bar + bar_w * (delta - dsd) / span
                hi = x_bar + bar_w * (delta + dsd) / span
                ax.plot([lo, hi], [y, y], color=INK, lw=1.0)
                for tip in (lo, hi):
                    ax.plot([tip, tip], [y - 0.05, y + 0.05], color=INK, lw=1.0)
                ax.text(x_bar + bar_w + 0.02, y, f"{delta:+.3f} ± {dsd:.3f}",
                        fontsize=12, color=c, va="center", ha="left", fontweight="bold")
            y -= row_h

        y -= 0.32

    ax.text(x_name, y + 0.22, FOOT, fontsize=10.5, color=MUTED, va="top")
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    build(pathlib.Path(__file__).parent.parent / "figures" / "results_table.png")
