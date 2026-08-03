import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1f2328"
MUTED = "#6b7280"

# One hue per stage, matching the poster's panels. Fills are light tints of the border
# hue so black text stays readable on all of them.
STAGES = [
    ("1", "Pool generation", "#3b6fd4", "#eef3fc",
     ["TabICLv2 prior, unmodified",
      "80,000 tables, drawn once (seed 0)",
      "2-60 features / 200 rows / up to 10 classes",
      "frozen: every run reads this same pool"]),
    ("2", "Difficulty scoring", "#c2621a", "#fdf3ea",
     ["measured per table, never scheduled",
      "axis 1: feature count, 2 to 60",
      "axis 2: in-context examples, 20 to 180",
      "weighted score; -1 reverses an axis"]),
    ("3", "Ordering", "#2f7d4f", "#edf7f0",
     ["sort the pool: the only free variable",
      "ascending / descending / shuffled",
      "7 orderings in total",
      "identical data and total compute"]),
    ("4", "Train and evaluate", "#6b4fa0", "#f3effa",
     ["nanoTabPFN, published recipe",
      "2,500 steps x batch 32 = one pass",
      "validation: 96 held-out tables",
      "TabArena, 16 tasks, seeds 42/1/2"]),
]


def build(out_path):
    fig = plt.figure(figsize=(18, 3.6), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    box_w, box_h, gap, x0, y0 = 4.05, 2.75, 0.45, 0.25, 0.35
    for i, (num, title, edge, fill, lines) in enumerate(STAGES):
        x = x0 + i * (box_w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), box_w, box_h,
                                    boxstyle="round,pad=0.06,rounding_size=0.12",
                                    fc=fill, ec=edge, lw=2.2))
        # numbered badge + stage title
        ax.add_patch(plt.Circle((x + 0.38, y0 + box_h - 0.38), 0.19, fc=edge, ec="none"))
        ax.text(x + 0.38, y0 + box_h - 0.385, num, ha="center", va="center",
                fontsize=13, color="white", fontweight="bold")
        ax.text(x + 0.70, y0 + box_h - 0.38, title, ha="left", va="center",
                fontsize=16.5, color=INK, fontweight="bold")
        for j, line in enumerate(lines):
            ax.text(x + 0.30, y0 + box_h - 0.95 - j * 0.48, line, ha="left", va="center",
                    fontsize=12.5, color=INK if j == 0 else MUTED)
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((x + box_w + 0.10, y0 + box_h / 2),
                                         (x + box_w + gap - 0.10, y0 + box_h / 2),
                                         arrowstyle="-|>", mutation_scale=26,
                                         color=INK, lw=2.2))

    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    build(pathlib.Path(__file__).parent.parent / "figures" / "pipeline.png")
