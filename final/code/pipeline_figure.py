import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1f2328"
MUTED = "#6b7280"
BLUE, ORANGE, GREEN, MAROON, PURPLE = "#3b6fd4", "#c2621a", "#2f7d4f", "#8c2f4a", "#6b4fa0"

PANEL_W, PANEL_H, GAP, X0, Y0 = 3.00, 4.45, 0.35, 0.25, 0.30
FIG_W = X0 * 2 + PANEL_W * 4 + GAP * 3  # 13.55, three quarters of the old 18


def rbox(ax, x, y, w, h, ec, fc="white", lw=1.4, r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.03,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


def header(ax, x, top, num, title, color):
    ax.add_patch(plt.Circle((x + 0.34, top - 0.36), 0.175, fc=color, ec="none"))
    ax.text(x + 0.34, top - 0.365, num, ha="center", va="center", fontsize=12,
            color="white", fontweight="bold")
    ax.text(x + 0.62, top - 0.36, title, ha="left", va="center", fontsize=14,
            color=color, fontweight="bold")


def panel_pool(ax, x):
    top = Y0 + PANEL_H
    header(ax, x, top, "1", "Pool generation", BLUE)
    # prior box (kept clear of the number badge above it)
    rbox(ax, x + 0.25, top - 1.14, PANEL_W - 0.50, 0.52, BLUE)
    ax.text(x + PANEL_W / 2, top - 0.81, "TabICLv2 prior", ha="center", va="center",
            fontsize=11.5, color=INK, fontweight="bold")
    ax.text(x + PANEL_W / 2, top - 1.03, "unmodified  ·  seed 0", ha="center",
            va="center", fontsize=9, color=MUTED)
    ax.add_patch(FancyArrowPatch((x + PANEL_W / 2, top - 1.21), (x + PANEL_W / 2, top - 1.40),
                                 arrowstyle="-|>", mutation_scale=16, color=BLUE, lw=1.6))
    # 80k tables box with a little grid of table icons
    rbox(ax, x + 0.25, top - 2.70, PANEL_W - 0.50, 1.22, BLUE)
    ax.text(x + PANEL_W / 2, top - 1.66, "80,000 tables", ha="center", va="center",
            fontsize=11.5, color=INK, fontweight="bold")
    ax.text(x + PANEL_W / 2, top - 1.88, "drawn once, then frozen", ha="center",
            va="center", fontsize=9, color=MUTED)
    for r in range(2):
        for c in range(5):
            ax.add_patch(plt.Rectangle((x + 0.48 + c * 0.42, top - 2.28 - r * 0.28),
                                       0.30, 0.18, fc="white", ec=BLUE, lw=1.0))
    # property chips in one row
    for i, txt in enumerate(["2-60 feat.", "200 rows", "10 classes"]):
        cx = x + 0.25 + i * 0.86
        rbox(ax, cx, top - 3.20, 0.78, 0.32, "#c7d5f0", fc="#f4f7fd", lw=1.0, r=0.05)
        ax.text(cx + 0.39, top - 3.04, txt, ha="center", va="center", fontsize=7.8, color=INK)
    ax.text(x + PANEL_W / 2, Y0 + 0.15, "every run reads this same pool", ha="center",
            va="center", fontsize=9, color=BLUE, style="italic")


def mini_tables(ax, x, y, n=4):
    # tables of growing width: axis 1 icon
    for i in range(n):
        w = 0.14 + i * 0.10
        ax.add_patch(plt.Rectangle((x, y), w, 0.28, fc="white", ec=ORANGE, lw=1.1))
        ax.add_patch(plt.Rectangle((x, y + 0.21), w, 0.07, fc=ORANGE, ec="none"))
        x += w + 0.11
    return x


def grow_bars(ax, x, y, n=5):
    for i in range(n):
        h = 0.08 + i * 0.07
        ax.add_patch(plt.Rectangle((x, y), 0.18, h, fc=ORANGE, ec="none"))
        x += 0.28
    return x


def panel_difficulty(ax, x):
    top = Y0 + PANEL_H
    header(ax, x, top, "2", "Difficulty scoring", ORANGE)
    ax.text(x + PANEL_W / 2, top - 0.70, "measured per table, never scheduled",
            ha="center", va="center", fontsize=8.6, color=MUTED, style="italic")
    rows = [("axis 1  ·  feature count", "2 → 60"),
            ("axis 2  ·  in-context examples", "20 → 180"),
            ("combined  ·  both axes", "w₁ + w₂")]
    ys = [top - 1.00, top - 2.05, top - 3.10]
    for (label, right), yy in zip(rows, ys):
        rbox(ax, x + 0.25, yy - 0.78, PANEL_W - 0.50, 0.80, "#e8c9ae", lw=1.1, r=0.06)
        ax.text(x + 0.38, yy - 0.18, label, ha="left", va="center", fontsize=9,
                color=ORANGE, fontweight="bold")
        ax.text(x + PANEL_W - 0.38, yy - 0.55, right, ha="right", va="center",
                fontsize=8.6, color=MUTED)
    mini_tables(ax, x + 0.40, ys[0] - 0.70)
    grow_bars(ax, x + 0.40, ys[1] - 0.70)
    # combined: growing squares
    sx = x + 0.40
    for i in range(5):
        s = 0.09 + i * 0.05
        ax.add_patch(plt.Rectangle((sx, ys[2] - 0.70), s, s, fc=ORANGE, ec="none"))
        sx += s + 0.13
    ax.text(x + PANEL_W / 2, Y0 + 0.15, "a negative weight reverses an axis", ha="center",
            va="center", fontsize=9, color=ORANGE, style="italic")


def strip(ax, x, y, w, widths, color):
    # a row of segments whose widths encode difficulty
    total = sum(widths) + 0.05 * (len(widths) - 1)
    scale = w / total
    cx = x
    for wd in widths:
        ax.add_patch(plt.Rectangle((cx, y), wd * scale, 0.24, fc=color, ec="none"))
        cx += (wd + 0.05) * scale


def panel_ordering(ax, x):
    top = Y0 + PANEL_H
    header(ax, x, top, "3", "Ordering", GREEN)
    ax.text(x + PANEL_W / 2, top - 0.70, "sort the pool, the only free variable",
            ha="center", va="center", fontsize=8.6, color=MUTED, style="italic")
    rng = np.random.default_rng(3)
    rows = [("CURRICULUM", "easy → hard", GREEN, list(np.linspace(0.3, 1.2, 8))),
            ("REVERSAL", "hard → easy", MAROON, list(np.linspace(1.2, 0.3, 8))),
            ("BASELINE", "random", BLUE, list(rng.uniform(0.3, 1.2, 8)))]
    ys = [top - 1.00, top - 1.93, top - 2.86]
    for (tag, sub, color, widths), yy in zip(rows, ys):
        rbox(ax, x + 0.25, yy - 0.68, PANEL_W - 0.50, 0.70, color, lw=1.1, r=0.06)
        ax.text(x + 0.38, yy - 0.16, tag, ha="left", va="center", fontsize=8,
                color=color, fontweight="bold")
        ax.text(x + PANEL_W - 0.38, yy - 0.16, sub, ha="right", va="center",
                fontsize=8.6, color=INK)
        strip(ax, x + 0.38, yy - 0.56, PANEL_W - 0.76, widths, color)
    ax.text(x + PANEL_W / 2, top - 3.80,
            "features ×2  ·  in-context ×2", ha="center", va="center",
            fontsize=8.6, color=INK)
    ax.text(x + PANEL_W / 2, top - 4.02,
            "combined  ·  sawtooth  ·  random", ha="center", va="center",
            fontsize=8.6, color=INK)
    ax.text(x + PANEL_W / 2, Y0 + 0.15, "7 orderings  ·  same data & compute",
            ha="center", va="center", fontsize=9, color=GREEN, style="italic")


def panel_train(ax, x):
    top = Y0 + PANEL_H
    header(ax, x, top, "4", "Train & evaluate", PURPLE)
    # model box with three layer bars (kept clear of the number badge above it)
    rbox(ax, x + 0.25, top - 1.58, PANEL_W - 0.50, 0.96, PURPLE)
    ax.text(x + PANEL_W / 2, top - 0.80, "nanoTabPFN", ha="center", va="center",
            fontsize=11.5, color=INK, fontweight="bold")
    for i in range(3):
        ax.add_patch(plt.Rectangle((x + 0.65, top - 1.04 - i * 0.13), PANEL_W - 1.30, 0.08,
                                   fc="#c9bce4", ec="none"))
    ax.text(x + PANEL_W / 2, top - 1.50, "published recipe, unchanged", ha="center",
            va="center", fontsize=8.2, color=MUTED)
    ax.add_patch(FancyArrowPatch((x + PANEL_W / 2, top - 1.66), (x + PANEL_W / 2, top - 1.84),
                                 arrowstyle="-|>", mutation_scale=14, color=PURPLE, lw=1.5))
    rbox(ax, x + 0.25, top - 2.34, PANEL_W - 0.50, 0.42, PURPLE)
    ax.text(x + PANEL_W / 2, top - 2.13, "2,500 steps × 32 = one pass", ha="center",
            va="center", fontsize=9.5, color=INK, fontweight="bold")
    # validation box with easy/medium/hard chips
    rbox(ax, x + 0.25, top - 3.28, PANEL_W - 0.50, 0.80, PURPLE)
    ax.text(x + PANEL_W / 2, top - 2.66, "validation  ·  96 held-out tables", ha="center",
            va="center", fontsize=8.8, color=INK)
    for i, (band, shade) in enumerate([("easy", "#ded5f0"), ("medium", "#b9a5dd"),
                                       ("hard", "#6b4fa0")]):
        cx = x + 0.40 + i * 0.78
        ax.add_patch(FancyBboxPatch((cx, top - 3.14), 0.66, 0.30,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=shade, ec="none"))
        ax.text(cx + 0.33, top - 2.99, band, ha="center", va="center", fontsize=8,
                color="white" if i == 2 else INK)
    rbox(ax, x + 0.25, top - 4.06, PANEL_W - 0.50, 0.56, PURPLE)
    ax.text(x + PANEL_W / 2, top - 3.68, "TabArena  ·  ROC-AUC", ha="center", va="center",
            fontsize=9.8, color=INK, fontweight="bold")
    ax.text(x + PANEL_W / 2, top - 3.92, "16 tasks  ·  seeds 42/1/2", ha="center",
            va="center", fontsize=8.6, color=MUTED)
    ax.text(x + PANEL_W / 2, Y0 + 0.15, "identical for every run", ha="center",
            va="center", fontsize=9, color=PURPLE, style="italic")


def build(out_path):
    fig = plt.figure(figsize=(FIG_W, 5.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    fills = ["#eef3fc", "#fdf3ea", "#edf7f0", "#f3effa"]
    edges = [BLUE, ORANGE, GREEN, PURPLE]
    panels = [panel_pool, panel_difficulty, panel_ordering, panel_train]
    for i, draw in enumerate(panels):
        x = X0 + i * (PANEL_W + GAP)
        ax.add_patch(FancyBboxPatch((x, Y0), PANEL_W, PANEL_H,
                                    boxstyle="round,pad=0.06,rounding_size=0.14",
                                    fc=fills[i], ec=edges[i], lw=2.2))
        draw(ax, x)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + PANEL_W + 0.09, Y0 + PANEL_H / 2),
                                         (x + PANEL_W + GAP - 0.09, Y0 + PANEL_H / 2),
                                         arrowstyle="-|>", mutation_scale=22, color=INK, lw=2.0))

    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    build(pathlib.Path(__file__).parent.parent / "figures" / "pipeline.png")
