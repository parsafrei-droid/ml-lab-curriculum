import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1f2328"
MUTED = "#6b7280"
BLUE, ORANGE, GREEN, MAROON, PURPLE = "#3b6fd4", "#c2621a", "#2f7d4f", "#8c2f4a", "#6b4fa0"

PANEL_W, PANEL_H, GAP, X0, Y0 = 4.05, 4.45, 0.45, 0.25, 0.30


def rbox(ax, x, y, w, h, ec, fc="white", lw=1.4, r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.03,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


def header(ax, x, top, num, title, color):
    ax.add_patch(plt.Circle((x + 0.38, top - 0.36), 0.185, fc=color, ec="none"))
    ax.text(x + 0.38, top - 0.365, num, ha="center", va="center", fontsize=13,
            color="white", fontweight="bold")
    ax.text(x + 0.68, top - 0.36, title, ha="left", va="center", fontsize=16,
            color=color, fontweight="bold")


def panel_pool(ax, x):
    top = Y0 + PANEL_H
    header(ax, x, top, "1", "Pool generation", BLUE)
    # prior box
    rbox(ax, x + 0.30, top - 1.05, PANEL_W - 0.60, 0.52, BLUE)
    ax.text(x + PANEL_W / 2, top - 0.72, "TabICLv2 prior", ha="center", va="center",
            fontsize=12.5, color=INK, fontweight="bold")
    ax.text(x + PANEL_W / 2, top - 0.94, "unmodified  ·  generation seed 0", ha="center",
            va="center", fontsize=9.5, color=MUTED)
    ax.add_patch(FancyArrowPatch((x + PANEL_W / 2, top - 1.12), (x + PANEL_W / 2, top - 1.32),
                                 arrowstyle="-|>", mutation_scale=16, color=BLUE, lw=1.6))
    # 80k tables box with a little grid of table icons
    rbox(ax, x + 0.30, top - 2.62, PANEL_W - 0.60, 1.22, BLUE)
    ax.text(x + PANEL_W / 2, top - 1.58, "80,000 tables", ha="center", va="center",
            fontsize=12.5, color=INK, fontweight="bold")
    ax.text(x + PANEL_W / 2, top - 1.80, "drawn once, then frozen", ha="center",
            va="center", fontsize=9.5, color=MUTED)
    for r in range(2):
        for c in range(7):
            ax.add_patch(plt.Rectangle((x + 0.55 + c * 0.44, top - 2.20 - r * 0.28),
                                       0.32, 0.18, fc="white", ec=BLUE, lw=1.0))
    # property chips in one row
    for i, txt in enumerate(["2-60 features", "200 rows", "up to 10 classes"]):
        cx = x + 0.30 + i * 1.20
        rbox(ax, cx, top - 3.12, 1.10, 0.34, "#c7d5f0", fc="#f4f7fd", lw=1.0, r=0.05)
        ax.text(cx + 0.55, top - 2.95, txt, ha="center", va="center", fontsize=8.6, color=INK)
    ax.text(x + PANEL_W / 2, Y0 + 0.15, "every run reads this same pool", ha="center",
            va="center", fontsize=10, color=BLUE, style="italic")


def mini_tables(ax, x, y, n=5):
    # tables of growing width: axis 1 icon
    for i in range(n):
        w = 0.16 + i * 0.11
        ax.add_patch(plt.Rectangle((x, y), w, 0.30, fc="white", ec=ORANGE, lw=1.1))
        ax.add_patch(plt.Rectangle((x, y + 0.22), w, 0.08, fc=ORANGE, ec="none"))
        x += w + 0.13
    return x


def grow_bars(ax, x, y, n=5):
    for i in range(n):
        h = 0.09 + i * 0.075
        ax.add_patch(plt.Rectangle((x, y), 0.22, h, fc=ORANGE, ec="none"))
        x += 0.34
    return x


def panel_difficulty(ax, x):
    top = Y0 + PANEL_H
    header(ax, x, top, "2", "Difficulty scoring", ORANGE)
    ax.text(x + PANEL_W / 2, top - 0.70, "measured per table, never scheduled",
            ha="center", va="center", fontsize=9.5, color=MUTED, style="italic")
    rows = [("axis 1  ·  feature count", "2 → 60"),
            ("axis 2  ·  in-context examples", "20 → 180"),
            ("combined  ·  both axes at once", "w₁ + w₂")]
    ys = [top - 1.00, top - 2.05, top - 3.10]
    for (label, right), yy in zip(rows, ys):
        rbox(ax, x + 0.30, yy - 0.78, PANEL_W - 0.60, 0.80, "#e8c9ae", lw=1.1, r=0.06)
        ax.text(x + 0.45, yy - 0.18, label, ha="left", va="center", fontsize=10,
                color=ORANGE, fontweight="bold")
        ax.text(x + PANEL_W - 0.45, yy - 0.55, right, ha="right", va="center",
                fontsize=9.5, color=MUTED)
    mini_tables(ax, x + 0.50, ys[0] - 0.70)
    grow_bars(ax, x + 0.50, ys[1] - 0.70)
    # combined: growing squares
    sx = x + 0.50
    for i in range(5):
        s = 0.10 + i * 0.06
        ax.add_patch(plt.Rectangle((sx, ys[2] - 0.70), s, s, fc=ORANGE, ec="none"))
        sx += s + 0.16
    ax.text(x + PANEL_W / 2, Y0 + 0.15, "a negative weight reverses an axis", ha="center",
            va="center", fontsize=10, color=ORANGE, style="italic")


def strip(ax, x, y, w, widths, color):
    # a row of segments whose widths encode difficulty
    total = sum(widths) + 0.05 * (len(widths) - 1)
    scale = w / total
    cx = x
    for wd in widths:
        ax.add_patch(plt.Rectangle((cx, y), wd * scale, 0.26, fc=color, ec="none"))
        cx += (wd + 0.05) * scale


def panel_ordering(ax, x):
    top = Y0 + PANEL_H
    header(ax, x, top, "3", "Ordering", GREEN)
    ax.text(x + PANEL_W / 2, top - 0.70, "sort the pool, the only free variable",
            ha="center", va="center", fontsize=9.5, color=MUTED, style="italic")
    rng = np.random.default_rng(3)
    rows = [("CURRICULUM", "ascending  ·  easy → hard", GREEN,
             list(np.linspace(0.3, 1.2, 9))),
            ("REVERSAL CONTROL", "descending  ·  hard → easy", MAROON,
             list(np.linspace(1.2, 0.3, 9))),
            ("BASELINE", "shuffled  ·  random", BLUE,
             list(rng.uniform(0.3, 1.2, 9)))]
    ys = [top - 1.00, top - 1.93, top - 2.86]
    for (tag, sub, color, widths), yy in zip(rows, ys):
        rbox(ax, x + 0.30, yy - 0.68, PANEL_W - 0.60, 0.70, color, lw=1.1, r=0.06)
        ax.text(x + 0.45, yy - 0.16, tag, ha="left", va="center", fontsize=8.5,
                color=color, fontweight="bold")
        ax.text(x + PANEL_W - 0.45, yy - 0.16, sub, ha="right", va="center",
                fontsize=9, color=INK)
        strip(ax, x + 0.45, yy - 0.58, PANEL_W - 0.90, widths, color)
    # which orderings exist
    chips = [("features", "×2"), ("in-context", "×2"), ("combined", "×1"),
             ("sawtooth", "×1"), ("random", "×1")]
    cw = 0.66
    for i, (name, count) in enumerate(chips):
        cx = x + 0.32 + i * (cw + 0.06)
        rbox(ax, cx, top - 4.02, cw, 0.30, "#bcd8c6", fc="#f0f7f2", lw=0.9, r=0.05)
        ax.text(cx + cw / 2, top - 3.87, name, ha="center", va="center", fontsize=7.8, color=INK)
        ax.text(cx + cw / 2, top - 4.14, count, ha="center", va="center", fontsize=7.8, color=MUTED)
    ax.text(x + PANEL_W / 2, Y0 + 0.15, "7 orderings  ·  identical data & compute",
            ha="center", va="center", fontsize=10, color=GREEN, style="italic")


def panel_train(ax, x):
    top = Y0 + PANEL_H
    header(ax, x, top, "4", "Train & evaluate", PURPLE)
    # model box with three layer bars
    rbox(ax, x + 0.30, top - 1.52, PANEL_W - 0.60, 0.96, PURPLE)
    ax.text(x + PANEL_W / 2, top - 0.74, "nanoTabPFN", ha="center", va="center",
            fontsize=12, color=INK, fontweight="bold")
    for i in range(3):
        ax.add_patch(plt.Rectangle((x + 0.75, top - 0.98 - i * 0.13), PANEL_W - 1.50, 0.08,
                                   fc="#c9bce4", ec="none"))
    ax.text(x + PANEL_W / 2, top - 1.44, "published recipe, unchanged", ha="center",
            va="center", fontsize=8.8, color=MUTED)
    ax.add_patch(FancyArrowPatch((x + PANEL_W / 2, top - 1.60), (x + PANEL_W / 2, top - 1.78),
                                 arrowstyle="-|>", mutation_scale=14, color=PURPLE, lw=1.5))
    rbox(ax, x + 0.30, top - 2.28, PANEL_W - 0.60, 0.42, PURPLE)
    ax.text(x + PANEL_W / 2, top - 2.07, "2,500 steps  ×  32  =  one pass", ha="center",
            va="center", fontsize=10.5, color=INK, fontweight="bold")
    # validation box with easy/medium/hard chips
    rbox(ax, x + 0.30, top - 3.22, PANEL_W - 0.60, 0.80, PURPLE)
    ax.text(x + PANEL_W / 2, top - 2.60, "validation  ·  96 held-out tables", ha="center",
            va="center", fontsize=9.5, color=INK)
    for i, (band, shade) in enumerate([("easy", "#ded5f0"), ("medium", "#b9a5dd"),
                                       ("hard", "#6b4fa0")]):
        cx = x + 0.55 + i * 1.05
        ax.add_patch(FancyBboxPatch((cx, top - 3.08), 0.90, 0.30,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=shade, ec="none"))
        ax.text(cx + 0.45, top - 2.93, band, ha="center", va="center", fontsize=8.8,
                color="white" if i == 2 else INK)
    rbox(ax, x + 0.30, top - 4.02, PANEL_W - 0.60, 0.58, PURPLE)
    ax.text(x + PANEL_W / 2, top - 3.63, "TabArena  ·  ROC-AUC", ha="center", va="center",
            fontsize=10.5, color=INK, fontweight="bold")
    ax.text(x + PANEL_W / 2, top - 3.87, "16 tasks  ·  seeds 42/1/2", ha="center",
            va="center", fontsize=9, color=MUTED)
    ax.text(x + PANEL_W / 2, Y0 + 0.15, "identical for every run", ha="center",
            va="center", fontsize=10, color=PURPLE, style="italic")


def build(out_path):
    fig = plt.figure(figsize=(18, 5.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 18)
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
            ax.add_patch(FancyArrowPatch((x + PANEL_W + 0.10, Y0 + PANEL_H / 2),
                                         (x + PANEL_W + GAP - 0.10, Y0 + PANEL_H / 2),
                                         arrowstyle="-|>", mutation_scale=26, color=INK, lw=2.2))

    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    build(pathlib.Path(__file__).parent.parent / "figures" / "pipeline.png")
