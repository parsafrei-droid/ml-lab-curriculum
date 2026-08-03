"""The Introduction schematic: what a tabular foundation model does, and where our
question sits.

Left: pretraining happens once and carries almost all the compute; the synthetic tables
arrive in some order, and that order is the free variable we study. Right: afterwards a
new dataset is solved in a single forward pass, with no gradient steps.

Deliberately pictorial (table icons, labelled vs unlabelled rows) so it does not read as
a second copy of the Method pipeline, which is a box-and-bullet diagram.
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1f2328"
MUTED = "#6b7280"
BLUE, ORANGE, PURPLE = "#3b6fd4", "#e8862e", "#6b4fa0"
GRID = "#c7d5f0"
GREY = "#b6bcc4"

FIG_W, FIG_H = 13.55, 2.75
MID = 1.25          # vertical centre line of the flow
HEAD_Y = 2.52


def rbox(ax, x, y, w, h, ec, fc="white", lw=1.4, r=0.07):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.03,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


def arrow(ax, x0, x1, y, color=INK, lw=1.9, scale=18):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=scale, color=color, lw=lw))


def table_icon(ax, x, y, w, h, color):
    """A little table: outlined body with a filled header strip."""
    ax.add_patch(plt.Rectangle((x, y), w, h, fc="white", ec=color, lw=1.2))
    ax.add_patch(plt.Rectangle((x, y + h - 0.09), w, 0.09, fc=color, ec="none"))


def model_block(ax, x, y, w, h, label, sub=None):
    rbox(ax, x, y, w, h, PURPLE)
    ax.text(x + w / 2, y + h - 0.24, label, ha="center", va="center", fontsize=11,
            color=INK, fontweight="bold")
    for i in range(3):
        ax.add_patch(plt.Rectangle((x + 0.16, y + h - 0.46 - i * 0.13), w - 0.32, 0.08,
                                   fc="#c9bce4", ec="none"))
    if sub:
        ax.text(x + w / 2, y + 0.13, sub, ha="center", va="center", fontsize=8.4,
                color=MUTED, style="italic")


def build(out_path):
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # ---------------- left: pretraining ----------------
    ax.text(0.30, HEAD_Y, "Pretraining  ·  once  ·  carries almost all the compute",
            ha="left", va="center", fontsize=11.5, color=BLUE, fontweight="bold")

    rbox(ax, 0.35, MID - 0.40, 1.30, 0.80, BLUE, fc="#f4f7fd")
    ax.text(1.00, MID + 0.22, "hand-designed", ha="center", va="center", fontsize=8.6,
            color=MUTED)
    ax.text(1.00, MID, "prior", ha="center", va="center", fontsize=12, color=INK,
            fontweight="bold")
    ax.text(1.00, MID - 0.19, "TabICLv2", ha="center", va="center", fontsize=8.6,
            color=MUTED)
    arrow(ax, 1.75, 2.10, MID, color=BLUE)

    # the stream of synthetic tables, widths varying to hint at the feature axis
    widths = [0.20, 0.32, 0.16, 0.28, 0.22, 0.35, 0.18, 0.30]
    x = 2.22
    for w in widths:
        table_icon(ax, x, MID - 0.25, w, 0.50, BLUE)
        x += 0.42
    stream_l, stream_r = 2.22, x - 0.42 + widths[-1]
    mid_stream = (stream_l + stream_r) / 2

    ax.text(mid_stream, MID - 0.47, "millions of synthetic tables", ha="center",
            va="center", fontsize=9, color=MUTED)

    # the question: in what order?
    ax.plot([stream_l, stream_r], [MID + 0.37, MID + 0.37], color=ORANGE, lw=1.4)
    for tx in (stream_l, stream_r):
        ax.plot([tx, tx], [MID + 0.29, MID + 0.37], color=ORANGE, lw=1.4)
    ax.text(mid_stream, MID + 0.75, "in what order?", ha="center", va="center",
            fontsize=13, color=ORANGE, fontweight="bold")
    ax.text(mid_stream, MID + 0.51, "the one thing we change", ha="center", va="center",
            fontsize=8.8, color=MUTED, style="italic")

    arrow(ax, 5.72, 6.07, MID, color=BLUE)
    model_block(ax, 6.17, MID - 0.48, 1.32, 0.96, "nanoTabPFN", "trained once")

    # ---------------- divider ----------------
    ax.plot([7.85, 7.85], [0.40, 2.42], color="#d9dce1", lw=1.4, ls=(0, (3, 3)))

    # ---------------- right: inference ----------------
    ax.text(8.15, HEAD_Y, "New dataset  ·  one forward pass, no gradients",
            ha="left", va="center", fontsize=11.5, color=PURPLE, fontweight="bold")

    # dataset: 4 labelled rows + 2 unlabelled rows, x-cells plus a y-cell
    cell_w, cell_h, vgap = 0.20, 0.145, 0.035
    n_cols, n_rows = 4, 6
    grid_h = n_rows * (cell_h + vgap) - vgap
    dx, dy = 8.20, MID - grid_h / 2
    for r in range(n_rows):
        yy = dy + (n_rows - 1 - r) * (cell_h + vgap)
        for c in range(n_cols):
            ax.add_patch(plt.Rectangle((dx + c * (cell_w + 0.03), yy), cell_w, cell_h,
                                       fc="#eef3fc", ec=GRID, lw=0.8))
        yx = dx + n_cols * (cell_w + 0.03) + 0.08
        if r < 4:
            ax.add_patch(plt.Rectangle((yx, yy), cell_w, cell_h, fc=ORANGE, ec="none"))
        else:
            ax.add_patch(plt.Rectangle((yx, yy), cell_w, cell_h, fc="white", ec=GREY,
                                       lw=0.9))
            ax.text(yx + cell_w / 2, yy + cell_h / 2, "?", ha="center", va="center",
                    fontsize=8.5, color=MUTED, fontweight="bold")
    right_edge = dx + n_cols * (cell_w + 0.03) + 0.08 + cell_w

    # legend above the grid, clear of it: colour plus wording, never colour alone
    for i, (fc, ec, txt) in enumerate([
            (ORANGE, "none", "labelled rows, given as in-context examples"),
            ("white", GREY, "rows to predict")]):
        ly = dy + grid_h + 0.40 - i * 0.22
        ax.add_patch(plt.Rectangle((dx, ly - 0.055), 0.15, 0.11, fc=fc, ec=ec, lw=0.9))
        ax.text(dx + 0.23, ly, txt, ha="left", va="center", fontsize=8.6, color=INK)

    ax.text(dx, dy - 0.22, "no gradient steps on this data", ha="left", va="center",
            fontsize=8.6, color=MUTED, style="italic")

    arrow(ax, right_edge + 0.16, right_edge + 0.52, MID, color=PURPLE)
    model_block(ax, right_edge + 0.62, MID - 0.48, 1.32, 0.96, "nanoTabPFN", "frozen")
    arrow(ax, right_edge + 2.10, right_edge + 2.46, MID, color=PURPLE)

    # predictions: the two unknown cells, now filled
    px = right_edge + 2.56
    for r in range(2):
        yy = MID - 0.02 + (1 - r) * (cell_h + vgap)
        ax.add_patch(plt.Rectangle((px, yy), cell_w, cell_h, fc=ORANGE, ec="none"))
    ax.text(px + cell_w / 2, MID - 0.28, "predictions", ha="center", va="center",
            fontsize=8.8, color=MUTED)

    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    build(pathlib.Path(__file__).parent.parent / "figures" / "intro.png")
