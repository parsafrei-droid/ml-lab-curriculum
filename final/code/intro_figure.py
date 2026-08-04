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

FIG_W, FIG_H = 9.50, 1.85
MID = 0.64          # vertical centre of the flow
HEAD_Y = 1.62
DIVIDER_X = 5.52


def rbox(ax, x, y, w, h, ec, fc="white", lw=1.3, r=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.025,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


def arrow(ax, x0, x1, y, color=INK, lw=1.7, scale=15):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=scale, color=color, lw=lw))


def table_icon(ax, x, y, w, h, color):
    """A little table: outlined body with a filled header strip."""
    ax.add_patch(plt.Rectangle((x, y), w, h, fc="white", ec=color, lw=1.0))
    ax.add_patch(plt.Rectangle((x, y + h - 0.07), w, 0.07, fc=color, ec="none"))


def model_block(ax, x, y, w, h, label, sub=None):
    rbox(ax, x, y, w, h, PURPLE)
    ax.text(x + w / 2, y + h - 0.19, label, ha="center", va="center", fontsize=9.5,
            color=INK, fontweight="bold")
    for i in range(3):
        ax.add_patch(plt.Rectangle((x + 0.13, y + h - 0.36 - i * 0.10), w - 0.26, 0.062,
                                   fc="#c9bce4", ec="none"))
    if sub:
        ax.text(x + w / 2, y + 0.11, sub, ha="center", va="center", fontsize=7.4,
                color=MUTED, style="italic")


def build(out_path):
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # ---------------- left: pretraining ----------------
    ax.text(0.30, HEAD_Y, "Pretraining  ·  once  ·  almost all the compute",
            ha="left", va="center", fontsize=9.8, color=BLUE, fontweight="bold")

    rbox(ax, 0.32, MID - 0.36, 1.02, 0.72, BLUE, fc="#f4f7fd")
    ax.text(0.83, MID + 0.19, "hand-designed", ha="center", va="center", fontsize=7.4,
            color=MUTED)
    ax.text(0.83, MID - 0.01, "prior", ha="center", va="center", fontsize=10.5, color=INK,
            fontweight="bold")
    ax.text(0.83, MID - 0.19, "TabICLv2", ha="center", va="center", fontsize=7.4,
            color=MUTED)
    arrow(ax, 1.41, 1.67, MID, color=BLUE)

    # the stream of synthetic tables, widths varying to hint at the feature axis
    widths = [0.16, 0.26, 0.13, 0.23, 0.18, 0.28]
    x = 1.77
    for w in widths:
        table_icon(ax, x, MID - 0.21, w, 0.42, BLUE)
        x += 0.36
    stream_l, stream_r = 1.77, x - 0.36 + widths[-1]
    mid_stream = (stream_l + stream_r) / 2

    ax.text(mid_stream, MID - 0.36, "millions of synthetic tables", ha="center",
            va="center", fontsize=7.6, color=MUTED)

    # the question: in what order?
    ax.plot([stream_l, stream_r], [MID + 0.30, MID + 0.30], color=ORANGE, lw=1.2)
    for tx in (stream_l, stream_r):
        ax.plot([tx, tx], [MID + 0.24, MID + 0.30], color=ORANGE, lw=1.2)
    ax.text(mid_stream, MID + 0.63, "in what order?", ha="center", va="center",
            fontsize=10.5, color=ORANGE, fontweight="bold")
    ax.text(mid_stream, MID + 0.44, "the one thing we change", ha="center", va="center",
            fontsize=7.4, color=MUTED, style="italic")

    arrow(ax, stream_r + 0.10, stream_r + 0.36, MID, color=BLUE)
    model_block(ax, stream_r + 0.44, MID - 0.40, 1.16, 0.80, "nanoTabPFN", "trained once")

    # ---------------- divider ----------------
    ax.plot([DIVIDER_X, DIVIDER_X], [0.18, 1.42], color="#d9dce1", lw=1.3, ls=(0, (3, 3)))

    # ---------------- right: inference ----------------
    ax.text(5.76, HEAD_Y, "New dataset  ·  one forward pass",
            ha="left", va="center", fontsize=9.8, color=PURPLE, fontweight="bold")

    # dataset: 4 labelled rows + 2 unlabelled rows, x-cells plus a y-cell
    cell_w, cell_h, vgap, hgap = 0.16, 0.108, 0.026, 0.022
    n_cols, n_rows = 4, 6
    grid_h = n_rows * (cell_h + vgap) - vgap
    dx, dy = 5.78, MID - grid_h / 2
    for r in range(n_rows):
        yy = dy + (n_rows - 1 - r) * (cell_h + vgap)
        for c in range(n_cols):
            ax.add_patch(plt.Rectangle((dx + c * (cell_w + hgap), yy), cell_w, cell_h,
                                       fc="#eef3fc", ec=GRID, lw=0.7))
        yx = dx + n_cols * (cell_w + hgap) + 0.05
        if r < 4:
            ax.add_patch(plt.Rectangle((yx, yy), cell_w, cell_h, fc=ORANGE, ec="none"))
        else:
            ax.add_patch(plt.Rectangle((yx, yy), cell_w, cell_h, fc="white", ec=GREY,
                                       lw=0.8))
            ax.text(yx + cell_w / 2, yy + cell_h / 2, "?", ha="center", va="center",
                    fontsize=6.6, color=MUTED, fontweight="bold")
    right_edge = dx + n_cols * (cell_w + hgap) + 0.05 + cell_w

    # legend above the grid: colour plus wording, never colour alone
    for i, (fc, ec, txt) in enumerate([
            (ORANGE, "none", "labelled rows = in-context examples"),
            ("white", GREY, "rows to predict")]):
        ly = dy + grid_h + 0.33 - i * 0.18
        ax.add_patch(plt.Rectangle((dx, ly - 0.042), 0.11, 0.084, fc=fc, ec=ec, lw=0.8))
        ax.text(dx + 0.17, ly, txt, ha="left", va="center", fontsize=7.4, color=INK)

    arrow(ax, right_edge + 0.12, right_edge + 0.38, MID, color=PURPLE)
    model_block(ax, right_edge + 0.46, MID - 0.40, 1.16, 0.80, "nanoTabPFN", "frozen, no gradients")
    px_arrow = right_edge + 1.62
    arrow(ax, px_arrow + 0.08, px_arrow + 0.34, MID, color=PURPLE)

    # predictions: the two unknown cells, now filled
    px = px_arrow + 0.42
    for r in range(2):
        yy = MID - 0.02 + (1 - r) * (cell_h + vgap)
        ax.add_patch(plt.Rectangle((px, yy), cell_w, cell_h, fc=ORANGE, ec="none"))
    ax.text(px + cell_w / 2, MID - 0.22, "predictions", ha="center", va="center",
            fontsize=7.4, color=MUTED)

    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}  ({FIG_W} x {FIG_H} in)")


if __name__ == "__main__":
    build(pathlib.Path(__file__).parent.parent / "figures" / "intro.png")
