"""Path vs endpoint: which part of an ordering carries the gain?

Seven arms, five seeds each, all on the same rebuilt pool at identical compute. Each row
pairs the SHAPE of the ordering (mean features per step, read straight from log.csv) with
the score it produced, so the reader can see the design and the outcome together.

The arms that begin on the smallest tables and work upward all land near 0.801. The arms
that begin on a shuffled mixture and only sort the tail all land near 0.795, however long
they hold the top of the range. Reading the shapes is the argument: the ending is
disposable, the early climb is not.

Data: experiments/robustness/runs_path and runs_endpoint (branch path-endpoint).
"""

import csv
import io
import json
import pathlib
import statistics
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1f2328"
MUTED = "#6b7280"
RULE = "#d9dce1"
# Validated pair: CVD dE 26.7, contrast >= 3:1 on white. Group identity is also carried
# by the block headings and the row order, never by colour alone.
CLIMB = "#c2621a"
NOCLIMB = "#3b6fd4"

REPO = pathlib.Path(__file__).resolve().parents[2]   # the ml-lab-curriculum checkout
REF = "origin/path-endpoint"
SEEDS = ("s42", "s1", "s2", "s3", "s4")

# (folder, arm, label, note) in the order they should be drawn
GROUPS = [
    ("Starts at the bottom, climbs upward", CLIMB, [
        ("runs_path", "full_sort", "Full sort", "one smooth climb, 2 to 60 features"),
        ("runs_path", "coarse_4", "Coarse staircase", "four blocks instead of a sort"),
        ("runs_path", "sorted_head", "Climb, ending removed", "last 700 steps shuffled"),
    ]),
    ("Shuffled start, only the tail sorted", NOCLIMB, [
        ("runs_endpoint", "tail_300", "Top held 300 steps", "sorted tail only"),
        ("runs_endpoint", "tail_700", "Top held 700 steps", "sorted tail only"),
        ("runs_endpoint", "tail_700_pure", "Top held 700, pure", "sorted tail only"),
        ("runs_endpoint", "tail_1300", "Top held 1300 steps", "sorted tail only"),
    ]),
]

BASELINE = 0.7911   # shuffled order, original pool, 3 seeds


def git_show(path):
    out = subprocess.run(["git", "-C", str(REPO), "show", f"{REF}:{path}"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"cannot read {path}: {out.stderr.strip()}")
    return out.stdout


def arm_scores(folder, arm):
    vals = []
    for s in SEEDS:
        doc = json.loads(git_show(f"experiments/robustness/{folder}/{arm}_{s}/tabarena_scores.json"))
        per = [v["roc_auc"] if isinstance(v, dict) else v for v in doc["per_dataset"].values()]
        per = [v for v in per if v is not None]
        vals.append(statistics.mean(per))
    return vals


def arm_trajectory(folder, arm):
    rows = list(csv.DictReader(io.StringIO(
        git_show(f"experiments/robustness/{folder}/{arm}_s42/log.csv"))))
    return ([int(r["step"]) for r in rows], [float(r["mean_features"]) for r in rows])


def build(out_path):
    data = {}
    for _, _, arms in GROUPS:
        for folder, arm, _, _ in arms:
            data[arm] = (arm_scores(folder, arm), arm_trajectory(folder, arm))

    row_h = 0.56
    head_block = 0.45 + 0.34 + 0.50 + 0.30       # title, subtitle, axis, gap
    groups_h = sum(0.78 + len(a) * row_h for _, _, a in GROUPS)
    fig_h = head_block + groups_h + 1.05         # + baseline label and footer
    fig = plt.figure(figsize=(9.60, fig_h), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 9.60)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    x_name = 0.20                       # left edge of the labels
    spark_l, spark_r = 2.55, 3.95       # sparkline band
    axis_l, axis_r = 4.55, 9.20         # score axis
    lo, hi = 0.780, 0.820               # score axis limits

    def sx(v):
        return axis_l + (v - lo) / (hi - lo) * (axis_r - axis_l)

    y = fig_h - 0.45
    ax.text(x_name, y, "Is it where training ends, or how it gets there?",
            fontsize=15, fontweight="bold", color=INK, va="center")
    y -= 0.34
    ax.text(x_name, y, "7 orderings  ·  5 seeds each  ·  same pool, same compute  ·  "
            "dot = mean, whisker = sd, small dots = seeds",
            fontsize=10, color=MUTED, va="center")
    y -= 0.50

    # score axis, drawn once at the top
    ax.plot([axis_l, axis_r], [y, y], color=INK, lw=1.2)
    for v in (0.780, 0.790, 0.800, 0.810, 0.820):
        ax.plot([sx(v), sx(v)], [y, y + 0.06], color=INK, lw=1.0)
        ax.text(sx(v), y + 0.17, f"{v:.3f}", fontsize=9, color=MUTED, ha="center", va="center")
    ax.text(spark_l, y + 0.17, "shape of the ordering", fontsize=9.5, color=MUTED,
            ha="left", va="center")
    y -= 0.30

    first_row_y, last_row_y = None, None
    for title, color, arms in GROUPS:
        y -= 0.34
        ax.text(x_name, y, title, fontsize=12, fontweight="bold", color=color, va="center")
        y -= 0.44

        for folder, arm, label, note in arms:
            vals, (steps, feats) = data[arm]
            m, sd = statistics.mean(vals), statistics.stdev(vals)
            if first_row_y is None:
                first_row_y = y
            last_row_y = y

            ax.text(x_name, y + 0.09, label, fontsize=11.5, color=INK, va="center")
            ax.text(x_name, y - 0.14, note, fontsize=9, color=MUTED, va="center")

            # sparkline: mean features per step, same scale for every row
            sy0, sy1 = y - 0.19, y + 0.19
            ax.plot([spark_l, spark_r], [sy0, sy0], color=RULE, lw=0.8)
            xs = [spark_l + (s / max(steps)) * (spark_r - spark_l) for s in steps]
            ys = [sy0 + (f / 60.0) * (sy1 - sy0) for f in feats]
            ax.fill_between(xs, sy0, ys, color=color, alpha=0.18, linewidth=0)
            ax.plot(xs, ys, color=color, lw=1.6)

            # per-seed dots, then the mean with an sd whisker
            for v in vals:
                ax.plot([sx(v)], [y], marker="o", ms=3.4, mfc="white", mec=color,
                        mew=1.0, zorder=3)
            ax.plot([sx(m - sd), sx(m + sd)], [y, y], color=color, lw=1.5, zorder=4)
            ax.plot([sx(m)], [y], marker="o", ms=8, mfc=color, mec="white", mew=1.2,
                    zorder=5)
            ax.text(sx(m), y + 0.22, f"{m:.3f}", fontsize=10.5, color=color, ha="center",
                    va="center", fontweight="bold")
            y -= row_h

    # the shuffled baseline, drawn behind the marks as a reference
    ax.plot([sx(BASELINE), sx(BASELINE)], [last_row_y - 0.30, first_row_y + 0.42],
            color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(sx(BASELINE), last_row_y - 0.40, "shuffled baseline", fontsize=9,
            color=MUTED, ha="center", va="center")

    ax.text(x_name, last_row_y - 0.62,
            "Holding the top of the range longer buys nothing (0.795 either way), and "
            "deleting the ending costs nothing (-0.001 ± 0.006).\nWhat separates the two "
            "blocks is the early climb from the smallest tables.",
            fontsize=10, color=MUTED, va="top", linespacing=1.5)

    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    build(pathlib.Path(__file__).parent.parent / "figures" / "path_endpoint.png")
