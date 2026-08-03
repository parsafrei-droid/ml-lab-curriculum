import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = pathlib.Path(__file__).parent
POOL = BASE.parent / "Second_Wave" / "results"

# Blue/orange pair validated for colour-vision deficiency (all checks pass). The orange
# is below 3:1 contrast on white, so both curves carry direct ink labels too.
RUNS = {
    "baseline": (POOL / "baseline_a100_s42" / "log.csv", "#3b6fd4", "Baseline"),
    "curriculum": (POOL / "curriculum_features_a100_s42" / "log.csv", "#e8862e", "Curriculum"),
}

INK = "#1f2328"
MUTED = "#6b7280"


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def series(rows, xcol):
    x = np.array([float(r[xcol]) for r in rows])
    y = np.array([float(r["val_auc"]) for r in rows])
    return x, y


def first_reach(x, y, target):
    for xi, yi in zip(x, y):
        if yi >= target:
            return xi
    return None


def make(xcol, xlabel, fname, unit, scale=1.0, fig_label="Figure 1"):
    curves = {k: series(read(v[0]), xcol) for k, v in RUNS.items()}
    bx, by = curves["baseline"]
    cx, cy = curves["curriculum"]
    target = by.max()
    t_base = first_reach(bx, by, target)
    t_curr = first_reach(cx, cy, target)

    fig = plt.figure(figsize=(9.0, 5.2))
    ax = fig.add_axes([0.10, 0.17, 0.86, 0.71])

    for k, (x, y) in curves.items():
        _, color, label = RUNS[k]
        ax.plot(x / scale, y, color=color, lw=2.4, label=label,
                marker="o", ms=3, mfc=color, mec="none")

    # direct labels in ink, where the curves are far apart
    i_c = len(cx) // 3
    ax.annotate("Curriculum", (cx[i_c] / scale, cy[i_c]), xytext=(-8, 10),
                textcoords="offset points", ha="right", fontsize=10, color=INK)
    i_b = len(bx) // 2
    ax.annotate("Baseline", (bx[i_b] / scale, by[i_b]), xytext=(10, -14),
                textcoords="offset points", ha="left", fontsize=10, color=INK)

    ax.axhline(target, ls=":", c="#999999", lw=1)
    ax.annotate("", xy=(t_base / scale, target), xytext=(t_curr / scale, target),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.6))
    gap = (t_base - t_curr) / scale
    factor = t_base / t_curr
    gap_txt = f"{gap:.0f}" if gap >= 10 else f"{gap:.2f}"
    mid = 0.5 * (t_base + t_curr) / scale
    ax.text(mid, target + 0.004,
            f"reaches target {gap_txt} {unit} earlier  ({factor:.1f}x)",
            ha="center", va="bottom", fontsize=10, color="#333333")
    for tx in (t_curr, t_base):
        ax.plot([tx / scale, tx / scale], [by.min() - 0.01, target],
                ls=":", c="#bbbbbb", lw=1, zorder=0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("ROC-AUC (validation)")
    ax.set_title("Same pool, same total compute: the curriculum reaches the\n"
                 f"baseline's best quality {factor:.1f}x sooner", fontsize=11)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.13)

    fig.text(0.10, 0.03,
             f"{fig_label} - validation ROC-AUC vs {xlabel.lower().split(' (')[0]} (seed 42, A100)",
             va="bottom", ha="left", fontsize=9.5, color=MUTED)

    out = BASE / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / fname, dpi=140)
    plt.close(fig)
    print(f"saved -> {out / fname}  (target={target:.4f}, "
          f"curr={t_curr:.1f}, base={t_base:.1f}, factor={factor:.2f})")


def main():
    make("cum_time_s", "Pretraining time (s)", "time_curve.png", "s")
    make("cum_flops", "Pretraining FLOPs", "flops_curve.png", "PFLOP", scale=1e15,
         fig_label="Figure 1 (FLOPs view)")


if __name__ == "__main__":
    main()
