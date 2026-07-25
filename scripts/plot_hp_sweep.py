"""Plot scripts/hp_random_search.sh's learning curves and the lr-mistuning trend.

Two figures, written to experiments/:
  hp_sweep_curves.png   - one small-multiple per trial: toy_tabarena.csv's
                           real-OpenML AUC over training steps, baseline vs
                           curriculum. This is the per-checkpoint signal that
                           tracked the final real TabArena result far better
                           than the synthetic val_acc curve did for this
                           sweep (see rank_hp_sweep.py's summary) - it's the
                           one worth looking at to see *when* a run
                           converges/diverges, not just the final number.
  hp_sweep_lr_trend.png - final TabArena mean_roc_auc_binary delta
                          (curriculum - baseline) vs signed log10(lr/optimum),
                          so the too-low (monotonic gain) vs too-high (cliff:
                          rescue right at the edge, no rescue further out)
                          shape is visible in one plot.

Trials are sorted/labelled the same way as scripts/rank_hp_sweep.py.

    python scripts/plot_hp_sweep.py
"""

import csv
import json
import math
import pathlib

import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).parent.parent
RESULTS = BASE / "results"
MANIFEST = BASE / "experiments" / "configs" / "hp_sweep" / "manifest.csv"
OUT = BASE / "experiments"
PAPER_OPTIMUM_LR = 0.003892

BASE_COLOR = "#2a78d6"   # palette slot 1 (blue) - baseline / control, fixed order
CURR_COLOR = "#eb6834"   # palette slot 2 (orange) - curriculum / treatment


def load_toy_curve(name):
    p = RESULTS / name / "toy_tabarena.csv"
    if not p.exists():
        return [], []
    rows = list(csv.reader(p.open()))[1:]
    steps = [int(s) for s, _ in rows]
    aucs = [float(a) for _, a in rows]
    return steps, aucs


def load_tabarena_auc(name):
    p = RESULTS / name / "tabarena_scores.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("mean_roc_auc_binary", d.get("mean_roc_auc"))


def main():
    rows = list(csv.DictReader(MANIFEST.open()))
    trials = []
    for r in rows:
        trial, lr, ndp = int(r["trial"]), float(r["lr"]), int(r["num_datapoints"])
        dist = abs(math.log10(lr) - math.log10(PAPER_OPTIMUM_LR))
        trials.append((dist, trial, lr, ndp))
    trials.sort(key=lambda t: -t[0])

    # --- Figure 1: per-trial toy-TabArena curves, baseline vs curriculum ---
    n = len(trials)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows), sharex=True)
    axes = axes.flatten()
    for ax, (dist, trial, lr, ndp) in zip(axes, trials):
        b_steps, b_aucs = load_toy_curve(f"paper_binary_hp{trial}_s42")
        c_steps, c_aucs = load_toy_curve(f"early_ramp_hp{trial}_s42")
        ax.plot(b_steps, b_aucs, color=BASE_COLOR, lw=2, label="baseline")
        ax.plot(c_steps, c_aucs, color=CURR_COLOR, lw=2, label="curriculum")
        ax.set_title(f"hp{trial}  (lr={lr:.4g}, n={ndp})", fontsize=9)
        ax.set_ylim(0.35, 1.02)
        ax.tick_params(labelsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.supxlabel("training steps")
    fig.supylabel("toy-TabArena ROC-AUC (real OpenML, per checkpoint)")
    fig.suptitle("hp sweep: baseline vs curriculum, sorted by |log10(lr/optimum)| descending", y=1.05, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "hp_sweep_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: final TabArena AUC delta vs signed lr-distance from optimum ---
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    xs, ys, labels = [], [], []
    for dist, trial, lr, ndp in trials:
        b_auc = load_tabarena_auc(f"paper_binary_hp{trial}_s42")
        c_auc = load_tabarena_auc(f"early_ramp_hp{trial}_s42")
        if b_auc is None or c_auc is None:
            continue
        signed_dist = math.log10(lr) - math.log10(PAPER_OPTIMUM_LR)  # + = too high, - = too low
        xs.append(signed_dist)
        ys.append(c_auc - b_auc)
        labels.append(f"hp{trial}")
    colors = [CURR_COLOR if y > 0 else BASE_COLOR for y in ys]
    ax2.axhline(0, color="#888888", lw=1, ls="--")
    ax2.axvline(0, color="#888888", lw=1, ls="--")
    ax2.scatter(xs, ys, c=colors, s=70, zorder=3, edgecolor="white", linewidth=0.5)
    for x, y, lbl in zip(xs, ys, labels):
        ax2.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax2.set_xlabel("log10(lr / paper optimum)  [negative = too low, positive = too high]")
    ax2.set_ylabel("curriculum - baseline  (TabArena mean_roc_auc_binary)")
    ax2.set_title("Curriculum's edge vs how mistuned lr is")
    fig2.tight_layout()
    fig2.savefig(OUT / "hp_sweep_lr_trend.png", dpi=120)
    plt.close(fig2)

    print(f"wrote {OUT / 'hp_sweep_curves.png'}")
    print(f"wrote {OUT / 'hp_sweep_lr_trend.png'}")


if __name__ == "__main__":
    main()
