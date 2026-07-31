"""Plot the hp sweep's learning curves and lr-mistuning trend, multi-seed.

Reads experiments/hp_sweep_data_multiseed.json (built from every
results/{paper_binary,early_ramp}_hp<i>_s<seed>/ run) and writes two figures
to experiments/, both showing mean-across-seeds with the min-max seed range:

  hp_sweep_curves.png   - one small-multiple per trial: toy_tabarena.csv's
                          real-OpenML AUC over training steps, baseline vs
                          curriculum, mean line + shaded seed-range band.
  hp_sweep_lr_trend.png - final TabArena mean_roc_auc_binary delta
                          (curriculum - baseline) vs signed log10(lr/optimum):
                          mean point + min-max whisker per config. The
                          too-low-lr side's consistent gains and the
                          too-high-lr side's seed-flipping (hp0: +0.27 to
                          -0.25) read directly off the whiskers.

Trials are sorted/labelled the same way as scripts/rank_hp_sweep.py.

    python scripts/plot_hp_sweep.py
"""

import json
import math
import pathlib

import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).parent.parent
DATA = BASE / "experiments" / "hp_sweep_data_multiseed.json"
OUT = BASE / "experiments"

BASE_COLOR = "#2a78d6"   # palette slot 1 (blue) - baseline / control, fixed order
CURR_COLOR = "#eb6834"   # palette slot 2 (orange) - curriculum / treatment


def main():
    data = json.loads(DATA.read_text())
    trials = data["trials"]  # already sorted most-lr-mistuned first
    n_seeds = len(data["overall"]["seeds"])

    # --- Figure 1: per-trial toy-TabArena mean curves with seed-range bands ---
    n = len(trials)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows), sharex=True)
    axes = axes.flatten()
    for ax, t in zip(axes, trials):
        steps = t["steps"]
        ax.fill_between(steps, t["base_toy_lo"], t["base_toy_hi"], color=BASE_COLOR, alpha=0.16, linewidth=0)
        ax.fill_between(steps, t["curr_toy_lo"], t["curr_toy_hi"], color=CURR_COLOR, alpha=0.16, linewidth=0)
        ax.plot(steps, t["base_toy_mean"], color=BASE_COLOR, lw=2, label="baseline (mean)")
        ax.plot(steps, t["curr_toy_mean"], color=CURR_COLOR, lw=2, label="curriculum (mean)")
        ax.set_title(f"hp{t['trial']}  (lr={t['lr']:.4g}, n={t['num_datapoints']})", fontsize=9)
        ax.set_ylim(0.35, 1.02)
        ax.tick_params(labelsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.supxlabel("training steps")
    fig.supylabel("toy-TabArena ROC-AUC (real OpenML, per checkpoint)")
    fig.suptitle(f"hp sweep: baseline vs curriculum, mean of {n_seeds} seeds (band = seed min-max), "
                 "sorted by |log10(lr/optimum)| descending", y=1.05, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "hp_sweep_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: final TabArena AUC delta vs signed lr-distance, mean + range ---
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    ax2.axhline(0, color="#888888", lw=1, ls="--")
    ax2.axvline(0, color="#888888", lw=1, ls="--")
    for t in trials:
        if t["delta_mean"] is None:
            continue
        x = t["log_dist"]
        c = CURR_COLOR if t["delta_mean"] > 0 else BASE_COLOR
        ax2.plot([x, x], [t["delta_min"], t["delta_max"]], color=c, lw=2, alpha=0.55, zorder=2)
        ax2.scatter([x], [t["delta_mean"]], c=c, s=70, zorder=3, edgecolor="white", linewidth=0.5)
        ax2.annotate(f"hp{t['trial']}", (x, t["delta_mean"]), textcoords="offset points",
                     xytext=(6, 4), fontsize=8)
    ax2.set_xlabel("log10(lr / paper optimum)  [negative = too low, positive = too high]")
    ax2.set_ylabel("curriculum - baseline  (TabArena mean_roc_auc_binary)")
    ax2.set_title(f"Curriculum's edge vs lr mistuning\n(mean of {n_seeds} seeds, whisker = seed min-max)", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(OUT / "hp_sweep_lr_trend.png", dpi=120)
    plt.close(fig2)

    print(f"wrote {OUT / 'hp_sweep_curves.png'}")
    print(f"wrote {OUT / 'hp_sweep_lr_trend.png'}")


if __name__ == "__main__":
    main()
