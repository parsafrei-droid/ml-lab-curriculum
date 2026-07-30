"""Plot the Third-Wave 2x2 as two panels sharing an x-axis (wall-clock) but NOT a y-axis.

Our model's val_auc is a synthetic-validation metric (~0.59); modded's is TabArena
average ROC-AUC (~0.807). Putting them on one y-axis would imply a comparison that does
not exist, so each model gets its own panel and the cross-model story is stated as a
speed ratio in the caption instead.
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "Third_Wave" / "results"
TARGET = 0.8068462330697953

BASE_C = "#4C72B0"
CURR_C = "#DD8452"


def main():
    with open(RESULTS / "summary.json") as f:
        data = json.load(f)
    runs = data["runs"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # ---- panel 1: our model (synthetic val_auc) ----
    ax = axes[0]
    for key, label, color in [
        ("run1_ours_baseline", "baseline (random order)", BASE_C),
        ("run2_ours_curriculum", "feature curriculum", CURR_C),
    ]:
        r = runs.get(key)
        if not r:
            continue
        xs = [p["cum_time_s"] for p in r["trajectory"]]
        ys = [p["val_auc"] for p in r["trajectory"]]
        ax.plot(xs, ys, label=label, color=color, lw=1.8)
    ax.set_title("ours (nanoTabPFN, 80k pool)\nsynthetic validation ROC-AUC")
    ax.set_xlabel("wall-clock (s)")
    ax.set_ylabel("val ROC-AUC")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, lw=0.6)

    # ---- panel 2: modded (TabArena ROC-AUC) ----
    ax = axes[1]
    plotted = False
    for key, label, color in [
        ("run3_modded_baseline", "baseline (dump order)", BASE_C),
        ("run4_modded_curriculum", "feature curriculum", CURR_C),
    ]:
        r = runs.get(key)
        if not r:
            continue
        xs = [p["cum_train_s"] for p in r["trajectory"]]
        ys = [p["avg_roc_auc"] for p in r["trajectory"]]
        ax.plot(xs, ys, label=label, color=color, lw=1.8)
        plotted = True
        if r.get("time_to_target_s"):
            ax.axvline(r["time_to_target_s"], color=color, ls=":", lw=1.2)
    if plotted:
        ax.axhline(TARGET, color="0.35", ls="--", lw=1.0)
        ax.annotate(f"target {TARGET:.4f}", (0.02, TARGET), xycoords=("axes fraction", "data"),
                    va="bottom", fontsize=8, color="0.35")
    else:
        ax.text(0.5, 0.5, "runs 3 & 4 pending", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
    ax.set_title("modded-nanoTabPFN (own dump)\nTabArena average ROC-AUC")
    ax.set_xlabel("training time (s)")
    ax.set_ylabel("TabArena ROC-AUC")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, lw=0.6)

    fig.suptitle("Third Wave: feature curriculum vs random order, same A100 "
                 "(separate y-axes: different metrics)", fontsize=11)
    fig.tight_layout()
    out = RESULTS / "third_wave.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
