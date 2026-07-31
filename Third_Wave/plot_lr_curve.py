import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).parent

LR = 0.00392393
PAPER_LR = 0.003892

STEPS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
         1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]

BASE_MEAN = [0.3624, 0.6883, 0.7509, 0.8153, 0.7037, 0.7280, 0.7796, 0.7751,
             0.8629, 0.8986, 0.8920, 0.9127, 0.9193, 0.9330, 0.9678, 0.9810,
             0.9841, 0.9859, 0.9885, 0.9921, 0.9912, 0.9921, 0.9925, 0.9925, 0.9943]
BASE_LO = [0.1495, 0.2183, 0.3981, 0.5926, 0.2500, 0.3267, 0.4749, 0.4497,
           0.7090, 0.8333, 0.8122, 0.8690, 0.8743, 0.8955, 0.9458, 0.9590,
           0.9722, 0.9683, 0.9788, 0.9894, 0.9828, 0.9841, 0.9868, 0.9854, 0.9907]
BASE_HI = [0.7288, 0.9630, 0.9841, 0.9881, 0.9934, 0.9974, 0.9987, 0.9921,
           0.9934, 0.9934, 0.9960, 0.9974, 0.9974, 0.9974, 0.9960, 0.9960,
           0.9960, 0.9960, 0.9947, 0.9947, 0.9960, 0.9974, 0.9960, 0.9974, 0.9974]

CURR_MEAN = [0.6296, 0.9682, 0.9766, 0.9797, 0.9810, 0.9943, 0.9947, 0.9952,
             0.9943, 0.9930, 0.9934, 0.9938, 0.9943, 0.9938, 0.9943, 0.9947,
             0.9943, 0.9952, 0.9947, 0.9938, 0.9934, 0.9947, 0.9934, 0.9947, 0.9947]
CURR_LO = [0.0807, 0.9537, 0.9365, 0.9497, 0.9497, 0.9881, 0.9921, 0.9921,
           0.9921, 0.9921, 0.9921, 0.9921, 0.9921, 0.9921, 0.9934, 0.9934,
           0.9907, 0.9921, 0.9907, 0.9907, 0.9934, 0.9934, 0.9921, 0.9921, 0.9934]
CURR_HI = [0.9114, 0.9907, 0.9987, 0.9960, 0.9987, 0.9987, 0.9974, 0.9987,
           0.9987, 0.9947, 0.9960, 0.9960, 0.9960, 0.9960, 0.9960, 0.9960,
           0.9974, 0.9974, 0.9987, 0.9960, 0.9934, 0.9960, 0.9947, 0.9987, 0.9960]

BASELINE_COLOR = "#3b6fd4"
CURRICULUM_COLOR = "#e8862e"

NOTE = [
    "Learning-rate sweep (this panel: hp5)",
    "",
    f"lr = {LR:.4g}  -  the sampled rate",
    f"closest to the paper's optimum ({PAPER_LR:.4g}).",
    "",
    "Setup: exact paper baseline, 2500 steps,",
    "binary, TabArena eval. 10 random learning",
    "rates from the paper's own search space,",
    "each trained twice (baseline vs early-ramp",
    "curriculum), 3 seeds. Band = seed min-max.",
    "",
    "At a well-tuned lr:",
    "- Curriculum reaches ~0.97 AUC by step 200",
    "  and stays there.",
    "- Baseline is erratic early and only catches",
    "  up near the end.",
    "- Both finish ~0.99 (final delta +0.003) - so",
    "  here the curriculum buys faster, steadier",
    "  convergence, not a higher ceiling.",
    "",
    "The large rescues appear at badly-tuned (low)",
    "learning rates, not at this one.",
]


def make(fname):
    fig = plt.figure(figsize=(12, 5.2))
    ax = fig.add_axes([0.07, 0.13, 0.5, 0.76])

    ax.fill_between(STEPS, BASE_LO, BASE_HI, color=BASELINE_COLOR, alpha=0.15, lw=0)
    ax.fill_between(STEPS, CURR_LO, CURR_HI, color=CURRICULUM_COLOR, alpha=0.15, lw=0)
    ax.plot(STEPS, BASE_MEAN, color=BASELINE_COLOR, lw=2.4, marker="o", ms=3,
            mec="none", label="Baseline")
    ax.plot(STEPS, CURR_MEAN, color=CURRICULUM_COLOR, lw=2.4, marker="o", ms=3,
            mec="none", label="Curriculum")

    ax.set_xlim(0, 2500)
    ax.set_ylim(0.35, 1.02)
    ax.set_xlabel("Training step")
    ax.set_ylabel("toy-TabArena ROC-AUC (real OpenML)")
    ax.set_title("At the paper's learning rate: the curriculum converges almost\n"
                 "immediately, the baseline catches up only late", fontsize=11)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.13)

    fig.text(0.61, 0.92, "\n".join(NOTE), va="top", ha="left",
             fontsize=9.5, family="monospace")

    out = BASE / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / fname, dpi=140)
    plt.close(fig)
    print(f"saved -> {out / fname}")


if __name__ == "__main__":
    make("lr_curve.png")
