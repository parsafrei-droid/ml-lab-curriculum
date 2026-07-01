"""Which knobs actually make a dataset harder?

Two experiments, both cheap (no training):

  1. Single-knob sweeps. Vary one knob, hold the rest, average difficulty over
     many datasets. Result (see the figure): every knob comes out roughly FLAT.
     Changing one knob while TabICL's ~16 other hyper-parameters keep sampling
     randomly just buries the signal in per-dataset noise.

  2. Regime contrast. Compare a "narrow" prior (every knob turned down at once)
     against the full default prior. THIS shows a clear gap - so difficulty is a
     property of the whole regime, not any single knob.

We measure difficulty with learnability_difficulty (how badly a cheap kNN does),
not the geometric class-separation score - the geometric one saturated and
couldn't tell any of these apart.

    python scripts/sweep_difficulty.py
"""

import pathlib
import sys

BASE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "TFM-Playground"))
sys.path.insert(0, str(BASE / "tabicl"))
sys.path.insert(0, str(BASE))

import matplotlib.pyplot as plt
import numpy as np

from curriculum.difficulty import learnability_difficulty
from curriculum.prior import make_prior, sample_dataset

N_PER_SETTING = 30
NUM_DATAPOINTS = 200
FIXED_F, FIXED_C = 20, 5
SEED = 0

SWEEPS = [
    ("max_features", [4, 10, 20, 40, 60],
     lambda v: dict(max_features=v, max_classes=FIXED_C)),
    ("max_classes", [2, 4, 7, 10],
     lambda v: dict(max_features=FIXED_F, max_classes=v)),
    ("noise_std", [0.01, 0.05, 0.1, 0.2, 0.3],
     lambda v: dict(max_features=FIXED_F, max_classes=FIXED_C, knobs={"noise_std": v})),
    ("num_layers", [2, 3, 4, 5, 6],
     lambda v: dict(max_features=FIXED_F, max_classes=FIXED_C, knobs={"num_layers": v})),
    ("hidden_dim", [8, 16, 32, 64, 128],
     lambda v: dict(max_features=FIXED_F, max_classes=FIXED_C, knobs={"hidden_dim": v})),
    ("num_causes", [2, 4, 6, 9, 12],
     lambda v: dict(max_features=FIXED_F, max_classes=FIXED_C, knobs={"num_causes": v})),
]

# the two ends of the "whole regime" axis the curriculum actually ramps along
EASY_REGIME = dict(max_features=4, max_classes=2, min_features=2,
                   knobs={"noise_std": 0.02, "num_layers": 2, "hidden_dim": 16, "num_causes": 2})
HARD_REGIME = dict(max_features=60, max_classes=10, min_features=2,
                   knobs={"noise_std": 0.3, "num_layers": 6, "hidden_dim": 128, "num_causes": 12})


def measure(max_features, max_classes, min_features=None, knobs=None, n=N_PER_SETTING):
    """Mean +/- std learnability difficulty over n datasets."""
    prior = make_prior(max_features=max_features, max_classes=max_classes,
                       min_features=min_features or max_features,
                       num_datapoints=NUM_DATAPOINTS, num_steps=n, knobs=knobs)
    scores = [learnability_difficulty(*sample_dataset(prior)) for _ in range(n)]
    return float(np.mean(scores)), float(np.std(scores))


def single_knob_figure():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (knob, values, build) in zip(axes.ravel(), SWEEPS):
        means, stds = [], []
        print(f"\n=== sweeping {knob} ===")
        for v in values:
            m, s = measure(**build(v))
            means.append(m)
            stds.append(s)
            print(f"  {knob}={v}: difficulty {m:.3f} +/- {s:.3f}")
        ax.errorbar(values, means, yerr=stds, marker="o", capsize=4)
        ax.set(xlabel=knob, ylabel="learnability difficulty", title=knob, ylim=(0, 1))
    fig.suptitle(f"Single knobs are flat - none controls difficulty on its own "
                 f"(avg over {N_PER_SETTING} datasets)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(BASE / "experiments" / "difficulty_sweep.png", dpi=120)


def regime_contrast_figure():
    print("\n=== regime contrast (all knobs together) ===")
    labels, means, stds = [], [], []
    for label, cfg in [("easy\n(narrow)", EASY_REGIME), ("hard\n(full default)", HARD_REGIME)]:
        m, s = measure(n=40, **cfg)
        labels.append(label)
        means.append(m)
        stds.append(s)
        print(f"  {label.splitlines()[0]:6s}: difficulty {m:.3f} +/- {s:.3f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, means, yerr=stds, capsize=6, color=["tab:green", "tab:red"])
    ax.set(ylabel="learnability difficulty", ylim=(0, 1),
           title="The whole regime DOES move difficulty")
    fig.tight_layout()
    fig.savefig(BASE / "experiments" / "regime_contrast.png", dpi=120)


def main():
    np.random.seed(SEED)
    (BASE / "experiments").mkdir(exist_ok=True)
    single_knob_figure()
    regime_contrast_figure()
    print("\nsaved -> experiments/difficulty_sweep.png and experiments/regime_contrast.png")


if __name__ == "__main__":
    main()
