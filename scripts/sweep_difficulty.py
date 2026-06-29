"""Which knobs actually make a dataset harder?

We sweep one knob at a time, generate a bunch of datasets at each setting, and
average the measured difficulty. Averaging is the whole point - a single dataset
is far too noisy to trust (we saw that the first time we plotted).

This is the cheap experiment to run before committing GPU time: it tells us which
knobs to build the curriculum on. Run from the project root:

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

from curriculum.difficulty import measured_difficulty
from curriculum.prior import make_prior, sample_dataset

N_PER_SETTING = 30   # datasets averaged per knob value
NUM_DATAPOINTS = 200
SEED = 0

# each sweep holds the other knob fixed and varies one
SWEEPS = [
    {"knob": "max_features", "values": [4, 10, 20, 40, 60], "fixed": {"max_classes": 2}},
    {"knob": "max_classes", "values": [2, 4, 7, 10], "fixed": {"max_features": 20}},
]


def measure(max_features, max_classes):
    """Mean +/- std of measured difficulty over N_PER_SETTING datasets."""
    prior = make_prior(max_features=max_features, max_classes=max_classes,
                       num_datapoints=NUM_DATAPOINTS)
    prior.num_steps = N_PER_SETTING
    scores = [measured_difficulty(*sample_dataset(prior)) for _ in range(N_PER_SETTING)]
    return float(np.mean(scores)), float(np.std(scores))


def main():
    np.random.seed(SEED)

    fig, axes = plt.subplots(1, len(SWEEPS), figsize=(6 * len(SWEEPS), 4), squeeze=False)

    for ax, sweep in zip(axes[0], SWEEPS):
        knob, values, fixed = sweep["knob"], sweep["values"], sweep["fixed"]
        means, stds = [], []
        print(f"\n=== sweeping {knob} (fixed {fixed}) ===")
        for v in values:
            kwargs = {"max_features": v if knob == "max_features" else fixed.get("max_features"),
                      "max_classes": v if knob == "max_classes" else fixed.get("max_classes")}
            m, s = measure(**kwargs)
            means.append(m)
            stds.append(s)
            print(f"  {knob}={v:3d}: difficulty {m:.3f} +/- {s:.3f}")

        ax.errorbar(values, means, yerr=stds, marker="o", capsize=4)
        ax.set_xlabel(knob)
        ax.set_ylabel("measured difficulty")
        ax.set_title(f"difficulty vs {knob}")
        ax.set_ylim(0, 1)

    fig.suptitle(f"Does the knob move difficulty? (avg over {N_PER_SETTING} datasets)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = BASE / "experiments"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "difficulty_sweep.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
