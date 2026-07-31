"""Look at what the prior actually produces, easy vs hard.

For a few difficulty settings we sample some datasets, squash them down to 2D
with PCA and scatter them coloured by class. The point is a sanity check we can
put on the poster: do the "easy" settings really give cleanly separated classes,
and the "hard" ones overlapping ones?

Run from the project root:
    python scripts/visualize_prior.py
"""

import pathlib
import sys

BASE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "TFM-Playground"))
sys.path.insert(0, str(BASE / "tabicl"))
sys.path.insert(0, str(BASE))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from curriculum.difficulty import difficulty_score, learnability_difficulty
from curriculum.prior import make_prior, sample_dataset

# The stages we want to look at. These mirror the "easy/medium/hard" idea from
# the brief - start small and narrow, grow features and classes from there.
STAGES = [
    {"name": "easy",   "max_features": 4,  "max_classes": 2},
    {"name": "medium", "max_features": 20, "max_classes": 5},
    {"name": "hard",   "max_features": 60, "max_classes": 10},
]

SAMPLES_PER_STAGE = 3   # how many datasets to draw per stage
NUM_DATAPOINTS = 200    # rows per dataset
SEED = 0


def to_2d(X):
    """Project features down to 2D so we can scatter them. PCA if we have to."""
    if X.shape[1] <= 2:
        # pad a single-feature dataset with zeros so it still plots
        if X.shape[1] == 1:
            return np.hstack([X, np.zeros_like(X)])
        return X
    return PCA(n_components=2).fit_transform(X)


def main():
    np.random.seed(SEED)

    fig, axes = plt.subplots(
        len(STAGES), SAMPLES_PER_STAGE,
        figsize=(4 * SAMPLES_PER_STAGE, 4 * len(STAGES)),
        squeeze=False,
    )

    for row, stage in enumerate(STAGES):
        prior = make_prior(
            max_features=stage["max_features"],
            max_classes=stage["max_classes"],
            num_datapoints=NUM_DATAPOINTS,
        )

        # planned difficulty from the knobs - same number for the whole row
        planned = difficulty_score(stage["max_features"], stage["max_classes"])
        print(f"\n=== stage '{stage['name']}'  "
              f"(max_features={stage['max_features']}, max_classes={stage['max_classes']})  "
              f"planned difficulty={planned:.2f} ===")

        for col in range(SAMPLES_PER_STAGE):
            X, y = sample_dataset(prior)
            measured = learnability_difficulty(X, y)

            # show the shapes so we can see the data structure, not just the plot
            print(f"  sample {col}: X={X.shape}, y={y.shape}, "
                  f"classes={len(np.unique(y))}, measured difficulty={measured:.2f}")

            P = to_2d(X)
            ax = axes[row][col]
            ax.scatter(P[:, 0], P[:, 1], c=y, cmap="tab10", s=12, alpha=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{stage['name']}\nplanned {planned:.2f}", fontsize=11)
            ax.set_title(f"measured {measured:.2f}", fontsize=10)

    fig.suptitle("TabICL prior: easy -> hard (2D PCA, coloured by class)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir = BASE / "experiments"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "prior_visualization.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nsaved figure -> {out_path}")


if __name__ == "__main__":
    main()
