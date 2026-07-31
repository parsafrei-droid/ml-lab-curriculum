"""Quick check that the curriculum scheduler really changes the prior.

We walk a handful of global steps through a tiny schedule and, every time the
stage flips, pull a dataset out and print how wide it is. If the curriculum is
wired up correctly the feature count should jump at the thresholds we set.

Run from the project root:
    python scripts/demo_scheduler.py
"""

import pathlib
import sys

BASE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "TFM-Playground"))
sys.path.insert(0, str(BASE / "tabicl"))
sys.path.insert(0, str(BASE))

import numpy as np

from curriculum.prior import make_prior, sample_dataset
from curriculum.scheduler import CurriculumScheduler

# start easy (4 features), then widen at step 5 and again at step 10
SCHEDULE = {
    0:  {"max_features": 4,  "max_classes": 2},
    5:  {"max_features": 20, "max_classes": 5},
    10: {"max_features": 50, "max_classes": 10},
}


def main():
    np.random.seed(0)

    prior = make_prior(max_features=4, max_classes=2, num_datapoints=100)
    scheduler = CurriculumScheduler(prior, SCHEDULE)

    for global_step in range(15):
        scheduler.step(global_step)

        # only sample right after a stage change, to keep this fast
        if global_step in SCHEDULE:
            X, y = sample_dataset(prior)
            print(f"  step {global_step:2d}: sampled X={X.shape}, "
                  f"classes={len(np.unique(y))}")


if __name__ == "__main__":
    main()
