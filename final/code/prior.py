import random

import numpy as np
import torch
from tfmplayground.external_priors import TabICLPriorDataLoader

FEATURE_BANDS = {"easy": (2, 10), "medium": (10, 30), "hard": (30, 60)}


def build_loader(min_features, max_features, max_classes, num_datapoints,
                 num_steps=1, batch_size=1, device="cpu"):
    return TabICLPriorDataLoader(
        num_steps=num_steps,
        batch_size=batch_size,
        num_datapoints_min=num_datapoints,
        num_datapoints_max=num_datapoints + 1,
        min_features=min_features,
        max_features=max_features,
        max_num_classes=max_classes,
        device=device,
    )


# The same validation set for every run, split into three feature bands so we can
# see whether a run got better on easy tables, hard ones, or both. We seed the
# three RNGs TabICL touches and put them back afterwards, so building it never
# shifts the training stream.
def build_validation(device, max_classes, num_datapoints=200, per_band=32, seed=12345):
    np_state, torch_state, py_state = np.random.get_state(), torch.get_rng_state(), random.getstate()
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        bands = {}
        for name, (lo, hi) in FEATURE_BANDS.items():
            loader = build_loader(lo, hi, max_classes, num_datapoints,
                                  num_steps=per_band, batch_size=1, device=device)
            bands[name] = [(b["x"], b["y"], b["train_test_split_index"]) for b in loader]
        return bands
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        random.setstate(py_state)
