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


def set_max_features(loader, value):
    for target in (loader, loader.pd, loader.pd.prior):
        if hasattr(target, "max_features"):
            setattr(target, "max_features", value)


def next_batch(loader):
    b = next(iter(loader))
    return b["x"], b["y"], b["train_test_split_index"]


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
