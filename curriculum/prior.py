"""Small helpers around TabICL's prior so the rest of the code stays short.

make_prior() builds a loader pinned to a starting difficulty, and gives it a
*private* copy of the sampled-HP ranges so the curriculum can safely tweak
internal knobs (noise, MLP depth/width, ...) without touching TabICL's globals.

sample_dataset() pulls one (X, y) out as plain numpy - all the plotting and
difficulty code wants.
"""

from copy import deepcopy

import numpy as np
import torch
from tabicl.prior._prior_config import DEFAULT_SAMPLED_HP
from tfmplayground.external_priors import TabICLPriorDataLoader

from curriculum.scheduler import apply_knobs

# The "full" prior regime - what baseline trains on and every curriculum ends at.
# We use it to build the shared validation set below.
FULL_REGIME = dict(
    max_features=60, max_classes=10, min_features=2,
    knobs={"noise_std": 0.3, "num_layers": 6, "hidden_dim": 128, "num_causes": 12},
)


def make_prior(max_features, max_classes, min_features=2, num_datapoints=200,
               num_steps=1, batch_size=1, device="cpu", knobs=None):
    """A TabICL prior loader, ready for the curriculum to drive.

    max_features / max_classes set the starting difficulty. `knobs` is an
    optional {knob: value} dict applied on top (e.g. {"noise_std": 0.05}).
    num_datapoints is the rows per dataset (the +1 is because TabICL samples the
    length from a range).
    """
    prior = TabICLPriorDataLoader(
        num_steps=num_steps,
        batch_size=batch_size,
        num_datapoints_min=num_datapoints,
        num_datapoints_max=num_datapoints + 1,
        min_features=min_features,
        max_features=max_features,
        max_num_classes=max_classes,
        device=device,
    )
    # Give this loader its own copy of the sampled-HP ranges. TabICL's default is
    # a shared module-level dict; mutating that would leak across every prior, so
    # we deep-copy it here and let the scheduler edit this private version.
    prior.pd.prior.sampled_hp = deepcopy(DEFAULT_SAMPLED_HP)

    if knobs:
        apply_knobs(prior, knobs)
    return prior


def make_validation_batches(device, n=16, num_datapoints=200, seed=12345, max_classes=10):
    """A FIXED set of validation datasets, identical for every training run.

    This is the whole point of comparable evaluation: training loss can't be
    compared across scenarios because each ends on different-difficulty data, so
    a curriculum that finishes on easy data looks "better" for free. Instead we
    score every model on this one shared set each epoch.

    max_classes must not exceed the model's number of outputs, or the labels
    won't fit - so we cap it to num_outputs at the call site. We seed the RNG to
    a fixed value so all runs get the *same* validation data, and restore it
    afterwards so training reproducibility is untouched. Returns a list of
    (x, y, train_test_split_index) with tensors on `device`.
    """
    regime = dict(FULL_REGIME)
    regime["max_classes"] = max_classes
    np_state, torch_state = np.random.get_state(), torch.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        prior = make_prior(
            max_features=regime["max_features"], max_classes=regime["max_classes"],
            min_features=regime.get("min_features", 2), num_datapoints=num_datapoints,
            num_steps=n, batch_size=1, device=device, knobs=regime.get("knobs"),
        )
        batches = []
        for _ in range(n):
            b = next(iter(prior))
            batches.append((b["x"], b["y"], b["train_test_split_index"]))
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
    return batches


def sample_dataset(prior):
    """Grab one dataset from the prior as numpy (X, y).

    X is (rows, features), y is (rows,). The loader already trims X down to the
    features that actually carry information, so what we get back is ready to use.
    """
    batch = next(iter(prior))
    X = batch["x"][0].cpu().numpy()
    y = batch["y"][0].cpu().numpy()
    return X, y
