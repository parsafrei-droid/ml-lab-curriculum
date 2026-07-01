"""Small helpers around TabICL's prior so the rest of the code stays short.

make_prior() builds a loader pinned to a starting difficulty, and gives it a
*private* copy of the sampled-HP ranges so the curriculum can safely tweak
internal knobs (noise, MLP depth/width, ...) without touching TabICL's globals.

sample_dataset() pulls one (X, y) out as plain numpy - all the plotting and
difficulty code wants.
"""

from copy import deepcopy

from tabicl.prior._prior_config import DEFAULT_SAMPLED_HP
from tfmplayground.external_priors import TabICLPriorDataLoader

from curriculum.scheduler import apply_knobs


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


def sample_dataset(prior):
    """Grab one dataset from the prior as numpy (X, y).

    X is (rows, features), y is (rows,). The loader already trims X down to the
    features that actually carry information, so what we get back is ready to use.
    """
    batch = next(iter(prior))
    X = batch["x"][0].cpu().numpy()
    y = batch["y"][0].cpu().numpy()
    return X, y
