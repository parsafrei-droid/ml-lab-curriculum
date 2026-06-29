"""Small helpers around TabICL's prior so the rest of the code stays short.

make_prior() builds a loader pinned to a single difficulty setting, and
sample_dataset() pulls one (X, y) out of it as plain numpy arrays - which is all
the plotting and difficulty code wants.
"""

from tfmplayground.external_priors import TabICLPriorDataLoader


def make_prior(max_features, max_classes, num_datapoints=200, batch_size=1, device="cpu"):
    """A prior fixed at one difficulty.

    We pin min_features == max_features so every dataset has the same width,
    which keeps the easy/hard comparison clean. num_datapoints is the number of
    rows per dataset (the +1 is because TabICL samples the length from a range).
    """
    return TabICLPriorDataLoader(
        num_steps=1,
        batch_size=batch_size,
        num_datapoints_min=num_datapoints,
        num_datapoints_max=num_datapoints + 1,
        min_features=max_features,
        max_features=max_features,
        max_num_classes=max_classes,
        device=device,
    )


def sample_dataset(prior):
    """Grab one dataset from the prior as numpy (X, y).

    X is (rows, features), y is (rows,). The loader already trims X down to the
    features that actually carry information, so what we get back is ready to use.
    """
    batch = next(iter(prior))
    X = batch["x"][0].cpu().numpy()
    y = batch["y"][0].cpu().numpy()
    return X, y
