import argparse
import pathlib
import random
import sys

BASE = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE.parent.parent / "TFM-Playground"))
sys.path.insert(0, str(BASE.parent.parent / "tabicl"))
sys.path.insert(0, str(BASE))

import numpy as np
import torch

from prior import build_loader


# This is the dump. We draw every synthetic table once, up front, and save them in
# generation order. Nothing here is sorted - the ordering happens at training time,
# which is what lets every run share the exact same data.
def build_pool(out_path, size, min_features, max_features, max_classes, num_datapoints, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    loader = build_loader(min_features, max_features, max_classes, num_datapoints,
                          num_steps=size, batch_size=1, device="cpu")
    items = []
    for b in loader:
        x, y = b["x"], b["y"]
        if torch.isnan(x).any() or torch.isnan(y).any():
            continue
        items.append({
            "x": x.cpu().contiguous(),
            "y": y.cpu().contiguous(),
            "split": int(b["train_test_split_index"]),
            "n_features": int(x.shape[2]),
        })
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "items": items,
        "meta": {
            "size": len(items), "min_features": min_features, "max_features": max_features,
            "max_classes": max_classes, "num_datapoints": num_datapoints, "seed": seed,
        },
    }, out_path)
    feats = [it["n_features"] for it in items]
    print(f"built {len(items)} datasets -> {out_path} | features {min(feats)}..{max(feats)} "
          f"mean {sum(feats) / len(feats):.1f}", flush=True)
    return len(items)


def load_pool(path):
    return torch.load(path, weights_only=False)["items"]


AXIS_DEFAULTS = {
    "curriculum": ["features"],
    "curriculum_context": ["context"],
    "curriculum_combined": ["features", "context"],
}


# Two difficulty axes. More features is harder. Fewer in-context examples is harder,
# so we negate the split index to keep "bigger number = harder" on both.
def axis_values(items, axis):
    if axis == "features":
        return [it["n_features"] for it in items]
    if axis == "context":
        return [-it["split"] for it in items]
    raise ValueError(f"unknown axis {axis!r}")


def normalised(values):
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return [0.0] * len(values)
    return [(v - lo) / span for v in values]


def difficulty_scores(items, axes, weights):
    total = [0.0] * len(items)
    for axis, w in zip(axes, weights):
        for i, v in enumerate(normalised(axis_values(items, axis))):
            total[i] += w * v
    return total


# We record the Spearman correlation between training position and each axis, so the
# ordering a run actually got ends up in its meta.json instead of being assumed.
def ordering_profile(items, order):
    from scipy.stats import spearmanr

    position = list(range(len(order)))
    profile = {}
    for axis in ("features", "context"):
        values = axis_values(items, axis)
        profile[axis] = round(float(spearmanr(position, [values[i] for i in order]).statistic), 3)
    return profile


# The only thing that separates our runs: shuffle for the baseline, or sort by a
# weighted difficulty score. A negative weight reverses that axis.
def order_indices(items, mode, seed, restarts=3, axes=None, weights=None):
    idx = list(range(len(items)))
    if mode == "shuffle":
        np.random.default_rng(seed).shuffle(idx)
        return idx
    if mode == "curriculum_restart":
        ordered = sorted(idx, key=lambda i: items[i]["n_features"])
        buckets = [[] for _ in range(restarts)]
        for j, i in enumerate(ordered):
            buckets[j % restarts].append(i)
        idx = []
        for b in buckets:
            idx += sorted(b, key=lambda i: items[i]["n_features"])
        return idx
    if mode not in AXIS_DEFAULTS:
        raise ValueError(f"unknown order {mode!r}")
    use_axes = axes or AXIS_DEFAULTS[mode]
    use_weights = weights or [1.0] * len(use_axes)
    score = difficulty_scores(items, use_axes, use_weights)
    idx.sort(key=lambda i: score[i])
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="pools/main.pt")
    parser.add_argument("--size", type=int, default=80000)
    parser.add_argument("--min-features", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=60)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--num-datapoints", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build_pool(BASE / args.out, args.size, args.min_features, args.max_features,
               args.max_classes, args.num_datapoints, args.seed)


if __name__ == "__main__":
    main()
