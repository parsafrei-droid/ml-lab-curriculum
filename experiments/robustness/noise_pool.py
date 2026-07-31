import argparse
import copy
import pathlib
import random
import sys

BASE = pathlib.Path(__file__).parent
ROOT = BASE.parent.parent
sys.path.insert(0, str(ROOT / "TFM-Playground"))
sys.path.insert(0, str(ROOT / "tabicl"))
sys.path.insert(0, str(ROOT / "final" / "code"))

import numpy as np
import torch

from prior import build_loader
from tabicl.prior._prior_config import DEFAULT_SAMPLED_HP


# Same idea as the feature pool, but the axis is noise. TabICL samples noise_std from a
# log-scaled range, so we pin both ends of that range per dataset. That way each table has
# one known noise level we can sort by, instead of a level we hope moved.
def build_noise_pool(out_path, size, min_noise, max_noise, min_features, max_features,
                     max_classes, num_datapoints, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    loader = build_loader(min_features, max_features, max_classes, num_datapoints,
                          num_steps=size, batch_size=1, device="cpu")
    loader.pd.prior.sampled_hp = copy.deepcopy(DEFAULT_SAMPLED_HP)
    spec = loader.pd.prior.sampled_hp["noise_std"]

    levels = np.exp(np.random.uniform(np.log(min_noise), np.log(max_noise), size))
    items = []
    stream = iter(loader)
    for level in levels:
        spec["min_mean"] = spec["max_mean"] = float(level)
        b = next(stream)
        x, y = b["x"], b["y"]
        if torch.isnan(x).any() or torch.isnan(y).any():
            continue
        items.append({
            "x": x.cpu().contiguous(),
            "y": y.cpu().contiguous(),
            "split": int(b["train_test_split_index"]),
            "n_features": int(x.shape[2]),
            "noise": float(level),
        })

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "items": items,
        "meta": {
            "size": len(items), "min_noise": min_noise, "max_noise": max_noise,
            "min_features": min_features, "max_features": max_features,
            "max_classes": max_classes, "num_datapoints": num_datapoints, "seed": seed,
        },
    }, out_path)
    noises = [it["noise"] for it in items]
    print(f"built {len(items)} datasets -> {out_path} | noise {min(noises):.5f}..{max(noises):.5f}",
          flush=True)
    return len(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="pools/noise.pt")
    parser.add_argument("--size", type=int, default=80000)
    parser.add_argument("--min-noise", type=float, default=0.0001)
    parser.add_argument("--max-noise", type=float, default=0.3)
    parser.add_argument("--min-features", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=60)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--num-datapoints", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build_noise_pool(BASE / args.out, args.size, args.min_noise, args.max_noise,
                     args.min_features, args.max_features, args.max_classes,
                     args.num_datapoints, args.seed)


if __name__ == "__main__":
    main()
