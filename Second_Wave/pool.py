import argparse
import pathlib
import sys

BASE = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE.parent / "TFM-Playground"))
sys.path.insert(0, str(BASE.parent / "tabicl"))
sys.path.insert(0, str(BASE))

import numpy as np
import torch

from prior import build_loader


def build_pool(out_path, size, min_features, max_features, max_classes, num_datapoints, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
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


def order_indices(items, mode, seed):
    idx = list(range(len(items)))
    if mode == "curriculum":
        idx.sort(key=lambda i: items[i]["n_features"])
    elif mode == "shuffle":
        np.random.default_rng(seed).shuffle(idx)
    else:
        raise ValueError(f"unknown order {mode!r}")
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
