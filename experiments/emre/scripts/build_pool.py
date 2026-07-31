"""Build a fixed pool of synthetic datasets for curriculum.pool's order-based
training mode (scripts/run_pool.py).

    python scripts/build_pool.py --config experiments/configs/pool_binary_main.yaml

The pool-build config is NOT a training config - it's consumed only here. It
holds a `pool_regimes:` schedule (same shape as a training config's `schedule:`
block: {threshold_step: {knob: value}}) that carves the pool into difficulty
segments, one curriculum.prior.make_prior(knobs=...) call per segment - not a
bare TabICLPriorDataLoader, since internal-knob control (noise_std, num_layers,
hidden_dim) requires make_prior's private sampled_hp copy.

Segment sizes are threshold-deltas x batch_size (the config's own batch_size,
matching what a training run's accumulate_gradients will be - see run_pool.py),
so the pool's item boundaries land exactly on the boundaries training will draw
at, mirroring how curriculum_*.yaml schedules line up with CHECKPOINT_STEPS.

--stop-after-count / --append let a build be split across multiple jobs/chunks:
each call generates up to `count` additional items (skipping regimes already
fully generated) and appends them to the existing pool file, so a slow build
can be resumed rather than restarted.
"""

import argparse
import pathlib
import sys

BASE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "TFM-Playground"))
sys.path.insert(0, str(BASE / "tabicl"))
sys.path.insert(0, str(BASE))

import torch
import yaml

from curriculum.prior import make_prior
from tfmplayground.utils import get_default_device


def _segments(pool_regimes, target_steps, batch_size):
    """{threshold: knobs} -> [(regime_index, knobs, n_items), ...] covering the
    whole pool. threshold deltas x batch_size = items in that segment; the last
    segment runs to target_steps x batch_size (a schedule's implicit "hold until
    the run ends", same convention as curriculum/scheduler.py).

    Each stage's knobs dict in pool_regimes is typically PARTIAL (e.g. a stage
    that only changes noise_std, same as a curriculum_*.yaml schedule stage) -
    curriculum.scheduler.apply_knobs mutates one persistent prior object across
    stages, so a knob not mentioned in a stage simply keeps whatever the previous
    stage last set it to. A fresh make_prior() call per regime has no such
    persistent state, so we have to replicate that accumulation explicitly here:
    each stage's dict is merged ON TOP of the running cumulative knob state
    (later keys override earlier ones, matching apply_knobs' per-key semantics),
    and the FULL accumulated dict - not just that stage's own partial dict - is
    what actually gets built. Skipping this would silently reset every
    unmentioned knob (typically num_layers/hidden_dim on a noise_std-only stage)
    to TabICL's stock default instead of holding the intended prior value."""
    thresholds = sorted(pool_regimes)
    segments = []
    accumulated = {}
    for i, t in enumerate(thresholds):
        accumulated = {**accumulated, **pool_regimes[t]}
        end = thresholds[i + 1] if i + 1 < len(thresholds) else target_steps
        n_items = (end - t) * batch_size
        segments.append((i, dict(accumulated), n_items))
    return segments


def build_regime_pool(out_path, target_steps, batch_size, num_datapoints, max_classes,
                       min_features, max_features, pool_regimes, seed,
                       stop_after_count=None, append=False, device="cpu"):
    out_path = pathlib.Path(out_path)
    items = []
    if append and out_path.exists():
        items = torch.load(out_path, weights_only=False)["items"]
        print(f"resuming: {len(items)} items already in {out_path}", flush=True)

    # Each --append call is a SEPARATE process (one per SLURM chunk). Seeding
    # with the same fixed `seed` every time would restart random/np/torch's
    # global RNG from the identical state on every chunk, instead of a single
    # continuously-advancing stream across the whole build - no segment gets
    # regenerated (the have/cursor skip logic below still prevents that), but
    # different chunks' make_prior() calls would all start from the same RNG
    # position rather than genuinely independent draws. Offsetting by how many
    # items already exist gives each chunk its own deterministic-but-distinct
    # seed, cheaply, with no effect on a non-chunked (single-call) build where
    # `items` is always empty at this point (offset 0, unchanged behavior).
    import numpy as np
    import random
    chunk_seed = seed + len(items)
    random.seed(chunk_seed)
    np.random.seed(chunk_seed)
    torch.manual_seed(chunk_seed)

    segments = _segments(pool_regimes, target_steps, batch_size)
    total_size = sum(n for _, _, n in segments)
    have = len(items)
    # which segment/offset does `have` fall into, so a resumed build picks up
    # mid-segment rather than re-generating from the start of the pool
    cursor = 0
    generated_this_call = 0
    for regime_index, knobs, n_items in segments:
        seg_start, seg_end = cursor, cursor + n_items
        cursor = seg_end
        if have >= seg_end:
            continue  # this whole segment is already built
        need_from = max(have, seg_start)
        need = seg_end - need_from
        if stop_after_count is not None:
            need = min(need, stop_after_count - generated_this_call)
        if need <= 0:
            break

        internal_knobs = {k: v for k, v in knobs.items() if k not in ("min_features", "max_features")}
        prior = make_prior(
            max_features=knobs.get("max_features", max_features),
            max_classes=max_classes,
            min_features=knobs.get("min_features", min_features),
            num_datapoints=num_datapoints,
            num_steps=need * 2,  # headroom: some draws get NaN-filtered below
            batch_size=1, device=device, knobs=internal_knobs or None,
        )
        prior_iter = iter(prior)
        got = 0
        while got < need:
            try:
                b = next(prior_iter)
            except StopIteration:
                # ran out of headroom (unusually high NaN rate for this regime) -
                # rebuild the loader and keep going rather than under-filling
                prior_iter = iter(prior)
                continue
            x, y = b["x"], b["y"]
            if torch.isnan(x).any() or torch.isnan(y).any():
                continue
            items.append({
                "x": x.cpu().contiguous(), "y": y.cpu().contiguous(),
                "split": int(b["train_test_split_index"]),
                "n_features": int(x.shape[2]), "n_classes": int(y.unique().numel()),
                "regime_index": regime_index,
            })
            got += 1
            generated_this_call += 1
        print(f"  regime {regime_index} ({knobs}): +{got} items ({len(items)}/{total_size} total)", flush=True)
        if stop_after_count is not None and generated_this_call >= stop_after_count:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "items": items,
        "meta": {
            "size": len(items), "target_size": total_size, "target_steps": target_steps,
            "batch_size": batch_size, "min_features": min_features, "max_features": max_features,
            "max_classes": max_classes, "num_datapoints": num_datapoints, "seed": seed,
            "regimes": pool_regimes,
        },
    }, out_path)
    done = len(items) >= total_size
    print(f"{'built' if done else 'partial'}: {len(items)}/{total_size} items -> {out_path}", flush=True)
    return len(items), total_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a pool-build YAML")
    parser.add_argument("--append", action="store_true",
                        help="resume/extend an existing pool file instead of overwriting it")
    parser.add_argument("--stop-after-count", type=int, default=None,
                        help="generate at most this many new items this run, then exit (for "
                             "chunking a slow build, or for a timing pilot)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pool_regimes = {int(k): v for k, v in cfg["pool_regimes"].items()}

    n, total = build_regime_pool(
        out_path=BASE / cfg["pool_out"],
        target_steps=cfg["target_steps"],
        batch_size=cfg.get("batch_size", 1),
        num_datapoints=cfg.get("num_datapoints", 200),
        max_classes=cfg.get("max_classes", 2),
        min_features=cfg.get("min_features", 2),
        max_features=cfg.get("max_features", 2),
        pool_regimes=pool_regimes,
        seed=cfg.get("seed", 0),
        stop_after_count=args.stop_after_count,
        append=args.append,
        device=get_default_device(),
    )
    if n < total:
        print(f"=== incomplete: {n}/{total}; re-run with --append to continue ===", flush=True)


if __name__ == "__main__":
    main()
