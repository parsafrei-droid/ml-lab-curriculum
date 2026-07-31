"""A second curriculum mechanism: order, not live ramping.

curriculum.scheduler.CurriculumScheduler changes the prior's own knobs mid
training - every batch is freshly sampled at whatever difficulty the current
stage dictates. This module implements a different idea (imported from a
collaborator's approach on another branch): pre-generate one large, fixed pool
of datasets up front - built in difficulty *regimes*, same knobs/values as a
scheduler.py schedule, just applied once per regime instead of live - then
train by walking that static pool in a chosen ORDER. "Curriculum" here means
presentation order over fixed data, not changing data.

    pool = load_pool("pools/binary_main.pt")
    order = order_indices(pool["items"], "curriculum", seed=42)
    loader = PoolCurriculumLoader(pool["items"], order, num_steps=..., device=...)

PoolCurriculumLoader duck-types the same __iter__/__len__/num_steps interface
as CurriculumScheduler, so it drops into tfmplayground.train.train()'s prior=
argument unchanged.
"""

import pathlib

import numpy as np
import torch


def load_pool(path):
    """Load a pool built by scripts/build_pool.py. Returns {"items": [...], "meta": {...}}."""
    return torch.load(pathlib.Path(path), weights_only=False)


def order_indices(items, mode, seed, restarts=3):
    """Return a list of indices into `items`, in the order training should visit them.

    Each item is tagged with a "regime_index" at build time (its ordinal position
    in the pool-build config's regime schedule - low = easy, high = hard). That's
    the axis "curriculum" sorts by here, not feature/class count: in this repo's
    binary scope every item shares the same feature and class count (min_features
    == max_features, max_classes == 2), so those would be degenerate sort keys.
    A multiclass-scope pool (varying feature count per item) would sort by
    "n_features" instead - order_indices falls back to that automatically when an
    item has no "regime_index" (see the key() helper below).

    Modes:
      curriculum         - ascending difficulty (regime_index, or n_features)
      curriculum_classes - ascending class count (n_classes) - only meaningful
                            for a multiclass-scope pool; degenerate otherwise
      curriculum_restart - `restarts` interleaved ascending passes (sawtooth:
                            easy->hard, easy->hard, ...) instead of one long ramp
      shuffle             - baseline: fully randomized, seeded for reproducibility
    """

    def key(i):
        it = items[i]
        return it["regime_index"] if "regime_index" in it else it["n_features"]

    idx = list(range(len(items)))
    if mode == "curriculum":
        idx.sort(key=key)
    elif mode == "curriculum_classes":
        idx.sort(key=lambda i: items[i]["n_classes"])
    elif mode == "curriculum_restart":
        ordered = sorted(idx, key=key)
        buckets = [[] for _ in range(restarts)]
        for j, i in enumerate(ordered):
            buckets[j % restarts].append(i)
        idx = []
        for b in buckets:
            idx += sorted(b, key=key)
    elif mode == "shuffle":
        np.random.default_rng(seed).shuffle(idx)
    else:
        raise ValueError(f"unknown order {mode!r}. allowed: curriculum, curriculum_classes, "
                          f"curriculum_restart, shuffle")
    return idx


class PoolCurriculumLoader:
    """Walks a fixed pool in a fixed order. Duck-types CurriculumScheduler's
    interface (__iter__/__len__/num_steps/step()) so tfmplayground.train.train()
    and scripts/run.py's resume logic both work unmodified.

    Structurally simpler than CurriculumScheduler: there's no live knob mutation
    or stage-transition bookkeeping, just an integer cursor into `order`, which
    cycles via modulo once training has walked the whole pool.

    global_step counts individual pool-item draws (one item = one micro-batch =
    one dataset), NOT optimizer steps - see scripts/run_pool.py for why pool-mode
    needs accumulate_gradients=batch_size and how the resume cursor accounts for
    that multiplier.
    """

    def __init__(self, items, order, num_steps, device):
        self.items = items
        self.order = order
        self.num_steps = num_steps
        self.device = device
        self.global_step = 0

    def step(self, global_step):
        # Pure repositioning, no side effects beyond the cursor itself - safe to
        # call repeatedly or out of order, unlike CurriculumScheduler.step() which
        # only fires on stage transitions.
        self.global_step = global_step

    def __iter__(self):
        for _ in range(self.num_steps):
            it = self.items[self.order[self.global_step % len(self.order)]]
            self.global_step += 1
            x = it["x"].to(self.device)
            y = it["y"].to(self.device)
            yield {"x": x, "y": y, "target_y": y, "train_test_split_index": it["split"]}

    def __len__(self):
        return self.num_steps
