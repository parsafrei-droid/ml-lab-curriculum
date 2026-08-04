"""Path vs endpoint: does the ordering effect need the gradual climb?

WHY THIS EXISTS
---------------
The endpoint-vs-duration run answered its own question and opened a new one. Three
arms that all finish at the top of the feature range, holding it for 300 / 700 /
1300 steps, scored 0.7947 / 0.7960 / 0.7958 -- flat, and all at baseline (0.7911)
rather than at the ascending curriculum's 0.8107. Even tail_1300, whose realised
features-Spearman is 0.888, captured none of the +0.020.

Those arms were built as "shuffled prefix + sorted tail". That isolates the ending,
but it also destroys the smooth low-to-high progression: the model sees a random mix
of low/mid tables, then jumps to the high band. The real curriculum instead climbs
monotonically through every band from step 1 (4 -> 14 -> 25 -> 37 -> 48 -> 60).

So the open question is no longer "endpoint or duration" -- both are ruled out. It
is whether the effect needs the PATH.

THE ARMS
--------
  full_sort     the plain ascending sort, at 5 seeds.
                A replication of curriculum_features, which was only run at 3 seeds.
                This is the anchor: if it does not reproduce ~0.811, the original
                effect was seed noise and everything downstream changes.

  sorted_head   sorted prefix + SHUFFLED tail. The mirror image of the tail arms:
                keeps the climb, throws away the ending. If the path is what
                matters this should recover part of the gain; if the ending is what
                matters it should sit at baseline.

  coarse_4      the climb kept but quantised into 4 ascending blocks, shuffled
                within each block. Same direction and same endpoint as full_sort,
                but a staircase instead of a smooth ramp. Separates "monotone
                trend" from "strictly sorted".

Together with the tail arms already measured, these four points (baseline, tail_*,
coarse_4, sorted_head, full_sort) span the space between shuffle and full sort.

EQUAL DATA AND COMPUTE
----------------------
Every arm is a permutation of the whole pool: each table used exactly once, no
repeats, no filtering. cum_flops therefore matches by construction, and the summary
below verifies it rather than assuming it. Any arm that breaks this is not evidence.

RUN IT
------
One cell on Kaggle (GPU on, internet on):

    !python experiments/robustness/kaggle_path_vs_endpoint.py

Resumable: every stage checks for its own output first, so re-running the same cell
in a fresh session skips finished work and continues. Set ARTIFACT_DIR to somewhere
outside the clone so a re-clone does not destroy the pool build.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent.parent
CODE = ROOT / "final" / "code"

_ART = os.environ.get("ARTIFACT_DIR")
ARTIFACTS = pathlib.Path(_ART).resolve() if _ART else BASE
POOLS = ARTIFACTS / "pools"
CONFIGS = ARTIFACTS / "configs_path"
RESULTS = CODE / "results"
OUT = ARTIFACTS / "results_path"

# Same pool, same seed, same builder args as the seven reported runs.
POOL = POOLS / "main_p0.pt"
POOL_SIZE = 80000
POOL_SEED = 0
TOTAL_STEPS = 2500
GRAD_ACCUM = 32
SEEDS = (42, 1, 2, 3, 4)

# arm name -> order mode string understood by the injected patch
ARMS = {
    "full_sort": "path_full_sort",
    "sorted_head": "path_sorted_head_700",   # last 700 steps shuffled
    "coarse_4": "path_coarse_4",             # 4 ascending blocks
}
HEAD_TAIL_STEPS = 700   # matches tail_700 so the two are mirror images

# Reference numbers from runs already completed, printed in the summary.
REFERENCE = {
    "curriculum_features (3 seeds)": 0.8107,
    "baseline shuffle   (3 seeds)": 0.7911,
    "tail_300           (5 seeds)": 0.7947,
    "tail_700           (5 seeds)": 0.7960,
    "tail_1300          (5 seeds)": 0.7958,
}

T_START = time.time()


def hrs():
    return (time.time() - T_START) / 3600


def log(msg):
    print(f"\n[{hrs():5.2f}h] {msg}", flush=True)


def child_env():
    """tabicl is a src-layout package, so the repo-root entry the tracked scripts
    insert has no importable package and shadows the editable install. Push the real
    package roots to the front for the child process."""
    env = dict(os.environ)
    extra = []
    for name in ("tabicl", "TFM-Playground"):
        d = ROOT / name
        p = d / "src" if (d / "src").is_dir() else d
        if p.is_dir():
            extra.append(str(p))
    if extra:
        env["PYTHONPATH"] = os.pathsep.join(extra + [env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def sh(cmd, env=None):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env or child_env())
    if r.returncode != 0:
        raise RuntimeError(f"exit {r.returncode} after {(time.time() - t0) / 60:.1f} min")
    print(f"[ok {(time.time() - t0) / 60:.1f} min]", flush=True)


def setup():
    log("SETUP: pinned deps + install")
    sh([sys.executable, BASE / "kaggle_setup.py"])


def build_pool():
    if POOL.exists():
        log(f"pool exists, skipping: {POOL.name}")
        return
    POOLS.mkdir(parents=True, exist_ok=True)
    log(f"BUILD {POOL.name} (80k tables, CPU-bound, ~1.7 h)")
    sh([sys.executable, CODE / "pool.py", "--out", POOL, "--size", POOL_SIZE,
        "--min-features", 2, "--max-features", 60, "--max-classes", 10,
        "--num-datapoints", 200, "--seed", POOL_SEED])


# ------------------------------------------------- the orderings under test

ORDER_PATCH = '''
# --- injected by kaggle_path_vs_endpoint.py ---------------------------------
# Path-controlled orderings. Every function returns a permutation of ALL indices,
# so data and compute are identical to the reported runs by construction.
import numpy as _np


def _full_sort(items, seed):
    """The plain ascending feature sort. Identical in effect to the repo's
    order='curriculum' with default axes, restated here so this arm does not
    depend on AXIS_DEFAULTS staying unchanged."""
    return sorted(range(len(items)), key=lambda i: items[i]["n_features"])


def _sorted_head(items, tail_steps, grad_accum, seed):
    """Sorted prefix + SHUFFLED tail: keep the climb, destroy the ending.

    Mirror image of _tail_partition from the endpoint experiment. The prefix is the
    low-feature portion in ascending order; the tail is the remaining (high-feature)
    portion shuffled, so the run no longer ends on a clean high band.
    """
    n_tail = tail_steps * grad_accum
    if n_tail > len(items):
        raise ValueError(f"tail {n_tail} exceeds pool {len(items)}")
    by_feat = sorted(range(len(items)), key=lambda i: items[i]["n_features"])
    head, tail = by_feat[:-n_tail], by_feat[-n_tail:]
    _np.random.default_rng(seed).shuffle(tail)   # ending is now random
    return list(head) + list(tail)


def _coarse(items, n_blocks, seed):
    """Ascending staircase: sort, cut into n_blocks equal blocks, shuffle within
    each block. Keeps the monotone trend and the endpoint, removes the fine sort."""
    by_feat = sorted(range(len(items)), key=lambda i: items[i]["n_features"])
    rng = _np.random.default_rng(seed)
    n = len(by_feat)
    out = []
    for b in range(n_blocks):
        lo = b * n // n_blocks
        hi = (b + 1) * n // n_blocks
        block = by_feat[lo:hi]
        rng.shuffle(block)
        out += list(block)
    return out
'''


def write_order_module():
    mod = CODE / "_path_orders.py"
    mod.write_text(f"TOTAL_STEPS = {TOTAL_STEPS}\n" + ORDER_PATCH)
    return mod


TRAIN_PATCH = '''
# --- injected by kaggle_path_vs_endpoint.py ---------------------------------
import _path_orders as _po
import pool as _pool

_orig_order_indices = _pool.order_indices


def _patched_order_indices(items, mode, seed, restarts=3, axes=None, weights=None):
    if isinstance(mode, str) and mode.startswith("path_"):
        spec = mode[len("path_"):]
        if spec == "full_sort":
            return _po._full_sort(items, seed)
        if spec.startswith("sorted_head_"):
            return _po._sorted_head(items, int(spec[len("sorted_head_"):]), 32, seed)
        if spec.startswith("coarse_"):
            return _po._coarse(items, int(spec[len("coarse_"):]), seed)
        raise ValueError(f"unknown path mode {mode!r}")
    return _orig_order_indices(items, mode, seed, restarts, axes, weights)


_pool.order_indices = _patched_order_indices
'''


def write_runner():
    """Thin launcher: install the patch, then hand off to the repo's real train.py,
    which stays byte-identical."""
    runner = CODE / "_path_train.py"
    runner.write_text(
        "import pathlib, runpy, sys\n"
        "BASE = pathlib.Path(__file__).resolve().parent\n"
        "ROOT = BASE.parent.parent\n"
        "sys.path.insert(0, str(BASE))\n"
        "for _n in ('tabicl', 'TFM-Playground'):\n"
        "    _d = ROOT / _n\n"
        "    _p = _d / 'src' if (_d / 'src').is_dir() else _d\n"
        "    if _p.is_dir() and str(_p) not in sys.path:\n"
        "        sys.path.insert(0, str(_p))\n"
        f"{TRAIN_PATCH}\n"
        "sys.argv[0] = str(BASE / 'train.py')\n"
        "runpy.run_path(str(BASE / 'train.py'), run_name='__main__')\n"
    )
    return runner


def write_config(arm, order_mode):
    """Clone the repo's curriculum_features config, change only pool / order / name,
    so architecture, lr, steps and accumulation stay exactly as reported."""
    src = (CODE / "configs" / "curriculum_features.yaml").read_text()
    out = []
    for ln in src.splitlines():
        if ln.startswith("pool:"):
            out.append(f"pool: {POOL.resolve().as_posix()}")
        elif ln.startswith("order:"):
            out.append(f"order: {order_mode}")
        elif ln.startswith("name:"):
            out.append(f"name: {arm}")
        else:
            out.append(ln)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    dst = CONFIGS / f"{arm}.yaml"
    dst.write_text("\n".join(out) + "\n")
    return dst


def collect(name):
    src, dst = RESULTS / name, OUT / name
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("meta.json", "tabarena_scores.json", "log.csv", "config.yaml"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)


def train_eval(runner, config_path, name, seed):
    if (RESULTS / name / "checkpoint.pth").exists():
        log(f"train done, skipping: {name}")
    else:
        log(f"TRAIN {name}")
        sh([sys.executable, runner, "--config", config_path, "--name", name, "--seed", seed])

    if (RESULTS / name / "tabarena_scores.json").exists():
        log(f"eval done, skipping: {name}")
    else:
        log(f"EVAL {name} (TabArena)")
        sh([sys.executable, CODE / "evaluate.py",
            "--checkpoint", RESULTS / name / "checkpoint.pth"])
    collect(name)


# ----------------------------------------------------------------- summary

def load(name, field, path="meta.json"):
    p = OUT / name / path
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(field)


def summarise():
    print("\n" + "=" * 96, flush=True)
    print("PATH vs ENDPOINT", flush=True)
    print("=" * 96, flush=True)

    table = {}
    for arm in ARMS:
        aucs, flops, spear = [], set(), []
        for s in SEEDS:
            a = load(f"{arm}_s{s}", "mean_roc_auc", "tabarena_scores.json")
            f = load(f"{arm}_s{s}", "cum_flops")
            prof = load(f"{arm}_s{s}", "ordering_profile") or {}
            if isinstance(a, (int, float)):
                aucs.append(a)
            if f is not None:
                flops.add(round(float(f)))
            if "features" in prof:
                spear.append(prof["features"])
        table[arm] = (aucs, flops, spear)

    print(f"{'arm':14s} {'n':>3s} {'mean ROC-AUC':>13s} {'spread':>8s} {'features-Spearman':>19s}",
          flush=True)
    print("-" * 96, flush=True)
    for arm in ARMS:
        aucs, _, spear = table[arm]
        if aucs:
            m = sum(aucs) / len(aucs)
            sp = f"{sum(spear)/len(spear):.3f}" if spear else "-"
            print(f"{arm:14s} {len(aucs):>3d} {m:>13.4f} {max(aucs)-min(aucs):>8.4f} {sp:>19s}",
                  flush=True)
        else:
            print(f"{arm:14s} {0:>3d} {'-':>13s} {'-':>8s} {'-':>19s}", flush=True)

    print("\nalready measured, same pool recipe:", flush=True)
    for k, v in REFERENCE.items():
        print(f"  {k:32s} {v:.4f}", flush=True)

    # Equal-compute check: all arms are whole-pool permutations, so this must hold.
    print("\nequal-compute check (all arms must share one cum_flops):", flush=True)
    all_flops = set()
    for arm in ARMS:
        _, f, _ = table[arm]
        all_flops |= f
        print(f"  {arm:14s} cum_flops={sorted(f) if f else '-'}", flush=True)
    if len(all_flops) == 1:
        print("  OK: identical across all arms.", flush=True)
    elif all_flops:
        print("  FAIL: arms differ in compute -- comparison is INVALID.", flush=True)

    # The reading, computed rather than eyeballed.
    fs = table["full_sort"][0]
    sh_ = table["sorted_head"][0]
    c4 = table["coarse_4"][0]
    base, curr = 0.7911, 0.8107
    print("\nreading:", flush=True)
    if fs:
        m = sum(fs) / len(fs)
        if m >= base + 0.010:
            print(f"  full_sort {m:.4f} reproduces the original effect at 5 seeds.", flush=True)
        else:
            print(f"  full_sort {m:.4f} does NOT reproduce {curr:.4f}. The original", flush=True)
            print("  3-seed result was optimistic -- this is the headline, report it.", flush=True)
    if sh_ and fs:
        m = sum(sh_) / len(sh_)
        print(f"  sorted_head {m:.4f}: ", end="", flush=True)
        if m >= base + 0.010:
            print("climb alone recovers the gain -> the PATH matters, not the ending.", flush=True)
        else:
            print("climb alone is not enough -> the ending is also required.", flush=True)
    if c4:
        m = sum(c4) / len(c4)
        print(f"  coarse_4 {m:.4f}: ", end="", flush=True)
        print("a 4-step staircase is " + ("enough." if m >= base + 0.010 else
              "NOT enough -- the fine sort matters."), flush=True)
    print(f"\nartifacts: {OUT}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if _ART is None:
        print("NOTE: ARTIFACT_DIR unset -- results live inside the clone and will not "
              "survive a re-clone.", flush=True)
    setup()
    build_pool()
    write_order_module()
    runner = write_runner()

    failures = []
    # full_sort first: it is the anchor, so if the session dies early that is the
    # arm that survived.
    for arm in ("full_sort", "sorted_head", "coarse_4"):
        cfg = write_config(arm, ARMS[arm])
        for s in SEEDS:
            try:
                train_eval(runner, cfg, f"{arm}_s{s}", s)
            except Exception as e:
                failures.append(f"{arm}_s{s}: {e}")
                traceback.print_exc()

    summarise()
    log(f"TOTAL {hrs():.2f} h")
    if failures:
        print("\nSTAGES THAT FAILED (re-run this same cell to resume):", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        sys.exit(1)
    print("\nALL DONE - every stage finished.", flush=True)


if __name__ == "__main__":
    main()
