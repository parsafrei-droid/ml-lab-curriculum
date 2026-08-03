"""Endpoint vs duration: does WHERE a run finishes matter, or HOW LONG it finishes there?

WHY THIS TEST EXISTS
--------------------
FINDINGS.md claims the ranking of orderings is explained by "where you finish".
Our own sawtooth run refutes the simple form of that claim:

    features ascending   ends at 60 features   +0.020
    restart sawtooth     ends at 60 features   -0.002

Both end at the maximum feature count -- pool.py's curriculum_restart deals the
feature-sorted pool into `restarts` interleaved buckets and re-sorts each bucket
ascending, so the final bucket climbs to the top just like the plain ramp. Same
endpoint, 0.022 apart. Endpoint alone cannot be the explanation.

What actually differs is the *duration of the final contiguous stretch* in the
high-feature regime: ascending spends ~700 unbroken steps there, sawtooth splits
the same total across three visits, the last worth ~1/3 as much.

The seven existing runs cannot separate these, because sorting the pool ties the
path and the endpoint together. This script holds the endpoint fixed and varies
only the tail length.

DESIGN (3 primary arms, all 2500 steps on one shared pool)
----------------------------------------------------------
Each arm: a shuffled prefix, then a contiguous high-feature tail. All arms end on
the pool's highest-feature tables, so the ENDPOINT IS CONSTANT and only tail
length changes.

    tail_300    2200 shuffled + 300 high      (short final phase)
    tail_700    1800 shuffled + 700 high      (matches the ascending run's ~700)
    tail_1300   1200 shuffled + 1300 high     (long final phase)

Prediction if "endpoint" is what matters: all three land together.
Prediction if "duration" is what matters: score rises with tail length.
This is the discriminating comparison the current design cannot make.

EQUAL COMPUTE, PRESERVED
------------------------
The headline result depends on every run seeing identical data and compute
(cum_flops = 5.562e12 for all seven existing runs). We keep that exactly, by
PARTITIONING rather than filtering: the top-N feature tables are reserved for the
tail, the remaining N are shuffled for the prefix. Every table is used exactly
once, all 80,000 are consumed, so FLOPs match by construction. A naive "shuffle,
then draw high-feature tables" would reuse tables and break this -- which is the
very confound the design is meant to remove.

THE HONEST LIMITATION
---------------------
A longer tail must reserve more tables, so it reaches deeper down the feature
range: tail_300 spans ~53-60 features, tail_700 ~44-60, tail_1300 ~30-60. Tail
length and tail purity therefore move together in the primary arms. This is a far
tighter test than the sorted orderings -- the endpoint is genuinely held fixed --
but it is not perfectly clean, and the writeup must say so.

    tail_700_pure   (secondary) 1800 shuffled + 700 steps drawn ONLY from the
                    >=44 band, cycling with repetition.

It matches tail_1300's length ambition while holding purity at tail_700's level.
Because it repeats tables it does NOT preserve equal data, so it is a diagnostic
that brackets the confound, never primary evidence. Its cum_flops will differ and
that is expected.

WHAT WOULD FALSIFY WHAT
-----------------------
  scores flat across tails      -> endpoint hypothesis survives; sawtooth needs
                                   another explanation
  scores rise with tail length  -> duration hypothesis; FINDINGS.md section 6
                                   must be rewritten
  tail_300 ~= ascending's +0.020 with only 300 steps -> neither; something about
                                   the shuffled prefix itself is doing the work

Any of the three is a publishable answer. The current poster claim is only safe
in the first case.

SEEDS
-----
Between-seed spread is ~0.01 and the gap we must resolve is 0.022, so 3 seeds is
not enough. This runs 5 seeds per arm.

USAGE (single Kaggle cell, GPU on, internet on)
-----------------------------------------------
    !python experiments/robustness/kaggle_endpoint_vs_duration.py

Resumable: every stage checks for its own output, so re-running the same cell in
a fresh session skips finished work and continues. Kaggle commits cap at 12 h;
the pool build alone is ~1.7 h and the 20 train+eval stages will not fit in one
session. Expect to run the cell 2-3 times. Nothing here is committed or pushed.
"""

import json
import pathlib
import shutil
import subprocess
import sys
import time
import traceback

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent.parent
CODE = ROOT / "final" / "code"
POOLS = BASE / "pools"
CONFIGS = BASE / "configs_endpoint"
RESULTS = CODE / "results"
OUT = BASE / "results_endpoint"

POOL = POOLS / "main_p0.pt"
POOL_SIZE = 80000
POOL_SEED = 0
TOTAL_STEPS = 2500
GRAD_ACCUM = 32
SEEDS = (42, 1, 2, 3, 4)

# tail length in steps -> arm name. The endpoint is identical in all three.
PRIMARY_ARMS = {"tail_300": 300, "tail_700": 700, "tail_1300": 1300}
PURE_BAND_MIN_FEATURES = 44  # matches tail_700's natural band, for the secondary arm

T_START = time.time()


def hrs():
    return (time.time() - T_START) / 3600


def log(msg):
    print(f"\n[{hrs():5.2f}h] {msg}", flush=True)


def child_env():
    """tabicl is src-layout, so the repo-root path the tracked scripts insert has no
    importable package and shadows the editable install. Push the real package roots
    in via PYTHONPATH so evaluate.py / pool.py work without editing tracked files."""
    import os

    extra = []
    for name in ("tabicl", "TFM-Playground"):
        d = ROOT / name
        p = d / "src" if (d / "src").is_dir() else d
        if p.is_dir():
            extra.append(str(p))
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra + ([existing] if existing else []))
    return env


def sh(cmd, env=None):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env or child_env())
    if r.returncode != 0:
        raise RuntimeError(f"exit {r.returncode} after {(time.time() - t0) / 60:.1f} min")
    print(f"[ok {(time.time() - t0) / 60:.1f} min]", flush=True)


# ------------------------------------------------------------------ setup

def setup():
    log("SETUP: pinned deps + install")
    sh([sys.executable, BASE / "kaggle_setup.py"])


def build_pool():
    """Same builder, same args, same seed as the original second-wave pool, so the
    arms are comparable to the seven runs already reported."""
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
# --- injected by kaggle_endpoint_vs_duration.py -----------------------------
# Adds endpoint-controlled orderings to the repo's pool.py without editing the
# tracked file. Imported by the patched train.py at runtime.
import numpy as _np


def _tail_partition(items, tail_steps, grad_accum, seed):
    """Shuffled prefix + contiguous high-feature tail, every table used once.

    Reserving the top-N by feature count (rather than filtering a shuffle) is what
    keeps total FLOPs identical across arms: the union of prefix and tail is always
    the whole pool, permuted.
    """
    n_tail = tail_steps * grad_accum
    if n_tail > len(items):
        raise ValueError(f"tail {n_tail} exceeds pool {len(items)}")
    by_feat = sorted(range(len(items)), key=lambda i: items[i]["n_features"])
    prefix, tail = by_feat[:-n_tail], by_feat[-n_tail:]
    _np.random.default_rng(seed).shuffle(prefix)      # prefix order is random
    tail.sort(key=lambda i: items[i]["n_features"])   # tail still ends at the max
    return list(prefix) + tail


def _tail_pure(items, tail_steps, grad_accum, seed, min_features):
    """Secondary diagnostic: hold tail PURITY fixed and stretch its LENGTH by
    cycling the >=min_features band with repetition.

    This deliberately breaks the equal-data rule (tables repeat, prefix tables are
    also seen in the tail). cum_flops will not match the primary arms. It exists
    only to bracket the length/purity confound, and must never be reported as
    primary evidence.
    """
    n_tail = tail_steps * grad_accum
    band = [i for i in range(len(items)) if items[i]["n_features"] >= min_features]
    if not band:
        raise ValueError("empty high-feature band")
    band.sort(key=lambda i: items[i]["n_features"])
    rng = _np.random.default_rng(seed)
    band_set = set(band)  # hoisted: rebuilding this per-item is O(n^2) and hangs
    prefix = [i for i in range(len(items)) if i not in band_set]
    rng.shuffle(prefix)
    n_prefix = (TOTAL_STEPS - tail_steps) * grad_accum
    if len(prefix) < n_prefix:                        # top up from the band if short
        extra = list(rng.choice(band, size=n_prefix - len(prefix), replace=True))
        prefix = prefix + extra
    prefix = prefix[:n_prefix]
    reps = (n_tail + len(band) - 1) // len(band)
    tail = (band * reps)[:n_tail]
    tail.sort(key=lambda i: items[i]["n_features"])
    return list(prefix) + tail
'''


def write_order_module():
    """Emit the ordering helpers as a standalone module next to the code."""
    mod = CODE / "_endpoint_orders.py"
    mod.write_text(f"TOTAL_STEPS = {TOTAL_STEPS}\n" + ORDER_PATCH)
    return mod


TRAIN_PATCH = '''
# --- injected by kaggle_endpoint_vs_duration.py -----------------------------
# train.py resolves the ordering through pool.order_indices. We wrap it so the
# new modes work without editing the tracked train.py or pool.py.
import _endpoint_orders as _eo
import pool as _pool

_orig_order_indices = _pool.order_indices


def _patched_order_indices(items, mode, seed, restarts=3, axes=None, weights=None):
    if isinstance(mode, str) and mode.startswith("endpoint_tail_"):
        spec = mode[len("endpoint_tail_"):]
        if spec.endswith("_pure"):
            steps = int(spec[:-len("_pure")])
            return _eo._tail_pure(items, steps, 32, seed, %d)
        return _eo._tail_partition(items, int(spec), 32, seed)
    return _orig_order_indices(items, mode, seed, restarts, axes, weights)


_pool.order_indices = _patched_order_indices
''' % PURE_BAND_MIN_FEATURES


def write_runner():
    """A thin launcher that installs the patch, then hands off to the repo's real
    train.py. train.py stays byte-identical -- fidelity matters more than tidiness."""
    runner = CODE / "_endpoint_train.py"
    runner.write_text(
        "import pathlib, runpy, sys\n"
        "BASE = pathlib.Path(__file__).resolve().parent\n"
        "ROOT = BASE.parent.parent\n"
        "sys.path.insert(0, str(BASE))\n"
        # train.py inserts ROOT/'tabicl', but tabicl is a src-layout package, so
        # that entry has no importable package and shadows the editable install.
        # Put the real package root ahead of it.
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
    """Clone the repo's own curriculum_features config, change only pool + order,
    so architecture / lr / steps / accum stay exactly as reported."""
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


# -------------------------------------------------------------------- runs

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
    print("\n" + "=" * 94, flush=True)
    print("ENDPOINT vs DURATION", flush=True)
    print("=" * 94, flush=True)

    table = {}
    for arm in list(PRIMARY_ARMS) + ["tail_700_pure"]:
        aucs, flops = [], set()
        for s in SEEDS:
            a = load(f"{arm}_s{s}", "mean_roc_auc", "tabarena_scores.json")
            f = load(f"{arm}_s{s}", "cum_flops")
            if isinstance(a, (int, float)):
                aucs.append(a)
            if f is not None:
                flops.add(round(float(f)))
        table[arm] = (aucs, flops)

    print(f"{'arm':16s} {'tail steps':>11s} {'n':>3s} {'mean ROC-AUC':>13s} {'spread':>8s}", flush=True)
    print("-" * 94, flush=True)
    for arm in PRIMARY_ARMS:
        aucs, _ = table[arm]
        if aucs:
            m = sum(aucs) / len(aucs)
            spread = max(aucs) - min(aucs)
            print(f"{arm:16s} {PRIMARY_ARMS[arm]:>11d} {len(aucs):>3d} {m:>13.4f} {spread:>8.4f}",
                  flush=True)
        else:
            print(f"{arm:16s} {PRIMARY_ARMS[arm]:>11d} {0:>3d} {'-':>13s} {'-':>8s}", flush=True)
    aucs, _ = table["tail_700_pure"]
    if aucs:
        m = sum(aucs) / len(aucs)
        print(f"{'tail_700_pure':16s} {700:>11d} {len(aucs):>3d} {m:>13.4f} "
              f"{max(aucs) - min(aucs):>8.4f}   (secondary: repeats tables)", flush=True)

    # Equal-compute check. The primary arms must agree to the digit; if they do not,
    # the partition logic broke and the comparison is void.
    print("\nequal-compute check (primary arms must share one cum_flops):", flush=True)
    all_flops = set()
    for arm in PRIMARY_ARMS:
        _, f = table[arm]
        all_flops |= f
        print(f"  {arm:16s} cum_flops={sorted(f) if f else '-'}", flush=True)
    if len(all_flops) == 1:
        print("  OK: identical across all primary arms.", flush=True)
    elif all_flops:
        print("  FAIL: primary arms differ in compute -- comparison is INVALID.", flush=True)

    # The actual verdict.
    means = {a: (sum(v[0]) / len(v[0])) for a, v in table.items() if a in PRIMARY_ARMS and v[0]}
    print("\nverdict:", flush=True)
    if len(means) < len(PRIMARY_ARMS):
        print("  incomplete -- re-run the cell to finish remaining seeds.", flush=True)
        return
    lo, hi = means["tail_300"], means["tail_1300"]
    span = max(means.values()) - min(means.values())
    seed_spread = max((max(v[0]) - min(v[0])) for a, v in table.items()
                      if a in PRIMARY_ARMS and v[0])
    print(f"  spread across tail lengths : {span:.4f}", flush=True)
    print(f"  worst within-arm seed spread: {seed_spread:.4f}", flush=True)
    if span < seed_spread:
        print("  -> FLAT: tail length does not matter beyond noise.", flush=True)
        print("     The endpoint account SURVIVES; sawtooth needs another explanation.", flush=True)
    elif hi > lo:
        print("  -> RISING with tail length: DURATION drives the effect.", flush=True)
        print("     FINDINGS.md section 6 and the poster's endpoint claim must be rewritten.", flush=True)
    else:
        print("  -> FALLING with tail length: neither hypothesis as stated.", flush=True)
        print("     Longer high-feature tails HURT -- report as-is, this is a real result.", flush=True)

    print("\nreference (already measured, same pool recipe):", flush=True)
    print("  features ascending  0.8107   (+0.020 vs baseline, ~700-step ramp)", flush=True)
    print("  baseline shuffle    0.7911", flush=True)
    print("  restart sawtooth    0.7895   (same endpoint, split tail)", flush=True)
    print(f"\nartifacts: {OUT}", flush=True)


# -------------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    build_pool()
    write_order_module()
    runner = write_runner()

    failures = []
    # Primary arms first, seed 42 across all three before deepening, so a session
    # that dies early still yields one complete comparison rather than one finished
    # arm and two empty ones.
    plan = [(arm, s) for s in SEEDS for arm in PRIMARY_ARMS]
    plan += [("tail_700_pure", s) for s in SEEDS]

    for arm, seed in plan:
        try:
            mode = (f"endpoint_tail_{PRIMARY_ARMS[arm]}" if arm in PRIMARY_ARMS
                    else "endpoint_tail_700_pure")
            cfg = write_config(arm, mode)
            train_eval(runner, cfg, f"{arm}_s{seed}", seed)
        except Exception as e:
            failures.append(f"{arm} seed {seed}: {e}")
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
