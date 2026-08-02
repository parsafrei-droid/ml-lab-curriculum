"""Kaggle runner for the two robustness tests, for when BwUniCluster is unavailable.

This is a drop-in replacement for `run_tests.sh`. It calls the SAME `final/code`
entry points with the SAME arguments, so a result produced here is comparable to
one produced by sbatch. Nothing in the pipeline is reimplemented: pool.py,
noise_pool.py, train.py and evaluate.py are invoked as subprocesses exactly as
the sbatch script invokes them.

What differs from Slurm, and why:

  * Slurm runs one config per job to stay under the 30 min cap. Kaggle has no
    per-job cap but kills the session on disconnect, so instead of splitting we
    make every stage resumable: a finished stage is detected and skipped, so
    re-running after a session dies continues where it stopped.
  * Pools are rebuilt only if missing, same `if [ ! -f "$POOL" ]` logic.

Usage inside a Kaggle notebook (GPU: P100 or T4 x2, Internet: ON):

    !python experiments/robustness/kaggle_run.py --test pool_seed --arg 1
    !python experiments/robustness/kaggle_run.py --test pool_seed --arg 2
    !python experiments/robustness/kaggle_run.py --test noise --arg noise_baseline
    !python experiments/robustness/kaggle_run.py --test curriculum_noise ...

or run every stage back to back, resuming as needed:

    !python experiments/robustness/kaggle_run.py --test all

Results land in final/code/results/ and are copied to
experiments/robustness/results/ (meta.json + tabarena_scores.json only, which is
what the README asks to send back).
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent.parent
CODE = ROOT / "final" / "code"
POOLS = BASE / "pools"
CONFIGS = BASE / "configs"
RESULTS = CODE / "results"
OUT = BASE / "results"

# Mirrors run_tests.sh exactly: same pool geometry, same two headline configs.
POOL_ARGS = dict(size=80000, min_features=2, max_features=60, max_classes=10,
                 num_datapoints=200)
POOL_SEED_CONFIGS = ["baseline", "curriculum_features"]
NOISE_CONFIGS = ["noise_baseline", "curriculum_noise", "curriculum_noise_reverse"]
SEED = 42


def sh(cmd):
    """Run a stage, streaming output so Kaggle's log shows progress live."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"FAILED (exit {r.returncode}) after {time.time() - t0:.0f}s")
    print(f"[ok {time.time() - t0:.0f}s]", flush=True)


def done_training(name):
    return (RESULTS / name / "checkpoint.pth").exists()


def done_eval(name):
    return (RESULTS / name / "tabarena_scores.json").exists()


def build_pool_if_missing(path, seed, noise=False):
    if path.exists():
        print(f"pool exists, skipping build: {path}", flush=True)
        return
    POOLS.mkdir(parents=True, exist_ok=True)
    if noise:
        # noise_pool.py resolves --out against its own BASE, so pass it relative
        # to experiments/robustness to land in the same place run_tests.sh uses.
        sh([sys.executable, BASE / "noise_pool.py", "--out", f"pools/{path.name}",
            "--size", POOL_ARGS["size"], "--seed", seed])
    else:
        sh([sys.executable, CODE / "pool.py", "--out", path,
            "--size", POOL_ARGS["size"],
            "--min-features", POOL_ARGS["min_features"],
            "--max-features", POOL_ARGS["max_features"],
            "--max-classes", POOL_ARGS["max_classes"],
            "--num-datapoints", POOL_ARGS["num_datapoints"],
            "--seed", seed])


def write_pool_seed_config(cfg, pool_path, arg):
    """The sed line from run_tests.sh, done in Python.

    train.py resolves cfg["pool"] as BASE / cfg["pool"] where BASE is final/code.
    pathlib keeps an absolute path absolute, so writing the absolute pool path
    here behaves exactly like the sed in the sbatch script.
    """
    src = (CODE / "configs" / f"{cfg}.yaml").read_text()
    lines = []
    for line in src.splitlines():
        if line.startswith("pool:"):
            line = f"pool: {pool_path.resolve().as_posix()}"
        lines.append(line)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    dst = CONFIGS / f"{cfg}_p{arg}.yaml"
    dst.write_text("\n".join(lines) + "\n")
    return dst


def train_and_eval(config_path, name):
    if done_training(name):
        print(f"training done, skipping: {name}", flush=True)
    else:
        sh([sys.executable, CODE / "train.py", "--config", config_path,
            "--name", name, "--seed", SEED])
    if done_eval(name):
        print(f"eval done, skipping: {name}", flush=True)
    else:
        sh([sys.executable, CODE / "evaluate.py",
            "--checkpoint", RESULTS / name / "checkpoint.pth"])
    collect(name)


def run_pool_seed(arg):
    pool = POOLS / f"main_p{arg}.pt"
    build_pool_if_missing(pool, seed=arg)
    for cfg in POOL_SEED_CONFIGS:
        cfg_path = write_pool_seed_config(cfg, pool, arg)
        train_and_eval(cfg_path, f"{cfg}_pool{arg}_s42")


def run_noise(cfg):
    build_pool_if_missing(POOLS / "noise.pt", seed=0, noise=True)
    train_and_eval(CONFIGS / f"{cfg}.yaml", f"{cfg}_s42")


def collect(name):
    """Copy the two artifacts the README asks for. Checkpoints stay put (gitignored)."""
    src, dst = RESULTS / name, OUT / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("meta.json", "tabarena_scores.json", "log.csv", "config.yaml"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)


def summarise():
    """Print ordering_profile and mean ROC-AUC per run, the two review checks."""
    print("\n" + "=" * 78)
    print(f"{'run':40s} {'profile':>22s} {'mean_roc_auc':>12s}")
    print("=" * 78)
    for d in sorted(OUT.iterdir()) if OUT.exists() else []:
        if not d.is_dir():
            continue
        prof, auc = "-", "-"
        if (d / "meta.json").exists():
            prof = json.dumps(json.loads((d / "meta.json").read_text())
                              .get("ordering_profile", {}))
        if (d / "tabarena_scores.json").exists():
            v = json.loads((d / "tabarena_scores.json").read_text()).get("mean_roc_auc")
            auc = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        print(f"{d.name:40s} {prof:>22s} {auc:>12s}")
    print("=" * 78)
    print("expect ordering_profile ~ +1.0 forward, ~ -1.0 reversed, ~0.0 shuffle", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True,
                   choices=["pool_seed", "noise", "all", "summarise", "build_pools"])
    p.add_argument("--arg", default=None,
                   help="pool seed (1|2) for pool_seed, config name for noise; "
                        "for build_pools one of 1|2|noise, or omit to build all three")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.test == "summarise":
        summarise()
        return
    if args.test == "build_pools":
        # Pool building is pure CPU (~5 h each). Run this in a CPU-only session so
        # it costs no GPU quota, then train in a GPU session with persistence on.
        todo = [args.arg] if args.arg else ["1", "2", "noise"]
        for t in todo:
            if t == "noise":
                build_pool_if_missing(POOLS / "noise.pt", seed=0, noise=True)
            else:
                build_pool_if_missing(POOLS / f"main_p{t}.pt", seed=t)
        print("\npools ready:", flush=True)
        for f in sorted(POOLS.glob("*.pt")):
            print(f"  {f.name}  {f.stat().st_size / 1e9:.2f} GB", flush=True)
        return
    if args.test == "pool_seed":
        run_pool_seed(args.arg)
    elif args.test == "noise":
        run_noise(args.arg)
    elif args.test == "all":
        for s in ("1", "2"):
            run_pool_seed(s)
        for c in NOISE_CONFIGS:
            run_noise(c)
    summarise()


if __name__ == "__main__":
    main()
