"""Kaggle environment setup for the robustness tests.

Reproduces what `run_tests.sh` gets for free on the cluster (module load + an
already-built .venv): the two upstream repos at the SAME pinned commits the
cluster env uses, plus the handful of real deps this pipeline imports.

    TFM-Playground  98e33be
    tabicl          8f665ed

Both pins matter. 98e33be is the commit that fixes the `tabicl.prior.dataset`
import path, so with this pin no source patch is needed. Leaving these unpinned
would silently change the prior, and therefore the data, which is the one thing
these tests must hold fixed.

Kaggle notebook, first cell (GPU on, Internet ON):

    !git clone --branch second-wave --single-branch \
        https://github.com/parsafrei-droid/ml-lab-curriculum.git /kaggle/working/ml-lab-curriculum
    %cd /kaggle/working/ml-lab-curriculum
    !python experiments/robustness/kaggle_setup.py
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

PIN = {
    "TFM-Playground": ("https://github.com/automl/TFM-Playground.git", "98e33be"),
    "tabicl": ("https://github.com/soda-inria/tabicl.git", "8f665ed"),
}

# Real deps only. schedulefree is the optimiser train.py uses; openml + the
# TabArena task fetch is what evaluate.py needs; scipy backs ordering_profile's
# spearmanr. Everything else is already in the Kaggle image.
DEPS = ["schedulefree", "einops", "huggingface-hub", "openml",
        "scikit-learn>=1.5", "scipy", "pyyaml", "pandas", "h5py", "requests"]


def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def clone_deps():
    for name, (url, commit) in PIN.items():
        d = ROOT / name
        if not d.exists():
            sh(f"git clone {url} {d}")
        # Pin every time, so a warm/restored working dir can't drift.
        sh(f"git -C {d} fetch --all --quiet", check=False)
        sh(f"git -C {d} checkout --quiet {commit}")
        head = subprocess.run(f"git -C {d} rev-parse --short HEAD", shell=True,
                              capture_output=True, text=True).stdout.strip()
        assert head.startswith(commit[:7]), f"{name} at {head}, expected {commit}"
        print(f"  {name} pinned at {head}", flush=True)


def install():
    sh(f"{sys.executable} -m pip -q install " + " ".join(f"'{d}'" for d in DEPS))
    # --no-deps so neither editable install drags torch off Kaggle's CUDA build.
    sh(f"{sys.executable} -m pip -q install --no-deps -e {ROOT / 'tabicl'}")
    sh(f"{sys.executable} -m pip -q install --no-deps -e {ROOT / 'TFM-Playground'}")


def self_check():
    """Import the exact chain train.py and evaluate.py use, in this process, so a
    missing name fails here with a clear message instead of mid-run."""
    # tabicl is a src-layout package (src/tabicl), TFM-Playground is flat
    # (tfmplayground/). Pointing at the repo root for a src-layout package puts a
    # directory with no importable package on sys.path, which then SHADOWS the
    # working `pip install -e` and makes `import tabicl` fail. Resolve each layout.
    search = [ROOT / "TFM-Playground", ROOT / "final" / "code"]
    for name in ("tabicl", "TFM-Playground"):
        d = ROOT / name
        search.append(d / "src" if (d / "src").is_dir() else d)
    for p in search:
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import warnings
    warnings.filterwarnings("ignore")

    import torch
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(), flush=True)
    else:
        print("WARNING: no CUDA — Settings > Accelerator > GPU.", flush=True)

    import schedulefree  # noqa: F401
    from scipy.stats import spearmanr  # noqa: F401  <- ordering_profile
    from tabicl.prior._prior_config import DEFAULT_SAMPLED_HP  # noqa: F401  <- noise_pool
    from tfmplayground.models.nanotabpfn import NanoTabPFNModel  # noqa: F401
    from tfmplayground.utils import get_default_device, set_randomness_seed  # noqa: F401
    from tfmplayground.external_priors import TabICLPriorDataLoader  # noqa: F401
    # evaluate.py's chain — the part that needs openml and network access.
    from tfmplayground.evaluation import TABARENA_TASKS, get_openml_predictions  # noqa: F401
    from tfmplayground.interface import NanoTabPFNClassifier  # noqa: F401
    print(f"eval chain OK, {len(TABARENA_TASKS)} TabArena tasks", flush=True)


if __name__ == "__main__":
    clone_deps()
    install()
    self_check()
    print("\nSETUP OK. Now run:", flush=True)
    print("  !python experiments/robustness/kaggle_run.py --test pool_seed --arg 1", flush=True)
