"""Kaggle environment setup for the robustness tests.

Reproduces what `run_tests.sh` gets for free on the cluster (module load + an
already-built .venv): the two upstream repos at the SAME pinned commits the
cluster env uses, plus the handful of real deps this pipeline imports.

    TFM-Playground  98e33be
    tabicl          8f665ed

Both pins matter: leaving them unpinned would silently change the prior, and
therefore the data, which is the one thing these tests must hold fixed.

The two pins are NOT compatible out of the box, and a one-line source patch IS
required. (An earlier version of this docstring claimed 98e33be removed the need
for it -- that was wrong, and it is why the Kaggle run failed here.) At 98e33be,
TFM-Playground does:

    from tabicl.prior.dataset import PriorDataset

but tabicl 8f665ed renamed that module to the private `_dataset.py` and re-exports
the class from the subpackage, so the public path is:

    from tabicl.prior import PriorDataset

`prior/dataset.py` does not exist at 8f665ed. The cluster .venv has this patch
applied by hand as an uncommitted edit in its TFM-Playground checkout, which is
why the cluster runs work and a clean clone does not. apply_source_patch() below
reproduces that edit so Kaggle matches the environment every existing result was
produced in.

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

# Installed with --no-deps and pinned to what TFM-Playground declares. pfns wants
# torch>=2.5 plus botorch/gpytorch; resolving those can replace Kaggle's CUDA torch
# build with a CPU wheel, which would silently cost us the GPU. We need exactly one
# symbol from it -- pfns.bar_distribution.FullSupportBarDistribution, imported at
# module level by tfmplayground/interface.py for the REGRESSION head we never use --
# so its transitive deps are dead weight here. self_check() imports that submodule
# directly to prove --no-deps left it usable.
NODEP_DEPS = ["pfns==0.3.0"]


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


"""(path, old, new) edits needed to make the two pinned commits work together.
Applied after checkout, since checkout resets the tree."""
SOURCE_PATCHES = [
    (
        "TFM-Playground/tfmplayground/external_priors/tabicl.py",
        "from tabicl.prior.dataset import PriorDataset as TabICLPriorDataset",
        "from tabicl.prior import PriorDataset as TabICLPriorDataset",
    ),
    # external_priors/__init__.py eagerly imports all four prior backends, so
    # `from tfmplayground.external_priors import TabICLPriorDataLoader` needs ticl
    # and tabpfn_prior installed -- two git dependencies we never call. This
    # pipeline only ever builds the TabICL prior (final/code/prior.py). Rather than
    # install heavy VCS deps (which also declare their own torch) to satisfy an
    # import we do not use, make the unused backends degrade to a clear error only
    # if something actually touches them. The TabICL path is untouched, so the
    # prior -- and therefore the data -- is bit-for-bit what the cluster produced.
    (
        "TFM-Playground/tfmplayground/external_priors/__init__.py",
        "from .tabpfn import TabPFNPriorDataLoader, build_tabpfn_prior\n"
        "from .ticl import TICLPriorDataLoader, build_ticl_prior",
        "try:  # optional: requires the tabpfn-v1-prior git dependency\n"
        "    from .tabpfn import TabPFNPriorDataLoader, build_tabpfn_prior\n"
        "except ModuleNotFoundError as _e:  # pragma: no cover\n"
        "    _tabpfn_err = _e\n"
        "    def _missing_tabpfn(*a, **k):\n"
        "        raise ModuleNotFoundError(\n"
        "            'TabPFN prior unavailable: ' + str(_tabpfn_err)) from _tabpfn_err\n"
        "    TabPFNPriorDataLoader = build_tabpfn_prior = _missing_tabpfn\n"
        "try:  # optional: requires the ticl git dependency\n"
        "    from .ticl import TICLPriorDataLoader, build_ticl_prior\n"
        "except ModuleNotFoundError as _e:  # pragma: no cover\n"
        "    _ticl_err = _e\n"
        "    def _missing_ticl(*a, **k):\n"
        "        raise ModuleNotFoundError(\n"
        "            'TICL prior unavailable: ' + str(_ticl_err)) from _ticl_err\n"
        "    TICLPriorDataLoader = build_ticl_prior = _missing_ticl",
    ),
]


def apply_source_patch():
    for rel, old, new in SOURCE_PATCHES:
        p = ROOT / rel
        if not p.exists():
            raise SystemExit(f"expected {rel} after checkout, not found")
        src = p.read_text()
        if new in src:
            print(f"  patch already applied: {rel}", flush=True)
            continue
        if old not in src:
            # Upstream moved: fail loudly rather than train against a prior that
            # silently differs from the one every existing result used.
            raise SystemExit(
                f"cannot patch {rel}: expected line not found.\n"
                f"  looking for: {old}\n"
                "Upstream changed; re-check the pins before running.")
        p.write_text(src.replace(old, new))
        print(f"  patched {rel}", flush=True)


def install():
    sh(f"{sys.executable} -m pip -q install " + " ".join(f"'{d}'" for d in DEPS))
    sh(f"{sys.executable} -m pip -q install --no-deps "
       + " ".join(f"'{d}'" for d in NODEP_DEPS))
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
    elif torch.version.cuda is None:
        # A CPU-only wheel means an install resolved torch and replaced Kaggle's
        # CUDA build. Training would fall back to CPU and never finish, so stop
        # here rather than burn a 12 h commit discovering it.
        raise SystemExit(
            f"torch {torch.__version__} is a CPU-only build — an install replaced "
            "Kaggle's CUDA torch. Start a fresh session; do not train on this env.")
    else:
        print("WARNING: no CUDA — Settings > Accelerator > GPU.", flush=True)

    import schedulefree  # noqa: F401
    from scipy.stats import spearmanr  # noqa: F401  <- ordering_profile
    # pfns is installed --no-deps, so confirm the one submodule we actually need
    # imports without botorch/gpytorch present.
    from pfns.bar_distribution import FullSupportBarDistribution  # noqa: F401
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
    apply_source_patch()  # after checkout, which resets the tree
    install()
    self_check()
    print("\nSETUP OK. Now run:", flush=True)
    print("  !python experiments/robustness/kaggle_run.py --test pool_seed --arg 1", flush=True)
