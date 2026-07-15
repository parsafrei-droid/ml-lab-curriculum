"""One-shot Colab setup for the Phase-2 runs. NO notebook, NO cell ordering.

Run this ONCE at the top of a fresh Colab (GPU) session:

    !git clone --branch curriculum-fixes --single-branch \
        https://github.com/parsafrei-droid/ml-lab-curriculum.git
    %cd ml-lab-curriculum
    !python phase2/colab_setup.py

It clones the two upstream deps at pinned commits, installs the minimal real deps,
and writes tiny stub packages for the prior backends this project imports at module
load but never CALLS on the classification path (pfns / tabpfn_prior / ticl). The
stub name set was verified against the tfmplayground source with an AST check, so it
is complete — not guessed.

After it prints "SETUP OK", run a training scenario, e.g.:

    !python scripts/run.py --config phase2/configs/curriculum_features.yaml \
        --epochs 5 --name smoke_curriculum_features_colab
"""

import site
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path("/content/ml-lab-curriculum")
PIN = {"TFM-Playground": ("https://github.com/automl/TFM-Playground.git", "98e33be"),
       "tabicl": ("https://github.com/soda-inria/tabicl.git", "8f665ed")}


def sh(cmd):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def clone_deps():
    for name, (url, commit) in PIN.items():
        d = REPO / name
        if not d.exists():
            sh(f"git clone {url} {d}")
        sh(f"git -C {d} checkout {commit}")


# Local patches that exist in our working copy but are NOT committed upstream, and
# must be reapplied on a fresh clone. Each: (file, old_line, new_line).
# - tabicl.py: TFM-Playground@98e33be imports `from tabicl.prior.dataset import
#   PriorDataset`, but the pinned tabicl (8f665ed) exposes PriorDataset only from
#   `tabicl.prior`. Without this, the very first prior import fails.
PATCHES = [
    (REPO / "TFM-Playground" / "tfmplayground" / "external_priors" / "tabicl.py",
     "from tabicl.prior.dataset import PriorDataset as TabICLPriorDataset",
     "from tabicl.prior import PriorDataset as TabICLPriorDataset"),
]


def patch_deps():
    for path, old, new in PATCHES:
        text = path.read_text()
        if new in text:
            print(f"patch already applied: {path.name}")
            continue
        if old not in text:
            raise RuntimeError(
                f"PATCH FAILED: expected line not found in {path}:\n  {old}\n"
                f"(the pinned commit may have changed — re-check the pin)")
        path.write_text(text.replace(old, new))
        print(f"patched {path.name}: '{old}' -> '{new}'")


def install():
    # openml is needed for the TabArena evaluation step (evaluation.py downloads the
    # OpenML datasets). The rest are the training-chain deps.
    sh(f"{sys.executable} -m pip -q install schedulefree einops huggingface-hub "
       f"'scikit-learn>=1.5' pandas requests h5py openml")
    # --no-deps so neither package drags torch to a different version than Colab's
    sh(f"{sys.executable} -m pip -q install --no-deps -e {REPO}/tabicl")
    sh(f"{sys.executable} -m pip -q install --no-deps -e {REPO}/TFM-Playground")


# module_relpath -> file contents. Verified complete against tfmplayground source.
STUB_FILES = {
    "pfns/__init__.py": "",
    "pfns/bar_distribution.py": textwrap.dedent("""\
        # stub: imported at module load; only used on the regression path, which the
        # classification smoke/real runs (CrossEntropyLoss) never execute.
        class FullSupportBarDistribution:
            pass

        def get_bucket_limits(*args, **kwargs):
            raise NotImplementedError("pfns stub: regression-only, not used here")
        """),
    "tabpfn_prior/__init__.py": "class TabPFNPriorDataLoader:  # stub\n    pass\n",
    "ticl/__init__.py": "",
    "ticl/dataloader.py": "class PriorDataLoader:  # stub\n    pass\n",
    "ticl/priors.py": textwrap.dedent("""\
        class BooleanConjunctionPrior: pass
        class ClassificationAdapterPrior: pass
        class GPPrior: pass
        class MLPPrior: pass
        class StepFunctionPrior: pass
        """),
}


def write_stubs():
    sp = Path(site.getsitepackages()[0])
    for rel, content in STUB_FILES.items():
        f = sp / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
    print(f"wrote {len(STUB_FILES)} stub files under {sp}")


def self_check():
    # Import EXACTLY the chain scripts/run.py uses, in THIS process, so any missing
    # name fails here with a clear message instead of mid-training.
    for p in (REPO, REPO / "TFM-Playground", REPO / "tabicl" / "src", REPO / "tabicl"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import warnings
    warnings.filterwarnings("ignore")

    import torch
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
    from tabicl.prior._prior_config import DEFAULT_SAMPLED_HP  # noqa: F401
    from tfmplayground.models.nanotabpfn import NanoTabPFNModel  # noqa: F401
    from tfmplayground.train import train  # noqa: F401
    from tfmplayground.utils import get_default_device  # noqa: F401
    from tfmplayground.external_priors import TabICLPriorDataLoader  # noqa: F401
    from curriculum.prior import make_prior  # noqa: F401  <- the real run.py entrypoint
    from curriculum.scheduler import CurriculumScheduler  # noqa: F401
    # also import the EVAL chain, so setup fails here (not after a long train) if the
    # TabArena eval deps (openml, etc.) are missing:
    from tfmplayground.evaluation import TABARENA_TASKS, get_openml_predictions  # noqa: F401
    from tfmplayground.interface import NanoTabPFNClassifier  # noqa: F401
    if not torch.cuda.is_available():
        print("WARNING: no CUDA — set Runtime > Change runtime type > T4 GPU.")


if __name__ == "__main__":
    clone_deps()
    patch_deps()
    install()
    write_stubs()
    self_check()
    print("\nSETUP OK — the full run.py import chain resolves. Now run:")
    print("  !python scripts/run.py --config phase2/configs/curriculum_features.yaml "
          "--epochs 5 --name smoke_curriculum_features_colab")
