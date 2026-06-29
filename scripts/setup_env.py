"""One command to get a teammate's machine (or the cluster) ready.

    python scripts/setup_env.py

It clones the two upstream repos, applies the one tiny import fix TFM-Playground
needs against the current TabICL, and installs everything into the active Python
environment. Cross-platform on purpose (laptops are Windows, the cluster is
Linux), so it's plain Python instead of a .ps1 / .sh pair.

Make and activate a virtual env first, e.g.:
    python -m venv .venv
    .venv\\Scripts\\activate        (Windows)
    source .venv/bin/activate       (Linux / cluster)
"""

import subprocess
import sys
import pathlib

BASE = pathlib.Path(__file__).parent.parent

REPOS = {
    "TFM-Playground": "https://github.com/automl/TFM-Playground.git",
    "tabicl": "https://github.com/soda-inria/tabicl.git",
}

DEPS = [
    "numpy", "pandas", "matplotlib", "seaborn", "scikit-learn",
    "h5py", "pyyaml", "jupyter", "ipykernel", "torch", "torchvision", "ruff",
]

# The current TabICL only exposes PriorDataset from tabicl.prior (the module was
# renamed to _dataset), but TFM-Playground still imports the old path. One line.
PATCH_FILE = BASE / "TFM-Playground" / "tfmplayground" / "external_priors" / "tabicl.py"
OLD_IMPORT = "from tabicl.prior.dataset import PriorDataset as TabICLPriorDataset"
NEW_IMPORT = "from tabicl.prior import PriorDataset as TabICLPriorDataset"


def run(cmd):
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def clone_repos():
    for name, url in REPOS.items():
        target = BASE / name
        if target.exists():
            print(f"  {name} already cloned, skipping")
        else:
            run(["git", "clone", url, str(target)])


def apply_patch():
    text = PATCH_FILE.read_text(encoding="utf-8")
    if NEW_IMPORT in text:
        print("  patch already applied")
    elif OLD_IMPORT in text:
        PATCH_FILE.write_text(text.replace(OLD_IMPORT, NEW_IMPORT), encoding="utf-8")
        print("  patched TFM-Playground import")
    else:
        print("  WARNING: couldn't find the import line to patch - check it by hand")


def install():
    # editable installs of the two repos, plus the rest of the deps
    run([sys.executable, "-m", "pip", "install",
         "-e", str(BASE / "TFM-Playground"), "-e", str(BASE / "tabicl"), *DEPS])


def main():
    print("1) cloning upstream repos")
    clone_repos()
    print("2) applying import patch")
    apply_patch()
    print("3) installing dependencies (this pulls torch, takes a while)")
    install()
    print("\ndone. quick check:")
    print("    python scripts/demo_scheduler.py")


if __name__ == "__main__":
    main()
