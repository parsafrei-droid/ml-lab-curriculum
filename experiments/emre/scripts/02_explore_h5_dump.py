"""
02_explore_h5_dump.py
Load and inspect the pretrained data dumps.

Usage:
    python scripts/02_explore_h5_dump.py
"""

import pathlib

BASE = pathlib.Path(__file__).parent.parent

import h5py

DATA_DIR = BASE / "data"

FILES = {
    "classification": DATA_DIR / "50x3_3_100k_classification.h5",
    "regression":     DATA_DIR / "50x3_1280k_regression.h5",
}

for name, path in FILES.items():
    print(f"\n{'=' * 60}")
    print(f"  {name.upper()} dump: {path.name}")
    print(f"{'=' * 60}")

    if not path.exists():
        print(f"  ✗ File not found: {path}")
        print("    Run setup.ps1 first to download the data dumps.")
        continue

    size_gb = path.stat().st_size / 1e9
    print(f"  File size : {size_gb:.2f} GB")

    with h5py.File(path, "r") as f:
        print(f"  Top-level keys: {list(f.keys())}")
        for key in list(f.keys())[:5]:
            ds = f[key]
            if hasattr(ds, "shape"):
                print(f"    [{key}]  shape={ds.shape}  dtype={ds.dtype}")
            else:
                # Group
                print(f"    [{key}]  (group) sub-keys: {list(ds.keys())[:5]}")

print("\nDone.")
