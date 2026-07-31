"""
03_run_smoke_pretrain.py
Smoke-test pretraining job: 2 epochs, 5 steps.
Mirrors Step 9 of the setup guide.

Usage:
    python scripts/03_run_smoke_pretrain.py
"""

import sys
import subprocess
import pathlib

BASE = pathlib.Path(__file__).parent.parent
TFM  = BASE / "TFM-Playground"
DATA = BASE / "data" / "50x3_3_100k_classification.h5"
PYTHON = sys.executable

if not TFM.exists():
    print("TFM-Playground not cloned yet — run setup.ps1 first.")
    sys.exit(1)

if not DATA.exists():
    print(f"Data dump not found: {DATA}")
    print("Run setup.ps1 to download it first.")
    sys.exit(1)

cmd = [
    PYTHON, str(TFM / "pretrain_classification.py"),
    "--epochs", "2",
    "--steps", "5",
    "--batchsize", "10",
    "--priordump", str(DATA),
]

print("Running:", " ".join(str(c) for c in cmd))
print()
result = subprocess.run(cmd, cwd=str(TFM))
sys.exit(result.returncode)
