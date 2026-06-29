"""
01_verify_env.py
Run this first to confirm every dependency is importable and the
NanoTabPFN classifier works end-to-end.

Usage:
    python scripts/01_verify_env.py
"""

import sys
import importlib

BASE = __file__.rsplit("scripts", 1)[0]
sys.path.insert(0, BASE + "TFM-Playground")
sys.path.insert(0, BASE + "tabicl")

REQUIRED = [
    "numpy", "pandas", "matplotlib", "seaborn",
    "sklearn", "torch", "torchvision", "h5py", "ruff",
    "jupyter", "notebook", "ipykernel",
]

print("=" * 60)
print("Environment Verification")
print("=" * 60)

all_ok = True
for pkg in REQUIRED:
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "?")
        print(f"  ✓ {pkg:<20} {ver}")
    except ImportError as e:
        print(f"  ✗ {pkg:<20} MISSING — {e}")
        all_ok = False

# TFM-Playground
print()
print("Testing NanoTabPFNClassifier...")
try:
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from tfmplayground import NanoTabPFNClassifier

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    clf = NanoTabPFNClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs[:, 1])
    print(f"  ✓ Accuracy : {acc:.4f}")
    print(f"  ✓ ROC AUC  : {auc:.4f}")
except Exception as e:
    print(f"  ✗ NanoTabPFNClassifier failed: {e}")
    all_ok = False

# TabICL prior
print()
print("Testing TabICL prior generation...")
try:
    from tfmplayground.external_priors import TabICLPriorDataLoader

    prior = TabICLPriorDataLoader(
        num_steps=3, batch_size=2, num_datapoints_max=50,
        min_features=3, max_features=3, max_num_classes=3, device="cpu"
    )
    batch = next(iter(prior))
    print(f"  ✓ Batch type : {type(batch)}")
    if hasattr(batch, "keys"):
        print(f"  ✓ Batch keys : {list(batch.keys())}")
except Exception as e:
    print(f"  ✗ TabICL prior failed: {e}")
    all_ok = False

print()
print("=" * 60)
if all_ok:
    print("ALL CHECKS PASSED — environment is ready!")
else:
    print("SOME CHECKS FAILED — see above for details.")
print("=" * 60)
