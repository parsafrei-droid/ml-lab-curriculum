"""Evaluate a trained checkpoint on TabArena (or the toy tasks) and save scores.

    python scripts/eval_tabarena.py --checkpoint results/scenario_A/checkpoint.pth
    python scripts/eval_tabarena.py --checkpoint results/scenario_A/checkpoint.pth --tasks tabarena

--tasks toy      -> 3 small sklearn-ish datasets, fast, for a sanity check
--tasks tabarena -> the full ~50-dataset benchmark, slow, for the real numbers

Writes results/<name>/tabarena_scores.json with per-dataset ROC-AUC and the mean.
The final comparison across scenarios is just reading these json files.
"""

import argparse
import json
import pathlib
import sys

BASE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "TFM-Playground"))
sys.path.insert(0, str(BASE / "tabicl"))

from sklearn.metrics import roc_auc_score

from tfmplayground.evaluation import TABARENA_TASKS, TOY_TASKS_CLASSIFICATION, get_openml_predictions
from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.utils import get_default_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to a checkpoint.pth")
    parser.add_argument("--tasks", choices=["toy", "tabarena"], default="toy")
    parser.add_argument(
        "--max-n-samples", type=int, default=5_000,
        help="skip OpenML tasks with more rows than this - datapoint attention is "
             "O(n^2) in rows and large tasks OOM the GPU (default 5000)",
    )
    args = parser.parse_args()

    device = get_default_device()
    # NanoTabPFNClassifier reads the architecture straight out of the checkpoint.
    classifier = NanoTabPFNClassifier(model=args.checkpoint, device=device)

    tasks = TOY_TASKS_CLASSIFICATION if args.tasks == "toy" else TABARENA_TASKS
    print(f"evaluating {args.checkpoint} on {len(tasks)} {args.tasks} task(s)..."
          f" (max_n_samples={args.max_n_samples})", flush=True)

    predictions = get_openml_predictions(
        model=classifier, classification=True, tasks=tasks,
        max_n_samples=args.max_n_samples,
    )

    scores = {}
    for name, (y_true, _y_pred, y_proba) in predictions.items():
        scores[name] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        print(f"  {name:30s} roc_auc={scores[name]:.4f}", flush=True)

    mean_auc = sum(scores.values()) / len(scores)
    out = {"checkpoint": args.checkpoint, "task_set": args.tasks,
           "mean_roc_auc": mean_auc, "per_dataset": scores}

    out_path = pathlib.Path(args.checkpoint).parent / "tabarena_scores.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nmean roc_auc = {mean_auc:.4f}  ->  {out_path}")


if __name__ == "__main__":
    main()
