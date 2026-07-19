import argparse
import json
import pathlib
import sys

BASE = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE.parent / "TFM-Playground"))
sys.path.insert(0, str(BASE.parent / "tabicl"))

from sklearn.metrics import roc_auc_score

from tfmplayground.evaluation import TABARENA_TASKS, get_openml_predictions
from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.utils import get_default_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-n-samples", type=int, default=5000)
    args = parser.parse_args()

    classifier = NanoTabPFNClassifier(model=args.checkpoint, device=get_default_device())
    print(f"evaluating {args.checkpoint} on {len(TABARENA_TASKS)} tasks", flush=True)

    predictions = get_openml_predictions(
        model=classifier, classification=True, tasks=TABARENA_TASKS,
        max_n_samples=args.max_n_samples,
    )

    scores = {}
    for name, (y_true, _y_pred, y_proba) in predictions.items():
        scores[name] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        print(f"  {name:30s} {scores[name]:.4f}", flush=True)

    mean_auc = sum(scores.values()) / len(scores)
    out_path = pathlib.Path(args.checkpoint).parent / "tabarena_scores.json"
    out_path.write_text(json.dumps(
        {"checkpoint": args.checkpoint, "mean_roc_auc": mean_auc, "per_dataset": scores},
        indent=2,
    ))
    print(f"mean roc_auc = {mean_auc:.4f} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
