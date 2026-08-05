"""Score the untrained model, so the curves can start at step 0 with a measured point.

At step 0 no ordering has been applied yet, so the value depends only on the seed and is
shared by every ordering of that seed. Writes final/figures/step0_val_auc.json, which
step_curve.py and auc_curve_table.py both read instead of assuming chance.

Runs on CPU in a couple of minutes; the cost is one validation pass per seed.
"""

import json
import pathlib
import sys

SECOND_WAVE = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "second_wave"
sys.path.insert(0, str(SECOND_WAVE.parent.parent / "TFM-Playground"))
sys.path.insert(0, str(SECOND_WAVE.parent.parent / "tabicl"))
sys.path.insert(0, str(SECOND_WAVE))

import torch
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import set_randomness_seed

from prior import build_validation
from train import run_validation

SEEDS = (42, 1, 2)


def main(out_path):
    device = torch.device("cpu")
    bands = build_validation(device, 10, 200, 32)
    criterion = nn.CrossEntropyLoss()

    scores = {}
    for seed in SEEDS:
        set_randomness_seed(seed)
        model = NanoTabPFNModel(num_attention_heads=4, embedding_size=96,
                                mlp_hidden_size=192, num_layers=3, num_outputs=10).to(device)
        model.eval()
        scores[f"s{seed}"] = round(run_validation(model, bands, criterion)["all"][2], 4)
        print(f"seed {seed}: {scores[f's{seed}']}", flush=True)

    pathlib.Path(out_path).write_text(json.dumps(scores, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main(pathlib.Path(__file__).parent.parent / "figures" / "step0_val_auc.json")
