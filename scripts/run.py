"""Train one curriculum scenario from a YAML config.

This is what each teammate runs on their own machine / the cluster:

    python scripts/run.py --config experiments/configs/scenario_A.yaml

It reads the config, builds the TabICL prior, wraps it in the curriculum
scheduler, and hands the whole thing to TFM-Playground's own train() loop. The
scheduler advances the difficulty as the loop pulls batches, so we don't touch
the upstream training code at all.

Outputs land in results/<name>/ :
    checkpoint.pth   - the trained model (architecture + weights)
    loss.csv         - loss per epoch
    loss_curve.png   - that same loss, plotted
    meta.json        - seed, total steps, wall-clock time
"""

import argparse
import json
import pathlib
import shutil
import sys
import time

BASE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "TFM-Playground"))
sys.path.insert(0, str(BASE / "tabicl"))
sys.path.insert(0, str(BASE))

import matplotlib.pyplot as plt
import yaml
from torch import nn

from curriculum.scheduler import CurriculumScheduler
from tfmplayground.callbacks import Callback
from tfmplayground.external_priors import TabICLPriorDataLoader
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed


class LossLoggerCallback(Callback):
    """Writes loss per epoch to a CSV and draws the loss curve at the end.

    We deliberately keep TabArena out of training - evaluating on ~300 real
    datasets every epoch would dwarf the training time. Evaluation happens once,
    afterwards, via eval_tabarena.py.
    """

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.rows = []  # (epoch, time, loss)
        self.csv_path = out_dir / "loss.csv"
        self.csv_path.write_text("epoch,time_s,loss\n")

    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        self.rows.append((epoch, epoch_time, loss))
        with self.csv_path.open("a") as f:
            f.write(f"{epoch},{epoch_time:.3f},{loss:.6f}\n")
        print(f"epoch {epoch:4d} | time {epoch_time:6.2f}s | loss {loss:.4f}", flush=True)

    def close(self):
        if not self.rows:
            return
        epochs = [r[0] for r in self.rows]
        losses = [r[2] for r in self.rows]
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, losses, marker="o", ms=3)
        plt.xlabel("epoch")
        plt.ylabel("mean loss")
        plt.title(self.out_dir.name)
        plt.tight_layout()
        plt.savefig(self.out_dir / "loss_curve.png", dpi=120)
        plt.close()


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # YAML keys under `schedule` come in as ints already (0:, 200: ...), good.
    cfg["schedule"] = {int(k): v for k, v in cfg["schedule"].items()}
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a scenario YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg["name"]
    schedule = cfg["schedule"]

    set_randomness_seed(cfg.get("seed", 42))
    device = get_default_device()

    # The model architecture is fixed for the whole run, so its number of output
    # classes has to cover the hardest stage we'll ever reach.
    num_outputs = max(stage.get("max_classes", 2) for stage in schedule.values())

    # Build the prior at the first stage's settings; the scheduler takes over from there.
    first = schedule[min(schedule)]
    num_datapoints = cfg.get("num_datapoints", 200)
    prior = TabICLPriorDataLoader(
        num_steps=cfg["steps"],
        batch_size=cfg.get("batch_size", 1),
        num_datapoints_min=num_datapoints,
        num_datapoints_max=num_datapoints + 1,
        min_features=first.get("min_features", 2),
        max_features=first["max_features"],
        max_num_classes=num_outputs,
        device=device,
    )
    scheduler = CurriculumScheduler(prior, schedule)

    model = NanoTabPFNModel(
        num_attention_heads=cfg.get("heads", 6),
        embedding_size=cfg.get("embedding_size", 192),
        mlp_hidden_size=cfg.get("hidden_size", 768),
        num_layers=cfg.get("layers", 6),
        num_outputs=num_outputs,
    )

    out_dir = BASE / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config.yaml")

    print(f"=== training '{name}' | {cfg['epochs']} epochs x {cfg['steps']} steps "
          f"| num_outputs={num_outputs} | device={device} ===")

    start = time.time()
    train(
        model=model,
        prior=scheduler,
        criterion=nn.CrossEntropyLoss(),
        epochs=cfg["epochs"],
        lr=cfg.get("lr", 1e-4),
        device=device,
        callbacks=[LossLoggerCallback(out_dir)],
        run_name=name,
    )
    elapsed = time.time() - start

    # train() saves to "workdir/<name>/" relative to wherever it ran, so check a
    # few likely spots and copy the checkpoint next to our results.
    candidates = [
        pathlib.Path.cwd() / "workdir" / name / "latest_checkpoint.pth",
        BASE / "workdir" / name / "latest_checkpoint.pth",
    ]
    ckpt_src = next((p for p in candidates if p.exists()), None)
    if ckpt_src:
        shutil.copy(ckpt_src, out_dir / "checkpoint.pth")
    else:
        print("warning: could not find the saved checkpoint to copy", flush=True)

    meta = {
        "name": name,
        "seed": cfg.get("seed", 42),
        "total_steps": cfg["epochs"] * cfg["steps"],
        "elapsed_s": round(elapsed, 1),
        "num_outputs": num_outputs,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {elapsed:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
