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
import torch
import yaml
from torch import nn

from curriculum.prior import make_prior, make_validation_batches
from curriculum.scheduler import CurriculumScheduler
from tfmplayground.callbacks import Callback
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed


class LossLoggerCallback(Callback):
    """Writes loss per epoch to a CSV and draws the loss curve at the end.

    We deliberately keep TabArena out of training - evaluating on ~300 real
    datasets every epoch would dwarf the training time. Evaluation happens once,
    afterwards, via eval_tabarena.py.
    """

    def __init__(self, out_dir, device):
        self.out_dir = out_dir
        self.device = device
        self.rows = []  # (epoch, loss)
        self.cum_time = 0.0
        self.peak_gpu_gb = 0.0
        self.csv_path = out_dir / "loss.csv"
        self.csv_path.write_text("epoch,epoch_time_s,cum_time_s,loss,gpu_mem_gb\n")

    def _gpu_gb(self):
        # peak memory since the last reset, in GB (0 on CPU)
        if str(self.device).startswith("cuda"):
            gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            return gb
        return 0.0

    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        self.cum_time += epoch_time
        gpu = self._gpu_gb()
        self.peak_gpu_gb = max(self.peak_gpu_gb, gpu)
        self.rows.append((epoch, loss))
        with self.csv_path.open("a") as f:
            f.write(f"{epoch},{epoch_time:.3f},{self.cum_time:.3f},{loss:.6f},{gpu:.3f}\n")
        print(f"epoch {epoch:4d} | time {epoch_time:6.2f}s | cum {self.cum_time:7.1f}s "
              f"| loss {loss:.4f} | gpu {gpu:.2f}GB", flush=True)

    def close(self):
        if not self.rows:
            return
        epochs = [r[0] for r in self.rows]
        losses = [r[1] for r in self.rows]
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, losses, marker="o", ms=3)
        plt.xlabel("epoch")
        plt.ylabel("mean loss")
        plt.title(self.out_dir.name)
        plt.tight_layout()
        plt.savefig(self.out_dir / "loss_curve.png", dpi=120)
        plt.close()


class FixedValidationCallback(Callback):
    """Scores the model each epoch on ONE shared validation set.

    Every scenario is judged on the same fixed synthetic datasets, so val_loss is
    directly comparable across runs - unlike training loss, which just reflects
    whatever difficulty a run ends on. This is the honest training-time signal;
    TabArena (via eval_tabarena.py) is the independent ground truth.
    """

    def __init__(self, out_dir, device, n=16):
        self.out_dir = out_dir
        self.batches = make_validation_batches(device, n=n)
        self.criterion = nn.CrossEntropyLoss()
        self.rows = []  # (epoch, val_loss, val_acc)
        self.final = None
        self.csv_path = out_dir / "val.csv"
        self.csv_path.write_text("epoch,val_loss,val_acc\n")

    @torch.no_grad()
    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        # train() has already put the model (and schedule-free optimizer) in eval
        # mode before calling us, so the weights here are the right ones to score.
        n_out = model.num_outputs
        losses, accs = [], []
        for x, y, split in self.batches:
            out = model((x, y[:, :split]), train_test_split_index=split).view(-1, n_out)
            tgt = y[:, split:].reshape(-1).long()
            losses.append(self.criterion(out, tgt).item())
            accs.append((out.argmax(-1) == tgt).float().mean().item())
        vl, va = float(sum(losses) / len(losses)), float(sum(accs) / len(accs))
        self.rows.append((epoch, vl, va))
        self.final = (vl, va)
        with self.csv_path.open("a") as f:
            f.write(f"{epoch},{vl:.6f},{va:.4f}\n")
        print(f"            | val_loss {vl:.4f} | val_acc {va:.4f}", flush=True)

    def close(self):
        if not self.rows:
            return
        epochs = [r[0] for r in self.rows]
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, [r[1] for r in self.rows], marker="o", ms=3)
        plt.xlabel("epoch")
        plt.ylabel("validation loss (shared set)")
        plt.title(self.out_dir.name)
        plt.tight_layout()
        plt.savefig(self.out_dir / "val_loss_curve.png", dpi=120)
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

    # Build the prior at the first stage's settings; the scheduler takes over from
    # there. make_prior gives it a private sampled_hp so internal knobs (noise etc.)
    # are controllable, and applies the first stage's knobs up front.
    first = schedule[min(schedule)]
    external = {k: v for k, v in first.items() if k in ("min_features", "max_features")}
    internal = {k: v for k, v in first.items() if k not in ("min_features", "max_features", "max_classes")}
    prior = make_prior(
        max_features=first["max_features"],
        max_classes=num_outputs,
        min_features=external.get("min_features", 2),
        num_datapoints=cfg.get("num_datapoints", 200),
        num_steps=cfg["steps"],
        batch_size=cfg.get("batch_size", 1),
        device=device,
        knobs=internal or None,
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

    logger = LossLoggerCallback(out_dir, device)
    validator = FixedValidationCallback(out_dir, device)
    start = time.time()
    train(
        model=model,
        prior=scheduler,
        criterion=nn.CrossEntropyLoss(),
        epochs=cfg["epochs"],
        lr=cfg.get("lr", 1e-4),
        device=device,
        callbacks=[logger, validator],
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

    total_steps = cfg["epochs"] * cfg["steps"]
    final_loss = logger.rows[-1][1] if logger.rows else None
    meta = {
        "name": name,
        "seed": cfg.get("seed", 42),
        "device": str(device),
        "total_steps": total_steps,
        "elapsed_s": round(elapsed, 1),
        # compute-resource summary, so we can compare "same compute" fairly
        "sec_per_step": round(elapsed / total_steps, 4),
        "steps_per_sec": round(total_steps / elapsed, 2),
        "peak_gpu_gb": round(logger.peak_gpu_gb, 3),
        "final_train_loss": round(final_loss, 4) if final_loss is not None else None,
        # comparable across scenarios (same validation set for everyone)
        "final_val_loss": round(validator.final[0], 4) if validator.final else None,
        "final_val_acc": round(validator.final[1], 4) if validator.final else None,
        "num_outputs": num_outputs,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {elapsed:.1f}s ({meta['sec_per_step']}s/step) -> {out_dir}")


if __name__ == "__main__":
    main()
