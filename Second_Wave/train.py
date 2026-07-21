import argparse
import csv
import json
import math
import pathlib
import sys
import time

BASE = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE.parent / "TFM-Playground"))
sys.path.insert(0, str(BASE.parent / "tabicl"))
sys.path.insert(0, str(BASE))

import numpy as np
import schedulefree
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import get_default_device, set_randomness_seed

from plot import plot_run
from pool import load_pool, n_classes, order_indices, ordering_profile
from prior import build_validation


def clean(xs):
    return [v for v in xs if v is not None and not (isinstance(v, float) and math.isnan(v))]


def mean(xs):
    xs = clean(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else round(v, 4)


def approx_flops(rows, features, model):
    e = model.embedding_size
    attention = rows * rows * e
    feedforward = rows * e * model.mlp_hidden_size
    encoding = rows * features * e
    forward = encoding + model.num_layers * (attention + feedforward)
    return 3.0 * forward


def gpu_peak_gb(device):
    if str(device).startswith("cuda"):
        gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        return gb
    return 0.0


def batch_auc(y_true, y_proba):
    classes = np.unique(y_true)
    if len(classes) < 2:
        return None
    try:
        if len(classes) == 2:
            pos = int(classes[1])
            return float(roc_auc_score(y_true == pos, y_proba[:, pos]))
        p = y_proba[:, classes]
        p = p / p.sum(axis=1, keepdims=True)
        return float(roc_auc_score(y_true, p, multi_class="ovr", labels=classes))
    except Exception:
        return None


@torch.no_grad()
def run_validation(model, bands, criterion):
    n_out = model.num_outputs
    results = {}
    pooled_loss, pooled_acc, pooled_auc = [], [], []
    for name, batches in bands.items():
        losses, accs, aucs = [], [], []
        for x, y, split in batches:
            out = model((x, y[:, :split]), train_test_split_index=split).view(-1, n_out)
            tgt = y[:, split:].reshape(-1).long()
            losses.append(criterion(out, tgt).item())
            accs.append((out.argmax(-1) == tgt).float().mean().item())
            probs = torch.softmax(out, dim=-1).cpu().numpy()
            aucs.append(batch_auc(tgt.cpu().numpy(), probs))
        results[name] = (mean(losses), mean(accs), mean(aucs))
        pooled_loss += losses
        pooled_acc += accs
        pooled_auc += aucs
    results["all"] = (mean(pooled_loss), mean(pooled_acc), mean(pooled_auc))
    return results


def save_checkpoint(model, path):
    torch.save({
        "architecture": {
            "num_attention_heads": int(model.num_attention_heads),
            "embedding_size": int(model.embedding_size),
            "mlp_hidden_size": int(model.mlp_hidden_size),
            "num_layers": int(model.num_layers),
            "num_outputs": int(model.num_outputs),
        },
        "model": model.state_dict(),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.name is not None:
        cfg["name"] = args.name
    if args.seed is not None:
        cfg["seed"] = args.seed

    name = cfg["name"]
    seed = cfg.get("seed", 42)
    set_randomness_seed(seed)
    device = get_default_device()

    max_classes = cfg.get("max_classes", 10)
    num_datapoints = cfg.get("num_datapoints", 200)
    total_steps = cfg["total_steps"]
    grad_accum = cfg.get("grad_accum", 32)
    eval_every = cfg.get("eval_every", 100)

    items = load_pool(BASE / cfg["pool"])
    order = order_indices(items, cfg["order"], seed, cfg.get("restarts", 3),
                          cfg.get("axes"), cfg.get("weights"))
    profile = ordering_profile(items, order)
    cursor = 0

    model = NanoTabPFNModel(
        num_attention_heads=cfg["heads"],
        embedding_size=cfg["embedding_size"],
        mlp_hidden_size=cfg["hidden_size"],
        num_layers=cfg["layers"],
        num_outputs=max_classes,
    ).to(device)

    bands = build_validation(device, max_classes, num_datapoints, cfg.get("val_per_band", 32))
    criterion = nn.CrossEntropyLoss()
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg["lr"], weight_decay=0.0)

    out_dir = BASE / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config) as f:
        (out_dir / "config.yaml").write_text(f.read())
    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "step", "mean_features", "mean_classes", "mean_context",
            "train_loss", "val_loss", "val_acc", "val_auc",
            "val_auc_easy", "val_auc_medium", "val_auc_hard",
            "cum_time_s", "cum_flops", "peak_gpu_gb",
        ])

    print(f"training {name} | order {cfg['order']} | {total_steps} steps x {grad_accum} accum "
          f"| pool {len(items)} | device {device}", flush=True)
    print(f"ordering profile (spearman of position vs axis): {profile}", flush=True)
    cum_time = 0.0
    cum_flops = 0.0
    overall_peak_gpu = 0.0
    gpu_peak_gb(device)
    for step in range(1, total_steps + 1):
        model.train()
        optimizer.train()
        t0 = time.time()
        optimizer.zero_grad()
        running = 0.0
        used = 0
        step_features = []
        step_classes = []
        step_context = []
        for _ in range(grad_accum):
            it = items[order[cursor % len(order)]]
            cursor += 1
            x, y, split = it["x"].to(device), it["y"].to(device), it["split"]
            out = model((x, y[:, :split]), train_test_split_index=split).view(-1, model.num_outputs)
            tgt = y[:, split:].reshape(-1).long()
            loss = criterion(out, tgt) / grad_accum
            loss.backward()
            running += loss.item() * grad_accum
            used += 1
            step_features.append(it["n_features"])
            step_classes.append(n_classes(it))
            step_context.append(split)
            cum_flops += approx_flops(x.shape[1], it["n_features"], model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        cum_time += time.time() - t0

        if step % eval_every == 0 or step == total_steps:
            peak = gpu_peak_gb(device)
            overall_peak_gpu = max(overall_peak_gpu, peak)
            model.eval()
            optimizer.eval()
            r = run_validation(model, bands, criterion)
            vl, va, vauc = r["all"]
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    step, round(mean(step_features), 2), round(mean(step_classes), 2),
                    round(mean(step_context), 2),
                    round(running / max(used, 1), 6),
                    round(vl, 6), round(va, 4), fmt(vauc),
                    fmt(r["easy"][2]), fmt(r["medium"][2]), fmt(r["hard"][2]),
                    round(cum_time, 2), round(cum_flops, 1), round(peak, 3),
                ])
            print(f"step {step:5d} | feat {mean(step_features):5.1f} | train {running / max(used, 1):.4f} "
                  f"| val_loss {vl:.4f} | val_auc {vauc:.4f} | {cum_time:.0f}s | {peak:.2f}GB", flush=True)

    model.eval()
    optimizer.eval()
    save_checkpoint(model, out_dir / "checkpoint.pth")

    meta = {
        "name": name,
        "seed": seed,
        "order": cfg["order"],
        "axes": cfg.get("axes") or None,
        "weights": cfg.get("weights") or None,
        "ordering_profile": profile,
        "device": str(device),
        "total_steps": total_steps,
        "grad_accum": grad_accum,
        "effective_batch": grad_accum,
        "pool_size": len(items),
        "elapsed_s": round(cum_time, 1),
        "cum_flops": round(cum_flops, 1),
        "peak_gpu_gb": round(overall_peak_gpu, 3),
        "num_outputs": max_classes,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    plot_run(out_dir)
    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
