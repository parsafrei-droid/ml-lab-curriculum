"""Train one pool-curriculum scenario from a YAML config.

    python scripts/run_pool.py --config experiments/configs/pool_curriculum_noise_binary.yaml

This is the training entrypoint for curriculum.pool's mechanism (a fixed,
pre-generated pool of datasets, walked in a chosen order - see curriculum/pool.py's
docstring for how this differs from scripts/run.py's live-ramped CurriculumScheduler).
Build the pool first with scripts/build_pool.py.

It's a separate script from run.py, not a run_with_toy_probe.py-style monkeypatch,
because more than the callback list differs between the two modes: config parsing
(no `schedule:` block to cast), the train() call itself (pool-mode must pass
accumulate_gradients=batch_size - see below - which scheduler-mode's call never
does), and the resume-cursor formula are all genuinely different. What *is*
shared - LossLoggerCallback, FixedValidationCallback, BandedValidationCallback,
CHECKPOINT_STEPS, find_checkpoint, write_meta - is imported from run.py rather
than duplicated.

Why accumulate_gradients=batch_size: scheduler-mode's TabICLPriorDataLoader stacks
`batch_size` same-call datasets into one forward/backward pass. Pool items are
drawn one at a time from independent make_prior() calls (potentially different
regimes at build time), so they can't be stacked the same way - training does
`batch_size` separate forward/backward passes (one pool item each) per optimizer
step, exactly like a normal gradient-accumulation loop. This also means one
pool-mode "epoch" (CHECKPOINT_STEPS=100 optimizer steps, to keep the step axis
comparable to scheduler-mode's) draws CHECKPOINT_STEPS * batch_size pool items,
not just CHECKPOINT_STEPS - see PoolCurriculumLoader's construction below and the
resume-cursor comment for why this matters.
"""

import argparse
import pathlib
import sys
import time

BASE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "TFM-Playground"))
sys.path.insert(0, str(BASE / "tabicl"))
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))

import shutil

import torch
import yaml
from torch import nn

import run as run_module  # scripts/run.py - reused helpers, not re-derived
from curriculum.pool import PoolCurriculumLoader, load_pool, order_indices
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed

CHECKPOINT_STEPS = run_module.CHECKPOINT_STEPS


def load_pool_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "schedule" in cfg:
        raise SystemExit(f"{path}: pool-mode config must not have a `schedule:` block "
                          f"(that's scheduler-mode - use scripts/run.py instead)")
    if "pool" not in cfg or "order" not in cfg:
        raise SystemExit(f"{path}: pool-mode config needs `pool:` and `order:` keys")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a pool-mode scenario YAML")
    parser.add_argument("--seed", type=int, default=None, help="override the config's seed")
    parser.add_argument("--name", type=str, default=None, help="override the run name (result folder)")
    parser.add_argument("--steps", type=int, default=None,
                        help="override the config's total steps - must equal the pool's own "
                             "target_steps (pool-mode's regime layout is baked in for one "
                             "specific step count; rebuild the pool to change the budget)")
    parser.add_argument("--lr", type=float, default=None, help="override the config's learning rate")
    parser.add_argument("--resume", action="store_true",
                        help="resume from workdir/<name>/latest_checkpoint.pth if it exists")
    parser.add_argument("--stop-after-step", type=int, default=None,
                        help="train only up to this many steps this run, then exit (for chunking "
                             f"a long run). Must be a multiple of {CHECKPOINT_STEPS}.")
    args = parser.parse_args()

    cfg = load_pool_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.name is not None:
        cfg["name"] = args.name
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.steps is not None:
        cfg["steps"] = args.steps

    name = cfg["name"]
    total_steps = cfg["steps"]
    assert total_steps % CHECKPOINT_STEPS == 0, (
        f"steps ({total_steps}) must be a multiple of the checkpoint granularity ({CHECKPOINT_STEPS})"
    )
    batch_size = cfg.get("batch_size", 1)
    num_outputs = cfg.get("max_classes", 2)

    set_randomness_seed(cfg.get("seed", 42))
    device = get_default_device()

    pool = load_pool(BASE / cfg["pool"])
    pool_target_steps = pool["meta"].get("target_steps")
    if args.steps is not None and args.steps != pool_target_steps:
        raise SystemExit(
            f"--steps {args.steps} != pool's target_steps {pool_target_steps} - pool-mode's "
            f"regime layout was baked in for a specific step count; rebuild the pool for this "
            f"budget instead of overriding --steps."
        )
    if pool["meta"].get("batch_size") != batch_size:
        print(f"warning: config batch_size ({batch_size}) differs from the pool's build-time "
              f"batch_size ({pool['meta'].get('batch_size')}) - regime segment boundaries won't "
              f"line up with optimizer-step boundaries", flush=True)
    if len(pool["items"]) < total_steps * batch_size:
        print(f"warning: pool has {len(pool['items'])} items, but {total_steps * batch_size} "
              f"draws are needed for one repeat-free pass - the pool will wrap around and repeat "
              f"items before training finishes", flush=True)

    order = order_indices(pool["items"], cfg["order"], seed=cfg.get("seed", 42), restarts=cfg.get("restarts", 3))

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

    loader = PoolCurriculumLoader(pool["items"], order, num_steps=CHECKPOINT_STEPS * batch_size, device=device)

    # --- optional resume, same purpose as run.py's - see that file's comment for
    # the general mechanism. The cursor formula here has an extra `* batch_size`
    # factor scheduler-mode doesn't need, because one pool-mode epoch draws
    # CHECKPOINT_STEPS * batch_size items (not CHECKPOINT_STEPS) - see this
    # script's module docstring. Getting this wrong doesn't crash anything; it
    # silently re-plays a fraction of the pool every resumed chunk instead of
    # advancing, which is why it's worked out explicitly here rather than copied
    # from run.py's (different) formula.
    ckpt = None
    ckpt_path = BASE / "workdir" / name / "latest_checkpoint.pth"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        loader.global_step = ckpt["epoch"] * CHECKPOINT_STEPS * batch_size
        loader.step(loader.global_step)
        print(f"=== RESUMING '{name}' from pool item {loader.global_step} "
              f"(step {ckpt['epoch'] * CHECKPOINT_STEPS}) ===", flush=True)

    target_checkpoints = total_steps // CHECKPOINT_STEPS
    if args.stop_after_step is not None:
        assert args.stop_after_step % CHECKPOINT_STEPS == 0, (
            f"--stop-after-step ({args.stop_after_step}) must be a multiple of {CHECKPOINT_STEPS}"
        )
        this_run_checkpoints = args.stop_after_step // CHECKPOINT_STEPS
    else:
        this_run_checkpoints = target_checkpoints
    resume = ckpt is not None

    print(f"=== training '{name}' (pool-mode, order={cfg['order']}) | up to step "
          f"{this_run_checkpoints * CHECKPOINT_STEPS}/{total_steps} | pool size {len(pool['items'])} "
          f"| num_outputs={num_outputs} | device={device} ===", flush=True)

    logger = run_module.LossLoggerCallback(out_dir, device, resume=resume)
    validator = run_module.FixedValidationCallback(out_dir, device, num_outputs, resume=resume)
    callbacks = [logger, validator]
    if "val_bands" in cfg:
        callbacks.append(run_module.BandedValidationCallback(
            out_dir, device, num_outputs, cfg["val_bands"], resume=resume))

    start = time.time()
    train(
        model=model,
        prior=loader,
        criterion=nn.CrossEntropyLoss(),
        epochs=this_run_checkpoints,
        accumulate_gradients=batch_size,
        lr=cfg.get("lr", 1e-4),
        device=device,
        callbacks=callbacks,
        run_name=name,
        ckpt=ckpt,
    )
    elapsed = time.time() - start

    last_step = logger.rows[-1][0] if logger.rows else 0
    if last_step < total_steps:
        print(f"=== chunk done at step {last_step}/{total_steps}; "
              f"resume with --resume to continue ===", flush=True)
        return

    ckpt_src = run_module.find_checkpoint(name)
    if ckpt_src:
        shutil.copy(ckpt_src, out_dir / "checkpoint.pth")
    else:
        print("warning: could not find the saved checkpoint to copy", flush=True)

    meta = run_module.write_meta(out_dir, name, cfg, total_steps, elapsed, logger, validator, num_outputs)
    print(f"done in {elapsed:.1f}s ({meta['sec_per_step']}s/step) -> {out_dir}")


if __name__ == "__main__":
    main()
