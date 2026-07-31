"""A cheap, frequent real-data probe to run alongside training.

FixedValidationCallback (in scripts/run.py) scores every checkpoint on a
fixed *synthetic* set, which is comparable across scenarios but never
touches real data. eval_tabarena.py is real data, but a full ~16-dataset
TabArena pass is too expensive to run every checkpoint, so it only runs
once, after training.

ToyTabArenaProbeCallback fills that gap: it scores the model on
TOY_TASKS_CLASSIFICATION (iris, wine, breast_cancer - ~900 rows total)
every checkpoint. Cheap enough to run every 100 steps, so you can see
whether the synthetic val_loss curve and real accuracy move together or
diverge over the course of a single run.

Deliberately not imported by scripts/run.py - wire it in via
scripts/run_with_toy_probe.py instead, so the default run path is unaffected.
"""

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score

from tfmplayground.callbacks import Callback
from tfmplayground.evaluation import TOY_TASKS_CLASSIFICATION, get_openml_predictions
from tfmplayground.interface import NanoTabPFNClassifier


class ToyTabArenaProbeCallback(Callback):
    """Scores the model on the 3 toy OpenML tasks every checkpoint.

    Mirrors FixedValidationCallback's CSV/resume/plot structure so its output
    (results/<name>/toy_tabarena.csv, toy_tabarena_curve.png) sits alongside
    loss.csv and val.csv the same way.
    """

    def __init__(self, out_dir, device, checkpoint_steps=100, resume=False):
        self.out_dir = out_dir
        self.device = device
        self.checkpoint_steps = checkpoint_steps
        self.rows = []  # (step, toy_roc_auc)
        self.csv_path = out_dir / "toy_tabarena.csv"
        if resume and self.csv_path.exists():
            with self.csv_path.open() as f:
                next(f, None)  # header
                for line in f:
                    s, auc = line.strip().split(",")
                    self.rows.append((int(s), float(auc)))
        else:
            self.csv_path.write_text("step,toy_roc_auc\n")

    @torch.no_grad()
    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        # train() has already put the model in eval mode before calling us
        # (see FixedValidationCallback for the same note) - safe to run
        # inference-only real-data predictions on it here.
        step = epoch * self.checkpoint_steps
        classifier = NanoTabPFNClassifier(model=model, device=self.device)
        predictions = get_openml_predictions(
            model=classifier, classification=True, tasks=TOY_TASKS_CLASSIFICATION,
        )
        # A model's classification head is architecturally fixed-width
        # (num_outputs, from training's schedule max max_classes) - iris/wine
        # are 3-class, so a binary-only model (num_outputs=2) can't represent
        # them; roc_auc_score would crash on the mismatch (predict_proba
        # silently clips instead of erroring, same failure mode as
        # eval_tabarena.py hit). Skip what doesn't fit instead of crashing the
        # whole training run over a probe that's meant to be a cheap side check.
        aucs = []
        for name, (y_true, _y_pred, y_proba) in predictions.items():
            try:
                aucs.append(roc_auc_score(y_true, y_proba, multi_class="ovr"))
            except ValueError as e:
                print(f"            | toy_tabarena probe: skipped {name} - {e}", flush=True)
        if not aucs:
            print(f"            | toy_tabarena_roc_auc n/a (no compatible toy task for this model)", flush=True)
            return
        auc = float(sum(aucs) / len(aucs))
        self.rows.append((step, auc))
        with self.csv_path.open("a") as f:
            f.write(f"{step},{auc:.4f}\n")
        print(f"            | toy_tabarena_roc_auc {auc:.4f} ({len(aucs)}/{len(predictions)} tasks)", flush=True)

    def close(self):
        if not self.rows:
            return
        steps = [r[0] for r in self.rows]
        aucs = [r[1] for r in self.rows]
        plt.figure(figsize=(6, 4))
        plt.plot(steps, aucs, marker="o", ms=3)
        plt.xlabel("training steps")
        plt.ylabel("toy TabArena ROC-AUC (iris, wine, breast_cancer)")
        plt.title(self.out_dir.name)
        plt.tight_layout()
        plt.savefig(self.out_dir / "toy_tabarena_curve.png", dpi=120)
        plt.close()
