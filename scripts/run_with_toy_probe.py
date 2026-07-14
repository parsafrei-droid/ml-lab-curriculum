"""scripts/run.py, plus a cheap real-data probe every checkpoint.

scripts/run.py is intentionally left untouched. This is a separate entry
point for anyone who wants the toy-TabArena curve (see toy_tabarena_probe.py)
alongside the usual loss.csv/val.csv - it reuses run.py's argument parsing,
config loading and training loop as-is, and only adds
ToyTabArenaProbeCallback to the callback list by wrapping the `train`
function run.py calls. No edits to run.py itself.

    python scripts/run_with_toy_probe.py --config experiments/configs/scenario_A.yaml

Accepts every flag scripts/run.py does (--seed, --name, --steps, --lr,
--resume, --stop-after-step) - it's the same argparse, untouched.

Writes an extra results/<name>/toy_tabarena.csv + toy_tabarena_curve.png
next to run.py's usual loss.csv / val.csv / checkpoint.pth / meta.json.
"""

import pathlib
import sys

BASE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "TFM-Playground"))
sys.path.insert(0, str(BASE / "tabicl"))
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))

import run as run_module  # scripts/run.py, unmodified

from toy_tabarena_probe import ToyTabArenaProbeCallback

_real_train = run_module.train


def _train_with_toy_probe(*args, **kwargs):
    # run.py always calls train(...) with these as keyword arguments (see
    # its train(...) call near the end of main()).
    out_dir = BASE / "results" / kwargs["run_name"]
    probe = ToyTabArenaProbeCallback(
        out_dir, kwargs.get("device"),
        checkpoint_steps=run_module.CHECKPOINT_STEPS,
        resume=kwargs.get("ckpt") is not None,
    )
    kwargs["callbacks"] = [*kwargs.get("callbacks", []), probe]
    return _real_train(*args, **kwargs)


# main() looks up the name `train` in this module's globals at call time, so
# reassigning it here redirects main()'s call without editing run.py.
run_module.train = _train_with_toy_probe


if __name__ == "__main__":
    run_module.main()
