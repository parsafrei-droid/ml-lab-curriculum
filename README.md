# ml-lab-curriculum

ML research environment for TFM-Playground + TabICL prior experiments.

## Structure

```
ml-lab-curriculum/
├── data/               # Pretrained H5 data dumps
├── experiments/        # Experiment configs & output logs
├── notebooks/          # Jupyter exploration notebooks
├── curriculum/         # Curriculum learning configs
├── results/            # Model checkpoints & metrics
├── scripts/            # Utility scripts
├── TFM-Playground/     # Cloned repo (after setup)
├── tabicl/             # Cloned repo (after setup)
├── .venv/              # Python 3.11 virtual environment (after setup)
├── pyproject.toml      # uv project manifest
├── requirements.txt    # Frozen deps snapshot (after setup)
└── setup.ps1           # One-shot setup script
```

## Quick Start

Open **PowerShell** (not the IDE terminal) and run:

```powershell
cd C:\Users\Lenovo\OneDrive\Desktop\NanoTab\ml-lab-curriculum
.\setup.ps1
```

This will automatically:
1. Install `uv` (if not present)
2. Create a Python 3.11 venv
3. Clone TFM-Playground and TabICL
4. Install all dependencies
5. Download the pretrained H5 data dumps
6. Run Steps 7–9 verification tests
7. Freeze `requirements.txt`
8. Initialise a git repo

## Curriculum experiments — 3-person workflow

Everything below runs from the project root with the venv active. The cloned
repos aren't in git; recreate them once with the setup script.

```powershell
# 0) one-time setup on each machine (clones repos, patches, installs)
python scripts/setup_env.py

# 1) explore difficulty (cheap, no training) — decide which knobs matter
python scripts/sweep_difficulty.py      # difficulty vs each knob, averaged
python scripts/visualize_prior.py       # easy -> hard datasets in 2D

# 2) train one scenario  (writes results/<name>/)
python scripts/run.py --config experiments/configs/scenario_A.yaml

# 3) evaluate that checkpoint on TabArena (writes tabarena_scores.json)
python scripts/eval_tabarena.py --checkpoint results/scenario_A/checkpoint.pth --tasks tabarena

# 4) compare: collect every results/*/tabarena_scores.json
```

Each scenario is one YAML in `experiments/configs/`. Runs are independent, so we
split them — nobody writes code, everyone just runs configs:

| person | scenarios |
|---|---|
| Parsa | baseline, scenario_A |
| Emre  | scenario_B, scenario_D |
| Omid  | scenario_E (reverse, sanity check) |

Smoke-test on a laptop (a couple of epochs), run the real 5000-step jobs on a
GPU / the cluster. A run writes `checkpoint.pth`, `loss.csv`, `loss_curve.png`
and `meta.json` to `results/<name>/`.

> Note: `noise_std` (scenario C in the brief) needs the scheduler to reach into
> TabICL's *internal* sampled HPs — not wired up yet, see `curriculum/scheduler.py`.

## Manual activation after setup

```powershell
.\.venv\Scripts\Activate.ps1
jupyter notebook
```

## Key packages

| Package | Purpose |
|---|---|
| `tfmplayground` | NanoTabPFN classifier & pretraining |
| `tabicl` | TabICL prior data generation |
| `torch` | Neural network backend |
| `scikit-learn` | Baseline models & evaluation |
| `h5py` | Reading pretrained data dumps |
| `jupyter` | Interactive exploration |
