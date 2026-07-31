# First attempt

The first version of the project, kept for the record. **Its main conclusion was wrong**, so
nothing here feeds the final result.

## What we did

Instead of ordering a fixed set of datasets, we ramped the TabICLv2 prior's own knobs during
training (`max_features`, `max_classes`, `noise_std`, `num_layers`, `hidden_dim`,
`num_causes`) and used a cheap kNN probe to decide which knobs actually controlled
difficulty. The sweep said no single knob mattered, only the whole regime together, so we
built curricula that moved everything at once.

## Why it was wrong

Reading the TabICLv2 source afterwards showed the sweep was measuring broken knobs:

- `noise_std`, `num_layers` and `hidden_dim` are sampled from a log-scaled distribution. We
  raised only `max_mean` and left `min_mean` tiny, so the sampled values barely moved.
- `num_causes` is overwritten with `num_features` whenever `is_causal=False`, which we never
  froze. That knob did nothing at all.
- `mix_probs = (0.7, 0.3)` means about 30% of datasets are tree-based and ignore every MLP
  knob.
- A kNN probe measures difficulty for a distance-based model, not for a transformer.

So "no single knob controls difficulty" was an artefact of the knobs not working, not a
finding about curricula. That is what pushed us to the fixed-pool design in
[`../second_wave/`](../second_wave/), where difficulty is a measured property of the data
rather than a knob we hope is doing something.

The external knobs that *do* work (number of features, number of classes, sequence length)
are the ones the later waves use.

## Contents

| path | what |
|---|---|
| `curriculum/` | The scheduler, prior wrapper and the kNN difficulty probe. |
| `scripts/` | Training, evaluation, sweeps and the cluster submission scripts. |
| `configs/` | One config per curriculum variant tried here. |
| `figures/` | The comparison plots and the difficulty sweep this attempt produced. |
| `results/` | Per-run logs, configs and TabArena scores. |
| `PRESENTATION.md` | The mid-project update written from these runs. |
| `setup.ps1`, `run_demo_scheduler.sh` | Environment setup and a scheduler demo. |
