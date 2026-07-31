# Experiments

Everything we ran on the way to the result. The finished product is in
[`../final/`](../final/); this folder is the working history, including the parts that did
not survive.

| folder | what it is | status |
|---|---|---|
| [`first_attempt/`](first_attempt/) | The first version. Ramped many prior knobs at once and used a kNN probe to rank difficulty. | Superseded. Most of the knobs turned out to be no-ops. |
| [`second_wave/`](second_wave/) | The fixed-pool design: one dump of 80,000 tables, runs differ only in order. | This is where the main result comes from. |
| [`third_wave/`](third_wave/) | The wall-clock test. Same ordering on modded-nanoTabPFN, a compute-bound speedrun, plus the compute history of the whole project. | Answered the "does it save real time" question. The answer was no. |
| [`emre/`](emre/) | A different approach: ramp the prior's own knobs during training instead of ordering a fixed pool, plus a learning-rate sweep. | Same direction as the pool result, weaker evidence. |

## Running any of it

All of it expects the two upstream repos next to this one at the repository root
(`TFM-Playground/` and `tabicl/`), and the shared virtualenv at `.venv/`. Paths in the
scripts are relative to the repository root, not to this folder.
