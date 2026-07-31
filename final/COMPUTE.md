# What the project cost

Across the whole project we ran 305 jobs, 299 on BwUniCluster and 6 Kaggle notebook runs,
totalling 56.7 hours of GPU time between 25 June and 30 July 2026. Most of the work ran on
NVIDIA H100 GPUs (230 runs, 42.6 h), with A100 (65 runs, 6.1 h) and Kaggle Tesla T4 (6 runs,
8.0 h) making up the rest. Waiting for the cluster cost far more than computing: 268.7 hours
queued against 48.7 hours of actual Slurm compute, which is why we split work into short
(<= 30 min) jobs run in parallel, up to 15 at once. The final experiment could not fit in a
short partition and waited up to 77 hours per job on the cluster, so we moved it to Kaggle
and accepted a 1.6x slower GPU to finish it the same day.

## GPUs

| GPU | Where | Runs | Run time | Share | Time queued |
|---|---|---:|---:|---:|---:|
| NVIDIA H100 94 GB | BwUniCluster 3.0 | 230 | 42.62 h | 75.2 % | 267.93 h |
| NVIDIA Tesla T4 | Kaggle notebooks | 6 | 7.96 h | 14.0 % | none |
| NVIDIA A100 80 GB | BwUniCluster 2.0 | 65 | 6.07 h | 10.7 % | 0.79 h |
| AMD MI300 | BwUniCluster 3.0 | 3 | 0.01 h | < 0.1 % | 0.01 h |
| CPU only | BwUniCluster 2.0 | 1 | 0.00 h | 0 % | 0.00 h |
| **Total** | | **305** | **56.67 h** | 100 % | **268.73 h** |

The A100 was used early for setup and the first sweeps, then again for the Third Wave, where
we wanted a fixed uncontended GPU so wall-clock comparisons between two models would be
fair. The MI300 was tried in 3 short probe jobs; our stack is CUDA only, so we did not
pursue it.

## Partitions

| Partition | Max wall-clock | Jobs | Run time | Queue time | Mean wait |
|---|---|---:|---:|---:|---:|
| `gpu_h100_short` | 30 min | 139 | 20.90 h | 40.53 h | 19.5 min |
| `gpu_h100` | 3 days | 52 | 17.23 h | 222.65 h | 477 min |
| `gpu_a100_short` | 30 min | 45 | 4.79 h | 0.21 h | 0.3 min |
| `dev_gpu_h100` | 30 min | 36 | 4.49 h | 4.75 h | 9.8 min |
| `dev_gpu_a100_il` | 30 min | 16 | 1.28 h | 0.58 h | 2.3 min |
| `gpu_mi300` | 3 days | 3 | 0.01 h | 0.01 h | 0.4 min |
| other | - | 8 | 0.00 h | 0.00 h | 0.0 min |

Same hardware, very different waits: 19.5 minutes on `gpu_h100_short` against 477 minutes on
`gpu_h100`. This is the most useful thing we learned about the cluster, and it shaped how we
ran everything after the First Wave.

## Per phase

| Phase | Dates | Jobs | Run time | Queue time |
|---|---|---:|---:|---:|
| Setup and GPU probes | Jun 25 - Jun 27 | 5 | 0.10 h | 0.03 h |
| First Wave | Jul 01 - Jul 06 | 170 | 32.39 h | 41.60 h |
| Second Wave | Jul 18 - Jul 28 | 109 | 14.94 h | 226.53 h |
| Second Wave on Kaggle | Jul 25 - Jul 27 | 6 | 7.96 h | none |
| Third Wave | Jul 30 | 15 | 1.28 h | 0.58 h |
| **Total** | Jun 25 - Jul 30 | **305** | **56.67 h** | **268.73 h** |

`figures/compute_dashboard.png` plots this. The full breakdown, including the per-job
records and the script that built it, is in
[`../experiments/third_wave/results/computations/`](../experiments/third_wave/results/computations/).
