"""Curriculum pretraining for nanoTabPFN.

Four small pieces:
  - difficulty.py : how we score a dataset as easy or hard
  - scheduler.py  : swaps the prior's knobs mid-training (easy -> hard)
  - prior.py      : tiny helpers to build a prior and pull one dataset out as numpy
  - pool.py       : a second curriculum mechanism - order over a fixed, pre-generated
                    pool instead of live-ramped knobs (see pool.py's own docstring)
"""

from curriculum.difficulty import difficulty_score, class_separation, measured_difficulty
from curriculum.scheduler import CurriculumScheduler, apply_knobs, KNOBS, EXTERNAL_KNOBS, INTERNAL_KNOBS
from curriculum.prior import make_prior, sample_dataset, make_validation_batches, make_banded_validation_batches
from curriculum.pool import load_pool, order_indices, PoolCurriculumLoader

__all__ = [
    "difficulty_score",
    "class_separation",
    "measured_difficulty",
    "CurriculumScheduler",
    "apply_knobs",
    "KNOBS",
    "EXTERNAL_KNOBS",
    "INTERNAL_KNOBS",
    "make_prior",
    "sample_dataset",
    "make_validation_batches",
    "make_banded_validation_batches",
    "load_pool",
    "order_indices",
    "PoolCurriculumLoader",
]
