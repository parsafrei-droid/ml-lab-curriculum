"""Curriculum pretraining for nanoTabPFN.

Three small pieces:
  - difficulty.py : how we score a dataset as easy or hard
  - scheduler.py  : swaps the prior's knobs mid-training (easy -> hard)
  - prior.py      : tiny helpers to build a prior and pull one dataset out as numpy
"""

from curriculum.difficulty import difficulty_score, class_separation, measured_difficulty
from curriculum.scheduler import CurriculumScheduler, KNOBS

__all__ = [
    "difficulty_score",
    "class_separation",
    "measured_difficulty",
    "CurriculumScheduler",
    "KNOBS",
]
