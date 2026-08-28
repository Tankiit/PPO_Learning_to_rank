"""Reproducible ARR August ranking-and-PPO pipeline.

The package is deliberately isolated from the historical ACL scripts.  Its public
objects are small, serialisable contracts that are shared by data preparation,
judges, evaluation and PPO.
"""

from .schema import Candidate, RankingGroup, ScoreRecord

__all__ = ["Candidate", "RankingGroup", "ScoreRecord"]
__version__ = "0.1.0"
