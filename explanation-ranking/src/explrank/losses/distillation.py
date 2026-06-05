"""Score distillation: frozen encoder RM (teacher) -> LLM judge (student).

RECONSTRUCTED from status-doc spec — verify against your src/losses.py
(`score_distillation_loss`, `CombinedRankingDistillLoss`).

Why this exists (your decision): "Add encoder RM score distillation as Signal 2
from prior work -> makes prior work load-bearing in new method, not just a
comparison." The teacher is the accepted paper's RM; matching the judge's
scores to it imports the prior work's separation directly.

Key anti-circularity constraint (your decision): the RANKING signal trains on
human-annotated e-SNLI only (no GPT-4 scores). Distillation is a SEPARATE
signal from a model, not from GPT-4, so it does not reintroduce circularity.
"""

import torch
import torch.nn.functional as F

from .ranking import listnet_loss


def score_distillation_loss(student_scores, teacher_scores):
    """MSE between student (judge) scores and frozen teacher (encoder RM) scores,
    per query group. Teacher scores are precomputed once (see
    scripts/precompute_teacher_scores.py)."""
    return F.mse_loss(student_scores, teacher_scores.detach())


class CombinedRankingDistillLoss:
    """L = lambda_rank * L_rank(student, gold_ranks)
         + lambda_distill * L_distill(student, teacher_scores)

    Ablation knobs (your matrix):
      - lambda_distill = 0  -> ListNet-only (ranking signal alone)
      - lambda_distill > 0  -> ListNet + distill (Signal 2 active)
    """

    def __init__(self, rank_loss_fn=listnet_loss, lambda_rank=1.0, lambda_distill=0.5):
        self.rank_loss_fn = rank_loss_fn
        self.lambda_rank = lambda_rank
        self.lambda_distill = lambda_distill

    def __call__(self, student_scores, gold_scores, teacher_scores=None):
        rank = self.rank_loss_fn(student_scores, gold_scores)
        if teacher_scores is None or self.lambda_distill == 0.0:
            return {"loss": self.lambda_rank * rank, "rank": rank,
                    "distill": torch.tensor(0.0)}
        distill = score_distillation_loss(student_scores, teacher_scores)
        total = self.lambda_rank * rank + self.lambda_distill * distill
        return {"loss": total, "rank": rank, "distill": distill}
