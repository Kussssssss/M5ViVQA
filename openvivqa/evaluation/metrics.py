"""
Compute VQA evaluation metrics using official implementations from the
OpenViVQA project.

Given lists of predictions and references, this module computes BLEU‑1/2/3/4,
METEOR, ROUGE‑L và CIDEr bằng cách sử dụng các lớp `Bleu`, `Meteor`,
`Rouge` và `Cider` trong thư mục evaluation bạn vừa sao chép.
"""

from __future__ import annotations
from typing import List, Dict

from evaluation.bleu.bleu import Bleu
from evaluation.meteor.meteor import Meteor
from evaluation.rouge.rouge import Rouge
from evaluation.cider.cider import Cider

def compute_vqa_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    assert len(predictions) == len(references), (
        "Number of predictions must match number of references"
    )
    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}

    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)

    meteor_score, _ = Meteor().compute_score(gts, res)
    rouge_score, _ = Rouge().compute_score(gts, res)
    cider_score, _ = Cider().compute_score(gts, res)

    return {
        "bleu1": float(bleu_scores[0]),
        "bleu2": float(bleu_scores[1]),
        "bleu3": float(bleu_scores[2]),
        "bleu4": float(bleu_scores[3]),
        "meteor": float(meteor_score),
        "rougeL": float(rouge_score),
        "cider": float(cider_score),
    }

__all__ = ["compute_vqa_metrics"]
