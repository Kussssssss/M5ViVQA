"""Compute ROUGE-L metric for VQA evaluation.

This module uses the evaluate library to compute the ROUGE-L score for a
list of predicted answers and reference answers.

Parameters
----------
predictions : list of str
    The list of predicted text outputs.
references : list of str
    The list of reference text outputs.

Returns
-------
dict
    A dictionary containing the ROUGE-L score under the key 'rougeL'.
"""

from __future__ import annotations
from typing import List, Dict
import evaluate


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-L score for predicted and reference answers.

    Parameters
    ----------
    predictions : list of str
        The list of predicted answers.
    references : list of str
        The list of reference answers.

    Returns
    -------
    dict
        Dictionary with key 'rougeL' containing the ROUGE-L score.
    """
    metric = evaluate.load("rouge")
    scores = metric.compute(predictions=predictions, references=references)
    # The evaluate library returns multiple ROUGE scores; we use rougeL
    return {"rougeL": scores.get("rougeL", 0.0)}


__all__ = ["compute_rouge"]
