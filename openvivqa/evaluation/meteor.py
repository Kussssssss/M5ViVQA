"""Compute METEOR metric for VQA evaluation.

This module uses the evaluate library to compute the METEOR score for a
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
    A dictionary containing the METEOR score under the key 'meteor'.
"""

from __future__ import annotations
from typing import List, Dict
import evaluate


def compute_meteor(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute METEOR score for predicted and reference answers.

    Parameters
    ----------
    predictions : list of str
        The list of predicted answers.
    references : list of str
        The list of reference answers.

    Returns
    -------
    dict
        Dictionary with key 'meteor' containing the METEOR score.
    """
    metric = evaluate.load("meteor")
    # The evaluate library expects references as a list of strings, not list of lists
    result = metric.compute(predictions=predictions, references=references)
    return {"meteor": result.get("meteor", 0.0)}


__all__ = ["compute_meteor"]
