"""BLEU evaluation metrics for VQA.

This module provides a function to compute BLEU-1, BLEU-2, BLEU-3 and BLEU-4 scores
using the `evaluate` library. The function expects a list of prediction strings and
corresponding reference strings, and returns a dictionary of the four scores.
"""

from typing import List, Dict
import evaluate


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BLEU-1 to BLEU-4 scores.

    Parameters
    ----------
    predictions : list of str
        Model output strings.
    references : list of str
        Ground truth answer strings.

    Returns
    -------
    dict
        Dictionary with keys ``bleu1``, ``bleu2``, ``bleu3``, ``bleu4`` containing the
        corresponding BLEU precision scores. If the evaluation library fails,
        all scores default to 0.0.
    """
    # Load the BLEU metric from the evaluate library. This may download data on first use.
    bleu_metric = evaluate.load("bleu")
    # Prepare references in the expected format: a list of lists
    references_wrapped = [[ref] for ref in references]
    bleu_scores = bleu_metric.compute(predictions=predictions, references=references_wrapped, max_order=4)
    # Extract precision scores. BLEU-4 is reported under the ``bleu`` key.
    result = {
        "bleu1": bleu_scores["precisions"][0] if len(bleu_scores["precisions"]) > 0 else 0.0,
        "bleu2": bleu_scores["precisions"][1] if len(bleu_scores["precisions"]) > 1 else 0.0,
        "bleu3": bleu_scores["precisions"][2] if len(bleu_scores["precisions"]) > 2 else 0.0,
        "bleu4": bleu_scores.get("bleu", 0.0),
    }
    return result


__all__ = ["compute_bleu"]
