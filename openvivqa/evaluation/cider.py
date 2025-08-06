"""Compute CIDEr metric for VQA evaluation.

This module implements a simple TF‑IDF based CIDEr score for evaluation of
Visual Question Answering (VQA) predictions against reference answers.

Parameters
----------
predictions : list of str
    A list of predicted answers.
references : list of str
    A list of reference answers.

Returns
-------
dict
    A dictionary containing the CIDEr score under the key 'cider'.
"""

from __future__ import annotations
from typing import List, Dict
import math


def compute_cider(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute a simplified CIDEr score for predicted and reference answers.

    This implementation is based on averaging TF‑IDF weighted n‑gram
    similarity over n=1..4 grams. It does not require external
    dependencies.

    Parameters
    ----------
    predictions : list of str
        Predicted answer strings.
    references : list of str
        Reference answer strings.

    Returns
    -------
    dict
        Dictionary with key 'cider' containing the averaged CIDEr score.
    """

    def get_ngrams(text: str, n: int) -> List[str]:
        tokens = text.lower().split()
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def compute_tf_idf(pred_ngrams: List[str], ref_ngrams_corpus: List[List[str]]) -> float:
        score = 0.0
        if not pred_ngrams:
            return score
        # Build vocabulary of all n-grams in the reference corpus
        vocab = set(ng for ref in ref_ngrams_corpus for ng in ref)
        for ng in pred_ngrams:
            if ng in vocab:
                tf = pred_ngrams.count(ng) / len(pred_ngrams)
                df = sum(1 for ref in ref_ngrams_corpus if ng in ref)
                idf = math.log((len(ref_ngrams_corpus)) / (df + 1))
                score += tf * idf
        return score

    scores: List[float] = []
    for pred, ref in zip(predictions, references):
        s = 0.0
        for n in range(1, 5):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = [get_ngrams(ref, n)]
            s += compute_tf_idf(pred_ngrams, ref_ngrams)
        # Average the score across n-grams
        scores.append(s / 4.0)

    avg_score = float(sum(scores) / len(scores)) if scores else 0.0
    return {"cider": avg_score}


__all__ = ["compute_cider"]
