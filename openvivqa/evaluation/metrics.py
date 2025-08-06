"""Các hàm tính metric đánh giá cho bài toán VQA.

Hàm chính `compute_vqa_metrics` được thiết kế để truyền vào `Seq2SeqTrainer`
và trả về dictionary chứa BLEU‑1, BLEU‑2, BLEU‑3, BLEU‑4, METEOR, ROUGE‑L
và CIDEr. Trong trường hợp lỗi khi decode, hàm trả về các giá trị 0.0.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

# Import individual metric computation functions from submodules
from .bleu import compute_bleu
from .meteor import compute_meteor
from .rouge import compute_rouge
from .cider import compute_cider


def compute_vqa_metrics(eval_pred: Tuple[Any, Any], tokenizer) -> Dict[str, float]:
    """Tính các metric BLEU, METEOR, ROUGE và CIDEr cho bài toán VQA.

    Hàm này có thể truyền trực tiếp vào tham số ``compute_metrics`` của
    :class:`transformers.Seq2SeqTrainer`. Tham số ``eval_pred`` là tuple chứa
    ``(predictions, labels)`` ở dạng tensor hoặc numpy. Tokenizer được
    truyền vào để giải mã (decode) về chuỗi.

    Parameters
    ----------
    eval_pred : tuple
        Tuple chứa ``(predictions, labels)`` dưới dạng tensor hoặc numpy.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer dùng để decode các chỉ số token thành chuỗi văn bản.

    Returns
    -------
    dict
        Dictionary gồm các khoá: ``bleu1``, ``bleu2``, ``bleu3``, ``bleu4``,
        ``meteor``, ``rougeL`` và ``cider``.
    """
    predictions, labels = eval_pred

    # Chuyển tensor sang numpy nếu cần thiết
    if hasattr(predictions, "cpu"):
        predictions = predictions.cpu().numpy()
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()

    # Bảo đảm giá trị trong khoảng vocab và kiểu int32
    vocab_size = tokenizer.vocab_size
    predictions = np.clip(predictions, 0, vocab_size - 1).astype(np.int32)
    labels = labels.astype(np.int32)

    decoded_preds: List[str] = []
    decoded_labels: List[str] = []
    try:
        for pred in predictions:
            try:
                text = tokenizer.decode(pred, skip_special_tokens=True)
                decoded_preds.append(text.strip())
            except Exception:
                decoded_preds.append("")
        for label in labels:
            label_cleaned = np.where(label != -100, label, tokenizer.pad_token_id)
            try:
                text = tokenizer.decode(label_cleaned, skip_special_tokens=True)
                decoded_labels.append(text.strip())
            except Exception:
                decoded_labels.append("")
    except Exception:
        # Lỗi không decode được
        return {
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "meteor": 0.0,
            "rougeL": 0.0,
            "cider": 0.0,
        }

    # Tính từng metric riêng biệt
    bleu_scores = compute_bleu(decoded_preds, decoded_labels)
    meteor_score = compute_meteor(decoded_preds, decoded_labels)
    rouge_score = compute_rouge(decoded_preds, decoded_labels)
    cider_score = compute_cider(decoded_preds, decoded_labels)

    # Gộp kết quả lại
    result: Dict[str, float] = {}
    result.update(bleu_scores)
    result.update(meteor_score)
    result.update(rouge_score)
    result.update(cider_score)
    return result


__all__ = ["compute_vqa_metrics"]
