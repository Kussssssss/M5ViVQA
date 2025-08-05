"""Các hàm tính metric đánh giá cho bài toán VQA.

Hàm chính `compute_vqa_metrics` được thiết kế để truyền vào `Seq2SeqTrainer`
và trả về dictionary chứa BLEU‑1, BLEU‑2, BLEU‑3, BLEU‑4, METEOR, ROUGE‑L
và CIDEr. Trong trường hợp lỗi khi decode, hàm trả về các giá trị 0.0.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import evaluate


def _compute_cider(predictions: List[str], references: List[str]) -> float:
    """Tính điểm CIDEr đơn giản dựa trên TF‑IDF cho n‑gram 1–4.

    Triển khai này mang tính tham khảo và không nhằm thay thế thư viện đầy đủ.

    Parameters
    ----------
    predictions : list of str
        Danh sách câu trả lời mô hình sinh ra.
    references : list of str
        Danh sách câu trả lời đúng tương ứng.

    Returns
    -------
    float
        Điểm CIDEr trung bình trên tất cả cặp dự đoán – nhãn.
    """

    def get_ngrams(text: str, n: int) -> List[str]:
        tokens = text.lower().split()
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def compute_tf_idf(pred_ngrams: List[str], ref_ngrams_corpus: List[List[str]]) -> float:
        score = 0.0
        if not pred_ngrams:
            return score
        # Tập hợp tất cả n‑gram xuất hiện trong tham chiếu
        vocab = set(ng for ref in ref_ngrams_corpus for ng in ref)
        for ng in pred_ngrams:
            if ng in vocab:
                tf = pred_ngrams.count(ng) / len(pred_ngrams)
                df = sum(1 for ref in ref_ngrams_corpus if ng in ref)
                # Thêm 1 vào mẫu để tránh chia cho 0
                idf = math.log((len(ref_ngrams_corpus)) / (df + 1))
                score += tf * idf
        return score

    scores: List[float] = []
    for pred, ref in zip(predictions, references):
        s = 0.0
        # CIDEr lấy trung bình điểm của 1‑4 gram
        for n in range(1, 5):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = [get_ngrams(ref, n)]
            s += compute_tf_idf(pred_ngrams, ref_ngrams)
        scores.append(s / 4.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def compute_vqa_metrics(eval_pred: Tuple[Any, Any], tokenizer) -> Dict[str, float]:
    """Tính BLEU, METEOR, ROUGE và CIDEr từ output của trainer.

    Hàm này có thể truyền trực tiếp vào tham số `compute_metrics` của
    `transformers.Seq2SeqTrainer`. Tham số `eval_pred` là tuple chứa
    `(predictions, labels)` ở dạng tensor hoặc numpy. Tokenizer được
    truyền vào để decode về chuỗi.

    Returns
    -------
    dict
        Dictionary gồm các khoá: `bleu1`, `bleu2`, `bleu3`, `bleu4`,
        `meteor`, `rougeL`, `cider`.
    """

    predictions, labels = eval_pred
    # Chuyển tensor sang numpy nếu cần
    if hasattr(predictions, "cpu"):
        predictions = predictions.cpu().numpy()
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()

    # Clip giá trị vượt quá vocab size và chuyển về int
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
        # Trong trường hợp không decode được, trả về các metric bằng 0
        return {
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "meteor": 0.0,
            "rougeL": 0.0,
            "cider": 0.0,
        }

    # Tính BLEU
    bleu_metric = evaluate.load("bleu")
    bleu_scores = bleu_metric.compute(
        predictions=decoded_preds, references=[[lbl] for lbl in decoded_labels], max_order=4
    )
    result: Dict[str, float] = {
        "bleu1": bleu_scores["precisions"][0] if len(bleu_scores["precisions"]) > 0 else 0.0,
        "bleu2": bleu_scores["precisions"][1] if len(bleu_scores["precisions"]) > 1 else 0.0,
        "bleu3": bleu_scores["precisions"][2] if len(bleu_scores["precisions"]) > 2 else 0.0,
        "bleu4": bleu_scores["bleu"],
    }
    # Tính METEOR
    meteor_metric = evaluate.load("meteor")
    meteor_score = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result["meteor"] = meteor_score.get("meteor", 0.0)
    # Tính ROUGE
    rouge_metric = evaluate.load("rouge")
    rouge_scores = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result["rougeL"] = rouge_scores.get("rougeL", 0.0)
    # Tính CIDEr
    result["cider"] = _compute_cider(decoded_preds, decoded_labels)
    return result


__all__ = ["compute_vqa_metrics"]