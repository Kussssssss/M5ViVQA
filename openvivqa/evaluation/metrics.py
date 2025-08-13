from __future__ import annotations
from typing import List, Dict
import logging as py_logging

# Import trực tiếp từ các module có sẵn
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider

def _to_string_list(texts: List[str]) -> List[str]:
    out: List[str] = []
    for t in texts:
        if t is None:
            out.append("")
        elif isinstance(t, str):
            out.append(t)
        else:
            out.append(str(t))
    return out

def _strip_only(texts: List[str]) -> List[str]:
    return [t.strip() for t in texts]

def _ensure_non_empty(texts: List[str], placeholder: str = "none") -> List[str]:
    return [t if len(t) > 0 else placeholder for t in texts]

def compute_vqa_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Tính toán các metrics đánh giá cho VQA task.
    
    Args:
        predictions: Danh sách các câu trả lời dự đoán
        references: Danh sách các câu trả lời tham chiếu
        
    Returns:
        Dictionary chứa các metrics: bleu1-4, meteor, rougeL, cider
    """
    assert len(predictions) == len(references), (
        f"Number of predictions ({len(predictions)}) must match number of references ({len(references)})"
    )
    
    if len(predictions) == 0:
        return {
            "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0,
            "meteor": 0.0, "rougeL": 0.0, "cider": 0.0
        }
    
    # Chuẩn bị dữ liệu cho các metrics (chuẩn hóa để tránh lỗi thư viện)
    predictions = _to_string_list(predictions)
    references = _to_string_list(references)
    predictions_norm = _ensure_non_empty(_strip_only(predictions))
    references_norm = _ensure_non_empty(_strip_only(references))
    gts = {i: [ref] for i, ref in enumerate(references_norm)}
    res = {i: [pred] for i, pred in enumerate(predictions_norm)}
    
    metrics = {}
    
    # Tính BLEU (retry một lần sau khi chuẩn hóa kỹ nếu cần)
    try:
        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(gts, res)
    except Exception as e:
        # Thử strip một lần nữa và thay thế rỗng bằng placeholder, vẫn dùng Bleu nội bộ
        py_logging.warning(f"BLEU compute_score lỗi, thử strip/placeholder: {e}")
        gts_retry = {i: [ (ref[0].strip() or "none") ] for i, ref in gts.items()}
        res_retry = {i: [ (pred[0].strip() or "none") ] for i, pred in res.items()}
        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(gts_retry, res_retry)
    metrics.update({
        "bleu1": float(bleu_scores[0]),
        "bleu2": float(bleu_scores[1]),
        "bleu3": float(bleu_scores[2]),
        "bleu4": float(bleu_scores[3]),
    })
    
    # METEOR (retry một lần nếu lỗi)
    try:
        meteor_score, _ = Meteor().compute_score(gts, res)
    except Exception as e:
        py_logging.warning(f"METEOR compute_score lỗi, thử strip/placeholder: {e}")
        gts_retry = {i: [ (ref[0].strip() or "none") ] for i, ref in gts.items()}
        res_retry = {i: [ (pred[0].strip() or "none") ] for i, pred in res.items()}
        meteor_score, _ = Meteor().compute_score(gts_retry, res_retry)
    metrics["meteor"] = float(meteor_score)
    
    # ROUGE (retry một lần nếu lỗi)
    try:
        rouge_score, _ = Rouge().compute_score(gts, res)
    except Exception as e:
        py_logging.warning(f"ROUGE compute_score lỗi, thử strip/placeholder: {e}")
        gts_retry = {i: [ (ref[0].strip() or "none") ] for i, ref in gts.items()}
        res_retry = {i: [ (pred[0].strip() or "none") ] for i, pred in res.items()}
        rouge_score, _ = Rouge().compute_score(gts_retry, res_retry)
    metrics["rougeL"] = float(rouge_score)
    
    # CIDEr (retry một lần nếu lỗi)
    try:
        cider_score, _ = Cider().compute_score(gts, res)
    except Exception as e:
        py_logging.warning(f"CIDEr compute_score lỗi, thử strip/placeholder: {e}")
        gts_retry = {i: [ (ref[0].strip() or "none") ] for i, ref in gts.items()}
        res_retry = {i: [ (pred[0].strip() or "none") ] for i, pred in res.items()}
        cider_score, _ = Cider().compute_score(gts_retry, res_retry)
    metrics["cider"] = float(cider_score)
    
    return metrics

__all__ = ["compute_vqa_metrics"]
