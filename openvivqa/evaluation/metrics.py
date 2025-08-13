from __future__ import annotations
from typing import List, Dict
import logging

# Import trực tiếp từ các module có sẵn
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider

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
    
    # Chuẩn bị dữ liệu cho các metrics
    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}
    
    metrics = {}
    
    try:
        # Tính BLEU scores
        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(gts, res)
        metrics.update({
            "bleu1": float(bleu_scores[0]),
            "bleu2": float(bleu_scores[1]), 
            "bleu3": float(bleu_scores[2]),
            "bleu4": float(bleu_scores[3])
        })
    except Exception as e:
        logging.warning(f"Lỗi khi tính BLEU: {e}")
        metrics.update({"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0})
    
    try:
        # Tính METEOR score
        meteor_score, _ = Meteor().compute_score(gts, res)
        metrics["meteor"] = float(meteor_score)
    except Exception as e:
        logging.warning(f"Lỗi khi tính METEOR: {e}")
        metrics["meteor"] = 0.0
    
    try:
        # Tính ROUGE score
        rouge_score, _ = Rouge().compute_score(gts, res)
        metrics["rougeL"] = float(rouge_score)
    except Exception as e:
        logging.warning(f"Lỗi khi tính ROUGE: {e}")
        metrics["rougeL"] = 0.0
    
    try:
        # Tính CIDEr score
        cider_score, _ = Cider().compute_score(gts, res)
        metrics["cider"] = float(cider_score)
    except Exception as e:
        logging.warning(f"Lỗi khi tính CIDEr: {e}")
        metrics["cider"] = 0.0
    
    return metrics

__all__ = ["compute_vqa_metrics"]
