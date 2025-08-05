"""Subpackage cho các hàm đánh giá.

Hiện tại chỉ cung cấp `compute_vqa_metrics` dùng cho Seq2SeqTrainer.
"""

from .metrics import compute_vqa_metrics

__all__ = ["compute_vqa_metrics"]