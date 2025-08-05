"""Top‑level package for OpenViVQA.

Module này định nghĩa những hàm và lớp thường dùng nhất của gói để người dùng
có thể import trực tiếp từ `openvivqa`. Nội dung chi tiết được tổ chức trong
subpackages `data`, `models`, `training` và `evaluation`.
"""

from .data.dataset import load_dataset, ViT5VQADataset, ViT5VQADataCollator
from .models.vit5_vqa import ViT5VQAModel
from .training.trainer import CustomSeq2SeqTrainer
from .evaluation.metrics import compute_vqa_metrics

__all__ = [
    "load_dataset",
    "ViT5VQADataset",
    "ViT5VQADataCollator",
    "ViT5VQAModel",
    "CustomSeq2SeqTrainer",
    "compute_vqa_metrics",
]
