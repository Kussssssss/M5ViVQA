"""Custom trainer để huấn luyện mô hình VQA.

Hàm `_save` được override để sử dụng phương thức `save_pretrained` của mô hình,
tránh các vấn đề khi lưu mô hình có trọng số chia sẻ (shared weights).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """Trainer tuỳ biến kế thừa từ `Seq2SeqTrainer`.

    Hàm `_save` được thay đổi để lưu mô hình thông qua `save_pretrained`
    nếu nó được định nghĩa trong model.
    """

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[dict] = None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Nếu model định nghĩa save_pretrained, sử dụng nó
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)
        else:
            model_path = os.path.join(output_dir, "pytorch_model.bin")
            # Lưu state_dict hoặc lấy từ model
            sd = state_dict if state_dict is not None else self.model.state_dict()
            torch.save(sd, model_path)
        # Lưu tokenizer nếu có
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        # Lưu training args để tái tạo
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


__all__ = ["CustomSeq2SeqTrainer"]