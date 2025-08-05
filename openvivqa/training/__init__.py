"""Package chứa các thành phần phục vụ huấn luyện mô hình.

* `trainer.py` định nghĩa lớp `CustomSeq2SeqTrainer`.
* `train.py` là script để khởi tạo và huấn luyện mô hình.
"""

from .trainer import CustomSeq2SeqTrainer

__all__ = ["CustomSeq2SeqTrainer"]
