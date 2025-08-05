"""Các hàm tiện ích cho xử lý dữ liệu.

Hàm `set_seed` giúp đặt trạng thái ngẫu nhiên đồng nhất cho
NumPy, Python và PyTorch để tái lập kết quả.
Hàm `show_example` hỗ trợ trực quan hóa một mẫu dữ liệu bằng cách hiển thị
ảnh và bảng câu hỏi/đáp án tương ứng.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import torch

try:
    # IPython.display.display được dùng khi chạy trong notebook để hiển thị DataFrame.
    from IPython.display import display
except ImportError:
    display = None  # Trong môi trường không có IPython, bỏ qua display


def set_seed(seed: int = 42) -> None:
    """Đặt seed cho tất cả các trình tạo số ngắu nhiên.

    Việc đặt seed giúp tái lập kết quả thí nghiệm. Hàm sẽ đặt seed cho
    random, NumPy và PyTorch, đồng thời tắt một số tối ưu hóa không
    định hình của cuDNN để đảm bảo tính quyết định.

    Parameters
    ----------
    seed : int
        Giá trị seed sử dụng. Mặc định là 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Vô hiệu hóa một số tối ưu hóa cuDNN để đảm bảo tính lặp lại
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def show_example(
    df: pd.DataFrame,
    image_dir: str,
    img_filename: Optional[str] = None,
    show_qa: bool = True,
) -> None:
    """Hiển thị một mẫu ảnh và câu hỏi/đáp án từ DataFrame.

    Nếu không cung cấp tên tệp ảnh, hàm sẻ chọn ngẫu nhiên một ảnh trong DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame chứa các cột `image_filename`, `question`, `answer`.
    image_dir : str
        Thư mục chứa ảnh.
    img_filename : Optional[str], optional
        Tên file ảnh muốn hiển thị, nếu None sẻ chọn ngẫu nhiên.
    show_qa : bool, optional
        Có hiển thị bảng câu hỏi và đáp án hay không, mặc định True.
    """
    if df is None or len(df) == 0:
        raise ValueError("DataFrame đầu vào rỗng.")
    # Chọn ảnh ngẫu nhiên nếu không chỉ định
    filenames = df["image_filename"].dropna().unique().tolist()
    if not filenames:
        raise ValueError("Không tìm thấy cột 'image_filename' trong DataFrame.")
    if img_filename is None:
        img_filename = random.choice(filenames)

    img_path = os.path.join(image_dir, img_filename)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")

    # Hiển thị ảnh
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.title(img_filename)
    plt.axis("off")
    plt.show()

    if show_qa:
        sample = df[df["image_filename"] == img_filename].reset_index(drop=True)
        # Nếu có IPython display sẻ hiển thị đẹp hơn
        if display is not None:
            display(sample[[col for col in sample.columns if col != "image_path"]])
        else:
            # In ra bảng đơn giản
            print(sample[[col for col in sample.columns if col != "image_path"]].to_string(index=False))


__all__ = ["set_seed", "show_example"]
