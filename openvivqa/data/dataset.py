"""Các lớp và hàm để tải và chuẩn bị dữ liệu cho bài toán VQA.

Module này cung cấp:
* `load_dataset` – đọc các tệp JSON chú thích (train/dev/test) và trả về
  ba DataFrame tương ứng.
* `ViT5VQADataset` – lớp kế thừa `torch.utils.data.Dataset` để cung cấp
  mẫu dữ liệu gồm câu hỏi, ảnh và câu trả lời cho mô hình.
* `ViT5VQADataCollator` – collator gom nhiều mẫu thành batch, xử lý pad token.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoImageProcessor



def load_dataset(
    data_dir: str,
    train_file: str = "vlsp2023_train_data.json",
    val_file: str = "vlsp2023_dev_data.json",
    test_file: str = "vlsp2023_test_data.json",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc các tệp JSON chú thích và trả về DataFrame cho train/val/test.

    Thư mục `data_dir` cần chứa ba tệp JSON với cấu trúc tương tự như bộ
    dữ liệu VLSP2023. Hàm này ánh xạ mỗi ảnh với câu hỏi/đáp án của nó và
    lưu đường dẫn ảnh đầy đủ vào cột `image_path`.

    Parameters
    ----------
    data_dir : str
        Thư mục chứa ảnh và tệp JSON. Thư mục con `images/images` phải chứa
        file ảnh.
    train_file : str, optional
        Tên tệp JSON cho tập huấn luyện, mặc định `vlsp2023_train_data.json`.
    val_file : str, optional
        Tên tệp JSON cho tập validation, mặc định `vlsp2023_dev_data.json`.
    test_file : str, optional
        Tên tệp JSON cho tập kiểm tra, mặc định `vlsp2023_test_data.json`.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Ba DataFrame tương ứng với train, validation và test.
    """
    annotation_files = {
        "train": os.path.join(data_dir, train_file),
        "validation": os.path.join(data_dir, val_file),
        "test": os.path.join(data_dir, test_file),
    }
    image_dir = os.path.join(data_dir, "images", "images")

    def _load_json(path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    records: List[Dict] = []
    for split, json_path in annotation_files.items():
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Không tìm thấy file chú thích: {json_path}")
        data = _load_json(json_path)
        # Lập bảng ánh xạ id → filename
        images_dict = {img["id"]: img.get("filename") for img in data.get("images", [])}
        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            filename = images_dict.get(img_id)
            question = ann.get("question", "").strip()
            # Bộ dữ liệu VLSP chứa danh sách `answers`; lấy phần tử đầu tiên
            answers = ann.get("answers", [""])
            answer = answers[0].strip() if answers else ""
            img_path = os.path.join(image_dir, filename) if filename else None
            records.append({
                "split": split,
                "image_id": img_id,
                "image_filename": filename,
                "question": question,
                "answer": answer,
                "image_path": img_path,
            })

    df = pd.DataFrame(records)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


class ViT5VQADataset(Dataset):
    """Dataset dành cho bài toán ViT5-VQA.

    Mỗi phần tử trả về bao gồm câu hỏi đã token hoá, mask, tensor ảnh và
    nhãn đáp án. Hàm token hoá và xử lý ảnh được cung cấp thông qua
    `tokenizer` và `image_processor`.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame đầu vào chứa các cột `question`, `answer` và `image_path`.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer dùng cho văn bản (ViT5).
    image_processor : transformers.FeatureExtractor
        Bộ xử lý ảnh (ViT). Thường là `ViTImageProcessor` hoặc `AutoImageProcessor`.
    max_length : int, optional
        Độ dài tối đa khi token hoá câu hỏi và đáp án, mặc định 128.
    """

    def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer, image_processor: AutoImageProcessor, max_length: int = 128):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        # Đọc ảnh và chuyển về RGB
        image = Image.open(image_path).convert("RGB")
        image_inputs = self.image_processor(image, return_tensors="pt")

        question = str(row["question"])
        answer = str(row["answer"])

        question_inputs = self.tokenizer(
            question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        answer_inputs = self.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "input_ids": question_inputs["input_ids"].squeeze(0),
            "attention_mask": question_inputs["attention_mask"].squeeze(0),
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "labels": answer_inputs["input_ids"].squeeze(0),
        }


@dataclass
class ViT5VQADataCollator:
    """Collator gom nhiều mẫu dữ liệu ViT5-VQA thành một batch.

    Padding được thực hiện thông qua token id của tokenizer. Những vị trí
    tương ứng với token pad sẽ được thay bằng -100 trong nhãn để PyTorch
    bỏ qua khi tính loss.
    """

    tokenizer: AutoTokenizer

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        # Thay thế pad_token_id bằng -100 để loss không tính tới
        labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


__all__ = ["load_dataset", "ViT5VQADataset", "ViT5VQADataCollator"]
