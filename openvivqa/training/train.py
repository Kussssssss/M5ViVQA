"""Script huấn luyện cho mô hình ViT5-VQA.

Sử dụng script này bằng cách chạy

```bash
python -m openvivqa.training.train --data_dir path/to/data --output_dir path/to/out
```

Hãy tham khảo README.md để biết thêm chi tiết về tham số và ví dụ sử dụng.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Callable

import torch
from transformers import Seq2SeqTrainingArguments

from ..data.dataset import load_dataset, ViT5VQADataset, ViT5VQADataCollator
from ..models.vit5_vqa import ViT5VQAModel
from ..evaluation.metrics import compute_vqa_metrics
from .trainer import CustomSeq2SeqTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình ViT5-VQA")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Thư mục chứa dữ liệu đã chuẩn bị"
    )
    parser.add_argument(
        "--output_dir", type=str, default="vit5-vqa-results", help="Thư mục lưu mô hình và log"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch huấn luyện")
    parser.add_argument(
        "--batch_size", type=int, default=24, help="Batch size mỗi thiết bị"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Số bước tích luỹ gradient",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Tốc độ học"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay cho optimizer"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500, help="Số bước warmup"
    )
    parser.add_argument(
        "--vit5_model", type=str, default="VietAI/vit5-base", help="Tên mô hình ViT5"
    )
    parser.add_argument(
        "--vit_model", type=str, default="google/vit-base-patch16-224-in21k", help="Tên mô hình ViT"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Số worker cho DataLoader"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=200,
        help="Khoảng cách bước giữa các lần đánh giá trong quá trình train",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Khoảng cách bước giữa các lần lưu checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info("Bắt đầu tải dữ liệu...")
    train_df, val_df, test_df = load_dataset(args.data_dir)
    logging.info(
        f"Đã tải dữ liệu: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    # Khởi tạo mô hình
    model = ViT5VQAModel(vit5_model_name=args.vit5_model, vit_model_name=args.vit_model)
    if torch.cuda.is_available():
        model.cuda()
        logging.info("Sử dụng GPU để huấn luyện")
    else:
        logging.info("Không phát hiện GPU, sử dụng CPU")

    # Chuẩn bị Dataset và collator
    train_dataset = ViT5VQADataset(train_df, model.tokenizer, model.image_processor)
    val_dataset = ViT5VQADataset(val_df, model.tokenizer, model.image_processor)
    test_dataset = ViT5VQADataset(test_df, model.tokenizer, model.image_processor)
    data_collator = ViT5VQADataCollator(model.tokenizer)

    # Cấu hình huấn luyện
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=2,
        disable_tqdm=False,
        save_safetensors=False,
        fp16=torch.cuda.is_available(),
    )

    # Hàm tính metric truyền vào trainer
    def metrics_fn(eval_pred) -> dict[str, float]:
        """
        Sanitize both prediction and label IDs by replacing negative values 
        (e.g. -100) with the pad_token_id. This prevents OverflowError from 
        the tokenizer's Rust backend.
        """
        import numpy as np
        
        # Lấy predictions và labels từ eval_pred
        pred_ids = eval_pred.predictions[0] if isinstance(
            eval_pred.predictions, tuple) else eval_pred.predictions
        label_ids = eval_pred.label_ids
        
        # Lấy pad_token_id từ tokenizer
        pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id or 0
        
        # Chuyển đổi sang numpy array và xử lý negative values
        pred_ids = np.array(pred_ids)
        label_ids = np.array(label_ids)
        
        # Thay thế các giá trị âm (như -100) bằng pad_token_id
        pred_ids = np.where(pred_ids < 0, pad_id, pred_ids)
        label_ids = np.where(label_ids < 0, pad_id, label_ids)
        
        # Decode predictions và references
        predictions = model.tokenizer.batch_decode(
            pred_ids.astype(int), skip_special_tokens=True)
        references = model.tokenizer.batch_decode(
            label_ids.astype(int), skip_special_tokens=True)
        
        # Làm sạch references (loại bỏ khoảng trắng thừa)
        references = [ref.strip() for ref in references]
        predictions = [pred.strip() for pred in predictions]
        
        # Lọc bỏ các câu trống hoặc quá ngắn
        valid_pairs = []
        for pred, ref in zip(predictions, references):
            if len(pred.strip()) > 0 and len(ref.strip()) > 0:
                valid_pairs.append((pred, ref))
        
        if not valid_pairs:
            return {
                "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0,
                "meteor": 0.0, "rougeL": 0.0, "cider": 0.0
            }
        
        # Tách predictions và references đã lọc
        filtered_predictions, filtered_references = zip(*valid_pairs)
        
        # Chuyển đổi format để phù hợp với compute_vqa_metrics
        # compute_vqa_metrics mong đợi: gts = {i: [ref]}, res = {i: [pred]}
        gts = {i: [ref] for i, ref in enumerate(filtered_references)}
        res = {i: [pred] for i, pred in enumerate(filtered_predictions)}
        
        return compute_vqa_metrics(filtered_predictions, filtered_references)

    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=metrics_fn,
        processing_class=model.tokenizer,
    )
    logging.info("Bắt đầu huấn luyện...")
    trainer.train()

    logging.info("Đánh giá trên tập kiểm tra...")
    test_results = trainer.evaluate(test_dataset)
    logging.info(f"Kết quả trên tập test: {test_results}")

    logging.info("Lưu mô hình cuối cùng...")
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    logging.info("Huấn luyện hoàn tất")


if __name__ == "__main__":
    main()
