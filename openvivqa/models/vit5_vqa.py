"""Định nghĩa mô hình ViT5-VQA.

Lớp `ViT5VQAModel` ghép nối mô hình thị giác ViT với mô hình ngôn ngữ
ViT5 nhằm giải quyết bài toán Visual Question Answering. Kiến trúc bao gồm
một tầng chiếu (projection) để đưa đặc trưng ảnh sang cùng không gian với
embedding văn bản, tiếp đó là một tầng attention đa đầu để trộn thông tin
hình ảnh và câu hỏi trước khi đưa vào encoder của ViT5. Phần decoder
ViT5 được dùng để sinh câu trả lời.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    ViTModel,
    ViTImageProcessor,
)


class ViT5VQAModel(nn.Module):
    """Kiến trúc kết hợp ViT (hình ảnh) và ViT5 (ngôn ngữ).

    Parameters
    ----------
    vit5_model_name : str, optional
        Tên mô hình ViT5 từ HuggingFace. Mặc định "VietAI/vit5-base".
    vit_model_name : str, optional
        Tên mô hình ViT dùng làm backbone thị giác. Mặc định
        "google/vit-base-patch16-224-in21k".
    """

    def __init__(
        self,
        vit5_model_name: str = "VietAI/vit5-base",
        vit_model_name: str = "google/vit-base-patch16-224-in21k",
    ) -> None:
        super().__init__()
        # Tải mô hình ngôn ngữ ViT5
        self.vit5 = AutoModelForSeq2SeqLM.from_pretrained(vit5_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(vit5_model_name)
        # Tải backbone hình ảnh ViT
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.image_processor = ViTImageProcessor.from_pretrained(vit_model_name)

        # Lưu cấu hình, generation_config của ViT5 để tiện sử dụng
        self.config = self.vit5.config
        self.generation_config = self.vit5.generation_config

        # Tầng chiếu để chuyển đặc trưng hình ảnh (hidden_size của ViT) sang
        # không gian d_model của ViT5
        self.image_projection = nn.Linear(self.vit.config.hidden_size, self.vit5.config.d_model)

        # Lớp attention đa đầu kết hợp feature ảnh và embedding câu hỏi
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=self.vit5.config.d_model,
            num_heads=8,
            batch_first=True,
        )

        # Lớp chuẩn hoá để thêm residual connection
        self.layer_norm = nn.LayerNorm(self.vit5.config.d_model)

    # Các hàm proxy cần thiết cho Seq2SeqTrainer
    def get_encoder(self):
        return self.vit5.encoder

    def get_decoder(self):
        return self.vit5.decoder

    def resize_token_embeddings(self, *args, **kwargs):
        return self.vit5.resize_token_embeddings(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.vit5.prepare_inputs_for_generation(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        """Lưu mô hình và tokenizer sang một thư mục.

        Hàm override này sử dụng `torch.save` thay vì safetensors để tránh
        vấn đề về định dạng. Nó lưu state_dict của toàn bộ module và
        thêm config, tokenizer, image_processor vào thư mục.
        """
        os.makedirs(save_directory, exist_ok=True)
        # Lưu trọng số mô hình
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        # Lưu cấu hình
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        # Lưu tokenizer và image_processor
        self.tokenizer.save_pretrained(save_directory)
        self.image_processor.save_pretrained(save_directory)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Lan truyền tiến để tính loss hoặc trả về encoder_outputs.

        Nếu `labels` được cung cấp thì đây là chế độ huấn luyện và mô hình sẽ
        trả về loss. Nếu không, hàm chỉ trả về `encoder_outputs` để hỗ trợ
        hàm generate.
        """
        # Encode hình ảnh
        image_outputs = self.vit(pixel_values=pixel_values)
        image_features = image_outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        # Chiếu sang không gian d_model
        projected_image = self.image_projection(image_features)
        # Lấy embedding câu hỏi từ token id
        text_embeddings = self.vit5.encoder.embed_tokens(input_ids)
        # Kết hợp thông tin ảnh với câu hỏi qua attention
        fused_features, _ = self.fusion_layer(
            query=text_embeddings,
            key=projected_image,
            value=projected_image,
        )
        # Thêm residual connection và layer norm
        fused_features = self.layer_norm(fused_features + text_embeddings)
        # Đưa vào encoder của ViT5
        encoder_outputs = self.vit5.encoder(
            inputs_embeds=fused_features,
            attention_mask=attention_mask,
            return_dict=True,
        )
        if labels is not None:
            # Huấn luyện: gọi ViT5 để tính loss với decoder
            outputs = self.vit5(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            return outputs
        else:
            # Suy luận: trả về encoder_outputs để dùng trong generate
            return encoder_outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Sinh ra câu trả lời dựa trên input.

        Hàm này tự thực hiện encode ảnh và fuse với câu hỏi trước khi gọi
        phương thức generate của ViT5. Các tham số bổ sung được truyền vào
        generate của ViT5 (ví dụ `max_length`, `num_beams`, …).
        """
        # Encode ảnh
        image_outputs = self.vit(pixel_values=pixel_values)
        image_features = image_outputs.last_hidden_state
        projected_image = self.image_projection(image_features)
        text_embeddings = self.vit5.encoder.embed_tokens(input_ids)
        fused_features, _ = self.fusion_layer(
            query=text_embeddings,
            key=projected_image,
            value=projected_image,
        )
        fused_features = self.layer_norm(fused_features + text_embeddings)
        encoder_outputs = self.vit5.encoder(
            inputs_embeds=fused_features,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return self.vit5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **kwargs,
        )


__all__ = ["ViT5VQAModel"]