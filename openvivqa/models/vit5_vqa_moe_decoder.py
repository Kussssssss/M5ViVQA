"""Mô hình ViT5-VQA với decoder tích hợp Mixture‑of‑Experts (MoE).

Mục tiêu của mô hình này là đưa cơ chế MoE vào phần giải mã (decoder) của
mô hình T5 trong kiến trúc ViT5-VQA. Thay vì chèn MoE ở đầu ra của encoder
như phiên bản trước, lớp MoE sẽ được áp dụng lên đầu ra ẩn cuối cùng của
decoder trước khi chuyển qua lớp ``lm_head`` để dự đoán token. Điều này
giúp bổ sung đa dạng chuyên gia vào quá trình sinh câu trả lời mà vẫn tận
dụng lại toàn bộ kiến trúc encoder/decoder có sẵn của T5.

Kiến trúc tổng quát:

* Ảnh được mã hoá bởi ViT (``vit_model``) và chiếu sang cùng không gian với
  embedding của câu hỏi bằng ``image_projection``.
* Câu hỏi (chuỗi token ``input_ids``) được embed bằng token embedding của
  T5 (``vit5.encoder.embed_tokens``).
* Hai nguồn thông tin ảnh và câu hỏi được trộn với nhau thông qua một lớp
  attention đa đầu ``fusion_layer`` và sau đó cộng residual và chuẩn hoá.
* Kết quả trộn này được đưa vào encoder của T5 để trích xuất biểu diễn ngữ
  cảnh (``encoder_outputs``).
* Khi huấn luyện, mô hình T5 gốc được gọi với ``encoder_outputs`` và
  ``labels`` để sinh ra ``decoder_hidden_states``. Một lớp MoE được áp dụng
  lên đầu ra ẩn cuối cùng của decoder trước khi đư¡ qua ``lm_head`` để tính
  logits và loss. Hàm mất mát được tính bằng cross entropy với nhãn.
* Khi suy luận, mô hình sử dụng ``generate`` của T5 với ``encoder_outputs``;
  MoE không được dùng trong quá trình giải mã tự hồi quy vì việc chèn MoE
  vào beam search phức tạp và nằm ngoài phạm vi thực nghiệm này.

Lớp MoE trong file này được cài đặt đơn giản: một mạng gating học xác suất
cho từng chuyên gia và ``num_experts`` mạng con (feed‑forward) vận hành
song song. Đầu ra của MoE là tổng có trọng số của các đầu ra chuyên gia.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    ViTModel,
    ViT
ImageProcessor,
)
from transformers.modeling_outputs import BaseModelOutput


class MoE(nn.Module):
    """Lớp Mixture‑of‑Experts đơn giản.

    Mệi chuyên gia là một mạng feed‑forward hai tầng ``Linear -> ReLU -> Linear``.
    Một mạng gating tính phân phối xác suất qua các chuyên gia dựa trên đầu vào.

    Parameters
    ----------
    input_dim : int
        Kích thước của vector đầu vào.
    hidden_dim : int
        Kích thước ẩn của mệi chuyên gia.
    num_experts : int
        Số lượng chuyên gia trong mô hình.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int = 4) -> None:
        super().__init__()
        self.num_experts = num_experts
        # Lớp gating đưa vào softmax để lấy trọng số chuyên gia
        self.gating = nn.Linear(input_dim, num_experts)
        # Tạo danh sách chuyên gia: mỗi chuyên gia là FFN hai tầng
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền qua MoE.

        Parameters
        ----------
        x : torch.Tensor
            Tensor đầu vào có shape [..., input_dim].

        Returns
        -------
        torch.Tensor
            Tensor có cùng shape như ``x`` sau khi tổng hợp các chuyên gia.
        """
        # Tính trọng số (batch, seq_len, num_experts)
        gate_logits = self.gating(x)
        gate_scores = torch.softmax(gate_logits, dim=-1)
        # Tính đầu ra cho từng chuyên gia
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        # Nhân trọng số và tổng hợp theo trục chuyên gia
        output = (expert_outputs * gate_scores.unsqueeze(-2)).sum(dim=-1)
        return output


class ViT5VQAModelMoEDecoder(nn.Module):
    """Kiến trúc ViT5-VQA tích hợp MoE ở phần decoder.

    Tham số ``num_experts`` kiểm soát số lượng chuyên gia trong lớp MoE.
    Tham số ``moe_hidden_dim`` đặt kích thước ẩn của các chuyên gia. Mặc định
    sử dụng ``4`` chuyên gia và ``moe_hidden_dim`` bằng ``4 * d_model`` của
    mô hình T5, tương tự như feed‑forward network gốc.
    """

    def __init__(
        self,
        vit5_model_name: str = "VietAI/vit5-base",
        vit_model_name: str = "google/vit-base-patch16-224-in21k",
        num_experts: int = 4,
        moe_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Tải mô hình ngôn ngữ ViT5
        self.vit5 = AutoModelForSeq2SeqLM.from_pretrained(vit5_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(vit5_model_name)
        # Tải backbone hình ảnh ViT
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.image_processor = ViTImageProcessor.from_pretrained(vit_model_name)

        # Lưu cấu hình và generation_config của ViT5 để tiện sử dụng
        self.config = self.vit5.config
        self.generation_config = self.vit5.generation_config

        # Tầng chiếu để chuyển đặc trưng hình ảnh sang không gian d_model của ViT5
        self.image_projection = nn.Linear(self.vit.config.hidden_size, self.vit5.config.d_model)
        # Lớp attention đa đầu kết hợp feature ảnh và embedding câu hỏi
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=self.vit5.config.d_model,
            num_heads=8,
            batch_first=True,
        )
        # Lớp chuẩn hoá và residual
        self.layer_norm = nn.LayerNorm(self.vit5.config.d_model)

        # Thiết lập lớp MoE cho decoder. Nếu ``moe_hidden_dim`` không được
        # cung cấp thì dùng 4 * d_model giông với feed‑forward network trong T5.
        if moe_hidden_dim is None:
            moe_hidden_dim = 4 * self.vit5.config.d_model
        self.moe_decoder = MoE(
            input_dim=self.vit5.config.d_model,
            hidden_dim=moe_hidden_dim,
            num_experts=num_experts,
        )

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

        Hàm override này sử dụng ``torch.save`` thay vì safetensors để tránh
        vấn đề về định dạng. Nó lưu state_dict của toàn bộ module và thêm
        config, tokenizer, image_processor vào thư mục.
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

        Nếu ``labels`` được cung cấp thì đây là chế độ huấn luyện. Mô hình
        thực hiện các bước sau:
        1. Mã hoá ảnh và câu hỏi, trộn với nhau bằng attention.
        2. Tính encoder_outputs qua encoder của T5.
        3. Gọi mô hình T5 để lấy ``decoder_hidden_states`` (yêu cầu
           ``output_hidden_states=True``).
        4. Áp dụng lớp MoE lên đầu ra ẩn cuối cùng của decoder.
        5. Tính logits qua ``lm_head`` và loss bằng cross entropy.

        Nếu ``labels`` không được cung cấp thì mô hình chỉ trả về
        ``encoder_outputs`` để hỗ trợ cho quá trình generate.
        """
        # 1. Encode hình ảnh và fuse với câu hỏi
        image_outputs = self.vit(pixel_values=pixel_values)
        image_features = image_outputs.last_hidden_state  # [batch, seq_len_img, hidden_size]
        projected_image = self.image_projection(image_features)
        text_embeddings = self.vit5.encoder.embed_tokens(input_ids)
        fused_features, _ = self.fusion_layer(
            query=text_embeddings,
            key=projected_image,
            value=projected_image,
        )
        fused_features = self.layer_norm(fused_features + text_embeddings)
        # 2. Tính encoder_outputs qua encoder của T5
        encoder_outputs = self.vit5.encoder(
            inputs_embeds=fused_features,
            attention_mask=attention_mask,
            return_dict=True,
        )
        if labels is None:
            # Suy luận: trả về encoder_outputs để dùng trong generate
            return encoder_outputs
        # 3. Gọi T5 với output_hidden_states để lấy decoder_hidden_states
        # Lưu ý: chúng ta bỏ qua loss và logits từ T5 gốc vì sẽ dùng MoE
        outputs = self.vit5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        # 4. Lấy hidden state cuối cùng của decoder và áp dụng MoE
        decoder_hidden = outputs.decoder_hidden_states[-1]  # [batch, seq_len_dec, d_model]
        moe_output = self.moe_decoder(decoder_hidden)
        # T5 áp dụng final layer norm trước khi lm_head. Làm tương tự
        moe_output = self.vit5.model.decoder.final_layer_norm(moe_output)
        # 5. Tính logits và loss
        logits = self.vit5.lm_head(moe_output)  # [batch, seq_len_dec, vocab_size]
        # Shift logits và labels để tính loss: bỏ token đầu của decoder
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        return {"loss": loss, "logits": logits}

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Sinh ra câu trả lời dựa trên input.

        Hàm này thực hiện encode ảnh và fuse với câu hỏi trước khi gọi
        phương thức ``generate`` của T5. MoE không được dùng trong quá trình
        generate do việc chèn MoE vào beam search phức tạp và nằm ngoài phạm vi.
        """
        # Encode ảnh và fuse
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


__all__ = ["ViT5VQAModelMoEDecoder"]
