
# M5ViVQA Project

**M5ViVQA** là một dự án nghiên cứu khoa học nhằm xây dựng và đánh giá mô hình Trả lời Câu hỏi Trực quan (Visual Question Answering – VQA) bằng tiếng Việt. Dự án sử dụng kiến trúc truyền thống để thực nghiệm và so sánh trên các bộ dữ liệu OpenViVQA, ViTextVQA và EVJVQA. Mục tiêu là xây dựng phương pháp SOTA trong bài toán VQA tiếng Việt trên các bộ dữ liệu được thực nghiệm.

## Cấu trúc thư mục

```
M5ViVQA/
├── openvivqa/                    # Gói Python chứa mã nguồn chính
│   ├── __init__.py              # Khai báo gói và import nhanh các thành phần
│   ├── data/                    # Module liên quan tới dữ liệu
│   │   ├── __init__.py
│   │   ├── dataset.py           # Định nghĩa hàm tải dữ liệu và lớp Dataset
│   │   └── utils.py             # Các tiện ích hỗ trợ (hiển thị ảnh, đặt seed)
│   ├── models/                  # Module chứa định nghĩa mô hình
│   │   ├── __init__.py
│   │   ├── vit5_vqa.py          # Kiến trúc ViT5-VQA chính
│   │   └── vit5_vqa_moe_decoder.py  # Biến thể sử dụng Mixture of Experts
│   ├── training/                # Module phục vụ huấn luyện
│   │   ├── __init__.py
│   │   ├── trainer.py           # Custom Seq2SeqTrainer xử lý lưu mô hình
│   │   └── train.py             # Script huấn luyện chính
│   └── evaluation/              # Module tính toán đánh giá
│       ├── __init__.py
│       ├── metrics.py           # Hàm tính BLEU, METEOR, ROUGE và CIDEr
│       ├── bleu/                # Implementation BLEU metric
│       ├── meteor/              # Implementation METEOR metric
│       ├── rouge/               # Implementation ROUGE metric
│       ├── cider/               # Implementation CIDEr metric
│       ├── accuracy/            # Implementation accuracy metric
│       ├── precision/           # Implementation precision metric
│       ├── recall/              # Implementation recall metric
│       └── f1/                  # Implementation F1 metric
├── scripts/                     # Scripts hỗ trợ
│   ├── create_sample_data.py    # Tạo dữ liệu mẫu để test
│   ├── download_data.py         # Tải dữ liệu từ Google Drive
│   └── download_data_openvivqa.py  # Tải dữ liệu OpenViVQA cụ thể
├── config.py                    # Cấu hình Python cho training (duy nhất)
├── main.py                      # Script chính để chạy training (wrapper cho training pipeline)
├── requirements.txt             # Liệt kê thư viện cần thiết
├── test_metrics.py              # Script test metrics
├── run_openvivqa_on_kaggle.ipynb  # Notebook hướng dẫn cho Kaggle
└── README.md                    # Tài liệu hướng dẫn (file hiện tại)
```

## Mô hình có sẵn

Project cung cấp 2 kiến trúc mô hình:

### 1. ViT5-VQA (Chính)
- **File**: `openvivqa/models/vit5_vqa.py`
- **Kiến trúc**: Kết hợp Vision Transformer (ViT) với mô hình ngôn ngữ ViT5
- **Đặc điểm**: 
  - Sử dụng Google ViT-base-patch16-224-in21k để trích xuất đặc trưng ảnh
  - Sử dụng VietAI/vit5-base làm backbone ngôn ngữ
  - Có tầng fusion để kết hợp thông tin ảnh và câu hỏi

### 2. ViT5-VQA với MoE Decoder
- **File**: `openvivqa/models/vit5_vqa_moe_decoder.py`
- **Kiến trúc**: Biến thể sử dụng Mixture of Experts trong decoder
- **Đặc điểm**: Cải thiện khả năng sinh câu trả lời đa dạng

## Cấu hình

Dự án sử dụng duy nhất `config.py` để quản lý cấu hình.
- `FULL_CONFIG`: cấu hình cho dữ liệu thật
- `SAMPLE_CONFIG`: cấu hình tối giản cho dữ liệu mẫu

Bạn có thể chọn cấu hình khi chạy:
```bash
python main.py --config full
# hoặc
python main.py --config sample
```

Ngoài ra, có thể override bằng CLI khi gọi trực tiếp `openvivqa.training.train`.

## Cài đặt phụ thuộc

Project này được thiết kế để chạy trên **Kaggle Notebooks** với GPU. Để cài đặt các thư viện cần thiết:

### Trên Kaggle:
```bash
# Cài đặt dependencies từ requirements.txt
pip install -r requirements.txt

# Hoặc cài đặt từng package chính
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers evaluate datasets
pip install rouge-score nltk sacrebleu
pip install gdown tqdm
```

### Trên môi trường local (tùy chọn):
```bash
pip install -r requirements.txt
```

**Lưu ý**: Kaggle đã có sẵn nhiều thư viện ML cơ bản, vì vậy `requirements.txt` chỉ chứa các dependencies cần thiết cho project cụ thể này.

## Sử dụng

### Tải bộ dữ liệu

Project cung cấp 3 script để tải dữ liệu:

#### 1. Tải dữ liệu OpenViVQA (Khuyến nghị):
```bash
python scripts/download_data_openvivqa.py
```

#### 2. Tải dữ liệu tùy chỉnh từ Google Drive:
```bash
python scripts/download_data.py \
  --raw_dir data/raw \
  --out_dir data/processed \
  --image_id 10z-92oXTvX2hIk0ds4yJOOHav5GGiEyc \
  --train_url https://drive.google.com/uc?export=download&id=16x3h386Q_2UfCxT_3vXmPuXLScxid9L6 \
  --dev_url https://drive.google.com/uc?export=download&id=1x8nW50igqUT90LUqmL5h66LoCYkkPTZA \
  --test_url https://drive.google.com/uc?export=download&id=10azOS9TzgQl8HrztbexlKh08pkyMb4m5
```

#### 3. Tạo dữ liệu mẫu để test:
```bash
python scripts/create_sample_data.py --output_dir sample_data
```

Sau khi chạy script, thư mục `out_dir` sẽ chứa ảnh và ba file
`vlsp2023_train_data.json`, `vlsp2023_dev_data.json`, `vlsp2023_test_data.json`.

### Huấn luyện mô hình (khuyến nghị)
```bash
# Dữ liệu mẫu
python scripts/create_sample_data.py --output_dir sample_data
python main.py --config sample

# Dữ liệu thật
python main.py --config full
```

### Huấn luyện bằng script training trực tiếp
```bash
python -m openvivqa.training.train \
  --data_dir sample_data \
  --output_dir sample_results \
  --epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --num_workers 0 \
  --eval_steps 1 \
  --save_steps 1
```

### Đánh giá

Dự án sử dụng các thước đo BLEU‑1/2/3/4, METEOR, ROUGE‑L và CIDEr được triển khai trong bộ mã gốc của OpenViVQA. Các hàm tính điểm này nằm trong thư mục `openvivqa/evaluation` và được gọi thông qua hàm `compute_vqa_metrics`. Nhờ vậy, kết quả đánh giá sẽ nhất quán với những báo cáo chính thức của cuộc thi VLSP 2023.

#### Metrics có sẵn:
- **BLEU-1/2/3/4**: Đo lường độ chính xác của n-gram trong câu trả lời
- **METEOR**: Đo lường độ tương đồng ngữ nghĩa giữa prediction và reference
- **ROUGE-L**: Đo lường độ dài của chuỗi con chung dài nhất
- **CIDEr**: Đo lường độ tương đồng dựa trên TF-IDF

#### Các metrics bổ sung:
- **Accuracy**: Độ chính xác đơn giản
- **Precision**: Độ chính xác dương
- **Recall**: Độ bao phủ
- **F1**: Trung bình điều hòa của precision và recall

Chức năng `compute_vqa_metrics` trong `openvivqa/evaluation/metrics.py` cung cấp
phương thức tính BLEU‑1/2/3/4, METEOR, ROUGE‑L và CIDEr cho dự đoán VQA. Hàm
được tích hợp sẵn trong script huấn luyện để đánh giá tự động.

## Đóng góp

Mọi đóng góp nhằm cải thiện code hoặc tài liệu đều được hoan nghênh. Hãy tạo
pull request hoặc issue nếu bạn gặp vấn đề.

## Chạy thử với dữ liệu mẫu

### Trên Kaggle (Khuyến nghị):
1. **Clone repository**:
```bash
!git clone https://github.com/Kussssssss/M5ViVQA.git
%cd M5ViVQA
```

2. **Cài đặt dependencies**:
```bash
!pip install -r requirements.txt
```

3. **Tạo dữ liệu mẫu**:
```bash
!python scripts/create_sample_data.py --output_dir sample_data
```

4. **Huấn luyện mô hình**:
```bash
!python main.py --config sample
```

### Trên môi trường local:

#### Cách 1: Sử dụng main.py (Khuyến nghị)
```bash
# Tạo dữ liệu mẫu
python scripts/create_sample_data.py --output_dir sample_data

# Huấn luyện với cấu hình mẫu (sử dụng training pipeline có sẵn)
python main.py --config sample
```

**Lưu ý**: `main.py` tự động chuyển đổi cấu hình thành command line arguments và gọi training pipeline từ `openvivqa.training.train`.

#### Cách 2: Sử dụng script training trực tiếp
```bash
# Tạo dữ liệu mẫu
python scripts/create_sample_data.py --output_dir sample_data

# Huấn luyện với script training
python -m openvivqa.training.train \
  --data_dir sample_data \
  --output_dir sample_results \
  --epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --num_workers 0 \
  --eval_steps 1 \
  --save_steps 1 \
  --generation_max_length 32 \
  --generation_num_beams 1
```

Thực nghiệm nhỏ này chỉ cần một GPU với VRAM khoảng **8 GB** (hoặc có thể chạy trên CPU, nhưng thời gian sẽ lâu hơn). Nhờ dữ liệu rất nhỏ và siêu tham số tối giản, bạn có thể kiểm tra toàn bộ quy trình huấn luyện và đánh giá mô hình trong vòng một vài phút.

## Test Metrics

Để kiểm tra xem các metrics có hoạt động đúng không:

### Trên Kaggle:
```bash
!python test_metrics.py
```

### Trên môi trường local:
```bash
python test_metrics.py
```

Script này sẽ test hàm `compute_vqa_metrics` với dữ liệu mẫu và đảm bảo rằng:
- BLEU-1/2/3/4, METEOR, ROUGE-L có giá trị trong khoảng [0, 1]
- CIDEr có giá trị không âm
- Tất cả các metrics được tính toán thành công

**Lưu ý**: Trên Kaggle, bạn có thể chạy trực tiếp các lệnh Python mà không cần `!` prefix nếu đang sử dụng Jupyter notebook.
