
# M5ViVQA Project

**M5ViVQA** là một dự án nghiên cứu khoa học nhằm xây dựng và đánh giá mô hình Trả lời Câu hỏi Trực quan (Visual Question Answering – VQA) bằng tiếng Việt. Dự án sử dụng kiến trúc truyền thống để thực nghiệm và so sánh trên các bộ dữ liệu OpenViVQA, ViTextVQA và EVJVQA. Mục tiêu là xây dựng phương pháp SOTA trong bài toán VQA tiếng Việt trên các bộ dữ liệu được thực nghiệm.

## Cấu trúc thư mục

```
openvivqa_project/
├── openvivqa/               # Gói Python chứa mã nguồn chính
│   ├── __init__.py          # Khai báo gói và import nhanh các thành phần
│   ├── data/                # Module liên quan tới dữ liệu
│   │   ├── __init__.py
│   │   ├── dataset.py       # Định nghĩa hàm tải dữ liệu và lớp Dataset
│   │   └── utils.py         # Các tiện ích hỗ trợ (hiển thị ảnh, đặt seed)
│   ├── models/              # Module chứa định nghĩa mô hình
│   │   ├── __init__.py
│   │   └── vit5_vqa.py      # Kiến trúc ViT5-VQA
│   ├── training/            # Module phục vụ huấn luyện
│   │   ├── __init__.py
│   │   ├── trainer.py       # Custom Seq2SeqTrainer xử lý lưu mô hình
│   │   └── train.py         # Script huấn luyện chính
│   └── evaluation/          # Module tính toán đánh giá
│       ├── __init__.py
│       └── metrics.py       # Hàm tính BLEU, METEOR, ROUGE và CIDEr
├── scripts/
│   └── download_data.py     # Script hỗ trợ tải và giải nén bộ dữ liệu gốc
├── requirements.txt         # Liệt kê thư viện cần thiết để chạy dự án
└── README.md                # Tài liệu hướng dẫn (file hiện tại)
```

## Cài đặt phụ thuộc

Để cài đặt toàn bộ các thư viện phụ thuộc, bạn có thể sử dụng pip:

```bash
pip install -r requirements.txt
```

## Sử dụng

### Tải bộ dữ liệu

Sử dụng script `scripts/download_data.py` để tải và giải nén bộ dữ liệu từ Google
Drive. Bạn cần cung cấp ID của file ảnh (zip) và các đường liên kết tới file
chú thích JSON. Ví dụ:

```bash
python scripts/download_data.py \
  --raw_dir data/raw \
  --out_dir data/processed \
  --image_id 10z-92oXTvX2hIk0ds4yJOOHav5GGiEyc \
  --train_url https://drive.google.com/uc?export=download&id=16x3h386Q_2UfCxT_3vXmPuXLScxid9L6 \
  --dev_url https://drive.google.com/uc?export=download&id=1x8nW50igqUT90LUqmL5h66LoCYkkPTZA \
  --test_url https://drive.google.com/uc?export=download&id=10azOS9TzgQl8HrztbexlKh08pkyMb4m5
```

Sau khi chạy script, thư mục `out_dir` sẽ chứa ảnh và ba file
`vlsp2023_train_data.json`, `vlsp2023_dev_data.json`, `vlsp2023_test_data.json`.

### Huấn luyện mô hình

Script `openvivqa/training/train.py` cung cấp điểm khởi đầu để huấn luyện
mô hình VQA. Bạn có thể chỉnh sửa các siêu tham số theo nhu cầu. Ví dụ
chạy huấn luyện với cấu hình mặc định:

```bash
python -m openvivqa.training.train \
  --data_dir data/processed \
  --output_dir runs/vit5-vqa \
  --epochs 3 \
  --batch_size 24 \
  --learning_rate 1e-4
```

Trong quá trình huấn luyện, mô hình tốt nhất theo `eval_loss` sẽ được lưu tại
`output_dir`. Sau khi huấn luyện, script cũng đánh giá trên tập kiểm tra và
lưu mô hình cuối cùng.

### Đánh giá

Chức năng `compute_vqa_metrics` trong `openvivqa/evaluation/metrics.py` cung cấp
phương thức tính BLEU‑1/2/3/4, METEOR, ROUGE‑L và CIDEr cho dự đoán VQA. Hàm
được tích hợp sẵn trong script huấn luyện để đánh giá tự động.

## Đóng góp

Mọi đóng góp nhằm cải thiện code hoặc tài liệu đều được hoan nghênh. Hãy tạo
pull request hoặc issue nếu bạn gặp vấn đề.

## Chạy thử với dữ liệu mẫu

Bạn có thể tạo một bộ dữ liệu nhỏ gọn để thử nghiệm nhanh toàn bộ quy trình huấn luyện.
Thực hiện script sau để sinh ba hình và chú thích câu hỏi:

```bash
python scripts/create_sample_data.py --output_dir sample_data
```

Sau khi tạo xong dữ liệu mẫu, bạn có thể huấn luyện mô hình với các siêu tham số tối giản như dưới đây (tương ứng nội dung trong `config_sample.yaml`):

```bash
python -m openvivqa.training.train --data_dir sample_data --output_dir sample_results --epochs 1 --batch_size 1 --gradient_accumulation_steps 1 --learning_rate 1e-4 --num_workers 0 --eval_steps 1 --save_steps 1 --generation_max_length 32 --generation_num_beams 1
```

Thực nghiệm nhỏ này chỉ cần một GPU với VRAM khoảng **8 GB** (hoặc có thể chạy trên CPU, nhưng thời gian sẽ lâu hơn). Nhờ dữ liệu rất nhỏ và siêu tham số tối giản, bạn có thể kiểm tra toàn bộ quy trình huấn luyện và đánh giá mô hình trong vòng một vài phút.
