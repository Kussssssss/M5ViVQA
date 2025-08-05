#!/usr/bin/env python3
"""Tải và giải nén bộ dữ liệu OpenViVQA từ Google Drive.

Sử dụng gdown để tải file zip ảnh bằng file ID và wget để tải các file JSON.
Sau khi tải xong, script sẽ giải nén ảnh vào thư mục `out_dir/images` và
sao chép các file JSON vào `out_dir`.

Ví dụ:

```bash
python scripts/download_data.py \
  --raw_dir data/raw \
  --out_dir data/processed \
  --image_id 10z-92oXTvX2hIk0ds4yJOOHav5GGiEyc \
  --train_url https://drive.google.com/uc?export=download&id=... \
  --dev_url https://drive.google.com/uc?export=download&id=... \
  --test_url https://drive.google.com/uc?export=download&id=...
```
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import gdown
import requests


def download_json(url: str, output_path: Path) -> None:
    """Tải một file JSON bằng requests và lưu ra đĩa."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OpenViVQA dataset")
    parser.add_argument("--raw_dir", type=str, required=True, help="Thư mục lưu tạm file tải về")
    parser.add_argument("--out_dir", type=str, required=True, help="Thư mục xuất dữ liệu đã giải nén")
    parser.add_argument("--image_id", type=str, required=True, help="ID file ảnh zip trên Google Drive")
    parser.add_argument("--train_url", type=str, required=True, help="URL tải file JSON train")
    parser.add_argument("--dev_url", type=str, required=True, help="URL tải file JSON dev")
    parser.add_argument("--test_url", type=str, required=True, help="URL tải file JSON test")
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tải file zip ảnh
    images_zip_path = raw_dir / "images.zip"
    if not images_zip_path.exists():
        print(f"Downloading images zip to {images_zip_path}...")
        gdown.download(id=args.image_id, output=str(images_zip_path), fuzzy=True)
    else:
        print(f"Images zip already exists at {images_zip_path}, skip download")

    # Tải các file JSON
    json_files = {
        "vlsp2023_train_data.json": args.train_url,
        "vlsp2023_dev_data.json": args.dev_url,
        "vlsp2023_test_data.json": args.test_url,
    }
    for filename, url in json_files.items():
        json_path = raw_dir / filename
        if json_path.exists():
            print(f"{filename} already exists, skip download")
        else:
            print(f"Downloading {filename}...")
            download_json(url, json_path)

    # Giải nén ảnh
    images_out_dir = out_dir / "images"
    images_out_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(images_zip_path, "r") as zf:
        print(f"Extracting images to {images_out_dir}...")
        zf.extractall(images_out_dir)

    # Sao chép file JSON sang out_dir
    for filename in json_files:
        src = raw_dir / filename
        dst = out_dir / filename
        print(f"Copying {src} → {dst}")
        shutil.copy(src, dst)

    # Hiển thị cấu trúc thư mục kết quả
    print(f"\n{out_dir.name}/")
    for root, dirs, files in os.walk(out_dir):
        level = root.replace(str(out_dir), "").count(os.sep)
        indent = "    " * level
        print(f"{indent}{os.path.basename(root)}/")
        file_count = len(files)
        if file_count > 5:
            for f in files[:3]:
                print(f"{indent}    {f}")
            print(f"{indent}    ... ({file_count - 3} more files)")
        else:
            for f in files:
                print(f"{indent}    {f}")


if __name__ == "__main__":
    main()
