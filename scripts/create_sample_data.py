"""
Utility script to generate a small sample dataset for quick experimentation.

This script creates a small dataset resembling the VLSP VQA format with a few
simple colored images and corresponding question/answer pairs. It can be used
on a local machine to verify that the training pipeline works end-to-end
without downloading large datasets.

Usage:
    python scripts/create_sample_data.py --output_dir sample_data

The generated directory structure will be:
    sample_data/
        images/
            images/
                red.jpg
                green.jpg
                blue.jpg
        vlsp2023_train_data.json
        vlsp2023_dev_data.json
        vlsp2023_test_data.json
"""

import os
import json
from argparse import ArgumentParser
from PIL import Image


def create_sample_data(output_dir: str = "sample_data") -> None:
    """Create a tiny VQA-style dataset with three colored images.

    The dataset includes three images (red, green, blue) and simple
    question/answer pairs asking about the dominant color. JSON annotation
    files for train, validation and test splits are generated to match the
    expected VLSP format.

    Parameters
    ----------
    output_dir : str
        Directory where the dataset will be created. If it does not exist,
        it will be created along with required subdirectories.
    """
    images_dir = os.path.join(output_dir, "images", "images")
    os.makedirs(images_dir, exist_ok=True)

    # Define colors and their RGB tuples
    colors = {
        "đỏ": (255, 0, 0),        # red
        "xanh lá": (0, 255, 0),    # green
        "xanh dương": (0, 0, 255) # blue
    }

    images = []
    annotations = []

    for idx, (name, color) in enumerate(colors.items(), start=1):
        filename = f"{idx}_{name.replace(' ', '_')}.jpg"
        img_path = os.path.join(images_dir, filename)
        # Create solid-color image
        img = Image.new('RGB', (224, 224), color)
        img.save(img_path)
        images.append({"id": idx, "filename": filename})
        annotations.append({
            "image_id": idx,
            "question": "Màu chủ đạo của ảnh này là gì?",
            "answers": [name]
        })

    # Split annotations into small train/val/test splits
    splits = {
        "train": annotations[:2],
        "validation": annotations[2:3],
        "test": annotations[2:3],
"""
Utility script to generate a small sample dataset for quick experimentation.

This script creates a small dataset resembling the VLSP VQA format with a few
simple colored images and corresponding question/answer pairs. It can be used
on a local machine to verify that the training pipeline works end-to-end
without downloading large datasets.

Usage:
    python scripts/create_sample_data.py --output_dir sample_data

The generated directory structure will be:
    sample_data/
        images/
            images/
                red.jpg
                green.jpg
                blue.jpg
        vlsp2023_train_data.json
        vlsp2023_dev_data.json
        vlsp2023_test_data.json
"""

import os
import json
from argparse import ArgumentParser
from PIL import Image


def create_sample_data(output_dir: str = "sample_data") -> None:
    """Create a tiny VQA-style dataset with three colored images.

    The dataset includes three images (red, green, blue) and simple
    question/answer pairs asking about the dominant color. JSON annotation
    files for train, validation and test splits are generated to match the
    expected VLSP format.

    Parameters
    ----------
    output_dir : str
        Directory where the dataset will be created. If it does not exist,
        it will be created along with required subdirectories.
    """
    images_dir = os.path.join(output_dir, "images", "images")
    os.makedirs(images_dir, exist_ok=True)

    # Define colors and their RGB tuples
    colors = {
        "đỏ": (255, 0, 0),        # red
        "xanh lá": (0, 255, 0),    # green
        "xanh dương": (0, 0, 255) # blue
    }

    images = []
    annotations = []

    for idx, (name, color) in enumerate(colors.items(), start=1):
        filename = f"{idx}_{name.replace(' ', '_')}.jpg"
        img_path = os.path.join(images_dir, filename)
        # Create solid-color image
        img = Image.new('RGB', (224, 224), color)
        img.save(img_path)
        images.append({"id": idx, "filename": filename})
        annotations.append({
            "image_id": idx,
            "question": "Màu chủ đạo của ảnh này là gì?",
            "answers": [name]
        })

    # Split annotations into small train/val/test splits
    splits = {
        "train": annotations[:2],
        "validation": annotations[2:3],
        "test": annotations[2:3],
    }

    for split, anns in splits.items():
        data = {
            "images": images,
            "annotations": anns,
        }
        json_path = os.path.join(output_dir, f"vlsp2023_{split}_data.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Sample dataset created in {output_dir}.")


def main() -> None:
    parser = ArgumentParser(description="Generate a small VQA sample dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sample_data",
        help="Directory where sample data will be generated",
    )
    args = parser.parse_args()
    create_sample_data(args.output_dir)


if __name__ == "__main__":
    main()
    }

    for split, anns in splits.items():
        data = {
            "images": images,
            "annotations": anns,
        }
        json_path = os.path.join(output_dir, f"vlsp2023_{split}_data.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Sample dataset created in {output_dir}.")


def main() -> None:
    parser = ArgumentParser(description="Generate a small VQA sample dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sample_data",
        help="Directory where sample data will be generated",
    )
    args = parser.parse_args()
    create_sample_data(args.output_dir)


if __name__ == "__main__":
    main()
