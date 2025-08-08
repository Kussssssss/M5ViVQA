#!/usr/bin/env python3
"""Download and prepare the OpenViVQA dataset from Google Drive.

This script is tailored for running on platforms like Kaggle where
hard‑coded paths are preferable and user interaction should be minimal.
Instead of requiring command line arguments for the Google Drive file IDs
and output directories, all necessary IDs and URLs are defined
statically within the script. Running this file will download the
compressed image archive and the train/dev/test JSON annotation files,
extract the images and copy the annotations into a clean output
directory, and finally print a summary of the resulting folder
structure.

Example usage on Kaggle:

```bash
python download_data_openvivqa.py
```

This will download the dataset into ``/kaggle/working/OpenViVQA_raw``,
prepare the processed dataset in ``/kaggle/working/OpenViVQA``, and
report the directory contents. Adjust the ``RAW_DATASET_DIR`` and
``OUT_DIR`` constants below if you wish to change these locations.

Note: ``gdown`` is used to download the image archive from Google
Drive. Ensure that the ``gdown`` Python package is installed. If not,
you can install it with ``pip install gdown`` prior to running this
script.
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict

import gdown
import requests


# ---------------------------------------------------------------------------
# Configuration
#
# These constants define where the raw data will be downloaded and where
# the processed dataset will be stored. They also specify the Google Drive
# ID for the image zip file and the direct download URLs for the JSON
# annotation files. Feel free to modify these values to suit your local
# environment or preferred directory layout.

# Directory where downloads will be stored before extraction/copy.
RAW_DATASET_DIR = Path("/kaggle/working/OpenViVQA_raw")
# Directory where the final dataset will be organised.
OUT_DIR = Path("/kaggle/working/OpenViVQA")

# Google Drive file ID for ``images.zip`` containing all images.
IMAGE_ZIP_ID = "10z-92oXTvX2hIk0ds4yJOOHav5GGiEyc"

# Direct download URLs for JSON annotation files. These are the canonical
# links provided by the dataset authors; they do not require any special
# cookies or tokens to download.
TRAIN_JSON_URL = "https://drive.google.com/uc?export=download&id=16x3h386Q_2UfCxT_3vXmPuXLScxid9L6"
DEV_JSON_URL = "https://drive.google.com/uc?export=download&id=1x8nW50igqUT90LUqmL5h66LoCYkkPTZA"
TEST_JSON_URL = "https://drive.google.com/uc?export=download&id=10azOS9TzgQl8HrztbexlKh08pkyMb4m5"


def download_json(url: str, output_path: Path) -> None:
    """Download a JSON file from a URL and write it to disk.

    Parameters
    ----------
    url: str
        The URL to download from.
    output_path: Path
        Path to write the downloaded file to.
    """
    print(f"Downloading {output_path.name} from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def ensure_dir(path: Path) -> None:
    """Ensure that a directory exists."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def download_dataset() -> None:
    """Download and prepare the OpenViVQA dataset.

    This function orchestrates the download of the image archive and
    annotation files, extracts the images into the output directory and
    copies the annotation files to the same location. It will skip
    downloading files that already exist on disk.
    """
    # Ensure raw and output directories exist
    ensure_dir(RAW_DATASET_DIR)
    ensure_dir(OUT_DIR)

    # Download the image archive if not already present
    images_zip_path = RAW_DATASET_DIR / "images.zip"
    if images_zip_path.exists():
        print(f"Images archive already exists at {images_zip_path}, skipping download")
    else:
        print(f"Downloading images archive to {images_zip_path}")
        gdown.download(id=IMAGE_ZIP_ID, output=str(images_zip_path), fuzzy=True)

    # Download JSON annotation files
    json_files: Dict[str, str] = {
        "vlsp2023_train_data.json": TRAIN_JSON_URL,
        "vlsp2023_dev_data.json": DEV_JSON_URL,
        "vlsp2023_test_data.json": TEST_JSON_URL,
    }
    for filename, url in json_files.items():
        json_path = RAW_DATASET_DIR / filename
        if json_path.exists():
            print(f"{filename} already exists, skipping download")
        else:
            download_json(url, json_path)

    # Extract images to output directory
    images_out_dir = OUT_DIR / "images"
    ensure_dir(images_out_dir)
    print(f"Extracting {images_zip_path} to {images_out_dir}")
    with zipfile.ZipFile(images_zip_path, "r") as zf:
        zf.extractall(images_out_dir)

    # Copy JSON files to output directory
    for filename in json_files:
        src = RAW_DATASET_DIR / filename
        dst = OUT_DIR / filename
        print(f"Copying {src} → {dst}")
        shutil.copy(src, dst)

    # Print summary of the output directory
    print(f"\nDataset prepared at {OUT_DIR}")
    for root, dirs, files in os.walk(OUT_DIR):
        level = root.replace(str(OUT_DIR), "").count(os.sep)
        indent = "    " * level
        print(f"{indent}{os.path.basename(root)}/")
        if level == 0:  # Only print file lists for top-level and one level below
            for f in files:
                print(f"{indent}    {f}")


if __name__ == "__main__":
    download_dataset()
