#!/usr/bin/env python
"""
Main script to run ViT5-VQA training and evaluation using Python configurations.

This script acts as a wrapper around the existing training pipeline in openvivqa.training.train.
Use the ``--config`` argument to choose between ``full`` and ``sample`` configurations.
"""

import argparse
import logging
import os
import sys

# Import configuration dictionaries
from config import FULL_CONFIG, SAMPLE_CONFIG

# Add the current directory to Python path to import openvivqa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openvivqa.training.train import main as train_main

os.environ["WANDB_DISABLED"] = "True"

def get_config(name: str):
    """Return the configuration dictionary based on the provided name."""
    if name == "full":
        return FULL_CONFIG
    if name == "sample":
        return SAMPLE_CONFIG
    raise ValueError(f"Unknown config name: {name}")


def run_experiment(cfg: dict) -> None:
    """Run training using the existing training pipeline with the specified configuration."""
    data_dir = cfg["dataset"]["data_dir"]
    output_dir = cfg["output"]["output_dir"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    os.makedirs(output_dir, exist_ok=True)

    # Set up command line arguments for the training script
    sys.argv = [
        "train.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--epochs", str(train_cfg["epochs"]),
        "--batch_size", str(train_cfg["batch_size"]),
        "--gradient_accumulation_steps", str(train_cfg["gradient_accumulation_steps"]),
        "--learning_rate", str(train_cfg["learning_rate"]),
        "--vit5_model", model_cfg["vit5_model"],
        "--vit_model", model_cfg["vit_model"],
        "--num_workers", str(train_cfg.get("num_workers", 4)),
        "--eval_steps", str(train_cfg.get("eval_steps", 50)),
        "--save_steps", str(train_cfg.get("save_steps", 50)),
    ]

    # Optional MoE decoder flags
    if model_cfg.get("use_moe_decoder", False):
        sys.argv.append("--use_moe_decoder")
        if model_cfg.get("num_experts") is not None:
            sys.argv.extend(["--num_experts", str(model_cfg["num_experts"])])
        if model_cfg.get("moe_hidden_dim") is not None:
            sys.argv.extend(["--moe_hidden_dim", str(model_cfg["moe_hidden_dim"])])

    # Add optional parameters if they exist
    if "weight_decay" in train_cfg:
        sys.argv.extend(["--weight_decay", str(train_cfg["weight_decay"])])
    if "warmup_steps" in train_cfg:
        sys.argv.extend(["--warmup_steps", str(train_cfg["warmup_steps"])])

    train_main()


def main() -> None:
    """Entry point of the script."""
    parser = argparse.ArgumentParser(
        description="Run ViT5-VQA training using existing training pipeline with specified configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["full", "sample"],
        default="full",
        help="Choose configuration: 'full' for the real dataset or 'sample' for the synthetic dataset",
    )
    args = parser.parse_args()
    cfg = get_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
