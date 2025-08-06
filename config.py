"""
Python configuration for ViT5-VQA project.

This module defines two dictionaries containing configuration settings for training the ViT5-VQA model. ``FULL_CONFIG`` should be used when training on the full dataset, while ``SAMPLE_CONFIG`` provides minimal settings for experiments on a small synthetic dataset.
"""

# Configuration for training on the full, real dataset
FULL_CONFIG = {
    "dataset": {
        "data_dir": "data/openvivqa",  # Path to prepared dataset directory
    },
    "output": {
        "output_dir": "vit5-vqa-results",  # Directory to save checkpoints and logs
    },
    "model": {
        "vit5_model": "VietAI/vit5-base",
        "vit_model": "google/vit-base-patch16-224-in21k",
    },
    "training": {
        "epochs": 3,
        "batch_size": 24,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-4,
        "num_workers": 4,
        "eval_steps": 200,
        "save_steps": 200,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "logging_steps": 200,
        "generation_max_length": 128,
        "generation_num_beams": 2,
    },
    "evaluation": {
        "metrics": ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rougeL", "cider"],
    },
}

# Configuration for training on the small synthetic sample dataset
SAMPLE_CONFIG = {
    "dataset": {
        "data_dir": "sample_data",
    },
    "output": {
        "output_dir": "sample_results",
    },
    "model": {
        "vit5_model": "VietAI/vit5-base",
        "vit_model": "google/vit-base-patch16-224-in21k",
    },
    "training": {
        "epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "num_workers": 0,
        "eval_steps": 1,
        "save_steps": 1,
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "logging_steps": 1,
        "generation_max_length": 32,
        "generation_num_beams": 1,
    },
    "evaluation": {
        "metrics": ["bleu1", "meteor", "rougeL", "cider"],
    },
}

__all__ = ["FULL_CONFIG", "SAMPLE_CONFIG"]
