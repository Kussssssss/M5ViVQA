#!/usr/bin/env python
"""
Main script to run ViT5-VQA training and evaluation using Python configurations.

This script allows you to train and evaluate the ViT5-VQA model on either the full dataset or a small synthetic dataset. Use the ``--config`` argument to choose between ``full`` and ``sample`` configurations.
"""

import argparse
import logging
import os
import torch

from openvivqa.data.dataset import load_dataset, ViT5VQADataset, ViT5VQADataCollator
from openvivqa.models.vit5_vqa import ViT5VQAModel
from openvivqa.training.trainer import CustomSeq2SeqTrainer
from openvivqa.evaluation.metrics import compute_vqa_metrics
from transformers import Seq2SeqTrainingArguments

# Import configuration dictionaries
from config import FULL_CONFIG, SAMPLE_CONFIG

os.environ["WANDB_DISABLED"] = "True"


def get_config(name: str):
    """Return the configuration dictionary based on the provided name."""
    if name == "full":
        return FULL_CONFIG
    if name == "sample":
        return SAMPLE_CONFIG
    raise ValueError(f"Unknown config name: {name}")


def run_experiment(cfg: dict) -> None:
    """Train and evaluate the ViT5-VQA model using the specified configuration."""
    data_dir = cfg["dataset"]["data_dir"]
    output_dir = cfg["output"]["output_dir"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logging.info("Loading dataset...")
    train_df, val_df, test_df = load_dataset(data_dir)
    logging.info(f"Loaded data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Initialize the model
    model = ViT5VQAModel(
        vit5_model_name=model_cfg["vit5_model"],
        vit_model_name=model_cfg["vit_model"],
    )
    if torch.cuda.is_available():
        model.cuda()
        logging.info("Using GPU for training")
    else:
        logging.info("Using CPU for training")

    # Prepare datasets and collator
    train_dataset = ViT5VQADataset(train_df, model.tokenizer, model.image_processor)
    val_dataset = ViT5VQADataset(val_df, model.tokenizer, model.image_processor)
    test_dataset = ViT5VQADataset(test_df, model.tokenizer, model.image_processor)
    collator = ViT5VQADataCollator(model.tokenizer)

    # Build training arguments
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=train_cfg.get("logging_steps", 200),
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 200),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        dataloader_num_workers=train_cfg.get("num_workers", 4),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        predict_with_generate=True,
        generation_max_length=train_cfg.get("generation_max_length", 128),
        generation_num_beams=train_cfg.get("generation_num_beams", 2),
        save_safetensors=False,
        fp16=torch.cuda.is_available(),
    )

    def metrics_fn(eval_pred):
        return compute_vqa_metrics(eval_pred, model.tokenizer)

    # Initialize trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=metrics_fn,
    )

    logging.info("Starting training...")
    trainer.train()

    logging.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    logging.info(f"Test results: {test_results}")

    logging.info("Saving final model...")
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    logging.info("Training complete")


def main() -> None:
    """Entry point of the script."""
    parser = argparse.ArgumentParser(
        description="Run ViT5-VQA training and evaluation using a specified configuration"
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
