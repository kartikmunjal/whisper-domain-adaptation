"""
Training loop for Whisper domain adaptation.

Uses HuggingFace Trainer with a custom data collator and WER callback.
The Seq2Seq trainer variant handles the generate() call during evaluation,
which is necessary for Whisper — the model's CTC-style encoder-decoder
architecture means teacher-forced perplexity and autoregressive WER can
diverge significantly (we care about WER, not perplexity).

Training considerations
-----------------------
gradient_checkpointing=True
    Whisper-small's encoder is a 12-layer transformer over 3000 time steps —
    activations are large. Checkpointing halves peak VRAM at ~30% compute cost.

fp16=True (GPU) / bf16 (Ampere+)
    Mixed precision is safe here because LoRA adapter updates are small and
    the frozen base weights don't accumulate gradient noise.

eval_strategy="steps" not "epoch"
    Medical dictation clips average ~5 seconds; 8,000 clips × 3 epochs finishes
    in ~2 hours on a single A100. Evaluating every 500 steps gives a good
    learning curve without excessive overhead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    EarlyStoppingCallback,
)

from ..data.feature_extraction import DataCollatorSpeechSeq2SeqWithPadding
from ..evaluation.wer import compute_wer_metrics

logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    output_dir: str = "checkpoints/medical"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 25
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    predict_with_generate: bool = True
    generation_max_length: int = 225
    dataloader_num_workers: int = 4
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    early_stopping_patience: int = 5
    push_to_hub: bool = False
    report_to: list[str] = field(default_factory=lambda: ["none"])


def run_finetune(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: Optional[FinetuneConfig] = None,
) -> Seq2SeqTrainer:
    """
    Fine-tune Whisper using Seq2SeqTrainer.

    Parameters
    ----------
    model         : PEFT-wrapped Whisper model from build_whisper_lora()
    processor     : matching WhisperProcessor
    train_dataset : HuggingFace Dataset with 'input_features' and 'labels'
    eval_dataset  : same schema, held-out split
    cfg           : training hyperparameters

    Returns
    -------
    Trainer instance after training completes (best model loaded automatically)
    """
    cfg = cfg or FinetuneConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gradient checkpointing requires disabling Whisper's input_requires_grad
    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def _compute_metrics(pred):
        return compute_wer_metrics(pred, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        predict_with_generate=cfg.predict_with_generate,
        generation_max_length=cfg.generation_max_length,
        dataloader_num_workers=cfg.dataloader_num_workers,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        push_to_hub=cfg.push_to_hub,
        report_to=cfg.report_to,
    )

    callbacks = []
    if cfg.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=callbacks,
    )

    logger.info("Starting fine-tuning: %d train / %d eval samples",
                len(train_dataset), len(eval_dataset))

    trainer.train()

    # Save the final adapter weights and processor
    adapter_save_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_save_path))
    processor.save_pretrained(str(adapter_save_path))
    logger.info("Saved adapter to %s", adapter_save_path)

    return trainer
