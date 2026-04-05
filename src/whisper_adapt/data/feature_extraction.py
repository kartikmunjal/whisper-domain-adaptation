"""
Log-mel spectrogram feature extraction and Whisper tokenisation.

Whisper's encoder consumes fixed-length 80-channel log-mel spectrograms
computed over 30-second windows (padded or truncated). The decoder uses
a byte-pair encoding tokeniser that operates on the transcript text.

This module wraps Transformers' WhisperProcessor to produce batches in the
format the Trainer expects. The only non-trivial logic is the label masking:
Whisper uses -100 as the ignore index in cross-entropy, so padding tokens
in the label sequence must be replaced before loss computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
from transformers import WhisperProcessor


@dataclass
class WhisperFeatureExtractor:
    """
    Wraps WhisperProcessor to produce ready-to-train batches.

    Parameters
    ----------
    model_id : HuggingFace model ID (e.g. "openai/whisper-small")
    language : target language code (default: "en")
    task     : "transcribe" or "translate"
    """

    model_id: str = "openai/whisper-small"
    language: str = "en"
    task: str = "transcribe"
    _processor: Optional[WhisperProcessor] = None

    def __post_init__(self) -> None:
        self._processor = WhisperProcessor.from_pretrained(self.model_id)
        self._processor.tokenizer.set_prefix_tokens(
            language=self.language, task=self.task
        )

    @property
    def processor(self) -> WhisperProcessor:
        assert self._processor is not None
        return self._processor

    def __call__(self, audio: np.ndarray, text: str) -> dict[str, torch.Tensor]:
        """
        Process a single (audio, transcript) pair.

        Returns
        -------
        dict with keys:
          input_features  — (1, 80, 3000) log-mel spectrogram
          labels          — (seq_len,) token IDs with -100 padding mask
        """
        features = self.processor.feature_extractor(
            audio,
            sampling_rate=16_000,
            return_tensors="pt",
        ).input_features  # (1, 80, 3000)

        label_ids = self.processor.tokenizer(
            text, return_tensors="pt"
        ).input_ids.squeeze(0)  # (seq_len,)

        # Replace padding token IDs with -100 so they're ignored in loss
        pad_id = self.processor.tokenizer.pad_token_id
        label_ids = label_ids.masked_fill(label_ids == pad_id, -100)

        return {"input_features": features.squeeze(0), "labels": label_ids}


def prepare_batch(
    batch: dict[str, Any],
    extractor: WhisperFeatureExtractor,
    audio_column: str = "audio",
    text_column: str = "sentence",
) -> dict[str, list]:
    """
    HuggingFace Datasets .map() compatible function.

    Usage
    -----
    >>> ds = ds.map(
    ...     lambda b: prepare_batch(b, extractor),
    ...     remove_columns=ds.column_names,
    ...     batched=True,
    ...     batch_size=8,
    ... )
    """
    input_features = []
    labels = []

    audios = batch[audio_column]
    texts = batch[text_column]

    for audio_data, text in zip(audios, texts):
        if isinstance(audio_data, dict):
            arr = np.array(audio_data["array"], dtype=np.float32)
        else:
            arr = np.array(audio_data, dtype=np.float32)

        processed = extractor(arr, text)
        input_features.append(processed["input_features"])
        labels.append(processed["labels"])

    return {"input_features": input_features, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom data collator for Whisper fine-tuning.

    Pads input_features and labels independently:
      - input_features: padded to max length in batch (or 3000 fixed)
      - labels: padded to max seq length with -100
    """

    processor: WhisperProcessor

    def __call__(
        self, features: list[dict[str, Union[list[int], torch.Tensor]]]
    ) -> dict[str, torch.Tensor]:
        # Separate input features and labels
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace pad token ID with -100 in labels
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove the initial BOS token if it was prepended by the collator
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
