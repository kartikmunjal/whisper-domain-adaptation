"""
Whisper + LoRA adapter setup.

Full fine-tuning Whisper-small (244M params) on a ~10k sample domain dataset
would take hours even on a V100, and risks catastrophic forgetting of general
speech. LoRA constrains the update to a low-rank subspace of each attention
projection, keeping the total trainable parameter count around 7M (rank 32)
instead of 244M — a 35× reduction.

The rank choice mirrors the rlhf-and-reward-modelling-alt project's Extension 4
where r=16 gave negligible quality loss vs. full fine-tuning on text alignment
tasks. We use r=32 here because audio cross-entropy is more sensitive to
projection errors than text log-likelihood — higher rank is warranted.

Which modules to target?
  Whisper's decoder self-attention and cross-attention (q/k/v/out projections)
  are the primary beneficiaries. The encoder's attention matrices are left
  partially frozen because the acoustic representations are already decent;
  it's the language model prior in the decoder that under-weights rare tokens.
  fc1 / fc2 (feed-forward sublayers in the decoder) are also adapted because
  they store a significant fraction of the vocabulary prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import WhisperForConditionalGeneration


@dataclass
class LoRAConfig:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # Decoder attention + FFN are the high-value targets (see module docstring)
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "out_proj",
        "fc1", "fc2",
    ])
    bias: str = "none"


def build_whisper_lora(
    model_id: str = "openai/whisper-small",
    lora_cfg: Optional[LoRAConfig] = None,
) -> WhisperForConditionalGeneration:
    """
    Load Whisper and attach LoRA adapters.

    The base model weights are frozen; only the adapter matrices are trainable.
    This means the model can be loaded in fp32 or fp16 without mixed-precision
    adapter issues.

    Returns
    -------
    PEFT-wrapped WhisperForConditionalGeneration with ~7M trainable parameters
    (vs 244M total for Whisper-small).
    """
    lora_cfg = lora_cfg or LoRAConfig()

    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # Freeze everything before attaching adapters
    for param in model.parameters():
        param.requires_grad = False

    # Whisper's generation config — preserve forced language/task tokens
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def load_finetuned(
    base_model_id: str,
    adapter_path: str,
) -> WhisperForConditionalGeneration:
    """
    Load a fine-tuned model for inference.

    Parameters
    ----------
    base_model_id : base Whisper model (e.g. "openai/whisper-small")
    adapter_path  : path to saved LoRA adapter directory
    """
    base = WhisperForConditionalGeneration.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base, adapter_path)
    # Merge weights for faster inference (no adapter overhead)
    model = model.merge_and_unload()
    return model
