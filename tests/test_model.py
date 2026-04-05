"""
Tests for model setup and LoRA configuration.

These tests don't run actual training; they verify the model architecture
is set up correctly and that the trainable parameter count is as expected.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whisper_adapt.models.whisper_lora import LoRAConfig


class TestLoRAConfig:
    def test_default_config(self):
        cfg = LoRAConfig()
        assert cfg.r == 32
        assert cfg.lora_alpha == 64
        assert "q_proj" in cfg.target_modules
        assert "fc1" in cfg.target_modules

    def test_custom_rank(self):
        cfg = LoRAConfig(r=16, lora_alpha=32)
        assert cfg.r == 16
        assert cfg.lora_alpha == 32

    def test_target_modules_immutable_default(self):
        # Ensure each instance gets its own list (not shared mutable default)
        cfg1 = LoRAConfig()
        cfg2 = LoRAConfig()
        cfg1.target_modules.append("test_module")
        assert "test_module" not in cfg2.target_modules


class TestBuildWhisperLora:
    """These tests require transformers + peft to be installed."""

    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers not installed",
    )
    def test_build_returns_model(self):
        from whisper_adapt.models.whisper_lora import build_whisper_lora
        model = build_whisper_lora(model_id="openai/whisper-tiny")
        assert model is not None

    @pytest.mark.skipif(
        not _transformers_available(),
        reason="transformers not installed",
    )
    def test_trainable_params_reduced(self):
        from whisper_adapt.models.whisper_lora import build_whisper_lora
        model = build_whisper_lora(model_id="openai/whisper-tiny", lora_cfg=LoRAConfig(r=8))
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # LoRA should keep trainable params well below 10% of total
        assert trainable / total < 0.10, (
            f"Expected < 10% trainable params, got {trainable/total:.1%}"
        )


def _transformers_available() -> bool:
    try:
        import transformers
        import peft
        return True
    except ImportError:
        return False
