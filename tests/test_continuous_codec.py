import pytest

torch = pytest.importorskip("torch")

from whisper_adapt.continuous_codec import (
    ContinuousAudioVAE,
    ContinuousCodecConfig,
    uniform_quantize,
)
from whisper_adapt.models.audio_codec import AudioCodecConfig


def tiny_config() -> ContinuousCodecConfig:
    return ContinuousCodecConfig(
        audio=AudioCodecConfig(hidden_dim=16, latent_dim=8),
        bottleneck_dim=3,
        quantization_bits=4,
    )


def test_continuous_codec_forward_and_fixed_width_rate():
    model = ContinuousAudioVAE(tiny_config())
    output = model(torch.randn(2, 1024))
    assert output["reconstruction"].shape == (2, 1, 1024)
    assert output["posterior_mean"].shape[-1] == 3
    assert output["kl_per_frame"].item() >= 0
    assert model.fixed_width_bitrate_bps == 600


def test_uniform_quantization_is_bounded_and_has_valid_codes():
    values = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
    restored, indices, saturation = uniform_quantize(values, bits=3, clip_value=2.0)
    assert indices.min() == 0 and indices.max() == 7
    assert restored.min() >= -2 and restored.max() <= 2
    assert saturation.item() == pytest.approx(0.4)


def test_continuous_checkpoint_roundtrip(tmp_path):
    model = ContinuousAudioVAE(tiny_config()).eval()
    checkpoint = tmp_path / "continuous.pt"
    model.save_checkpoint(checkpoint, seed=11)
    restored = ContinuousAudioVAE.from_checkpoint(checkpoint)
    audio = torch.randn(1, 1024)
    assert restored.reconstruct(audio, quantized=True).shape == (1, 1, 1024)


def test_chunked_reconstruction_preserves_length_for_mean_and_quantized():
    model = ContinuousAudioVAE(tiny_config()).eval()
    audio = torch.randn(1, 2500)
    mean = model.reconstruct_chunked(audio, chunk_samples=1024, overlap_samples=128)
    quantized = model.reconstruct_chunked(
        audio, quantization_bits=1, chunk_samples=1024, overlap_samples=128
    )
    assert mean.shape == quantized.shape == (1, 1, 2500)
