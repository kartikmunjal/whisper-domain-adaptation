import pytest

torch = pytest.importorskip("torch")

from whisper_adapt.models.audio_codec import (
    AudioCodecConfig,
    AudioVQVAE,
    FiniteScalarQuantizer,
    VectorQuantizer,
    codec_rate_hz,
)


def test_vq_codec_reconstructs_input_shape():
    cfg = AudioCodecConfig(hidden_dim=16, latent_dim=8, codebook_size=16)
    model = AudioVQVAE(cfg, quantizer="vq")
    audio = torch.randn(2, 1024)

    out = model(audio)

    assert out["reconstruction"].shape == (2, 1, 1024)
    assert out["indices"].shape[:2] == out["latents"].shape[:2]
    assert out["loss"].requires_grad


def test_fsq_codec_reconstructs_input_shape():
    cfg = AudioCodecConfig(hidden_dim=16, latent_dim=8, fsq_levels=(4, 4, 4, 4))
    model = AudioVQVAE(cfg, quantizer="fsq")
    audio = torch.randn(2, 1, 1024)

    out = model(audio)

    assert out["reconstruction"].shape == audio.shape
    assert out["quantizer_loss"].item() == 0.0


def test_vector_quantizer_outputs_codebook_indices():
    quantizer = VectorQuantizer(codebook_size=8, latent_dim=4)
    latents = torch.randn(2, 5, 4)

    quantized, info = quantizer(latents)

    assert quantized.shape == latents.shape
    assert info["indices"].shape == (2, 5)
    assert info["perplexity"].item() > 0


def test_fsq_requires_divisible_latent_dim():
    quantizer = FiniteScalarQuantizer(levels=(4, 4, 4))
    latents = torch.randn(2, 5, 8)

    try:
        quantizer(latents)
    except ValueError as exc:
        assert "latent_dim" in str(exc)
    else:
        raise AssertionError("Expected ValueError for incompatible FSQ latent dimension")


def test_codec_rate_hz_matches_two_stride_encoder():
    assert codec_rate_hz(16_000) == 4_000
