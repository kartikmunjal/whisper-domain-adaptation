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
    assert out["indices"].shape[-1] == 4


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
    assert codec_rate_hz(16_000) == 50


def test_inference_reconstruction_and_matched_rates(tmp_path):
    audio = torch.randn(1, 16_123)
    for quantizer, kwargs in (
        ("vq", {"codebook_size": 256}),
        ("fsq", {"fsq_levels": (4, 4, 4, 4)}),
    ):
        cfg = AudioCodecConfig(hidden_dim=16, latent_dim=8, **kwargs)
        model = AudioVQVAE(cfg, quantizer=quantizer)
        reconstruction = model.reconstruct(audio)
        assert reconstruction.shape == (1, 1, audio.shape[-1])
        assert not reconstruction.requires_grad
        assert model.nominal_bits_per_frame == 8
        assert model.nominal_bitrate_bps == 400

        checkpoint = tmp_path / f"{quantizer}.pt"
        torch.save(
            {
                "config": cfg.__dict__,
                "quantizer": quantizer,
                "state_dict": model.state_dict(),
            },
            checkpoint,
        )
        restored = AudioVQVAE.from_checkpoint(checkpoint)
        assert restored.reconstruct(audio).shape == reconstruction.shape


def test_chunked_reconstruction_preserves_long_shape():
    cfg = AudioCodecConfig(hidden_dim=16, latent_dim=8, codebook_size=16)
    model = AudioVQVAE(cfg, quantizer="vq")
    audio = torch.randn(1, 5000)
    reconstructed = model.reconstruct_chunked(
        audio, chunk_samples=2048, overlap_samples=320
    )
    assert reconstructed.shape == (1, 1, 5000)
    assert torch.isfinite(reconstructed).all()


def test_decode_vq_indices_shape_and_validation():
    cfg = AudioCodecConfig(hidden_dim=16, latent_dim=8, codebook_size=16)
    model = AudioVQVAE(cfg, quantizer="vq")
    audio = model.decode_vq_indices(torch.tensor([[0, 1, 2, 3]]))
    assert audio.shape == (1, 1, 4 * 320)
    with pytest.raises(ValueError, match="outside"):
        model.decode_vq_indices(torch.tensor([16]))
