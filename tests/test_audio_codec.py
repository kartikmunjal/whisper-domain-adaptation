import pytest

torch = pytest.importorskip("torch")

from whisper_adapt.models.audio_codec import (
    AudioCodecConfig,
    AudioVQVAE,
    FiniteScalarQuantizer,
    FSQRangeNormalizer,
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
    assert info["usage_histogram"].sum().item() == 10


def test_vq_ema_updates_without_embedding_gradients_and_resets_dead_codes():
    quantizer = VectorQuantizer(
        codebook_size=8, latent_dim=4, ema_decay=0.9, dead_code_batches=1
    )
    before = quantizer.embedding.weight.detach().clone()
    _, info = quantizer(torch.randn(2, 5, 4))

    assert quantizer.embedding.weight.grad is None
    assert not torch.equal(before, quantizer.embedding.weight)
    assert info["dead_codes_reset"].item() > 0
    assert quantizer.total_resets.item() == info["dead_codes_reset"].item()


def test_fsq_range_normalization_and_grid_indices():
    normalizer = FSQRangeNormalizer(dimensions=4, target_std=1.25)
    values = normalizer(
        torch.randn(30, 7, 4) * torch.tensor([0.001, 0.01, 0.1, 1.0])
        + torch.tensor([2.0, -3.0, 4.0, -5.0])
    )
    dimension_mean = values.mean(dim=(0, 1))
    dimension_std = values.std(dim=(0, 1), unbiased=False)
    assert torch.allclose(dimension_mean, torch.zeros_like(dimension_mean), atol=2e-2)
    assert torch.allclose(
        dimension_std, torch.full_like(dimension_std, 1.25), atol=2e-2
    )

    quantizer = FiniteScalarQuantizer(levels=(4, 4, 4, 4))
    _, info = quantizer(values)
    assert info["indices"].min() >= 0
    assert info["indices"].max() < 4
    assert info["unique_code_vectors"].item() > 1


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


def test_legacy_vq_checkpoint_loads_without_ema_buffers(tmp_path):
    cfg = AudioCodecConfig(hidden_dim=16, latent_dim=8, codebook_size=16)
    model = AudioVQVAE(cfg, quantizer="vq")
    state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("quantizer.ema_")
        and key not in {
            "quantizer.batches_since_use",
            "quantizer.total_resets",
        }
    }
    legacy_config = {
        key: value
        for key, value in cfg.__dict__.items()
        if not key.startswith("vq_") and key != "fsq_input_scale"
    }
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {"config": legacy_config, "quantizer": "vq", "state_dict": state},
        checkpoint,
    )
    restored = AudioVQVAE.from_checkpoint(checkpoint)
    assert restored.quantizer.ema_cluster_size.shape == (16,)


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
