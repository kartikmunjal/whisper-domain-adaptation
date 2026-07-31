import pytest

torch = pytest.importorskip("torch")

from whisper_adapt.models.codec_tts import (
    CodecTTSConfig,
    CodecTokenTTS,
    encode_text_bytes,
)


def tiny_model():
    return CodecTokenTTS(CodecTTSConfig(
        codebook_size=16,
        d_model=32,
        nhead=4,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=64,
        max_text_tokens=32,
        max_audio_tokens=24,
    ))


def test_byte_tokenizer_is_stable():
    assert encode_text_bytes("A") == [66]
    assert encode_text_bytes("é", max_length=1) == [196]


def test_forward_and_checkpoint(tmp_path):
    model = tiny_model()
    text = torch.tensor([[66, 67, 0]])
    audio = torch.tensor([[model.config.audio_bos_id, 1, 2]])
    logits = model(text, audio)
    assert logits.shape == (1, 3, model.config.audio_vocab_size)
    path = tmp_path / "tts.pt"
    model.save_checkpoint(path, seed=11)
    restored = CodecTokenTTS.from_checkpoint(path)
    assert restored(text, audio).shape == logits.shape


def test_generation_respects_limit():
    model = tiny_model()
    generated = model.generate(torch.tensor([[66, 67]]), max_new_tokens=3)
    assert generated.shape[1] <= 3


def test_duration_prediction_is_finite_and_batched():
    model = tiny_model()
    predicted = model.predict_log_lengths(torch.tensor([[66, 67, 0], [68, 0, 0]]))
    assert predicted.shape == (2,)
    assert torch.isfinite(predicted).all()


def test_duration_control_forces_eos_at_predicted_cap():
    model = tiny_model()
    with torch.no_grad():
        for parameter in model.duration_head.parameters():
            parameter.zero_()
        model.duration_head[-1].bias.fill_(torch.log1p(torch.tensor(4.0)))
        model.output.weight.zero_()
        model.output.bias.zero_()
        model.output.bias[1] = 10.0
    generated = model.generate(
        torch.tensor([[66, 67]]),
        max_new_tokens=20,
        use_duration_control=True,
        length_cap_multiplier=1.25,
    )
    assert generated.shape == (1, 5)
    assert generated[0, -1].item() == model.config.audio_eos_id


def test_generate_rejects_invalid_decoding_controls():
    model = tiny_model()
    text = torch.tensor([[66]])
    with pytest.raises(ValueError, match="length_cap_multiplier"):
        model.generate(text, length_cap_multiplier=0)
    with pytest.raises(ValueError, match="repetition_penalty"):
        model.generate(text, repetition_penalty=-0.1)


def test_legacy_checkpoint_without_duration_head_loads(tmp_path):
    model = tiny_model()
    path = tmp_path / "legacy.pt"
    state = {
        key: value for key, value in model.state_dict().items()
        if not key.startswith("duration_head.")
    }
    torch.save({"config": model.config.__dict__, "state_dict": state}, path)
    restored = CodecTokenTTS.from_checkpoint(path)
    assert restored(torch.tensor([[66]]), torch.tensor([[17]])).shape[-1] == 18
