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
