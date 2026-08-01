"""Small encoder-decoder Transformer for text-to-VQ-token generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import re

import torch
from torch import nn


@dataclass
class CodecTTSConfig:
    codebook_size: int = 256
    text_vocab_size: int = 258
    d_model: int = 256
    nhead: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_text_tokens: int = 256
    max_audio_tokens: int = 1024
    decoder_input_mode: str = "autoregressive"
    text_representation: str = "bytes"

    @property
    def audio_eos_id(self) -> int:
        return self.codebook_size

    @property
    def audio_bos_id(self) -> int:
        return self.codebook_size + 1

    @property
    def audio_vocab_size(self) -> int:
        return self.codebook_size + 2


def encode_text_bytes(text: str, max_length: int = 256) -> list[int]:
    """Stable UTF-8 byte tokenizer: PAD=0, byte values shifted by one."""
    return [byte + 1 for byte in text.encode("utf-8")[:max_length]]


@lru_cache(maxsize=1)
def _cmudict_resources() -> tuple[dict[str, list[list[str]]], tuple[str, ...]]:
    """Load CMUdict once per process instead of once per utterance."""
    import cmudict

    return cmudict.dict(), tuple(sorted(set(cmudict.symbols())))


def phoneme_vocabulary() -> list[str]:
    """Stable CMUdict phone vocabulary plus explicit OOV grapheme fallback."""
    _, phones = _cmudict_resources()
    return ["<wb>", "<unk>"] + phones + [f"G_{c}" for c in "abcdefghijklmnopqrstuvwxyz0123456789"]


def encode_text_phonemes(text: str, max_length: int = 256) -> tuple[list[int], int]:
    """Encode first CMUdict pronunciations; spell OOV words with marked graphemes."""
    vocab = phoneme_vocabulary(); ids = {token: i + 1 for i, token in enumerate(vocab)}
    dictionary, _ = _cmudict_resources(); output: list[int] = []; oov = 0
    for word in re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", text.lower()):
        if output: output.append(ids["<wb>"])
        pronunciations = dictionary.get(word)
        if pronunciations:
            output.extend(ids.get(phone, ids["<unk>"]) for phone in pronunciations[0])
        else:
            oov += 1
            output.extend(ids.get(f"G_{char}", ids["<unk>"]) for char in word if char.isalnum())
    return output[:max_length], oov


def encode_conditioning_text(text: str, config: CodecTTSConfig) -> list[int]:
    if config.text_representation == "bytes":
        return encode_text_bytes(text, config.max_text_tokens)
    if config.text_representation == "phonemes":
        return encode_text_phonemes(text, config.max_text_tokens)[0]
    raise ValueError(f"Unknown text_representation: {config.text_representation}")


class CodecTokenTTS(nn.Module):
    def __init__(self, config: CodecTTSConfig | None = None):
        super().__init__()
        self.config = config or CodecTTSConfig()
        cfg = self.config
        self.text_embedding = nn.Embedding(cfg.text_vocab_size, cfg.d_model, padding_idx=0)
        self.audio_embedding = nn.Embedding(cfg.audio_vocab_size, cfg.d_model)
        self.text_position = nn.Embedding(cfg.max_text_tokens, cfg.d_model)
        self.audio_position = nn.Embedding(cfg.max_audio_tokens + 1, cfg.d_model)
        self.output_query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            cfg.d_model,
            cfg.nhead,
            cfg.dim_feedforward,
            cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            cfg.d_model,
            cfg.nhead,
            cfg.dim_feedforward,
            cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, cfg.decoder_layers)
        self.output = nn.Linear(cfg.d_model, cfg.audio_vocab_size)
        self.duration_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def encode_text(self, text_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        text_positions = torch.arange(text_ids.shape[1], device=text_ids.device)
        memory = self.text_embedding(text_ids) + self.text_position(text_positions)
        text_padding = text_ids.eq(0)
        memory = self.encoder(memory, src_key_padding_mask=text_padding)
        return memory, text_padding

    def predict_log_lengths(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Predict log1p codec-token length from masked text representations."""
        memory, text_padding = self.encode_text(text_ids)
        keep = (~text_padding).unsqueeze(-1)
        pooled = (memory * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1)
        return self.duration_head(pooled).squeeze(-1)

    def forward(
        self,
        text_ids: torch.Tensor,
        decoder_ids: torch.Tensor,
    ) -> torch.Tensor:
        audio_positions = torch.arange(decoder_ids.shape[1], device=decoder_ids.device)
        memory, text_padding = self.encode_text(text_ids)
        if self.config.decoder_input_mode == "text_only":
            target = self.output_query.expand(decoder_ids.shape[0], decoder_ids.shape[1], -1) + self.audio_position(audio_positions)
            causal = None
        elif self.config.decoder_input_mode == "autoregressive":
            target = self.audio_embedding(decoder_ids) + self.audio_position(audio_positions)
            causal = nn.Transformer.generate_square_subsequent_mask(decoder_ids.shape[1], device=decoder_ids.device)
        else:
            raise ValueError(f"Unknown decoder_input_mode: {self.config.decoder_input_mode}")
        decoded = self.decoder(
            target,
            memory,
            tgt_mask=causal,
            memory_key_padding_mask=text_padding,
        )
        return self.output(decoded)

    @torch.inference_mode()
    def generate(
        self,
        text_ids: torch.Tensor,
        max_new_tokens: int = 600,
        use_duration_control: bool = False,
        length_cap_multiplier: float = 1.25,
        repetition_penalty: float = 0.0,
    ) -> torch.Tensor:
        self.eval()
        if length_cap_multiplier <= 0:
            raise ValueError("length_cap_multiplier must be positive")
        if repetition_penalty < 0:
            raise ValueError("repetition_penalty must be non-negative")
        hard_limit = min(max_new_tokens, self.config.max_audio_tokens)
        if use_duration_control:
            predicted_lengths = torch.expm1(self.predict_log_lengths(text_ids)).round()
            length_caps = (predicted_lengths * length_cap_multiplier).ceil().long()
            length_caps = length_caps.clamp(min=1, max=hard_limit)
        else:
            length_caps = torch.full(
                (text_ids.shape[0],), hard_limit, device=text_ids.device,
                dtype=torch.long,
            )
        if self.config.decoder_input_mode == "text_only":
            width = int(length_caps.max().item()) + 1
            placeholder = torch.full((text_ids.shape[0], width), self.config.audio_bos_id, dtype=torch.long, device=text_ids.device)
            predicted = self(text_ids, placeholder).argmax(dim=-1)
            positions = torch.arange(width, device=text_ids.device)[None, :]
            return predicted.masked_fill(positions >= length_caps[:, None], self.config.audio_eos_id)
        generated = torch.full(
            (text_ids.shape[0], 1),
            self.config.audio_bos_id,
            dtype=torch.long,
            device=text_ids.device,
        )
        finished = torch.zeros(text_ids.shape[0], dtype=torch.bool, device=text_ids.device)
        for step in range(hard_limit):
            logits = self(text_ids, generated)[:, -1]
            if repetition_penalty:
                counts = torch.zeros_like(logits)
                prior = generated[:, 1:]
                if prior.numel():
                    counts.scatter_add_(
                        1, prior, torch.ones_like(prior, dtype=logits.dtype)
                    )
                counts[:, self.config.audio_eos_id :] = 0
                logits = logits - repetition_penalty * counts
            if finished.any():
                logits = logits.clone()
                logits[finished] = -torch.inf
                logits[finished, self.config.audio_eos_id] = 0
            force_eos = torch.full_like(logits[:, 0], step + 1).ge(length_caps)
            if force_eos.any():
                logits = logits.clone()
                logits[force_eos] = -torch.inf
                logits[force_eos, self.config.audio_eos_id] = 0
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            finished |= next_token.eq(self.config.audio_eos_id)
            if finished.all():
                break
        return generated[:, 1:]

    def save_checkpoint(self, path: str | Path, **metadata: object) -> None:
        torch.save({
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "metadata": metadata,
        }, path)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> "CodecTokenTTS":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(CodecTTSConfig(**payload["config"]))
        incompatible = model.load_state_dict(payload["state_dict"], strict=False)
        allowed_missing = {
            "duration_head.0.weight", "duration_head.0.bias",
            "duration_head.2.weight", "duration_head.2.bias",
            "output_query",
        }
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Incompatible TTS checkpoint: missing={sorted(unexpected_missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
        return model
