"""Small encoder-decoder Transformer for text-to-VQ-token generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

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


class CodecTokenTTS(nn.Module):
    def __init__(self, config: CodecTTSConfig | None = None):
        super().__init__()
        self.config = config or CodecTTSConfig()
        cfg = self.config
        self.text_embedding = nn.Embedding(cfg.text_vocab_size, cfg.d_model, padding_idx=0)
        self.audio_embedding = nn.Embedding(cfg.audio_vocab_size, cfg.d_model)
        self.text_position = nn.Embedding(cfg.max_text_tokens, cfg.d_model)
        self.audio_position = nn.Embedding(cfg.max_audio_tokens + 1, cfg.d_model)
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

    def forward(
        self,
        text_ids: torch.Tensor,
        decoder_ids: torch.Tensor,
    ) -> torch.Tensor:
        text_positions = torch.arange(text_ids.shape[1], device=text_ids.device)
        audio_positions = torch.arange(decoder_ids.shape[1], device=decoder_ids.device)
        memory = self.text_embedding(text_ids) + self.text_position(text_positions)
        text_padding = text_ids.eq(0)
        memory = self.encoder(memory, src_key_padding_mask=text_padding)
        target = self.audio_embedding(decoder_ids) + self.audio_position(audio_positions)
        causal = nn.Transformer.generate_square_subsequent_mask(
            decoder_ids.shape[1], device=decoder_ids.device
        )
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
    ) -> torch.Tensor:
        self.eval()
        generated = torch.full(
            (text_ids.shape[0], 1),
            self.config.audio_bos_id,
            dtype=torch.long,
            device=text_ids.device,
        )
        finished = torch.zeros(text_ids.shape[0], dtype=torch.bool, device=text_ids.device)
        for _ in range(min(max_new_tokens, self.config.max_audio_tokens)):
            next_token = self(text_ids, generated)[:, -1].argmax(dim=-1)
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
        model.load_state_dict(payload["state_dict"])
        return model
