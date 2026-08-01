#!/usr/bin/env python3
"""Encode paired financial speech into VQ targets for neural-codec TTS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.models.audio_codec import AudioVQVAE
from whisper_adapt.models.codec_tts import encode_text_bytes, encode_text_phonemes, phoneme_vocabulary
from whisper_adapt.reproducibility import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--codec-checkpoint",
        default="checkpoints/codec_rate_grid/vq_400bps/seed_11/codec.pt",
    )
    parser.add_argument("--data-dir", default="data/financial_research")
    parser.add_argument("--output-dir", default="data/codec_tts_tokens")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text-representation", choices=("bytes", "phonemes"), default="bytes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / args.codec_checkpoint
    device = torch.device(args.device)
    codec = AudioVQVAE.from_checkpoint(checkpoint, map_location=device).to(device)
    codec.eval()
    if codec.quantizer_name != "vq":
        raise RuntimeError("Text-to-codec study is preregistered for a VQ codec")
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    counts = {}; total_words = 0; total_oov = 0
    for split in ("train", "validation", "test"):
        frame = pd.read_parquet(root / args.data_dir / f"{split}_manifest.parquet")
        rows = []
        for row in frame.to_dict("records"):
            audio, _ = librosa.load(
                root / row["path"], sr=codec.cfg.sample_rate, mono=True
            )
            tensor = torch.tensor(audio, dtype=torch.float32, device=device)[None]
            with torch.inference_mode():
                _, info = codec.quantize(codec.encode(tensor))
            tokens = info["indices"][0].cpu().tolist()
            if args.text_representation == "phonemes":
                text_ids, oov = encode_text_phonemes(row["sentence"], max_length=256)
                total_oov += oov
            else:
                text_ids = encode_text_bytes(row["sentence"], max_length=256)
            total_words += len(str(row["sentence"]).split())
            rows.append({
                **row,
                "text_token_ids": text_ids,
                "codec_token_ids": tokens,
                "n_codec_tokens": len(tokens),
                "codec_checkpoint": args.codec_checkpoint,
                "codec_sha256": sha256_file(checkpoint),
            })
        pd.DataFrame(rows).to_parquet(output / f"{split}.parquet", index=False)
        counts[split] = len(rows)
    report = {
        "schema_version": 1,
        "generator": "scripts/prepare_codec_tts_tokens.py",
        "codec_checkpoint": args.codec_checkpoint,
        "codec_sha256": sha256_file(checkpoint),
        "codebook_size": codec.cfg.codebook_size,
        "frame_rate_hz": codec.cfg.frame_rate_hz,
        "text_representation": args.text_representation,
        "text_vocab_size": 258 if args.text_representation == "bytes" else len(phoneme_vocabulary()) + 1,
        "phoneme_vocabulary": phoneme_vocabulary() if args.text_representation == "phonemes" else None,
        "oov_word_count": total_oov,
        "approximate_word_count": total_words,
        "split_counts": counts,
        "selection_rule": (
            "Preregistered median-rate VQ checkpoint, seed 11; selected without "
            "round-trip test WER."
        ),
    }
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
