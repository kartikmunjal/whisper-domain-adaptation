#!/usr/bin/env python3
"""Generate held-out waveforms with a trained text-to-codec-token model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.models.audio_codec import AudioVQVAE
from whisper_adapt.models.codec_tts import CodecTokenTTS, encode_text_bytes
from whisper_adapt.reproducibility import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts-checkpoint", required=True)
    parser.add_argument(
        "--codec-checkpoint",
        default="checkpoints/codec_rate_grid/vq_400bps/seed_11/codec.pt",
    )
    parser.add_argument("--manifest", default="data/financial_research/test_manifest.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    device = torch.device(args.device)
    tts = CodecTokenTTS.from_checkpoint(root / args.tts_checkpoint, device).to(device).eval()
    codec = AudioVQVAE.from_checkpoint(root / args.codec_checkpoint, device).to(device).eval()
    output = root / args.output_dir
    wav_dir = output / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(root / args.manifest)
    rows = []
    for row in frame.to_dict("records"):
        text_ids = torch.tensor(
            [encode_text_bytes(row["sentence"], tts.config.max_text_tokens)],
            dtype=torch.long,
            device=device,
        )
        generated = tts.generate(text_ids, max_new_tokens=args.max_new_tokens)[0]
        eos = generated.eq(tts.config.audio_eos_id).nonzero()
        if len(eos):
            generated = generated[: int(eos[0])]
        generated = generated[generated.lt(tts.config.codebook_size)]
        if not len(generated):
            raise RuntimeError(f"Model generated no codec tokens for {row['id']}")
        waveform = codec.decode_vq_indices(generated)[0, 0].cpu().numpy()
        wav_path = wav_dir / f"{row['id']}.wav"
        sf.write(wav_path, waveform, codec.cfg.sample_rate, subtype="PCM_16")
        rows.append({
            **row,
            "edge_tts_path": row["path"],
            "path": str(wav_path.relative_to(root)),
            "n_generated_codec_tokens": len(generated),
            "terminated_with_eos": bool(len(eos)),
        })
    generated_manifest = output / "generated_manifest.parquet"
    pd.DataFrame(rows).to_parquet(generated_manifest, index=False)
    report = {
        "schema_version": 1,
        "tts_checkpoint": args.tts_checkpoint,
        "tts_sha256": sha256_file(root / args.tts_checkpoint),
        "codec_checkpoint": args.codec_checkpoint,
        "codec_sha256": sha256_file(root / args.codec_checkpoint),
        "source_manifest": args.manifest,
        "source_manifest_sha256": sha256_file(root / args.manifest),
        "n_samples": len(rows),
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "eos_rate": sum(row["terminated_with_eos"] for row in rows) / len(rows),
    }
    (output / "generation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
