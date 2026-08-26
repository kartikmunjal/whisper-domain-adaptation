#!/usr/bin/env python3
"""Benchmark parallel text-to-codec generation and waveform decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.latency_benchmark import benchmark_callable, hardware_metadata
from whisper_adapt.models.audio_codec import AudioVQVAE
from whisper_adapt.models.codec_tts import CodecTokenTTS, encode_conditioning_text
from whisper_adapt.reproducibility import collect_provenance, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-checkpoint", required=True)
    parser.add_argument("--codec-checkpoint", required=True)
    parser.add_argument("--eval-manifest", default="data/financial_research/test_manifest.parquet")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--chunk-ms", type=int, nargs="+", default=[200,500,1000])
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    device = torch.device(args.device)
    tts_path = (root / args.tts_checkpoint).resolve()
    codec_path = (root / args.codec_checkpoint).resolve()
    tts = CodecTokenTTS.from_checkpoint(tts_path, map_location=device).to(device).eval()
    if tts.config.decoder_input_mode != "text_only":
        raise ValueError("locked benchmark requires the parallel text_only decoder")
    codec = AudioVQVAE.from_checkpoint(codec_path, map_location=device).to(device).eval()
    row = pd.read_parquet(root / args.eval_manifest).iloc[args.row_index]
    text = str(row["sentence"])
    text_ids = encode_conditioning_text(text, tts.config)
    text_tensor = torch.tensor([text_ids], device=device, dtype=torch.long)
    conditions=[]
    with torch.inference_mode():
        for chunk_ms in args.chunk_ms:
            duration=chunk_ms/1000; output_tokens=max(1,round(duration*codec.cfg.frame_rate_hz))
            def generate(): return tts.generate(text_tensor,max_new_tokens=output_tokens)
            def valid_tokens():
                tokens=generate()[0]; tokens=tokens[(tokens>=0)&(tokens<codec.cfg.codebook_size)]
                return tokens if len(tokens) else torch.zeros(1,dtype=torch.long,device=device)
            def end_to_end(): return codec.decode_vq_indices(valid_tokens(),target_length=round(duration*codec.cfg.sample_rate))
            sample_tokens=valid_tokens()
            generation=benchmark_callable(generate,device=device,audio_duration_seconds=duration,warmup_iterations=args.warmups,timed_iterations=args.iterations)
            combined=benchmark_callable(end_to_end,device=device,audio_duration_seconds=duration,warmup_iterations=args.warmups,timed_iterations=args.iterations)
            conditions.append({"chunk_ms":chunk_ms,"output_token_budget":output_tokens,"emitted_valid_tokens":int(len(sample_tokens)),"generation":generation.to_dict(),"generation_and_decode":combined.to_dict()})
    report = {
        "schema_version": 1,
        "benchmark_scope": "parallel batch generation; not production streaming",
        "hardware": hardware_metadata(device),
        "text": text,
        "row_id": str(row["id"]),
        "conditions": conditions,
        "tts_checkpoint_sha256": sha256_file(tts_path),
        "codec_checkpoint_sha256": sha256_file(codec_path),
        "provenance": collect_provenance(repo_root=root, arguments=vars(args), input_files=[tts_path, codec_path], seed=None),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
