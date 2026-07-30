#!/usr/bin/env python3
"""Evaluate base or LoRA Whisper on long-form manifest audio without truncation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.evaluation.wer import (
    DomainWERAnalyzer,
    bootstrap_wer_ci,
    load_domain_vocab,
)
from whisper_adapt.models.whisper_lora import load_finetuned
from whisper_adapt.reproducibility import collect_provenance, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--domain-vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="openai/whisper-small")
    parser.add_argument("--adapter-path")
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--chunk-length-seconds", type=float, default=30.0)
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if device >= 0 else torch.float32

    if args.adapter_path:
        model = load_finetuned(args.base_model, args.adapter_path)
        processor = WhisperProcessor.from_pretrained(args.adapter_path)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
        processor = WhisperProcessor.from_pretrained(args.base_model)
    model.eval()
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=args.chunk_length_seconds,
        stride_length_s=args.stride_seconds,
    )
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_parquet(args.eval_manifest)
    audio_root = Path(args.audio_root) if args.audio_root else root
    paths = [
        path if (path := Path(value)).is_absolute() else audio_root / path
        for value in frame.path
    ]
    audio_inputs = []
    for path in paths:
        samples, _ = librosa.load(path, sr=processor.feature_extractor.sampling_rate, mono=True)
        audio_inputs.append({
            "raw": samples,
            "sampling_rate": processor.feature_extractor.sampling_rate,
        })
    outputs = recognizer(
        audio_inputs,
        batch_size=1,
        generate_kwargs={"language": "en", "task": "transcribe"},
    )
    hypotheses = [item["text"] for item in outputs]
    references = frame.sentence.astype(str).tolist()
    analyzer = DomainWERAnalyzer(load_domain_vocab(args.domain_vocab))
    report = analyzer.analyze(references, hypotheses)
    mask = [analyzer._contains_domain_term(reference) for reference in references]
    slices = {
        "overall": (references, hypotheses),
        "domain_terms": (
            [r for r, keep in zip(references, mask) if keep],
            [h for h, keep in zip(hypotheses, mask) if keep],
        ),
        "common_terms": (
            [r for r, keep in zip(references, mask) if not keep],
            [h for h, keep in zip(hypotheses, mask) if not keep],
        ),
    }
    result = {
        "schema_version": 1,
        "model": args.base_model,
        "adapter_path": args.adapter_path,
        "eval_manifest": args.eval_manifest,
        "n_samples": len(frame),
        "n_trials": 1,
        "long_form": {
            "chunk_length_seconds": args.chunk_length_seconds,
            "stride_seconds": args.stride_seconds,
            "no_30_second_truncation": True,
        },
        "wer": {
            "overall": report.wer_overall,
            "domain_terms": report.wer_domain_terms,
            "common_terms": report.wer_common_terms,
            "n_domain_utterances": report.n_domain,
            "n_common_utterances": report.n_common,
        },
        "uncertainty": {
            name: bootstrap_wer_ci(
                refs, hyps, n_resamples=args.bootstrap_resamples, seed=args.seed
            )
            for name, (refs, hyps) in slices.items()
        },
        "predictions": [
            {"id": str(row.id), "reference": ref, "hypothesis": hyp}
            for row, ref, hyp in zip(frame.itertuples(), references, hypotheses)
        ],
        "provenance": collect_provenance(
            repo_root=root,
            arguments=vars(args),
            input_files=[args.eval_manifest, args.domain_vocab],
            seed=args.seed,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("n_samples", "wer", "uncertainty")}, indent=2))


if __name__ == "__main__":
    main()
