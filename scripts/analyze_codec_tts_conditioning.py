#!/usr/bin/env python3
"""Diagnose text use, free-running drift, and cross-attention in codec TTS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from whisper_adapt.models.codec_tts import CodecTokenTTS
from whisper_adapt.reproducibility import collect_provenance, sha256_file


def edit_alignment(reference: list[int], hypothesis: list[int]) -> list[tuple[str, int]]:
    """Return edit events with a stable reference-position index."""
    n, m = len(reference), len(hypothesis)
    cost = np.zeros((n + 1, m + 1), dtype=np.int32)
    back = np.empty((n + 1, m + 1), dtype=object)
    for i in range(1, n + 1): cost[i, 0], back[i, 0] = i, "delete"
    for j in range(1, m + 1): cost[0, j], back[0, j] = j, "insert"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [(cost[i-1, j] + 1, "delete"), (cost[i, j-1] + 1, "insert"),
                       (cost[i-1, j-1] + (reference[i-1] != hypothesis[j-1]), "equal" if reference[i-1] == hypothesis[j-1] else "substitute")]
            cost[i, j], back[i, j] = min(options, key=lambda x: (x[0], {"equal":0,"substitute":1,"delete":2,"insert":3}[x[1]]))
    events, i, j = [], n, m
    while i or j:
        op = back[i, j]
        events.append((op, max(i - 1, 0)))
        if op in ("equal", "substitute"): i, j = i - 1, j - 1
        elif op == "delete": i -= 1
        else: j -= 1
    return events[::-1]


def normalized_position_errors(reference: list[int], hypothesis: list[int], bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    errors = np.zeros(bins, dtype=np.int64); totals = np.zeros(bins, dtype=np.int64)
    length = max(len(reference), 1)
    for op, position in edit_alignment(reference, hypothesis):
        index = min(int(position / length * bins), bins - 1)
        if op != "insert": totals[index] += 1
        else: totals[index] += 1
        if op != "equal": errors[index] += 1
    return errors, totals


def token_edit_rate(left: list[int], right: list[int]) -> float:
    return sum(op != "equal" for op, _ in edit_alignment(left, right)) / max(len(left), len(right), 1)


def collate(rows: list[dict], model: CodecTokenTTS, device: torch.device):
    cfg = model.config
    max_text = min(max(len(x["text_token_ids"]) for x in rows), cfg.max_text_tokens)
    max_audio = min(max(len(x["codec_token_ids"]) for x in rows) + 1, cfg.max_audio_tokens)
    text = torch.zeros(len(rows), max_text, dtype=torch.long, device=device)
    decoder = torch.full((len(rows), max_audio), cfg.audio_eos_id, dtype=torch.long, device=device)
    labels = torch.full((len(rows), max_audio), -100, dtype=torch.long, device=device)
    references = []
    for i, row in enumerate(rows):
        text_ids = list(row["text_token_ids"])[:max_text]
        audio_ids = list(row["codec_token_ids"])[:max_audio-1]
        references.append(audio_ids)
        text[i, :len(text_ids)] = torch.tensor(text_ids, device=device)
        decoder[i, 0] = cfg.audio_bos_id
        if audio_ids:
            decoder[i, 1:len(audio_ids)+1] = torch.tensor(audio_ids, device=device)
            labels[i, :len(audio_ids)] = torch.tensor(audio_ids, device=device)
        labels[i, len(audio_ids)] = cfg.audio_eos_id
    return text, decoder, labels, references


def capture_cross_attention(model: CodecTokenTTS, text: torch.Tensor, decoder: torch.Tensor) -> list[np.ndarray]:
    captured, originals = [], []
    for layer in model.decoder.layers:
        module = layer.multihead_attn; original = module.forward; originals.append((module, original))
        def wrapped(*args, _original=original, **kwargs):
            kwargs["need_weights"] = True; kwargs["average_attn_weights"] = False
            output, weights = _original(*args, **kwargs)
            captured.append(weights.detach().cpu().numpy())
            return output, weights
        module.forward = wrapped
    try:
        with torch.inference_mode(): model(text, decoder)
    finally:
        for module, original in originals: module.forward = original
    return captured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokens", default="data/codec_tts_tokens/test.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--attention-examples", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]; device = torch.device(args.device)
    rows = pd.read_parquet(root / args.tokens).sort_values("id", kind="stable").to_dict("records")
    model = CodecTokenTTS.from_checkpoint(root / args.checkpoint, device).to(device).eval()
    rng = np.random.default_rng(20260731); permutation = rng.permutation(len(rows))
    if np.any(permutation == np.arange(len(rows))): permutation = np.roll(np.arange(len(rows)), 1)
    nll_true_sum = nll_shuffled_sum = 0.0; tokens = 0; sensitivities = []; errors = np.zeros(10, dtype=np.int64); totals = np.zeros(10, dtype=np.int64); example_rows = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start+args.batch_size]; shuffled_rows = [dict(row, text_token_ids=rows[permutation[start+i]]["text_token_ids"]) for i, row in enumerate(batch)]
        text, decoder, labels, references = collate(batch, model, device); shuffled_text, _, _, _ = collate(shuffled_rows, model, device)
        with torch.inference_mode():
            true_logits = model(text, decoder); shuffled_logits = model(shuffled_text, decoder)
            nll_true_sum += float(F.cross_entropy(true_logits.flatten(0,1), labels.flatten(), ignore_index=-100, reduction="sum"))
            nll_shuffled_sum += float(F.cross_entropy(shuffled_logits.flatten(0,1), labels.flatten(), ignore_index=-100, reduction="sum")); tokens += int(labels.ne(-100).sum())
            maximum = min(max(map(len, references)) + 100, model.config.max_audio_tokens)
            true_generated = model.generate(text, maximum, use_duration_control=True, repetition_penalty=0.0)
            shuffled_generated = model.generate(shuffled_text, maximum, use_duration_control=True, repetition_penalty=0.0)
        for row, reference, true_seq, shuffled_seq in zip(batch, references, true_generated, shuffled_generated):
            def clean(seq):
                eos = seq.eq(model.config.audio_eos_id).nonzero(); seq = seq[:int(eos[0])] if len(eos) else seq
                return seq[seq.lt(model.config.codebook_size)].cpu().tolist()
            true_tokens, shuffled_tokens = clean(true_seq), clean(shuffled_seq)
            sensitivities.append(token_edit_rate(true_tokens, shuffled_tokens)); e, t = normalized_position_errors(reference, true_tokens); errors += e; totals += t
            example_rows.append({"id": str(row["id"]), "reference_length": len(reference), "generated_length": len(true_tokens), "free_running_token_error_rate": token_edit_rate(reference, true_tokens), "true_vs_shuffled_generated_edit_rate": sensitivities[-1]})
    attention_rows = []
    for row in rows[:args.attention_examples]:
        text, decoder, _, _ = collate([row], model, device); layers = capture_cross_attention(model, text, decoder)
        text_length = int(text.ne(0).sum()); target_length = min(len(row["codec_token_ids"]) + 1, decoder.shape[1])
        for layer_index, weights in enumerate(layers):
            matrix = weights[0, :, :target_length, :text_length].mean(axis=0); positions = np.arange(text_length); centroid = (matrix * positions).sum(axis=1) / np.maximum(matrix.sum(axis=1), 1e-12)
            rho = float(np.corrcoef(np.arange(len(centroid)), centroid)[0,1]) if len(centroid) > 1 and np.std(centroid) > 0 else 0.0
            attention_rows.append({"id": str(row["id"]), "layer": layer_index, "centroid_monotonicity_r": rho, "attention_entropy": float(np.mean(-(matrix * np.log(np.maximum(matrix, 1e-12))).sum(axis=1) / np.log(max(text_length, 2)))), "matrix": matrix.tolist()})
    true_nll = nll_true_sum / tokens; shuffled_nll = nll_shuffled_sum / tokens
    result = {"schema_version":1, "seed":args.seed, "n_examples":len(rows), "conditioning":{"true_text_nll":true_nll,"shuffled_text_nll":shuffled_nll,"shuffled_minus_true_nll":shuffled_nll-true_nll,"mean_generated_true_vs_shuffled_edit_rate":float(np.mean(sensitivities)),"nll_gate":0.05,"generated_sensitivity_gate":0.05}, "free_running_position_error":{"n_bins":10,"error_counts":errors.tolist(),"event_counts":totals.tolist(),"rates":np.divide(errors, totals, out=np.zeros(10,dtype=float), where=totals>0).tolist()}, "attention":attention_rows, "examples":example_rows, "checkpoint_sha256":sha256_file(root/args.checkpoint), "provenance":collect_provenance(repo_root=root, arguments=vars(args), input_files=[root/args.checkpoint, root/args.tokens], seed=args.seed)}
    output=root/args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps({"seed":args.seed,"conditioning":result["conditioning"],"position_rates":result["free_running_position_error"]["rates"]},indent=2))


if __name__ == "__main__": main()
