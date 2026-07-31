#!/usr/bin/env python3
"""Test whether codec TTS can free-run on a deterministic tiny training set."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.models.codec_tts import CodecTTSConfig, CodecTokenTTS
from whisper_adapt.reproducibility import collect_provenance


def edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for i, x in enumerate(left, 1):
        current = [i]
        for j, y in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def batch(rows: list[dict], cfg: CodecTTSConfig):
    max_text = max(len(row["text_token_ids"]) for row in rows)
    max_audio = max(len(row["codec_token_ids"]) for row in rows) + 1
    text = torch.zeros(len(rows), max_text, dtype=torch.long)
    decoder = torch.full((len(rows), max_audio), cfg.audio_eos_id, dtype=torch.long)
    labels = torch.full((len(rows), max_audio), -100, dtype=torch.long)
    for index, row in enumerate(rows):
        text_ids = torch.tensor(list(row["text_token_ids"]), dtype=torch.long)
        audio_ids = torch.tensor(list(row["codec_token_ids"]), dtype=torch.long)
        text[index, : len(text_ids)] = text_ids
        decoder[index, 0] = cfg.audio_bos_id
        decoder[index, 1 : len(audio_ids) + 1] = audio_ids
        labels[index, : len(audio_ids)] = audio_ids
        labels[index, len(audio_ids)] = cfg.audio_eos_id
    return text, decoder, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/codec_tts_tokens")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-examples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = pd.read_parquet(root / args.data_dir / "train.parquet").to_dict("records")
    rows = sorted(rows, key=lambda row: (len(row["codec_token_ids"]), str(row["id"])))
    rows = rows[: args.n_examples]
    report = json.loads((root / args.data_dir / "dataset_report.json").read_text())
    cfg = CodecTTSConfig(codebook_size=report["codebook_size"])
    device = torch.device(args.device)
    text, decoder, labels = [x.to(device) for x in batch(rows, cfg)]
    model = CodecTokenTTS(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        logits = model(text, decoder)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            history.append({"step": step, "teacher_forced_nll": float(loss.detach())})
            print(history[-1])

    model.eval()
    maximum = max(len(row["codec_token_ids"]) for row in rows) + 100
    generated = model.generate(text, max_new_tokens=maximum)
    diagnostics = []
    for row, sequence in zip(rows, generated):
        eos = sequence.eq(cfg.audio_eos_id).nonzero()
        terminated = bool(len(eos))
        if terminated:
            sequence = sequence[: int(eos[0])]
        prediction = sequence[sequence.lt(cfg.codebook_size)].cpu().tolist()
        reference = list(row["codec_token_ids"])
        distance = edit_distance(reference, prediction)
        diagnostics.append({
            "id": str(row["id"]),
            "reference_length": len(reference),
            "generated_length": len(prediction),
            "terminated_with_eos": terminated,
            "token_error_rate": distance / max(len(reference), 1),
            "exact_match": prediction == reference,
        })
    result = {
        "schema_version": 1,
        "diagnostic": "free-running tiny-set memorization; not a generalization result",
        "n_examples": len(rows),
        "steps": args.steps,
        "mean_token_error_rate": float(np.mean([x["token_error_rate"] for x in diagnostics])),
        "exact_match_rate": float(np.mean([x["exact_match"] for x in diagnostics])),
        "eos_rate": float(np.mean([x["terminated_with_eos"] for x in diagnostics])),
        "history": history,
        "examples": diagnostics,
        "provenance": collect_provenance(
            repo_root=root,
            arguments=vars(args),
            input_files=[root / args.data_dir / "train.parquet"],
            seed=args.seed,
        ),
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "n_examples", "mean_token_error_rate", "exact_match_rate", "eos_rate"
    )}, indent=2))


if __name__ == "__main__":
    main()
