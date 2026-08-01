#!/usr/bin/env python3
"""Train the preregistered text-to-VQ-token Transformer."""

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
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.models.codec_tts import CodecTTSConfig, CodecTokenTTS
from whisper_adapt.reproducibility import collect_provenance, sha256_file


class TokenDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = pd.read_parquet(path).to_dict("records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(rows, cfg: CodecTTSConfig):
    max_text = min(max(len(row["text_token_ids"]) for row in rows), cfg.max_text_tokens)
    max_audio = min(
        max(len(row["codec_token_ids"]) for row in rows) + 1,
        cfg.max_audio_tokens,
    )
    text = torch.zeros(len(rows), max_text, dtype=torch.long)
    decoder = torch.full(
        (len(rows), max_audio), cfg.audio_eos_id, dtype=torch.long
    )
    labels = torch.full((len(rows), max_audio), -100, dtype=torch.long)
    for index, row in enumerate(rows):
        text_ids = list(row["text_token_ids"])[:max_text]
        audio_ids = list(row["codec_token_ids"])[: max_audio - 1]
        text[index, :len(text_ids)] = torch.tensor(text_ids)
        decoder[index, 0] = cfg.audio_bos_id
        if audio_ids:
            decoder[index, 1:len(audio_ids) + 1] = torch.tensor(audio_ids)
            labels[index, :len(audio_ids)] = torch.tensor(audio_ids)
        labels[index, len(audio_ids)] = cfg.audio_eos_id
    return text, decoder, labels


def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    token_count = 0
    correct = 0
    duration_absolute_error = 0.0
    duration_count = 0
    with torch.inference_mode():
        for text, decoder, labels in loader:
            text, decoder, labels = text.to(device), decoder.to(device), labels.to(device)
            logits = model(text, decoder)
            loss_sum += float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ))
            mask = labels.ne(-100)
            token_count += int(mask.sum())
            correct += int((logits.argmax(-1).eq(labels) & mask).sum())
            target_lengths = mask.sum(dim=1).sub(1).clamp_min(0)
            predicted_lengths = torch.expm1(model.predict_log_lengths(text)).clamp_min(0)
            duration_absolute_error += float(
                (predicted_lengths - target_lengths).abs().sum().cpu()
            )
            duration_count += len(text)
    return (
        loss_sum / token_count,
        correct / token_count,
        duration_absolute_error / duration_count,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/codec_tts_tokens")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--duration-loss-weight", type=float, default=0.1)
    parser.add_argument("--scheduled-sampling-max", type=float, default=0.25)
    parser.add_argument("--scheduled-sampling-warmup-fraction", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decoder-input-mode", choices=("autoregressive", "text_only"), default="autoregressive")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root = Path(__file__).resolve().parents[1]
    data = root / args.data_dir
    report = json.loads((data / "dataset_report.json").read_text())
    cfg = CodecTTSConfig(codebook_size=report["codebook_size"], decoder_input_mode=args.decoder_input_mode)
    train = TokenDataset(data / "train.parquet")
    validation = TokenDataset(data / "validation.parquet")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=generator, collate_fn=lambda rows: collate(rows, cfg)
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda rows: collate(rows, cfg)
    )
    device = torch.device(args.device)
    model = CodecTokenTTS(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        duration_total = 0.0
        scheduled_sampling_probability = args.scheduled_sampling_max * min(
            1.0,
            epoch / max(args.epochs * args.scheduled_sampling_warmup_fraction, 1.0),
        )
        if cfg.decoder_input_mode == "text_only":
            scheduled_sampling_probability = 0.0
        for text, decoder, labels in train_loader:
            text, decoder, labels = text.to(device), decoder.to(device), labels.to(device)
            teacher_logits = model(text, decoder)
            sampled_decoder = decoder
            if scheduled_sampling_probability > 0 and decoder.shape[1] > 1:
                sampled_decoder = decoder.clone()
                predicted_previous = teacher_logits[:, :-1].detach().argmax(dim=-1)
                valid_previous = labels[:, :-1].ne(-100)
                replace = (
                    torch.rand(valid_previous.shape, device=device)
                    < scheduled_sampling_probability
                ) & valid_previous
                sampled_decoder[:, 1:] = torch.where(
                    replace, predicted_previous, sampled_decoder[:, 1:]
                )
            logits = model(text, sampled_decoder)
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
            target_lengths = labels.ne(-100).sum(dim=1).sub(1).clamp_min(0)
            duration_loss = F.smooth_l1_loss(
                model.predict_log_lengths(text),
                torch.log1p(target_lengths.float()),
            )
            loss = token_loss + args.duration_loss_weight * duration_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens = int(labels.ne(-100).sum())
            total += float(token_loss.detach()) * tokens
            duration_total += float(duration_loss.detach()) * len(text)
            count += tokens
        val_loss, val_accuracy, val_duration_mae = evaluate(
            model, validation_loader, device
        )
        record = {
            "epoch": epoch,
            "train_nll": total / count,
            "validation_nll": val_loss,
            "validation_token_accuracy": val_accuracy,
            "train_duration_loss": duration_total / len(train),
            "validation_duration_mae_tokens": val_duration_mae,
            "scheduled_sampling_probability": scheduled_sampling_probability,
        }
        history.append(record)
        print(record)
        if val_loss < best_loss:
            best_loss = val_loss
            model.save_checkpoint(
                output / "model.pt",
                seed=args.seed,
                epoch=epoch,
                validation_nll=val_loss,
                codec_sha256=report["codec_sha256"],
            )
    (output / "run.json").write_text(json.dumps({
        "schema_version": 1,
        "seed": args.seed,
        "n_trials_planned": 5,
        "best_validation_nll": best_loss,
        "n_train_clips": len(train),
        "n_validation_clips": len(validation),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "optimizer_steps_per_epoch": len(train_loader),
        "planned_optimizer_steps": len(train_loader) * args.epochs,
        "duration_loss_weight": args.duration_loss_weight,
        "scheduled_sampling_max": args.scheduled_sampling_max,
        "scheduled_sampling_warmup_fraction": args.scheduled_sampling_warmup_fraction,
        "history": history,
        "train_manifest_sha256": sha256_file(data / "train.parquet"),
        "validation_manifest_sha256": sha256_file(data / "validation.parquet"),
        "config": cfg.__dict__,
        "model_sha256": sha256_file(output / "model.pt"),
        "provenance": collect_provenance(
            repo_root=root,
            arguments=vars(args),
            input_files=[
                data / "dataset_report.json",
                data / "train.parquet",
                data / "validation.parquet",
            ],
            seed=args.seed,
        ),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
