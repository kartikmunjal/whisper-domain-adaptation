#!/usr/bin/env python3
"""Regenerate ElevenLabs result blocks from scale-study aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

README_START = "<!-- BEGIN GENERATED ELEVENLABS RESULT -->"
README_END = "<!-- END GENERATED ELEVENLABS RESULT -->"
PLAN_START = "<!-- BEGIN GENERATED ELEVENLABS FINAL RESULT -->"
PLAN_END = "<!-- END GENERATED ELEVENLABS FINAL RESULT -->"
METRICS = ("overall", "domain_terms", "common_terms")


def replace_block(text: str, start: str, end: str, body: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"expected one generated block delimited by {start!r}")
    prefix, remainder = text.split(start, 1)
    _, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n{body.rstrip()}\n{end}{suffix}"


def wer_ci(values: list[float]) -> str:
    return (
        f"{values[0] * 100:.3f}% "
        f"(95% seed-bootstrap CI {values[1] * 100:.3f}–{values[2] * 100:.3f})"
    )


def render(summary: dict) -> tuple[str, str]:
    name = "elevenlabs_multilingual_v2"
    metrics = {
        metric: summary["metrics"][metric][name]["mean_ci95"]
        for metric in METRICS
    }
    provenance = summary["elevenlabs_generation_provenance"]
    sentence_count = provenance["n_samples"]
    model = provenance["model_id"]
    voice = provenance["voice_id"]
    seeds = summary["n_trials"]
    resamples = summary["bootstrap"]["n_resamples"]

    readme = f"""The fixed ElevenLabs `{model}` comparator on the same {sentence_count}
held-out sentences reaches {wer_ci(metrics['overall'])} overall,
{wer_ci(metrics['domain_terms'])} on domain sentences, and
{wer_ci(metrics['common_terms'])} on common controls across the same {seeds}
frozen adapters. It is included beside Piper and Edge-TTS in the
[scale-study table](experiments/results/codec_tts_scale_study/REPORT.md).
This round-trip content metric does not measure naturalness or preference, and
the TTS-on-TTS evaluation remains optimistic."""

    plan = f"""## Amendment 6 final result

The locked ElevenLabs `{model}` / `{voice}` run completed all
{sentence_count} held-out sentences, and all {seeds} frozen financial adapters
completed with clean provenance. With {resamples:,} seed-level bootstrap
resamples, overall WER is {wer_ci(metrics['overall'])}, domain WER is
{wer_ci(metrics['domain_terms'])}, and common-control WER is
{wer_ci(metrics['common_terms'])}. The comparator is more accurately
transcribed by these adapters than Piper or Edge-TTS on this narrow synthetic
content protocol. This is not evidence of perceptual preference or general TTS
superiority: voice, model family, latency, cost, naturalness, and real speech
were not evaluated, and the frozen adapters were trained on synthetic
financial speech. The complete table and machine-readable aggregation are in
`experiments/results/codec_tts_scale_study/`; the committed primary API report
and five ASR reports are in
`experiments/results/elevenlabs_multilingual_v2/`. Generated audio and the API
key are not retained in Git."""
    return readme, plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", default="experiments/results/codec_tts_scale_study/summary.json"
    )
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--research-plan", default="TTS_RESEARCH_PLAN.md")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    readme_body, plan_body = render(summary)
    targets = (
        (Path(args.readme), README_START, README_END, readme_body),
        (Path(args.research_plan), PLAN_START, PLAN_END, plan_body),
    )
    stale = []
    for path, start, end, body in targets:
        current = path.read_text(encoding="utf-8")
        generated = replace_block(current, start, end, body)
        if generated != current:
            stale.append(str(path))
            if not args.check:
                path.write_text(generated, encoding="utf-8", newline="\n")
    if args.check and stale:
        raise SystemExit("stale generated ElevenLabs docs: " + ", ".join(stale))


if __name__ == "__main__":
    main()
