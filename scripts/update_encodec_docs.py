#!/usr/bin/env python3
"""Regenerate EnCodec result blocks from the committed machine summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


README_START = "<!-- BEGIN GENERATED ENCODEC RESULT -->"
README_END = "<!-- END GENERATED ENCODEC RESULT -->"
PLAN_START = "<!-- BEGIN GENERATED ENCODEC FINAL RESULT -->"
PLAN_END = "<!-- END GENERATED ENCODEC FINAL RESULT -->"


def pct_ci(values: list[float]) -> str:
    return (
        f"{values[0] * 100:.2f}% "
        f"(95% CI {values[1] * 100:.2f}–{values[2] * 100:.2f})"
    )


def pp_ci(values: list[float]) -> str:
    return (
        f"{values[0] * 100:+.2f} points "
        f"(95% CI {values[1] * 100:.2f}–{values[2] * 100:.2f})"
    )


def ratio_ci(values: list[float]) -> str:
    return f"{values[0]:.2f}× (95% CI {values[1]:.2f}–{values[2]:.2f})"


def replace_block(text: str, start: str, end: str, body: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"expected one generated block delimited by {start!r}")
    prefix, remainder = text.split(start, 1)
    _, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n{body.rstrip()}\n{end}{suffix}"


def render(summary: dict) -> tuple[str, str]:
    encodec = summary["encodec"]
    vq = summary["custom_cells"]["vq_500bps"]
    fsq = summary["custom_cells"]["fsq_500bps"]
    vq_cmp = summary["comparisons"]["vq_500bps"]
    fsq_cmp = summary["comparisons"]["fsq_500bps"]
    absolute = encodec["wer"]["overall"]["adapted_whisper"][
        "absolute_wer_mean_ci95"
    ]
    delta = encodec["wer"]["overall"]["adapted_whisper"]["delta_wer_mean_ci95"]
    vq_ratio = vq_cmp["overall"]["custom_over_encodec_absolute_wer_ratio_mean_ci95"]
    fsq_ratio = fsq_cmp["overall"]["custom_over_encodec_absolute_wer_ratio_mean_ci95"]
    nominal_ratio = vq_cmp["rate_context"]["encodec_to_custom_nominal_bitrate_ratio"]
    vq_rate_ratio = vq_cmp["rate_context"]["encodec_to_custom_empirical_bitrate_ratio"]
    fsq_rate_ratio = fsq_cmp["rate_context"]["encodec_to_custom_empirical_bitrate_ratio"]

    readme = f"""The preregistered [external EnCodec benchmark](experiments/results/codec_medical_encodec/REPORT.md)
anchors those results to the pinned open `facebook/encodec_24khz` checkpoint.
At its lowest supported rate, {encodec['nominal_bitrate_bps'] / 1000:.1f} kbps, EnCodec's
five-adapter mean absolute WER is {pct_ci(absolute)}, and its reconstructed-minus-
original ΔWER is {pp_ci(delta)}. At 500 nominal bps,
VQ-VAE and FSQ have {ratio_ci(vq_ratio)} and {ratio_ci(fsq_ratio)} EnCodec's
absolute WER, respectively. Signal fidelity and utilization show the same gap:
EnCodec reaches {encodec['si_sdr_db']['mean']:.2f} dB SI-SDR and {encodec['utilization']['entropy_utilization'] * 100:.1f}%
entropy utilization, versus {vq['si_sdr_db']['mean']:.2f} dB/{vq['entropy_utilization']['mean_ci95'][0] * 100:.1f}% for
VQ-VAE and {fsq['si_sdr_db']['mean']:.2f} dB/{fsq['entropy_utilization']['mean_ci95'][0] * 100:.1f}% for FSQ. This is an
external anchor, not a matched-rate ranking: EnCodec receives {nominal_ratio:.0f}× the nominal
bitrate and {vq_rate_ratio:.2f}×/{fsq_rate_ratio:.2f}× the measured empirical entropy rate of
the VQ-VAE/FSQ cells. The elevated but improved custom-codec utilization is
therefore a plausible mechanism for part of the gap, not proof that utilization
alone causes it."""

    plan = f"""## Amendment 4 final result

The pinned EnCodec evaluation completed on all {summary['protocol']['n_clips']} clips, the
frozen Whisper baseline, and all {len(summary['protocol']['adapter_seeds'])} fixed medical adapters with clean
provenance and {summary['protocol']['bootstrap_resamples']:,}-resample uncertainty. EnCodec's mean
adapted absolute WER is {pct_ci(absolute)}, and reconstructed-minus-original
ΔWER is {pp_ci(delta)}. The corresponding
absolute-WER ratios are {ratio_ci(vq_ratio)} for corrective VQ-500 and
{ratio_ci(fsq_ratio)} for corrective FSQ-500.

EnCodec records {encodec['si_sdr_db']['mean']:.2f} dB mean SI-SDR, {encodec['log_mel_l1_db']['mean']:.2f} dB mean log-mel
distance, and {encodec['utilization']['entropy_utilization'] * 100:.1f}% pooled entropy utilization. VQ-500 records
{vq['si_sdr_db']['mean']:.2f} dB, {vq['log_mel_l1_db']['mean']:.2f} dB, and {vq['entropy_utilization']['mean_ci95'][0] * 100:.1f}%; FSQ-500 records
{fsq['si_sdr_db']['mean']:.2f} dB, {fsq['log_mel_l1_db']['mean']:.2f} dB, and {fsq['entropy_utilization']['mean_ci95'][0] * 100:.1f}%. This supports the
mechanical diagnosis that residual custom-codec under-utilization accompanies
materially worse reconstruction and transcription. It does not isolate
utilization as the sole cause. EnCodec receives {nominal_ratio:.0f}× the nominal bitrate and
{vq_rate_ratio:.2f}×/{fsq_rate_ratio:.2f}× the empirical entropy rate of VQ-500/FSQ-500, so no
matched-rate superiority claim is made.
The complete machine-generated tables, per-split intervals, hashes, and plot
are in `experiments/results/codec_medical_encodec/`."""
    return readme, plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", default="experiments/results/codec_medical_encodec/summary.json"
    )
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--research-plan", default="CODEC_RESEARCH_PLAN.md")
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
        if current != generated:
            stale.append(str(path))
            if not args.check:
                path.write_text(generated, encoding="utf-8", newline="\n")
    if args.check and stale:
        raise SystemExit("stale generated EnCodec docs: " + ", ".join(stale))


if __name__ == "__main__":
    main()
