#!/usr/bin/env python3
"""Build the preregistered 20-call Earnings-21 real-audio evaluation set.

References come from Rev's official timestamped NLP files. Audio comes from the
Hugging Face mirror. Selection is deterministic and never uses model output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

DATASET_ID = "Revai/earnings21"
REFERENCE_REPO = "revdotcom/speech-datasets"
SEED = 20260729
N_CLIPS = 20


def get_json(url: str) -> object:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def parse_nlp(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        fields = line.split("|")
        if len(fields) < 2 or fields[0].lower() in {"token", "word"}:
            continue
        token = fields[0].strip()
        speaker = fields[1].strip()
        punctuation = fields[4].strip() if len(fields) > 4 else ""
        if token:
            rows.append({"token": token + punctuation, "speaker": speaker})
    if not rows:
        raise ValueError("No timestamped tokens parsed from NLP reference")
    return rows


def group_runs(items: list[dict], speaker_key: str = "speaker") -> list[list[dict]]:
    groups: list[list[dict]] = []
    for item in items:
        if not groups or groups[-1][-1][speaker_key] != item[speaker_key]:
            groups.append([])
        groups[-1].append(item)
    return groups


def parse_rttm(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[0] == "SPEAKER":
            rows.append({
                "start": float(fields[3]),
                "end": float(fields[3]) + float(fields[4]),
                "speaker": fields[7],
            })
    return rows


def candidate_windows(
    tokens: list[dict], rttm: list[dict], domain_terms: set[str]
) -> list[dict]:
    """Select a complete 15--20 s speaker turn with an exact transcript run."""
    token_runs = group_runs(tokens)
    time_runs = group_runs(rttm)
    token_speakers = [run[0]["speaker"] for run in token_runs]
    time_speakers = [run[0]["speaker"] for run in time_runs]
    if token_speakers != time_speakers:
        raise ValueError("NLP and RTTM speaker-run sequences do not match")
    candidates = []
    for words, segments in zip(token_runs, time_runs):
        start, end = segments[0]["start"], segments[-1]["end"]
        sentence = " ".join(word["token"] for word in words)
        normalized = " ".join(re.findall(r"[a-z0-9]+", sentence.lower()))
        matched_terms = sorted(
            term for term in domain_terms
            if re.search(rf"\b{re.escape(term)}\b", normalized)
        )
        if 5.0 <= end - start <= 300.0 and len(words) >= 8:
            candidates.append({
                "start": start,
                "end": end,
                "speaker": words[0]["speaker"],
                "sentence": sentence,
                "matched_domain_terms": matched_terms,
            })
    return sorted(candidates, key=lambda item: item["end"] - item["start"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/earnings21_eval")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--reference-revision", default=None)
    parser.add_argument("--domain-vocab", default="configs/financial_terms.txt")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output_dir
    wav_dir = output / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    domain_vocab_path = root / args.domain_vocab
    domain_terms = {
        " ".join(re.findall(r"[a-z0-9]+", line.lower()))
        for line in domain_vocab_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    api = HfApi()
    hf_sha = api.dataset_info(DATASET_ID, revision=args.hf_revision).sha
    ref_revision = args.reference_revision or get_json(
        f"https://api.github.com/repos/{REFERENCE_REPO}/commits/main"
    )["sha"]
    listing = get_json(
        f"https://api.github.com/repos/{REFERENCE_REPO}/contents/"
        f"earnings21/transcripts/nlp_references?ref={ref_revision}"
    )
    references = sorted(
        (x for x in listing if x["name"].endswith(".nlp")),
        key=lambda item: item["name"],
    )
    random.Random(SEED).shuffle(references)

    eligible_calls = []
    hf_files = set(api.list_repo_files(DATASET_ID, repo_type="dataset", revision=hf_sha))
    for item in references:
        call_id = Path(item["name"]).stem
        audio_candidates = [
            name for name in hf_files
            if Path(name).stem == call_id and Path(name).suffix.lower() in {".wav", ".flac"}
        ]
        if not audio_candidates:
            continue
        ref_text = urllib.request.urlopen(item["download_url"]).read().decode("utf-8")
        rttm_url = (
            f"https://raw.githubusercontent.com/{REFERENCE_REPO}/{ref_revision}/"
            f"earnings21/rttms/{call_id}.rttm"
        )
        try:
            rttm_text = urllib.request.urlopen(rttm_url).read().decode("utf-8")
            windows = candidate_windows(
                parse_nlp(ref_text), parse_rttm(rttm_text), domain_terms
            )
        except (ValueError, urllib.error.HTTPError):
            continue
        if windows:
            eligible_calls.append((call_id, audio_candidates[0], item, windows))
    domain_entries = []
    common_entries = []
    for call_id, audio_name, item, windows in eligible_calls:
        domain = [window for window in windows if window["matched_domain_terms"]]
        common = [window for window in windows if not window["matched_domain_terms"]]
        if domain:
            domain_entries.append((call_id, audio_name, item, domain[0]))
        if common:
            common_entries.append((call_id, audio_name, item, common[0]))
    selected = domain_entries[:10]
    used_calls = {entry[0] for entry in selected}
    common_unused = [entry for entry in common_entries if entry[0] not in used_calls]
    selected.extend(common_unused[:10])
    if len(selected) < N_CLIPS:
        selected.extend(
            entry for entry in common_entries
            if entry[0] in used_calls and entry not in selected
        )
        selected = selected[:N_CLIPS]
    if len(selected) != N_CLIPS:
        raise RuntimeError(f"Only {len(selected)} eligible clips found")

    rows = []
    for call_id, audio_name, item, window in selected:
        source_path = hf_hub_download(
            DATASET_ID, audio_name, repo_type="dataset", revision=hf_sha
        )
        with sf.SoundFile(source_path) as source:
            sample_rate = source.samplerate
            source.seek(round(window["start"] * sample_rate))
            samples = source.read(round((window["end"] - window["start"]) * sample_rate))
        clip_id = hashlib.sha256(
            f"{call_id}|{window['start']:.6f}|{window['end']:.6f}".encode()
        ).hexdigest()[:16]
        clip_path = wav_dir / f"{clip_id}.wav"
        sf.write(clip_path, samples, sample_rate, subtype="PCM_16")
        rows.append({
            "id": clip_id,
            "path": str(clip_path.relative_to(root)),
            "sentence": window["sentence"],
            "call_id": call_id,
            "speaker": window["speaker"],
            "source_start_seconds": window["start"],
            "source_end_seconds": window["end"],
            "source_audio_file": audio_name,
            "source_reference_file": item["path"],
            "hf_revision": hf_sha,
            "reference_revision": ref_revision,
            "manual_verification": "pending",
            "matched_domain_terms": window["matched_domain_terms"],
        })
    frame = pd.DataFrame(rows)
    if frame.call_id.nunique() < 15:
        raise RuntimeError("Distinct-call coverage invariant failed")
    domain_count = int(frame.matched_domain_terms.map(bool).sum())
    if domain_count != 10:
        raise RuntimeError(f"Expected 10 domain and 10 common clips, got {domain_count}")
    frame.to_parquet(output / "eval_manifest.parquet", index=False)
    ledger = frame[[
        "id", "call_id", "path", "sentence", "source_start_seconds",
        "source_end_seconds", "manual_verification"
    ]]
    ledger.to_csv(output / "manual_verification_ledger.csv", index=False)
    report = {
        "schema_version": 1,
        "generator": "scripts/prepare_earnings21_eval.py",
        "seed": SEED,
        "n_clips": len(frame),
        "n_distinct_calls": frame.call_id.nunique(),
        "selection_order": "10 domain and 10 common; distinct calls before second clips",
        "hf_dataset": DATASET_ID,
        "hf_revision": hf_sha,
        "reference_repository": REFERENCE_REPO,
        "reference_revision": ref_revision,
        "license": "CC BY-SA 4.0",
        "selection_uses_model_outputs": False,
        "manual_verification_status": "pending",
        "domain_selection": {
            "vocabulary": args.domain_vocab,
            "vocabulary_sha256": hashlib.sha256(domain_vocab_path.read_bytes()).hexdigest(),
            "domain_clips": domain_count,
            "common_clips": N_CLIPS - domain_count,
            "balanced_split_fixed_before_inference": True,
        },
    }
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
