#!/usr/bin/env python3
"""Synthesize the frozen scale-study sentences with a fixed ElevenLabs voice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_adapt.reproducibility import collect_provenance, sha256_file


ENDPOINT = "https://api.elevenlabs.io/v1/text-to-speech"
MODEL_ID = "eleven_multilingual_v2"
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
OUTPUT_FORMAT = "mp3_44100_128"
VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True,
}
TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}


def synthesize(text: str, api_key: str, timeout: float, max_attempts: int):
    url = f"{ENDPOINT}/{VOICE_ID}?output_format={OUTPUT_FORMAT}"
    body = json.dumps(
        {"text": text, "model_id": MODEL_ID, "voice_settings": VOICE_SETTINGS}
    ).encode("utf-8")
    for attempt in range(max_attempts):
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(), response.headers.get("request-id")
        except urllib.error.HTTPError as exc:
            if exc.code not in TRANSIENT_STATUS or attempt + 1 == max_attempts:
                detail = exc.read(500).decode("utf-8", errors="replace")
                raise RuntimeError(f"ElevenLabs HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError:
            if attempt + 1 == max_attempts:
                raise
        delay = min(30.0, 2.0**attempt) + random.Random(attempt).random()
        time.sleep(delay)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="data/financial_research/test_manifest.parquet"
    )
    parser.add_argument(
        "--output-dir", default="experiments/results/elevenlabs_multilingual_v2"
    )
    parser.add_argument("--api-key-file")
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()

    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if args.api_key_file:
        if api_key:
            raise ValueError("provide either ELEVENLABS_API_KEY or --api-key-file")
        api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("set ELEVENLABS_API_KEY or pass --api-key-file")

    root = Path(__file__).resolve().parents[1]
    manifest_path = root / args.manifest
    output = root / args.output_dir
    wav_dir = output / "audio"
    wav_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_parquet(manifest_path).to_dict("records")
    rows, files = [], []
    for index, row in enumerate(records):
        audio_path = wav_dir / f"{index:04d}_{row['id']}.mp3"
        metadata_path = audio_path.with_suffix(".json")
        if not audio_path.exists() or not metadata_path.exists():
            audio, request_id = synthesize(
                row["sentence"], api_key, args.timeout_seconds, args.max_attempts
            )
            audio_path.write_bytes(audio)
            metadata_path.write_text(
                json.dumps(
                    {
                        "id": row["id"],
                        "request_id": request_id,
                        "sha256": hashlib.sha256(audio).hexdigest(),
                        "n_bytes": len(audio),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["sha256"] != sha256_file(audio_path):
            raise RuntimeError(f"hash mismatch for restart artifact {audio_path}")
        files.append(metadata)
        rows.append(
            {
                **row,
                "edge_tts_path": row["path"],
                "path": str(audio_path.relative_to(root)),
                "source": f"elevenlabs-{MODEL_ID}-{VOICE_ID}",
            }
        )

    generated_manifest = output / "generated_manifest.parquet"
    pd.DataFrame(rows).to_parquet(generated_manifest, index=False)
    report = {
        "schema_version": 1,
        "n_samples": len(rows),
        "endpoint": ENDPOINT,
        "model_id": MODEL_ID,
        "voice_id": VOICE_ID,
        "output_format": OUTPUT_FORMAT,
        "voice_settings": VOICE_SETTINGS,
        "input_manifest_sha256": sha256_file(manifest_path),
        "generated_manifest_sha256": sha256_file(generated_manifest),
        "files": files,
        "api_key_recorded": False,
        "provenance": collect_provenance(
            repo_root=root,
            arguments={
                "manifest": args.manifest,
                "output_dir": args.output_dir,
                "timeout_seconds": args.timeout_seconds,
                "max_attempts": args.max_attempts,
            },
            input_files=[manifest_path],
            seed=20260803,
        ),
    }
    (output / "generation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
