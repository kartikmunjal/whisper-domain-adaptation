#!/usr/bin/env python3
"""Expand only the codec-TTS training split; preserve frozen validation/test."""
from __future__ import annotations
import argparse,asyncio,hashlib,json,sys
from pathlib import Path
import librosa,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from whisper_adapt.data.financial import ALL_VOICES,TARGET_SR,SynthesisConfig,_synthesize_one
from whisper_adapt.data.medical import QualityThresholds,check_quality
from whisper_adapt.evaluation.wer import load_domain_vocab,normalize_text
from whisper_adapt.reproducibility import sha256_file

SCALE_TEMPLATES=(
 "The chief financial officer explained that {term} influenced the quarterly outlook.",
 "Analysts asked management to clarify the reported change in {term}.",
 "During the earnings call, the company provided additional detail about {term}.",
 "The prepared remarks compared this quarter's {term} with the prior-year period.",
 "Management said its updated guidance incorporates recent trends in {term}.",
 "The investor presentation includes a reconciliation and discussion of {term}.",
 "Our financial results reflect both recurring operations and movements in {term}.",
 "The question-and-answer session returned to the outlook for {term}.",
)
SCALE_COMMON=(
 "The operator introduced the speakers before the prepared remarks.",
 "Management thanked employees and customers for their continued support.",
 "The company will publish additional information on its investor website.",
 "Participants were reminded that the discussion included forward-looking statements.",
 "The leadership team reviewed priorities for the coming reporting period.",
 "Analysts joined the call from several research organizations.",
 "The presentation concluded before the question-and-answer session began.",
 "Management discussed execution across its principal operating regions.",
 "The company continues to monitor changes in the broader environment.",
 "The operator provided instructions for submitting a question.",
)

def args():
 p=argparse.ArgumentParser(); p.add_argument("--source-dir",default="data/financial_research"); p.add_argument("--output-dir",default="data/financial_tts_scaled"); p.add_argument("--dry-run",action="store_true"); return p.parse_args()

def main():
 a=args(); root=Path(__file__).resolve().parents[1]; source=root/a.source_dir; out=root/a.output_dir; voices=ALL_VOICES[:4]; terms=sorted(load_domain_vocab(root/"configs/financial_terms.txt")); planned=[]
 for template_id,template in enumerate(SCALE_TEMPLATES):
  for term in terms:
   for voice in voices: planned.append({"sentence":template.format(term=term),"term":term,"is_domain":True,"template_family":f"scale_domain_{template_id:02d}","voice":voice})
 for common_id,sentence in enumerate(SCALE_COMMON):
  for voice in voices: planned.append({"sentence":sentence,"term":None,"is_domain":False,"template_family":f"scale_common_{common_id:02d}","voice":voice})
 original=pd.read_parquet(source/"train_manifest.parquet"); target=len(original)+len(planned)
 if a.dry_run: print(json.dumps({"original_train":len(original),"new_train":len(planned),"target_train":target,"terms":len(terms),"voices":list(voices)},indent=2)); return
 wav=out/"wav"; wav.mkdir(parents=True,exist_ok=True); cfg=SynthesisConfig(); thresholds=QualityThresholds(min_snr_db=cfg.min_snr_db,max_silence_ratio=cfg.max_silence_ratio,min_duration_sec=cfg.min_duration_sec,max_duration_sec=cfg.max_duration_sec); accepted=[]; failures=[]
 frozen_text=set()
 for split in ("validation","test"):
  frame=pd.read_parquet(source/f"{split}_manifest.parquet"); frame.to_parquet(out/f"{split}_manifest.parquet",index=False); frozen_text|={normalize_text(x) for x in frame.sentence}
 for row in planned:
  if normalize_text(row["sentence"]) in frozen_text: raise RuntimeError("Expanded text overlaps frozen validation/test")
  identity=f"{row['sentence']}|{row['voice']}"; sid=hashlib.sha256(identity.encode()).hexdigest()[:16]; path=wav/f"{sid}.wav"
  if not path.exists() and not asyncio.run(_synthesize_one(row["sentence"],row["voice"],path)): failures.append({"id":sid,"reason":"synthesis_failed"}); continue
  audio,_=librosa.load(path,sr=TARGET_SR,mono=True); quality=check_quality(audio,TARGET_SR,thresholds)
  if not quality.passes: failures.append({"id":sid,"reason":quality.fail_reasons}); continue
  accepted.append({**row,"split":"train","id":sid,"path":str(path.relative_to(root)),"duration_sec":quality.duration_sec,"snr_db":quality.snr_db,"silence_ratio":quality.silence_ratio,"source":"edge-tts-scale","audio_sha256":sha256_file(path),"transcript_sha256":hashlib.sha256(normalize_text(row["sentence"]).encode()).hexdigest()})
 if failures: raise RuntimeError(f"{len(failures)} failures; refusing partial corpus: {failures[:5]}")
 expanded=pd.concat([original,pd.DataFrame(accepted)],ignore_index=True); assert len(expanded)==target; expanded.to_parquet(out/"train_manifest.parquet",index=False)
 report={"schema_version":1,"generator":"scripts/prepare_codec_tts_scaled_data.py","source_dir":a.source_dir,"original_train_count":len(original),"new_train_count":len(accepted),"train_count":len(expanded),"validation_count":len(pd.read_parquet(out/"validation_manifest.parquet")),"test_count":len(pd.read_parquet(out/"test_manifest.parquet")),"scale_templates":list(SCALE_TEMPLATES),"scale_common":list(SCALE_COMMON),"training_voices":list(voices),"frozen_validation_sha256":sha256_file(out/"validation_manifest.parquet"),"frozen_test_sha256":sha256_file(out/"test_manifest.parquet")}; (out/"dataset_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__":main()
