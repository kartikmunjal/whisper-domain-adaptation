#!/usr/bin/env python3
"""Synthesize the frozen held-out sentences with a pinned open Piper voice."""
from __future__ import annotations
import argparse,json,sys,wave
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from huggingface_hub import hf_hub_download
from piper import PiperVoice
from whisper_adapt.reproducibility import collect_provenance,sha256_file

REPO="rhasspy/piper-voices"; REVISION="5512791644e2148e4be301d4c7fc2a4bf51a5057"; PREFIX="en/en_US/lessac/low"; MODEL=f"{PREFIX}/en_US-lessac-low.onnx"; CONFIG=MODEL+".json"
def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifest",default="data/financial_research/test_manifest.parquet"); p.add_argument("--output-dir",default="experiments/results/piper_lessac_low"); a=p.parse_args(); root=Path(__file__).resolve().parents[1]; out=root/a.output_dir; wav_dir=out/"wav"; wav_dir.mkdir(parents=True,exist_ok=True)
 model=Path(hf_hub_download(REPO,MODEL,revision=REVISION)); config=Path(hf_hub_download(REPO,CONFIG,revision=REVISION)); voice=PiperVoice.load(str(model),config_path=str(config)); rows=[]
 for index,row in enumerate(pd.read_parquet(root/a.manifest).to_dict("records")):
  path=wav_dir/f"{index:04d}_{row['id']}.wav"
  with wave.open(str(path),"wb") as handle: voice.synthesize_wav(row["sentence"],handle)
  rows.append({**row,"edge_tts_path":row["path"],"path":str(path.relative_to(root)),"source":"piper-en_US-lessac-low"})
 manifest=out/"generated_manifest.parquet"; pd.DataFrame(rows).to_parquet(manifest,index=False); report={"schema_version":1,"n_samples":len(rows),"model_repo":REPO,"revision":REVISION,"model_file":MODEL,"model_sha256":sha256_file(model),"config_sha256":sha256_file(config),"voice_model_card":"https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/lessac/low/MODEL_CARD","piper_project":"https://github.com/OHF-Voice/piper1-gpl","provenance":collect_provenance(repo_root=root,arguments=vars(a),input_files=[root/a.manifest,model,config],seed=20260801)}; (out/"generation_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__":main()
