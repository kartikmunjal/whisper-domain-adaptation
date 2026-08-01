#!/usr/bin/env python3
"""Compare synthetic/real acoustics and diagnose real-audio ASR regressions."""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from whisper_adapt.evaluation.wer import DomainWERAnalyzer, load_domain_vocab
from whisper_adapt.reproducibility import collect_provenance

FUNCTION_WORDS=set("a an and are as at be been but by for from had has have he her his i in is it its of on or our she that the their they this to was we were will with you your".split())
NUMBER_WORDS=set("zero one two three four five six seven eight nine ten hundred thousand million billion percent".split())
OPS=("substitute","insert","delete")

def resolve_audio(root:Path,value:str)->Path:
    path=Path(value.replace("\\","/")); return path if path.is_absolute() else root/path

def acoustic_features(path:Path,sample_rate:int=16_000)->dict:
    audio,_=librosa.load(path,sr=sample_rate,mono=True); duration=len(audio)/sample_rate
    rms=librosa.feature.rms(y=audio,frame_length=400,hop_length=160,center=False)[0]; rms_db=20*np.log10(np.maximum(rms,1e-8)); peak=float(rms_db.max()) if len(rms_db) else -160.0; active=rms_db[(rms_db>=-80.0)&(rms_db>=peak-50.0)]
    centroid=librosa.feature.spectral_centroid(y=audio,sr=sample_rate,n_fft=400,hop_length=160,center=False)[0]
    spectrum=np.abs(librosa.stft(audio,n_fft=512,hop_length=160,win_length=400,center=False))**2; frequencies=librosa.fft_frequencies(sr=sample_rate,n_fft=512); keep=(frequencies>=200)&(frequencies<=4000); mean_power=np.maximum(spectrum[keep].mean(axis=1),1e-12)
    f0,voiced,_=librosa.pyin(audio,fmin=65,fmax=500,sr=sample_rate,frame_length=1024,hop_length=160); voiced_f0=f0[np.isfinite(f0)]
    return {"duration_seconds":duration,"rms_dbfs":float(20*np.log10(np.sqrt(np.mean(audio**2))+1e-8)),"heuristic_snr_db":float(np.percentile(active,90)-np.percentile(active,10)) if len(active)>=2 else None,"silence_fraction":float(np.mean(rms_db<peak-40)) if len(rms_db) else 1.0,"spectral_centroid_hz":float(np.mean(centroid)),"spectral_tilt_db_per_octave":float(np.polyfit(np.log2(frequencies[keep]/200),10*np.log10(mean_power),1)[0]),"voiced_fraction":float(np.mean(voiced)),"pitch_median_hz":float(np.median(voiced_f0)) if len(voiced_f0) else None,"pitch_range_hz":float(np.percentile(voiced_f0,95)-np.percentile(voiced_f0,5)) if len(voiced_f0) else None}

def cliffs_delta(left:list[float],right:list[float])->float:
    a=np.asarray(left)[:,None]; b=np.asarray(right)[None,:]; return float((np.sum(a>b)-np.sum(a<b))/(a.size*b.shape[1]))

def bootstrap_difference(left:list[float],right:list[float],n:int=10_000)->list[float]:
    rng=np.random.default_rng(20260731); a=np.asarray(left); b=np.asarray(right); values=np.asarray([rng.choice(b,len(b),True).mean()-rng.choice(a,len(a),True).mean() for _ in range(n)]); return np.quantile(values,[.025,.975]).tolist()

def words(text:str)->list[str]: return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?",text.lower())

def word_alignment(reference:str,hypothesis:str)->list[tuple[str,str,str]]:
    ref,hyp=words(reference),words(hypothesis); n,m=len(ref),len(hyp); cost=np.zeros((n+1,m+1),dtype=int); back=np.empty((n+1,m+1),dtype=object)
    for i in range(1,n+1): cost[i,0],back[i,0]=i,"delete"
    for j in range(1,m+1): cost[0,j],back[0,j]=j,"insert"
    rank={"equal":0,"substitute":1,"delete":2,"insert":3}
    for i in range(1,n+1):
        for j in range(1,m+1):
            choices=[(cost[i-1,j]+1,"delete"),(cost[i,j-1]+1,"insert"),(cost[i-1,j-1]+(ref[i-1]!=hyp[j-1]),"equal" if ref[i-1]==hyp[j-1] else "substitute")]; cost[i,j],back[i,j]=min(choices,key=lambda x:(x[0],rank[x[1]]))
    result=[]; i=n; j=m
    while i or j:
        op=back[i,j]
        if op in ("equal","substitute"): result.append((op,ref[i-1],hyp[j-1])); i-=1;j-=1
        elif op=="delete": result.append((op,ref[i-1],"")); i-=1
        else: result.append((op,"",hyp[j-1])); j-=1
    return result[::-1]

def word_class(word:str,financial:set[str])->str:
    if not word:return "gap"
    if word in FUNCTION_WORDS:return "function"
    if word in NUMBER_WORDS or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?",word):return "numeric"
    if word in financial:return "financial"
    return "other"

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--synthetic-manifest",default="data/financial_research/train_manifest.parquet"); p.add_argument("--real-manifest",default="data/earnings21_eval/eval_manifest.parquet"); p.add_argument("--baseline",default="experiments/results/earnings21/baseline.json"); p.add_argument("--adapted-template",default="experiments/results/earnings21/seed_{seed}/finetuned.json"); p.add_argument("--domain-vocab",default="configs/financial_terms.txt"); p.add_argument("--output",default="experiments/results/earnings21/regression_diagnosis.json"); p.add_argument("--bootstrap-resamples",type=int,default=10_000); args=p.parse_args(); root=Path(__file__).resolve().parents[1]
    synth=pd.read_parquet(root/args.synthetic_manifest); real=pd.read_parquet(root/args.real_manifest); corpora={"synthetic":[{"id":str(r["id"]),**acoustic_features(resolve_audio(root,r["path"]))} for r in synth.to_dict("records")],"real":[{"id":str(r["id"]),**acoustic_features(resolve_audio(root,r["path"]))} for r in real.to_dict("records")]}; metrics={}
    for name in [k for k in corpora["synthetic"][0] if k!="id"]:
        left=[x[name] for x in corpora["synthetic"] if x[name] is not None]; right=[x[name] for x in corpora["real"] if x[name] is not None]; metrics[name]={"synthetic_mean":float(np.mean(left)),"synthetic_median":float(np.median(left)),"real_mean":float(np.mean(right)),"real_median":float(np.median(right)),"real_minus_synthetic_mean_95_ci":bootstrap_difference(left,right,args.bootstrap_resamples),"cliffs_delta_real_vs_synthetic":cliffs_delta(right,left),"n_synthetic":len(left),"n_real":len(right)}
    vocab=load_domain_vocab(root/args.domain_vocab); financial=set(w for term in vocab for w in words(term)); analyzer=DomainWERAnalyzer(vocab); baseline=json.loads((root/args.baseline).read_text()); base_rows=baseline["predictions"]; common=[not analyzer._contains_domain_term(x["reference"]) for x in base_rows]; trials=[]; inputs=[root/args.synthetic_manifest,root/args.real_manifest,root/args.baseline,root/args.domain_vocab]
    for seed in (11,22,33,44,55):
        path=root/args.adapted_template.format(seed=seed); inputs.append(path); adapted=json.loads(path.read_text())["predictions"]
        if [x["id"] for x in base_rows]!=[x["id"] for x in adapted]:raise RuntimeError("Prediction IDs are not paired")
        transitions={op:{"introduced":0,"resolved":0,"retained":0} for op in OPS}; classes=Counter(); confusions=Counter()
        for base,after,keep in zip(base_rows,adapted,common):
            if not keep:continue
            b=word_alignment(base["reference"],base["hypothesis"]); a=word_alignment(after["reference"],after["hypothesis"]); bc=Counter(op for op,_,_ in b if op in OPS); ac=Counter(op for op,_,_ in a if op in OPS)
            for op in OPS: transitions[op]["introduced"]+=max(ac[op]-bc[op],0); transitions[op]["resolved"]+=max(bc[op]-ac[op],0); transitions[op]["retained"]+=min(bc[op],ac[op])
            for op,ref,hyp in a:
                if op in OPS: classes[word_class(ref or hyp,financial)]+=1; confusions[f"{ref} -> {hyp}"]+=op=="substitute"
        trials.append({"seed":seed,"transitions":transitions,"adapted_error_word_classes":dict(classes),"top_substitutions":confusions.most_common(20)})
    result={"schema_version":1,"acoustic_comparison":{"metrics":metrics,"clip_features":corpora},"common_control_error_analysis":{"n_common_clips":sum(common),"trials":trials},"provenance":collect_provenance(repo_root=root,arguments=vars(args),input_files=inputs,seed=20260731)}; output=root/args.output; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps({"metrics":metrics,"n_common_clips":sum(common)},indent=2))
if __name__=="__main__":main()
