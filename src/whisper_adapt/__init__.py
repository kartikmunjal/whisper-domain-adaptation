"""
whisper-domain-adaptation
=========================

Fine-tuning Whisper on domain-specific vocabulary where the base model falls short.

The core observation is simple: Whisper-small achieves ~4-5% WER on LibriSpeech
test-clean, but on medical dictation or financial earnings calls that number
balloons to 30-40%. The problem isn't general ASR capability — it's out-of-vocabulary
terminology. A cardiologist saying "echocardiogram" or an analyst saying "EBITDA"
is just as intelligible as any common word, but the language model component of
Whisper has low prior probability on those tokens.

Fine-tuning on ~8,000 domain-specific samples (real + TTS-synthesized) brings
WER on domain terms from ~45% down to ~22% with minimal degradation on general speech.

Pipeline overview
-----------------
1. prepare_medical_data.py   — download HuggingFace medical-speech-transcription,
                                run quality filtering (SNR, duration, silence)
2. prepare_financial_data.py — TTS-synthesize earnings call text using the same
                                Edge-TTS voices as the rlhf-and-reward-modelling-alt
                                pipeline, then quality-filter
3. run_finetune.py           — LoRA fine-tune Whisper-small on the curated corpus
4. evaluate_baseline.py      — WER breakdown: overall / domain terms / common terms
5. evaluate_finetuned.py     — same breakdown after fine-tuning

The quality filtering logic mirrors Audio-Data-Creation exactly (SNR >= 15 dB,
silence ratio < 0.4, duration 0.5–30 s), so datasets are directly comparable.
"""
