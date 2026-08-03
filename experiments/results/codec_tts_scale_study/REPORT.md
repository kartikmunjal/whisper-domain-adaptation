# Codec-TTS scale study

Five paired training seeds; intervals use 10,000 seed-level bootstrap resamples. WER may exceed 100% because insertions are unbounded.

| System | Overall WER mean [95% CI] | Domain WER mean [95% CI] | Common WER mean [95% CI] | Conditioning gate |
|---|---:|---:|---:|---:|
| text_forced | 291.34% [148.23, 534.16] | 302.15% [152.27, 565.27] | 162.00% [100.00, 286.00] | False |
| scaled_bytes | 240.12% [134.35, 373.77] | 222.74% [137.22, 309.38] | 448.00% [100.00, 1144.00] | False |
| scaled_phonemes | 273.98% [138.46, 437.70] | 288.53% [141.67, 465.93] | 100.00% [100.00, 100.00] | True |
| piper_lessac_low | 2.86% [2.76, 2.95] | 3.09% [2.99, 3.20] | 0.00% [0.00, 0.00] | external comparator |
| edge_tts | 1.18% [1.06, 1.27] | 1.27% [1.15, 1.38] | 0.00% [0.00, 0.00] | external comparator |
| elevenlabs_multilingual_v2 | 0.48% [0.31, 0.66] | 0.52% [0.33, 0.71] | 0.00% [0.00, 0.00] | external comparator |

## Paired intervention effects

Positive values are worse. Each interval is computed from the five seed-matched WER differences.

| Metric | Scaled bytes − text-forced | Scaled phonemes − scaled bytes |
|---|---:|---:|
| overall | -51.22 pp [-269.76, +127.39] | +33.86 pp [-54.39, +151.88] |
| domain_terms | -79.41 pp [-292.31, +66.31] | +65.79 pp [+0.86, +171.28] |
| common_terms | +286.00 pp [+0.00, +858.00] | -348.00 pp [-1044.00, +0.00] |

## Paired external-comparator contrasts

Negative values mean lower WER for ElevenLabs. Each interval uses the same five frozen adapters.

| Metric | ElevenLabs − Edge-TTS | ElevenLabs − Piper |
|---|---:|---:|
| overall | -0.69 pp [-0.83, -0.54] | -2.37 pp [-2.47, -2.27] |
| domain_terms | -0.75 pp [-0.90, -0.58] | -2.57 pp [-2.68, -2.47] |
| common_terms | +0.00 pp [+0.00, +0.00] | +0.00 pp [+0.00, +0.00] |
