# Synthetic-to-real ASR regression diagnosis

All acoustic values are computed per clip. Mean differences use 10,000 paired-independent bootstrap resamples; effect size is Cliff's delta (real versus synthetic).

| Metric | Synthetic mean | Real mean | Real−synthetic 95% CI | Cliff's δ |
|---|---:|---:|---:|---:|
| duration_seconds | 4.722 | 49.194 | [12.901, 84.571] | 0.933 |
| rms_dbfs | -20.809 | -23.406 | [-5.185, -0.248] | -0.145 |
| heuristic_snr_db | 26.071 | 28.970 | [-0.873, 5.797] | 0.448 |
| silence_fraction | 0.245 | 0.149 | [-0.140, -0.051] | -0.510 |
| spectral_centroid_hz | 1576.129 | 1175.847 | [-505.877, -302.781] | -0.777 |
| spectral_tilt_db_per_octave | -7.131 | -8.300 | [-2.041, -0.228] | -0.454 |
| voiced_fraction | 0.591 | 0.555 | [-0.097, 0.026] | -0.194 |
| pitch_median_hz | 159.839 | 146.606 | [-30.803, 4.106] | -0.139 |
| pitch_range_hz | 117.919 | 121.302 | [-29.263, 39.612] | -0.104 |

Locked augmentation trigger (|δ| ≥ 0.474 for SNR or silence): **met**.

## Common-slice error transitions

Common-control clips: 10

### Seed 11

| Edit | Introduced | Resolved | Retained |
|---|---:|---:|---:|
| substitute | 0 | 6 | 8 |
| insert | 0 | 5 | 2 |
| delete | 23 | 0 | 8 |

Error word classes: `{"function": 11, "numeric": 1, "other": 29}`

### Seed 22

| Edit | Introduced | Resolved | Retained |
|---|---:|---:|---:|
| substitute | 0 | 6 | 8 |
| insert | 0 | 5 | 2 |
| delete | 23 | 0 | 8 |

Error word classes: `{"function": 11, "numeric": 1, "other": 29}`

### Seed 33

| Edit | Introduced | Resolved | Retained |
|---|---:|---:|---:|
| substitute | 0 | 6 | 8 |
| insert | 0 | 5 | 2 |
| delete | 23 | 0 | 8 |

Error word classes: `{"function": 11, "numeric": 1, "other": 29}`

### Seed 44

| Edit | Introduced | Resolved | Retained |
|---|---:|---:|---:|
| substitute | 0 | 6 | 8 |
| insert | 0 | 5 | 2 |
| delete | 23 | 0 | 8 |

Error word classes: `{"function": 11, "numeric": 1, "other": 29}`

### Seed 55

| Edit | Introduced | Resolved | Retained |
|---|---:|---:|---:|
| substitute | 0 | 6 | 8 |
| insert | 0 | 5 | 2 |
| delete | 23 | 0 | 8 |

Error word classes: `{"function": 11, "numeric": 1, "other": 29}`
