# Codec rate–distortion study: 200–500 bps

Each signal point is a five-seed mean; WER uses the seed selected by median SI-SDR before ASR inference.

| Quantizer | Nominal bps | Empirical bps | SI-SDR | Log-mel L1 |
|---|---:|---:|---:|---:|
| FSQ | 200 | 44.2 | -33.602 | 15.364 |
| VQ | 200 | 54.8 | -25.541 | 13.568 |
| FSQ | 250 | 33.3 | -38.310 | 16.156 |
| VQ | 250 | 76.2 | -23.012 | 12.976 |
| FSQ | 300 | 56.1 | -29.054 | 14.150 |
| VQ | 300 | 101.5 | -22.756 | 12.954 |
| FSQ | 400 | 99.1 | -21.503 | 11.848 |
| VQ | 400 | 175.0 | -17.880 | 11.734 |
| FSQ | 500 | 66.8 | -28.438 | 13.946 |
| VQ | 500 | 200.3 | -15.731 | 12.513 |
