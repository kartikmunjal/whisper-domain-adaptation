# Codec-TTS conditioning diagnostic

Five deterministic trials (seeds 11, 22, 33, 44, 55). Decision: **conditioning passes; use the pre-registered duration-aware non-autoregressive path if drift is position-dependent**.

| Diagnostic | Mean [95% bootstrap CI] | Locked failure gate |
|---|---:|---:|
| Shuffled − true teacher-forced NLL (nats/token) | 0.0871 [0.0626, 0.1125] | ≤ 0.05 |
| True-vs-shuffled generated token edit rate | 0.2390 [0.1922, 0.3001] | ≤ 0.05 |
| Cross-attention centroid monotonicity | 0.5007 [0.4412, 0.5560] | descriptive |
| Normalized cross-attention entropy | 0.9453 [0.9396, 0.9508] | descriptive |

## Free-running error by normalized position

| Decile | Errors / events | Rate |
|---:|---:|---:|
| 1 | 12418 / 18207 | 0.6820 |
| 2 | 14108 / 17415 | 0.8101 |
| 3 | 14499 / 16944 | 0.8557 |
| 4 | 13387 / 16836 | 0.7951 |
| 5 | 14352 / 16318 | 0.8795 |
| 6 | 14701 / 16134 | 0.9112 |
| 7 | 14293 / 16063 | 0.8898 |
| 8 | 13455 / 15951 | 0.8435 |
| 9 | 7804 / 16006 | 0.4876 |
| 10 | 679 / 15924 | 0.0426 |
