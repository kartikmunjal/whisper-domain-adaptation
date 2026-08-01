# Codec-TTS conditioning diagnostic

Five deterministic trials (seeds 11, 22, 33, 44, 55). Decision: **conditioning is broken; repair the conditioning path**.

| Diagnostic | Mean [95% bootstrap CI] | Locked failure gate |
|---|---:|---:|
| Shuffled − true teacher-forced NLL (nats/token) | 0.0255 [0.0145, 0.0371] | ≤ 0.05 |
| True-vs-shuffled generated token edit rate | 0.1461 [0.1287, 0.1698] | ≤ 0.05 |
| Cross-attention centroid monotonicity | 0.1598 [0.0790, 0.2391] | descriptive |
| Normalized cross-attention entropy | 0.9635 [0.9599, 0.9668] | descriptive |

## Free-running error by normalized position

| Decile | Errors / events | Rate |
|---:|---:|---:|
| 1 | 12495 / 17953 | 0.6960 |
| 2 | 13333 / 16874 | 0.7902 |
| 3 | 13952 / 16476 | 0.8468 |
| 4 | 12706 / 16444 | 0.7727 |
| 5 | 14371 / 16200 | 0.8871 |
| 6 | 14750 / 16198 | 0.9106 |
| 7 | 14214 / 16035 | 0.8864 |
| 8 | 13439 / 15950 | 0.8426 |
| 9 | 7847 / 16046 | 0.4890 |
| 10 | 772 / 15970 | 0.0483 |
