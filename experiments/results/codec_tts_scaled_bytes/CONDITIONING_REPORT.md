# Codec-TTS conditioning diagnostic

Five deterministic trials (seeds 11, 22, 33, 44, 55). Decision: **conditioning is broken; repair the conditioning path**.

| Diagnostic | Mean [95% bootstrap CI] | Locked failure gate |
|---|---:|---:|
| Shuffled − true teacher-forced NLL (nats/token) | 0.0385 [0.0267, 0.0528] | ≤ 0.05 |
| True-vs-shuffled generated token edit rate | 0.2609 [0.2240, 0.2965] | ≤ 0.05 |
| Cross-attention centroid monotonicity | 0.6123 [0.5758, 0.6459] | descriptive |
| Normalized cross-attention entropy | 0.9374 [0.9309, 0.9439] | descriptive |

## Free-running error by normalized position

| Decile | Errors / events | Rate |
|---:|---:|---:|
| 1 | 12584 / 18537 | 0.6789 |
| 2 | 14148 / 17637 | 0.8022 |
| 3 | 14743 / 17368 | 0.8489 |
| 4 | 13330 / 16892 | 0.7891 |
| 5 | 14562 / 16531 | 0.8809 |
| 6 | 14761 / 16119 | 0.9158 |
| 7 | 14292 / 16045 | 0.8907 |
| 8 | 13460 / 15964 | 0.8431 |
| 9 | 7930 / 16145 | 0.4912 |
| 10 | 1124 / 16300 | 0.0690 |
