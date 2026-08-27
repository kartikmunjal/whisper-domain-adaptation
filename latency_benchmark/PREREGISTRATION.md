# Latency and Throughput Preregistration

Status: locked before benchmark results are inspected.

The benchmark reports codec encode, decode, and end-to-end time separately for
200 ms, 500 ms, 1 s, and full-clip inputs. The parallel text-to-codec decoder is
measured separately from waveform decoding. Each condition uses 10 warm-up and
100 timed trials, batch size one, inference mode, explicit accelerator
synchronization, and the same local device. Raw trial durations are retained.

Locked metrics are median and p95 latency in milliseconds, real-time factor
(wall time / represented audio duration), and audio-seconds processed per wall
second. Hardware, software versions, precision, model parameter count, input
shape, and checkpoint hash are recorded. RTF below 1 is only called faster than
real time on the documented hardware; this batch-model simulation is not called
a production streaming implementation. Sub-300 ms p95 end-to-end chunk latency
and RTF below 1 are the preregistered conversational-usefulness reference bars.

Results are reported regardless of direction. Any optimization performed after
viewing a benchmark is labeled exploratory and receives a fresh benchmark file.

## Execution amendment 1 — held-out inputs and full clips

Approved by the repository owner on 2026-08-26 before any benchmark outcome
was observed. Existing research artifacts and conclusions remain immutable.
Codec timing uses real audio from the English-only 24-clip medical manifest,
not zero-valued tensors. Chunk conditions take deterministic leading 200 ms,
500 ms, and 1 s windows; the full-clip condition retains observed duration.
Encode, decode, and combined encode-quantize-decode are reported separately
with raw synchronized trials. The fixed checkpoint is corrected VQ-400 seed
11, chosen by protocol identity rather than latency outcome.

Parallel TTS timing uses the scaled-phoneme text-only checkpoint and VQ decoder
at seed 11, with fixed 200 ms, 500 ms, and 1 s output-token budgets derived
from codec frame rate. Held-out test row zero supplies text conditioning.
Generation and generation-plus-waveform decoding are timed separately; emitted
valid-token count is recorded. Float32, batch one, 10 warmups, 100 synchronized
trials, and RTX 3070 hardware remain locked.

<!-- BEGIN GENERATED LATENCY FINAL RESULT -->
## Final result

All locked timing trials completed on the RTX 3070 with raw trials retained.
For 200-ms inputs, codec end-to-end median/p95 latency is
2.27/2.59 ms with median RTF
0.0113. Parallel TTS generation plus waveform decoding is
10.73/11.91 ms with median RTF
0.0536. Both clear the preregistered RTF < 1 and p95 < 300 ms
chunk-simulation bars on this hardware. Full-clip codec p95 is
2290.72 ms, retained separately from the
conversational chunk criterion. These are batch-one research measurements, not
production streaming, concurrency, tail-at-load, or enterprise throughput claims.
<!-- END GENERATED LATENCY FINAL RESULT -->
