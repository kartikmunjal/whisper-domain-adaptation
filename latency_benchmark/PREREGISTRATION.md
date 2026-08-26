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
