"""Device-aware benchmark primitives with raw-trial retention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
import statistics
import time
from typing import Callable

import torch


@dataclass(frozen=True)
class BenchmarkResult:
    warmup_iterations: int
    timed_iterations: int
    audio_duration_seconds: float
    latency_ms_median: float
    latency_ms_p95: float
    realtime_factor_median: float
    audio_seconds_per_second: float
    raw_latency_ms: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_callable(
    function: Callable[[], object],
    *,
    device: torch.device,
    audio_duration_seconds: float,
    warmup_iterations: int = 10,
    timed_iterations: int = 100,
) -> BenchmarkResult:
    if audio_duration_seconds <= 0:
        raise ValueError("audio_duration_seconds must be positive")
    if warmup_iterations < 0 or timed_iterations < 2:
        raise ValueError("need non-negative warmups and at least two timed iterations")
    for _ in range(warmup_iterations):
        function()
    synchronize(device)
    timings = []
    for _ in range(timed_iterations):
        synchronize(device)
        started = time.perf_counter_ns()
        function()
        synchronize(device)
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(timings)
    median = statistics.median(ordered)
    p95 = ordered[max(0, int(0.95 * len(ordered)) - 1)]
    rtf = median / (audio_duration_seconds * 1000)
    return BenchmarkResult(
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
        audio_duration_seconds=audio_duration_seconds,
        latency_ms_median=median,
        latency_ms_p95=p95,
        realtime_factor_median=rtf,
        audio_seconds_per_second=1.0 / rtf,
        raw_latency_ms=timings,
    )


def hardware_metadata(device: torch.device) -> dict:
    metadata = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device_type": device.type,
    }
    if device.type == "cuda":
        metadata.update({
            "device_name": torch.cuda.get_device_name(device),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        })
    elif device.type == "mps":
        metadata["device_name"] = "Apple Metal Performance Shaders"
    else:
        metadata["device_name"] = platform.processor() or "CPU"
    return metadata
