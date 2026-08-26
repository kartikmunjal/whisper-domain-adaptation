import pytest

torch = pytest.importorskip("torch")

from whisper_adapt.latency_benchmark import benchmark_callable, hardware_metadata


def test_benchmark_retains_trials_and_computes_rtf():
    result = benchmark_callable(
        lambda: torch.ones(2) + 1,
        device=torch.device("cpu"),
        audio_duration_seconds=0.2,
        warmup_iterations=1,
        timed_iterations=5,
    )
    assert len(result.raw_latency_ms) == 5
    assert result.latency_ms_p95 >= 0
    assert result.realtime_factor_median == pytest.approx(
        result.latency_ms_median / 200
    )
    assert result.audio_seconds_per_second > 0


def test_hardware_metadata_has_reproducibility_fields():
    metadata = hardware_metadata(torch.device("cpu"))
    assert {"platform", "python", "torch", "device_type", "device_name"} <= metadata.keys()


def test_p95_uses_retained_trials():
    result = benchmark_callable(
        lambda: None,
        device=torch.device("cpu"),
        audio_duration_seconds=1.0,
        warmup_iterations=0,
        timed_iterations=100,
    )
    assert len(result.raw_latency_ms) == 100
    assert result.latency_ms_p95 == sorted(result.raw_latency_ms)[94]
