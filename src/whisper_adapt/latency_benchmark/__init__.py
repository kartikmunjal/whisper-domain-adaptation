"""Auditable latency and throughput measurement utilities."""

from .timing import BenchmarkResult, benchmark_callable, hardware_metadata

__all__ = ["BenchmarkResult", "benchmark_callable", "hardware_metadata"]
