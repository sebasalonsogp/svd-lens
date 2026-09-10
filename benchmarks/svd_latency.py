"""Repeatable local latency measurements for the SVD interaction path.

Run from the repository root with::

    uv run python -m benchmarks.svd_latency

The numbers are intentionally informational. Shared CI machines are too noisy
for stable wall-clock performance assertions, so automated tests verify the
benchmark contract without enforcing a latency threshold.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np

from svd_lab.svd import SVDResult, decompose, reconstruct

DEFAULT_SIZES = (256, 384, 512)
DEFAULT_REPETITIONS = 5
DEFAULT_SEED = 20260910
DEFAULT_INTERACTIVE_RANK = 12


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Median and observed range for one measured operation."""

    median_ms: float
    fastest_ms: float
    slowest_ms: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Latency and memory measurements for one square working matrix."""

    size: int
    decomposition: LatencySummary
    interactive_rank: int
    interactive_reconstruction: LatencySummary
    stress_rank: int
    stress_reconstruction: LatencySummary
    factor_mib: float


def benchmark_size(
    size: int,
    *,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
) -> BenchmarkResult:
    """Measure the core operations for a square image matrix.

    Square matrices are the most expensive shape permitted for a given
    longest-side limit. The default rank mirrors the application's initial
    slider position; the quarter-rank measurement exercises a substantially
    heavier update without reconstructing at full rank.
    """
    _require_positive_int("size", size)
    _require_positive_int("repetitions", repetitions)

    matrix = np.random.default_rng(seed).random((size, size), dtype=np.float64)

    latest_decomposition: SVDResult | None = None

    def run_decomposition() -> None:
        nonlocal latest_decomposition
        latest_decomposition = decompose(matrix)

    decomposition_latency = _measure(run_decomposition, repetitions)
    assert latest_decomposition is not None

    interactive_rank = min(DEFAULT_INTERACTIVE_RANK, size)
    stress_rank = max(1, size // 4)
    interactive_latency = _measure(
        lambda: reconstruct(latest_decomposition, interactive_rank), repetitions
    )
    stress_latency = _measure(lambda: reconstruct(latest_decomposition, stress_rank), repetitions)
    factor_bytes = sum(
        factor.nbytes
        for factor in (
            latest_decomposition.u,
            latest_decomposition.singular_values,
            latest_decomposition.vt,
        )
    )

    return BenchmarkResult(
        size=size,
        decomposition=decomposition_latency,
        interactive_rank=interactive_rank,
        interactive_reconstruction=interactive_latency,
        stress_rank=stress_rank,
        stress_reconstruction=stress_latency,
        factor_mib=factor_bytes / (1024**2),
    )


def format_markdown(results: Sequence[BenchmarkResult]) -> str:
    """Format benchmark results as a Markdown table for easy comparison."""
    lines = [
        "| Side (px) | SVD median (range) | Interaction | Stress update | Factors |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.size} | "
            f"{_format_latency(result.decomposition)} | "
            f"Rank-{result.interactive_rank} update: "
            f"{_format_latency(result.interactive_reconstruction)} | "
            f"Rank-{result.stress_rank} update: "
            f"{_format_latency(result.stress_reconstruction)} | "
            f"{result.factor_mib:.2f} MiB |"
        )
    return "\n".join(lines)


def _measure(operation: Callable[[], object], repetitions: int) -> LatencySummary:
    durations_ms: list[float] = []
    for _ in range(repetitions):
        started_at = perf_counter_ns()
        operation()
        durations_ms.append((perf_counter_ns() - started_at) / 1_000_000)

    return LatencySummary(
        median_ms=statistics.median(durations_ms),
        fastest_ms=min(durations_ms),
        slowest_ms=max(durations_ms),
    )


def _format_latency(summary: LatencySummary) -> str:
    return f"{summary.median_ms:.1f} ms ({summary.fastest_ms:.1f}-{summary.slowest_ms:.1f})"


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and print its environment and results."""
    args = _parse_args()
    results = [
        benchmark_size(size, repetitions=args.repetitions, seed=args.seed) for size in args.sizes
    ]

    print(f"Platform: {platform.platform()}")
    print(
        f"Machine: {platform.machine()} | Python {sys.version.split()[0]} | NumPy {np.__version__}"
    )
    print(f"Repetitions: {args.repetitions} | Seed: {args.seed}")
    print()
    print(format_markdown(results))


if __name__ == "__main__":
    main()
