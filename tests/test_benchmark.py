import pytest

from benchmarks.svd_latency import benchmark_size, format_markdown


def test_benchmark_size_reports_decomposition_and_rank_updates() -> None:
    result = benchmark_size(8, repetitions=2, seed=7)

    assert result.size == 8
    assert result.interactive_rank == 8
    assert result.stress_rank == 2
    assert result.factor_mib > 0
    assert result.decomposition.median_ms >= 0
    assert result.interactive_reconstruction.median_ms >= 0
    assert result.stress_reconstruction.median_ms >= 0


@pytest.mark.parametrize(
    ("size", "repetitions"),
    [(0, 1), (8, 0), (True, 1), (8, True)],
)
def test_benchmark_size_rejects_invalid_arguments(size: int, repetitions: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        benchmark_size(size, repetitions=repetitions)


def test_format_markdown_builds_a_readable_results_table() -> None:
    result = benchmark_size(4, repetitions=1)

    markdown = format_markdown([result])

    assert "| Side (px) |" in markdown
    assert "| 4 |" in markdown
    assert "SVD median" in markdown
    assert "Rank-4 update" in markdown
