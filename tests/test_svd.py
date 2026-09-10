"""Behavioral tests for decomposition and rank-k reconstruction."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from svd_lab.svd import ApproximationMetrics, decompose, metrics_for, reconstruct


def test_decompose_returns_compact_factors_for_rectangular_matrix() -> None:
    matrix = np.array([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    result = decompose(matrix)

    assert result.original_shape == (2, 3)
    assert result.max_rank == 2
    assert result.u.shape == (2, 2)
    assert result.singular_values.shape == (2,)
    assert result.vt.shape == (2, 3)
    np.testing.assert_allclose(result.singular_values, [3.0, 2.0])


def test_full_rank_reconstruction_matches_input() -> None:
    matrix = np.array(
        [
            [4.0, 1.0, 3.0],
            [2.0, 5.0, 0.0],
            [1.0, 2.0, 6.0],
            [0.0, 1.0, 2.0],
        ]
    )

    result = decompose(matrix)
    reconstructed = reconstruct(result, result.max_rank)

    np.testing.assert_allclose(reconstructed, matrix, atol=1e-12)


def test_rank_one_reconstruction_recovers_rank_one_matrix() -> None:
    left = np.array([1.0, 2.0, -1.0])
    right = np.array([3.0, 0.5])
    matrix = np.outer(left, right)

    reconstructed = reconstruct(decompose(matrix), rank=1)

    np.testing.assert_allclose(reconstructed, matrix, atol=1e-12)


def test_decomposition_does_not_modify_input() -> None:
    matrix = np.arange(12.0).reshape(3, 4)
    original = matrix.copy()

    decompose(matrix)

    np.testing.assert_array_equal(matrix, original)


def test_decomposition_factors_are_read_only() -> None:
    result = decompose(np.eye(3))

    assert not result.u.flags.writeable
    assert not result.singular_values.flags.writeable
    assert not result.vt.flags.writeable


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([1.0, 2.0]),
        np.empty((0, 3)),
        np.array([[1.0, np.nan]]),
        np.array([[1.0, np.inf]]),
        np.array([[1.0 + 2.0j]]),
    ],
)
def test_decompose_rejects_invalid_matrices(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError):
        decompose(matrix)


@pytest.mark.parametrize("rank", [0, -1, 3, 1.5, True])
def test_reconstruct_rejects_invalid_rank(rank: object) -> None:
    result = decompose(np.ones((2, 3)))

    with pytest.raises(ValueError):
        reconstruct(result, cast(int, rank))


def test_metrics_describe_rank_one_approximation() -> None:
    result = decompose(np.diag([4.0, 3.0]))

    metrics = metrics_for(result, rank=1)

    assert isinstance(metrics, ApproximationMetrics)
    assert metrics.rank == 1
    assert metrics.retained_energy == pytest.approx(16 / 25)
    assert metrics.relative_error == pytest.approx(3 / 5)
    assert metrics.original_value_count == 4
    assert metrics.factorized_value_count == 5
    assert metrics.representation_ratio == pytest.approx(4 / 5)


def test_full_rank_metrics_report_complete_reconstruction() -> None:
    result = decompose(np.diag([4.0, 3.0]))

    metrics = metrics_for(result, rank=result.max_rank)

    assert metrics.retained_energy == pytest.approx(1.0)
    assert metrics.relative_error == pytest.approx(0.0)


def test_zero_matrix_has_no_defined_energy_ratio_and_no_error() -> None:
    result = decompose(np.zeros((3, 2)))

    metrics = metrics_for(result, rank=1)

    assert metrics.retained_energy is None
    assert metrics.relative_error == 0.0


def test_metrics_remain_finite_for_very_large_singular_values() -> None:
    result = decompose(np.diag([1e200, 1e200]))

    metrics = metrics_for(result, rank=1)

    assert metrics.retained_energy == pytest.approx(0.5)
    assert metrics.relative_error == pytest.approx(np.sqrt(0.5))


def test_quality_metrics_are_monotonic_as_rank_increases() -> None:
    matrix = np.arange(1.0, 31.0).reshape(5, 6)
    result = decompose(matrix)

    metrics = [metrics_for(result, rank) for rank in range(1, result.max_rank + 1)]
    energies = [item.retained_energy for item in metrics]
    errors = [item.relative_error for item in metrics]

    assert all(energy is not None for energy in energies)
    numeric_energies = [cast(float, energy) for energy in energies]
    assert np.all(np.diff(numeric_energies) >= -1e-15)
    assert np.all(np.diff(errors) <= 1e-15)


def test_energy_and_squared_error_partition_total_matrix_energy() -> None:
    result = decompose(np.diag([5.0, 4.0, 2.0, 1.0]))

    for rank in range(1, result.max_rank + 1):
        metrics = metrics_for(result, rank)

        assert metrics.retained_energy is not None
        assert metrics.retained_energy + metrics.relative_error**2 == pytest.approx(1.0)


@pytest.mark.parametrize("rank", [0, -1, 3, 1.5, True])
def test_metrics_reject_invalid_rank(rank: object) -> None:
    result = decompose(np.ones((2, 3)))

    with pytest.raises(ValueError):
        metrics_for(result, cast(int, rank))
