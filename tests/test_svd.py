"""Behavioral tests for decomposition and rank-k reconstruction."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from svd_lab.svd import decompose, reconstruct


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
