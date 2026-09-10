"""Singular value decomposition, reconstruction, and quality metrics.

The public functions accept and return ordinary NumPy values. This module has
no dependency on a web framework so any interface can reuse the same domain
behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatMatrix = NDArray[np.float64]
FloatVector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SVDResult:
    """Compact singular value decomposition of a two-dimensional matrix."""

    u: FloatMatrix
    singular_values: FloatVector
    vt: FloatMatrix
    original_shape: tuple[int, int]

    @property
    def max_rank(self) -> int:
        """Return the greatest valid reconstruction rank."""
        return int(self.singular_values.size)


@dataclass(frozen=True, slots=True)
class ApproximationMetrics:
    """Quality and representation measures for one reconstruction rank.

    ``representation_ratio`` is the original scalar-value count divided by the
    factorized count. Values above one indicate a smaller factorized
    representation; values below one indicate a larger one.
    """

    rank: int
    retained_energy: float | None
    relative_error: float
    original_value_count: int
    factorized_value_count: int
    representation_ratio: float


def decompose(matrix: ArrayLike) -> SVDResult:
    """Return the compact SVD of a finite, non-empty real matrix.

    Raises:
        ValueError: If ``matrix`` is not a finite, non-empty real 2D matrix.
    """
    values = _as_float_matrix(matrix)
    u, singular_values, vt = np.linalg.svd(values, full_matrices=False)

    for factor in (u, singular_values, vt):
        factor.setflags(write=False)

    return SVDResult(
        u=u,
        singular_values=singular_values,
        vt=vt,
        original_shape=(int(values.shape[0]), int(values.shape[1])),
    )


def reconstruct(result: SVDResult, rank: int) -> FloatMatrix:
    """Reconstruct a matrix from the leading ``rank`` singular components.

    Raises:
        ValueError: If ``rank`` is not an integer from 1 through
            ``result.max_rank``.
    """
    validated_rank = _validated_rank(rank, result.max_rank)
    scaled_u = result.u[:, :validated_rank] * result.singular_values[:validated_rank]
    return scaled_u @ result.vt[:validated_rank, :]


def metrics_for(result: SVDResult, rank: int) -> ApproximationMetrics:
    """Return quality and representation metrics for a reconstruction rank.

    Retained energy is ``None`` for a zero matrix because its energy ratio has a
    zero denominator. Its relative reconstruction error is defined as zero
    because every rank reconstructs it exactly.

    Raises:
        ValueError: If ``rank`` is not an integer from 1 through
            ``result.max_rank``.
    """
    validated_rank = _validated_rank(rank, result.max_rank)
    singular_values = result.singular_values
    largest_value = float(singular_values[0])

    if largest_value == 0.0:
        retained_energy = None
        relative_error = 0.0
    else:
        scaled_values = singular_values / largest_value
        squared_values = np.square(scaled_values)
        total_energy = float(np.sum(squared_values))
        retained = float(np.sum(squared_values[:validated_rank]) / total_energy)
        discarded = float(np.sum(squared_values[validated_rank:]) / total_energy)
        retained_energy = float(np.clip(retained, 0.0, 1.0))
        relative_error = float(np.sqrt(np.clip(discarded, 0.0, 1.0)))

    rows, columns = result.original_shape
    original_value_count = rows * columns
    factorized_value_count = validated_rank * (rows + columns + 1)

    return ApproximationMetrics(
        rank=validated_rank,
        retained_energy=retained_energy,
        relative_error=relative_error,
        original_value_count=original_value_count,
        factorized_value_count=factorized_value_count,
        representation_ratio=original_value_count / factorized_value_count,
    )


def _as_float_matrix(matrix: ArrayLike) -> FloatMatrix:
    try:
        candidate = np.asarray(matrix)
    except (TypeError, ValueError) as exc:
        raise ValueError("matrix must be convertible to a numeric array") from exc

    if np.iscomplexobj(candidate):
        raise ValueError("matrix must contain real values")

    try:
        values = np.asarray(candidate, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("matrix must contain numeric values") from exc

    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if 0 in values.shape:
        raise ValueError("matrix must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must contain only finite values")

    return values


def _validated_rank(rank: int, max_rank: int) -> int:
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError("rank must be an integer")

    validated_rank = int(rank)
    if not 1 <= validated_rank <= max_rank:
        raise ValueError(f"rank must be between 1 and {max_rank}")

    return validated_rank


__all__ = ["ApproximationMetrics", "SVDResult", "decompose", "metrics_for", "reconstruct"]
