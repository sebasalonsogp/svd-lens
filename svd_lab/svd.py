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


__all__ = ["SVDResult", "decompose", "reconstruct"]
