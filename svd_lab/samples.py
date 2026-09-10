"""Small, deterministic sample images for the zero-setup app experience."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SampleImage:
    """A named normalized grayscale image bundled through procedural data."""

    name: str
    description: str
    matrix: NDArray[np.float64]


def default_sample() -> SampleImage:
    """Return a geometric image with both low-frequency structure and detail."""
    height, width = 180, 240
    y, x = np.mgrid[:height, :width]
    matrix = np.full((height, width), 0.94, dtype=np.float64)

    # Smooth illumination gives the spectrum a gradual tail instead of an
    # artificially tiny exact rank.
    matrix -= 0.12 * np.exp(-(((x - 120) / 105) ** 2 + ((y - 90) / 75) ** 2))

    circle_distance = np.hypot(x - 67, y - 78)
    matrix[circle_distance <= 37] = 0.20
    matrix[np.abs(circle_distance - 37) <= 3] = 0.05
    matrix[42:116, 128:199] = 0.64
    matrix[49:109, 135:192] = 0.88

    diagonal = np.abs(y - (0.42 * x + 58)) <= 3
    matrix[diagonal] = 0.10

    for offset in range(5):
        top = 132 + offset * 7
        left = 25 + offset * 18
        matrix[top : top + 4, left : left + 42] = 0.30 + offset * 0.10

    np.clip(matrix, 0.0, 1.0, out=matrix)
    matrix.flags.writeable = False
    return SampleImage(
        name="Geometric study",
        description="Circles, edges, gradients, and fine lines reveal how structure returns.",
        matrix=matrix,
    )


__all__ = ["SampleImage", "default_sample"]
