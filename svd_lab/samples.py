"""Small, deterministic sample images for the zero-setup app experience."""

from dataclasses import dataclass
from functools import cache

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SampleImage:
    """A named normalized grayscale image bundled through procedural data."""

    name: str
    description: str
    matrix: NDArray[np.float64]


@cache
def available_samples() -> tuple[SampleImage, ...]:
    """Return the curated sample catalog in display order."""
    return (_geometric_study(), _soft_bands(), _woven_detail())


def default_sample() -> SampleImage:
    """Return the sample shown on a visitor's first app load."""
    return available_samples()[0]


def _geometric_study() -> SampleImage:
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

    return SampleImage(
        name="Geometric study",
        description="Circles, edges, gradients, and fine lines reveal how structure returns.",
        matrix=_ready(matrix),
    )


def _soft_bands() -> SampleImage:
    height, width = 180, 240
    y, x = np.mgrid[:height, :width]
    matrix = (
        0.52
        + 0.20 * np.sin(2 * np.pi * x / width)
        + 0.16 * np.cos(3 * np.pi * y / height)
        + 0.10 * np.cos(2 * np.pi * (x + y) / (width + height))
    )
    matrix -= 0.22 * np.exp(-(((x - 164) / 46) ** 2 + ((y - 78) / 58) ** 2))
    return SampleImage(
        name="Soft bands",
        description="Smooth, repeated gradients show why broad structure needs only a few ranks.",
        matrix=_ready(matrix),
    )


def _woven_detail() -> SampleImage:
    height, width = 180, 240
    y, x = np.mgrid[:height, :width]
    curved_threads = np.sin(x / 4.8 + np.sin(y / 18) * 2.4)
    cross_threads = np.cos(y / 5.4 + np.sin(x / 22) * 2.0)
    diagonal_threads = np.sin((x + 1.6 * y) / 7.0)
    matrix = 0.50 + 0.17 * curved_threads + 0.16 * cross_threads + 0.11 * diagonal_threads

    ring = np.abs(np.hypot(x - 120, y - 90) - 54) <= 3
    matrix[ring] = 0.94
    return SampleImage(
        name="Woven detail",
        description="Crossing textures create a longer spectrum and reward retaining more ranks.",
        matrix=_ready(matrix),
    )


def _ready(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    np.clip(matrix, 0.0, 1.0, out=matrix)
    matrix.flags.writeable = False
    return matrix


__all__ = ["SampleImage", "available_samples", "default_sample"]
