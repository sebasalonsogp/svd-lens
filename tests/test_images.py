from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from svd_lab.images import PreparedImage, prepare_image


def encode_image(
    mode: str,
    size: tuple[int, int],
    color: int | tuple[int, ...],
    *,
    image_format: str = "PNG",
    exif: Image.Exif | None = None,
) -> bytes:
    image = Image.new(mode, size, color)
    output = BytesIO()
    image.save(output, format=image_format, exif=exif)
    return output.getvalue()


def test_prepare_image_returns_normalized_read_only_grayscale_matrix() -> None:
    upload = encode_image("RGB", (3, 2), (255, 0, 0))

    result = prepare_image(upload)

    assert isinstance(result, PreparedImage)
    assert result.source_format == "PNG"
    assert result.original_size == (3, 2)
    assert result.processed_size == (3, 2)
    assert result.matrix.shape == (2, 3)
    assert result.matrix.dtype == np.float64
    assert result.matrix.min() >= 0.0
    assert result.matrix.max() <= 1.0
    assert result.matrix[0, 0] == pytest.approx(76 / 255)
    assert not result.matrix.flags.writeable


def test_prepare_image_downsamples_while_preserving_aspect_ratio() -> None:
    upload = encode_image("RGB", (800, 400), (20, 40, 60))

    result = prepare_image(upload, max_dimension=512)

    assert result.original_size == (800, 400)
    assert result.processed_size == (512, 256)
    assert result.matrix.shape == (256, 512)


def test_prepare_image_does_not_enlarge_small_images() -> None:
    upload = encode_image("L", (8, 4), 100)

    result = prepare_image(upload, max_dimension=512)

    assert result.processed_size == (8, 4)
    assert result.matrix.shape == (4, 8)


def test_prepare_image_applies_exif_orientation() -> None:
    exif = Image.Exif()
    exif[274] = 6
    upload = encode_image("RGB", (2, 3), (10, 20, 30), image_format="JPEG", exif=exif)

    result = prepare_image(upload)

    assert result.source_format == "JPEG"
    assert result.original_size == (3, 2)
    assert result.processed_size == (3, 2)
    assert result.matrix.shape == (2, 3)


def test_prepare_image_composites_transparency_over_white() -> None:
    upload = encode_image("RGBA", (1, 1), (0, 0, 0, 0))

    result = prepare_image(upload)

    assert result.matrix[0, 0] == pytest.approx(1.0)
