from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from svd_lab.images import ImageValidationError, PreparedImage, prepare_image


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


@pytest.mark.parametrize("upload", [b"", b"this is not an image"])
def test_prepare_image_rejects_invalid_image_bytes(upload: bytes) -> None:
    with pytest.raises(ImageValidationError, match="valid PNG or JPEG"):
        prepare_image(upload)


def test_prepare_image_rejects_unsupported_detected_format() -> None:
    upload = encode_image("RGB", (4, 4), (1, 2, 3), image_format="GIF")

    with pytest.raises(ImageValidationError, match="valid PNG or JPEG"):
        prepare_image(upload)


def test_prepare_image_rejects_truncated_image_data() -> None:
    upload = encode_image("RGB", (20, 20), (1, 2, 3))

    with pytest.raises(ImageValidationError, match="valid PNG or JPEG"):
        prepare_image(upload[: len(upload) // 2])


def test_prepare_image_rejects_upload_before_decoding_when_byte_limit_is_exceeded() -> None:
    upload = encode_image("RGB", (4, 4), (1, 2, 3))

    with pytest.raises(ImageValidationError, match="upload-size limit"):
        prepare_image(upload, max_upload_bytes=len(upload) - 1)


def test_prepare_image_rejects_excessive_decoded_dimensions() -> None:
    upload = encode_image("RGB", (5, 5), (1, 2, 3))

    with pytest.raises(ImageValidationError, match="pixel limit"):
        prepare_image(upload, max_decoded_pixels=24)


def test_prepare_image_treats_pillow_decompression_warning_as_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload = encode_image("RGB", (2, 3), (1, 2, 3))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 4)

    with pytest.raises(ImageValidationError, match="pixel limit"):
        prepare_image(upload)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("max_dimension", 0),
        ("max_dimension", True),
        ("max_upload_bytes", -1),
        ("max_decoded_pixels", 0),
    ],
)
def test_prepare_image_rejects_invalid_processing_limits(option: str, value: int) -> None:
    upload = encode_image("RGB", (2, 2), (1, 2, 3))

    with pytest.raises(ValueError, match=option):
        prepare_image(upload, **{option: value})


def test_prepare_image_requires_bytes() -> None:
    with pytest.raises(TypeError, match="data must be bytes"):
        prepare_image("not bytes")  # type: ignore[arg-type]
