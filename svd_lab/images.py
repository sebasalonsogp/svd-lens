"""Framework-independent preparation of uploaded images for SVD."""

import warnings
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_DIMENSION = 512
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_DECODED_PIXELS = 20_000_000
ALLOWED_FORMATS = ("PNG", "JPEG")


class ImageValidationError(ValueError):
    """Raised when uploaded bytes cannot be processed safely as an image."""


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """A decoded image and the normalized grayscale matrix used by the SVD core.

    Sizes use Pillow's ``(width, height)`` convention. ``matrix`` is a read-only
    two-dimensional float64 array with values in the closed interval [0, 1].
    """

    matrix: NDArray[np.float64]
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    source_format: str


def prepare_image(
    data: bytes,
    *,
    max_dimension: int = MAX_IMAGE_DIMENSION,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    max_decoded_pixels: int = MAX_DECODED_PIXELS,
) -> PreparedImage:
    """Decode PNG/JPEG bytes and return a bounded normalized grayscale image."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    _require_positive_int("max_dimension", max_dimension)
    _require_positive_int("max_upload_bytes", max_upload_bytes)
    _require_positive_int("max_decoded_pixels", max_decoded_pixels)

    if len(data) > max_upload_bytes:
        raise ImageValidationError("Image exceeds the upload-size limit.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data), formats=ALLOWED_FORMATS) as opened:
                if opened.width * opened.height > max_decoded_pixels:
                    raise ImageValidationError("Image exceeds the decoded-pixel limit.")
                source_format = opened.format
                opened.load()
                oriented = ImageOps.exif_transpose(opened).copy()
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ImageValidationError("Image exceeds the decoded-pixel limit.") from error
    except (OSError, SyntaxError, UnidentifiedImageError) as error:
        raise ImageValidationError("Upload must be a valid PNG or JPEG image.") from error

    original_size = oriented.size
    image = _composite_transparency(oriented)
    image = _resize_to_fit(image, max_dimension)
    grayscale = image.convert("L")

    matrix = np.asarray(grayscale, dtype=np.float64).copy()
    matrix /= 255.0
    matrix.flags.writeable = False

    return PreparedImage(
        matrix=matrix,
        original_size=original_size,
        processed_size=grayscale.size,
        source_format=source_format,
    )


def to_display_image(matrix: ArrayLike) -> Image.Image:
    """Convert a finite 2D numeric matrix to an 8-bit grayscale image.

    Values outside the normalized [0, 1] range are clipped, which keeps small
    floating-point overshoots from an SVD reconstruction display-safe.
    """
    try:
        values = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("matrix must contain numeric values") from error

    if values.ndim != 2 or values.size == 0:
        raise ValueError("matrix must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("matrix must contain only finite values")

    pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(pixels)


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _composite_transparency(image: Image.Image) -> Image.Image:
    if "A" not in image.getbands() and "transparency" not in image.info:
        return image

    foreground = image.convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    return Image.alpha_composite(background, foreground).convert("RGB")


def _resize_to_fit(image: Image.Image, max_dimension: int) -> Image.Image:
    if max(image.size) <= max_dimension:
        return image

    scale = max_dimension / max(image.size)
    target_size = tuple(max(1, round(dimension * scale)) for dimension in image.size)
    return image.resize(target_size, Image.Resampling.LANCZOS, reducing_gap=3.0)
