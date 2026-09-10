"""Framework-independent preparation of uploaded images for SVD."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

MAX_IMAGE_DIMENSION = 512
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_DECODED_PIXELS = 20_000_000
ALLOWED_FORMATS = ("PNG", "JPEG")


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
    del max_upload_bytes, max_decoded_pixels  # Enforced by the validation slice.

    with Image.open(BytesIO(data), formats=ALLOWED_FORMATS) as opened:
        source_format = opened.format
        opened.load()
        oriented = ImageOps.exif_transpose(opened)

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
