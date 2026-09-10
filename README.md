# SVD Lens

An interactive visual explanation of singular value decomposition and low-rank
image approximation.

> Upload an image and drag rank from 1 upward to watch structure emerge while
> reconstruction error falls.

## Status

The framework-independent mathematical and image-processing core is complete.
It implements compact SVD, rank reconstruction and metrics, plus safe PNG/JPEG
decoding, orientation, resizing, grayscale normalization, and reconstruction
display conversion. The interactive experience is the next implementation
layer.

## Planned experience

Visitors will be able to start with a curated sample or upload a PNG or JPEG,
compare the original with a rank-`k` reconstruction, and inspect:

- Retained singular-value energy
- Relative Frobenius reconstruction error
- Estimated matrix representation size
- The singular-value spectrum

This is an educational portfolio prototype, not a replacement for production
image codecs.

## Architecture

- `app.py` contains only the Streamlit interface and session orchestration.
- `svd_lab/svd.py` owns decomposition, reconstruction, and mathematical metrics.
- `svd_lab/images.py` owns safe image preparation.
- `svd_lab/plots.py` creates reusable Plotly figures.
- `svd_lab/explanations.py` creates plain-language interpretations.
- `tests/` verifies numerical behavior, image handling, the application shell,
  and the boundary between domain logic and UI frameworks.

The `svd_lab` package deliberately contains no Streamlit imports, allowing a
future Gradio adapter to reuse the mathematical and presentation logic.

### Mathematical API

`svd_lab.svd` exposes three operations:

- `decompose(matrix)` validates a real, finite, non-empty matrix and returns
  immutable compact SVD factors.
- `reconstruct(result, rank)` returns the approximation formed from the leading
  singular components.
- `metrics_for(result, rank)` returns quality and representation metrics without
  reconstructing the matrix solely to measure its error.

Ranks are one-based and must fall between `1` and `result.max_rank`. Invalid
matrices and ranks raise `ValueError`. Retained energy is reported as `None` for
an all-zero matrix because its energy ratio has a zero denominator.

### Image API

`svd_lab.images.prepare_image(data)` accepts PNG or JPEG bytes and returns a
`PreparedImage` containing a read-only, normalized grayscale matrix and its
source and processed dimensions. The default limits are 10 MiB per upload, 20
million decoded pixels, and 512 pixels on the longest processed side. Images
are never enlarged, aspect ratio is preserved, EXIF orientation is applied,
and transparent pixels are composited over white.

`svd_lab.images.to_display_image(matrix)` converts a finite two-dimensional
matrix back to an 8-bit Pillow grayscale image, clipping small numerical
overshoots to the displayable range.

## Local development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```powershell
uv sync
uv run streamlit run app.py
```

Run the quality checks with:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The first `uv sync` will create `uv.lock`. Commit that lockfile so local, CI,
and hosted environments resolve the same dependency versions.

## Deployment

The first public deployment target is Streamlit Community Cloud with `app.py`
as the entry point. The GitHub repository remains the source of truth.

## Uploaded images

The application processes uploaded bytes in memory and does not intentionally
persist them or use the supplied filename. Uploads are restricted by encoded
size, decoded pixel count, and detected image format before entering the
numerical pipeline.
