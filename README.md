# SVD Lens

An interactive visual explanation of singular value decomposition and low-rank
image approximation.

> Upload an image and drag rank from 1 upward to watch structure emerge while
> reconstruction error falls.

## Status

The core interactive path is working: visitors can begin with a generated
geometric sample or upload a PNG/JPEG, select a retained rank, compare the
processed original with its reconstruction, and inspect quality metrics and the
singular-value spectrum. A reproducible performance baseline now supports the
512-pixel processing cap. Visual refinement, additional samples, and richer
explanatory states remain before deployment.

## Current experience

Visitors can start with a zero-setup sample or upload a PNG or JPEG,
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
- `svd_lab/samples.py` supplies the deterministic zero-setup sample.
- `svd_lab/plots.py` creates reusable Plotly figures.
- `svd_lab/explanations.py` is reserved for richer plain-language interpretations.
- `benchmarks/svd_latency.py` measures the framework-independent numerical path.
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

## Performance

The 512-pixel longest-side limit is measurement-backed rather than arbitrary.
On the documented local baseline, a worst-case 512 × 512 matrix had SVD medians
of 187.6 ms and 194.3 ms across two trials; cached rank reconstructions remained
below 1 ms at both the initial and quarter-rank test points. Results vary by
machine, so hosted responsiveness will be rechecked during deployment.

See [the benchmark method and decision](docs/performance.md), or reproduce it:

```powershell
uv run python -m benchmarks.svd_latency --sizes 256 384 512 --repetitions 7
```

## Deployment

The first public deployment target is Streamlit Community Cloud with `app.py`
as the entry point. The GitHub repository remains the source of truth.

## Uploaded images

The application processes uploaded bytes in memory and does not intentionally
persist them or use the supplied filename. Uploads are restricted by encoded
size, decoded pixel count, and detected image format before entering the
numerical pipeline.
