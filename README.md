# SVD Lens

An interactive visual explanation of singular value decomposition and low-rank
image approximation.

> Upload an image and drag rank from 1 upward to watch structure emerge while
> reconstruction error falls.

## Status

The project architecture and application shell are scaffolded. The first
vertical slice will add image preparation, SVD computation, rank-based
reconstruction, metrics, and the singular-value visualization.

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

The finished application will process uploaded images temporarily for the
active session and will not intentionally persist them.
