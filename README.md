# SVD Lens

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://svd-lens.streamlit.app)
[![CI](https://github.com/sebasalonsogp/svd-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/sebasalonsogp/svd-lens/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-2EC4A6.svg)](LICENSE)

SVD Lens is an interactive explanation of **singular value decomposition
(SVD)** and low-rank image approximation. Choose a built-in experiment or
upload a photo, then move one slider to watch the image rebuild from broad
structure to fine detail.

This portfolio project was inspired by concepts I encountered in my **MTH 520** class.
It turns the underlying linear algebra into something visible and hands-on,
while demonstrating numerical programming, data visualization, input handling,
testing, performance measurement, and interface design.

**[Launch the live app](https://svd-lens.streamlit.app)**

## What the app does

A grayscale image can be treated as a spreadsheet of numbers: every pixel is a
cell whose value describes its brightness. SVD reorganizes that large grid into
an ordered collection of reusable image patterns.

The first patterns explain the strongest, broadest structure. Later patterns
add progressively smaller details such as edges and texture. **Rank** is simply
the number of those patterns kept when rebuilding the image:

- A low rank gives a rough, simplified image.
- A middle rank usually recovers the recognizable subject and major edges.
- A high rank adds fine detail and approaches the processed original.

The key idea is not that a single rank is always best. The app lets you see the
tradeoff: fewer patterns make a smaller mathematical representation, while more
patterns produce a closer reconstruction.

## Use it online

1. Open the [hosted app](https://svd-lens.streamlit.app).
2. Start with one of the three generated samples, or upload a PNG or JPEG.
3. Try the **Structure**, **Balanced**, and **Detail** buttons.
4. Drag **Rank — patterns kept** and compare the reconstruction with the
   processed original.
5. Turn on **Show difference map** to reveal where the reconstruction still
   differs.
6. Read the spectrum and metrics to connect what changed visually with what
   changed mathematically.

Uploaded images may be at most 10 MiB and 20 million decoded pixels. The app
resizes them to at most 512 pixels on the longest side, converts them to
grayscale, and processes them in memory. The application does not intentionally
save uploads or use their filenames. For a sensitive image, run the project
locally instead of sending it to a hosted service.

## Read the results

| Display | Plain-language meaning |
| --- | --- |
| **Rank** | The number of image-building patterns used in the reconstruction. The maximum is the smaller of the processed image's height and width. |
| **Retained energy** | The share of the matrix's squared strength captured by the selected patterns. A high value means the reconstruction preserves most mathematical variation, but it does not guarantee that every visually important edge is sharp. |
| **Relative error** | The overall difference between the reconstruction and processed original, scaled to the original. Lower is better; it reaches approximately zero at full rank. |
| **Representation ratio** | Original matrix values divided by values in the retained SVD factors. Above `1×` means the factors use fewer scalar values. This is an educational estimate, not a claim about PNG or JPEG file size. |
| **Difference map** | A visual map of the remaining pixel differences. Brighter areas differ more. It is normalized for visibility, so compare its shape rather than brightness between ranks. |

### How to read the singular-value spectrum

The spectrum orders all available patterns from strongest to weakest:

- Each point is one pattern, and its height shows its strength relative to the
  strongest pattern.
- Green points and the shaded region are included at the current rank.
- The gold dotted line marks the cutoff; points to its right are not yet used.
- A steep early drop means a few patterns carry most of the image's broad
  structure. A long, gradual tail means many smaller patterns contribute
  texture and fine detail.

Try rank 1 first, then move the slider slowly. Watch the reconstructed image,
the cutoff line, retained energy, and relative error together. This makes the
graph a map of *why* the picture improves as rank rises, rather than a score to
maximize on its own.

## How it works

After an image is oriented, resized, and converted to a normalized grayscale
matrix `A`, NumPy computes its compact decomposition:

```text
A = U Σ Vᵀ
```

The singular values on the diagonal of `Σ` rank the components by strength. A
rank-`k` reconstruction keeps only the first `k` components:

```text
Aₖ = Uₖ Σₖ Vₖᵀ
```

This reconstruction is the closest rank-`k` approximation to `A` under the
Frobenius norm. In practical terms, no other rank-`k` matrix has a smaller
overall squared pixel error. The app computes the decomposition once per image,
caches it, and reuses the factors while the rank slider changes.

SVD Lens deliberately uses one grayscale matrix instead of decomposing three
color channels. That keeps the mathematical story direct and the prototype
responsive.

## Technical design

| Area | Choice | Role |
| --- | --- | --- |
| Interface | Streamlit | Uploads, controls, layout, session state, and caching |
| Numerical work | NumPy | Compact SVD, reconstruction, and matrix metrics |
| Image handling | Pillow | Validation, EXIF orientation, transparency, resizing, and grayscale conversion |
| Visualization | Plotly | Interactive singular-value spectrum |
| Quality | pytest and Ruff | Behavioral tests, architecture checks, linting, and formatting |
| Environment | uv | Python and dependency locking for local development and CI |
| Delivery | GitHub Actions and Streamlit Community Cloud | Automated quality gate and hosted app |

The code keeps the web framework at the edge:

```text
Browser
  └─ app.py                    Streamlit UI and orchestration
      ├─ svd_lab/images.py     Validate and prepare uploaded images
      ├─ svd_lab/samples.py    Generate the zero-setup sample catalog
      ├─ svd_lab/svd.py        Decompose, reconstruct, and calculate metrics
      ├─ svd_lab/plots.py      Build the Plotly spectrum
      └─ svd_lab/explanations.py
                               Rank presets and contextual plain-language copy
```

`svd_lab` has no Streamlit dependency. The numerical and presentation logic can
therefore be tested without a browser and reused by another interface, such as
Gradio, without rewriting the core.

Additional repository pieces are intentionally small:

- `assets/styles.css` provides the responsive layout and keyboard-focus layer.
- `benchmarks/svd_latency.py` measures the UI-independent numerical path.
- `docs/performance.md` records the measurement method, results, and image-size
  decision.
- `tests/` covers numerical behavior, image safety, samples, plotting,
  explanations, the app shell, and framework boundaries.
- `.github/workflows/ci.yml` runs linting, formatting checks, and the full test
  suite on every push and pull request.

## Run locally

### Requirements

- Git
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

The project targets Python 3.12 or 3.13. `uv` can install the pinned Python
version and all project dependencies for you.

### Start the app

```powershell
git clone https://github.com/sebasalonsogp/svd-lens.git
cd svd-lens
uv sync
uv run streamlit run app.py
```

Streamlit will print a local address, normally
[`http://localhost:8501`](http://localhost:8501). Open it in a browser, use a
built-in sample first, then upload your own PNG or JPEG. Press `Ctrl+C` in the
terminal when you are finished.

`uv.lock` is committed, so local development, CI, and the hosted app resolve
the same dependency set.

### Run the quality checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

### Reproduce the performance benchmark

```powershell
uv run python -m benchmarks.svd_latency --sizes 256 384 512 --repetitions 7
```

The measured local baseline and the reasoning behind the 512-pixel limit are in
[`docs/performance.md`](docs/performance.md).

## Deployment

The public app is deployed from the `main` branch on Streamlit Community Cloud
with `app.py` as its entry point and Python 3.12 as its runtime. Community Cloud
uses the committed `uv.lock` to install dependencies. GitHub Actions tests every
push, while Community Cloud watches `main` and updates the hosted app
automatically.

No secrets, database, user accounts, or external API keys are required.

## Project scope

SVD Lens is a focused educational portfolio prototype. It demonstrates the
mathematics and implementation of low-rank approximation; it is not a
production image compressor or a substitute for codecs such as JPEG and WebP.
It intentionally excludes color-channel decomposition, video, arbitrary matrix
editing, saved sessions, authentication, and large-scale image processing.

## License

Released under the [MIT License](LICENSE).
