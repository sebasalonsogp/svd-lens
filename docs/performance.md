# Performance baseline

SVD Lens keeps the 512-pixel longest-side processing limit. At that size, the
local worst-case square matrix stayed within the prototype's latency and memory
budgets while preserving enough detail for side-by-side visual comparison.

## Decision

The working limits are:

- 512 pixels on the longest processed side
- 10 MiB of encoded upload data
- 20 million decoded pixels before resizing

The byte and decoded-pixel limits protect the upload boundary. The 512-pixel
limit bounds the numerical work. They address different risks and should remain
independent.

For the local numerical path, the working budgets are a median below 250 ms for
the initial SVD and below 10 ms for reconstruction after a rank change. These
are development targets, not CI assertions: wall-clock checks on shared runners
would be noisy and flaky.

## Method

Measurements were taken on September 10, 2026, on Windows 11 (AMD64, 20 logical
processors visible) with Python 3.12.14 and NumPy 2.5.3. The benchmark uses
deterministic square `float64` matrices because a square image is the most
expensive shape allowed by a longest-side cap.

Each primary trial ran seven repetitions with seed `20260910`. The table reports
the median from two separate trials. Rank 12 matches the app's initial slider
position; quarter-rank is a heavier slider update.

| Side | SVD medians | Rank-12 medians | Quarter-rank medians | Full SVD factors |
| ---: | ---: | ---: | ---: | ---: |
| 256 px | 15.7 / 16.8 ms | 0.1 / 0.1 ms | 0.2 / 0.2 ms | 1.00 MiB |
| 384 px | 95.2 / 79.1 ms | 0.3 / 0.3 ms | 0.4 / 0.4 ms | 2.25 MiB |
| 512 px | 187.6 / 194.3 ms | 0.5 / 0.5 ms | 0.9 / 0.9 ms | 4.00 MiB |

The factor figure is the combined NumPy array storage for `U`, the singular
values, and `Vt`; it is not peak process memory.

A separate five-repetition scaling probe measured a 768-pixel SVD at 257.4 ms
and a 1024-pixel SVD at 619.3 ms. The 1024-pixel factors used 16.01 MiB—four
times the 512-pixel factor storage—for limited value in the app's displayed
image area. The extra cost and reduced hosting headroom do not fit this
portfolio prototype's scope, so the larger caps were rejected.

## Reproduce the measurement

From the repository root:

```powershell
uv run python -m benchmarks.svd_latency --sizes 256 384 512 --repetitions 7
```

The command prints the runtime environment and a Markdown results table. Its
automated tests verify the benchmark interface but deliberately avoid fixed
timing assertions.

## Hosted verification

The initial Streamlit Community Cloud deployment was smoke-tested on September
10, 2026. The generated sample loaded successfully, the rank-1 preset updated
the reconstruction, metrics, interpretation, and spectrum, and the difference
view rendered without a memory-related restart.

That smoke test confirms the deployed interaction path but is not a repeatable
end-to-end timing benchmark. For a future quantitative hosted check, repeat five
uncached uploads of a 512-pixel square image and five rank changes. The working
targets remain:

- first uncached result visible within 2.5 seconds;
- rank-change result visible within 250 ms;
- no memory-related restart during the sequence.

If the hosted target misses consistently, first lower the working cap to 384
pixels and remeasure. Algorithm changes or additional infrastructure are out of
scope unless that simpler adjustment fails.
