"""Streamlit entry point for SVD Lens.

Keep framework-specific rendering, widget state, and caching in this module.
The reusable numerical and presentation logic belongs in ``svd_lab``.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from svd_lab import PRODUCT_NAME
from svd_lab.images import ImageValidationError, prepare_image, to_display_image
from svd_lab.plots import singular_value_figure
from svd_lab.samples import default_sample
from svd_lab.svd import SVDResult, decompose, metrics_for, reconstruct


# Streamlit recommends st.cache_data for serializable computational results,
# including NumPy transformations. Source:
# https://docs.streamlit.io/develop/concepts/architecture/caching
@st.cache_data(show_spinner=False)
def cached_decomposition(matrix: np.ndarray) -> SVDResult:
    """Reuse an image decomposition when a widget reruns the app."""
    return decompose(matrix)


def main() -> None:
    """Render the first complete SVD exploration flow."""
    st.set_page_config(
        page_title=PRODUCT_NAME,
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title(PRODUCT_NAME)
    st.caption("See how singular value decomposition rebuilds an image, one rank at a time.")
    st.markdown(
        "Start with the geometric sample below or upload your own image. "
        "Move the rank slider and watch visual detail return as error falls."
    )

    # The widget limit complements the stricter byte and decoded-pixel checks in
    # svd_lab.images. Source:
    # https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
    upload = st.file_uploader(
        "Upload your own image",
        type=["png", "jpg", "jpeg"],
        max_upload_size=10,
        help="PNG or JPEG, up to 10 MiB. Images are processed in memory and not saved.",
    )

    if upload is None:
        sample = default_sample()
        matrix = sample.matrix
        source_caption = (
            f"Sample: {sample.name} · {matrix.shape[1]} × {matrix.shape[0]} px — "
            f"{sample.description}"
        )
    else:
        try:
            prepared = prepare_image(upload.getvalue())
        except ImageValidationError as error:
            st.error(str(error), icon=":material/error:")
            st.stop()
        matrix = prepared.matrix
        source_caption = (
            f"Uploaded {prepared.source_format} · {prepared.processed_size[0]} × "
            f"{prepared.processed_size[1]} px"
        )

    st.caption(source_caption)

    with st.spinner("Computing the singular value decomposition…"):
        decomposition = cached_decomposition(matrix)

    initial_rank = min(12, decomposition.max_rank)
    rank = st.slider(
        "Retained rank",
        min_value=1,
        max_value=decomposition.max_rank,
        value=initial_rank,
        help="The number of singular components used to rebuild the image.",
        key="rank",
    )
    reconstruction = reconstruct(decomposition, rank)
    metrics = metrics_for(decomposition, rank)

    metric_columns = st.columns(4, gap="small")
    metric_columns[0].metric("Rank", f"{rank} / {decomposition.max_rank}", border=True)
    retained_energy = "—" if metrics.retained_energy is None else f"{metrics.retained_energy:.1%}"
    metric_columns[1].metric("Retained energy", retained_energy, border=True)
    metric_columns[2].metric("Relative error", f"{metrics.relative_error:.1%}", border=True)
    metric_columns[3].metric(
        "Representation ratio",
        f"{metrics.representation_ratio:.2f}×",
        help="Original matrix values divided by values in the rank-k factors.",
        border=True,
    )

    original_column, reconstruction_column = st.columns(2, gap="medium")
    with original_column:
        st.subheader("Processed original")
        st.image(
            to_display_image(matrix),
            caption="The grayscale matrix used for decomposition.",
            width="stretch",
        )
    with reconstruction_column:
        st.subheader(f"Rank-{rank} reconstruction")
        st.image(
            to_display_image(reconstruction),
            caption=f"Rebuilt from the first {rank} singular components.",
            width="stretch",
        )

    st.plotly_chart(
        singular_value_figure(decomposition, rank),
        width="stretch",
        config={"displayModeBar": False},
        key="singular-value-spectrum",
    )

    if metrics.retained_energy is None:
        st.info("This image has no variation, so every rank reconstructs it exactly.")
    else:
        st.markdown(
            f"At **rank {rank}**, the reconstruction retains "
            f"**{metrics.retained_energy:.1%}** of the matrix energy while using "
            f"**{metrics.factorized_value_count:,}** factor values instead of "
            f"**{metrics.original_value_count:,}** original matrix values."
        )


if __name__ == "__main__":
    main()
