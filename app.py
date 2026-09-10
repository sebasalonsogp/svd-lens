"""Streamlit entry point for SVD Lens.

Keep framework-specific rendering, widget state, and caching in this module.
The reusable numerical and presentation logic belongs in ``svd_lab``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from svd_lab import PRODUCT_NAME
from svd_lab.explanations import rank_presets, reconstruction_caption, reconstruction_insight
from svd_lab.images import ImageValidationError, prepare_image, to_display_image
from svd_lab.plots import singular_value_figure
from svd_lab.samples import available_samples
from svd_lab.svd import SVDResult, decompose, metrics_for, reconstruct

STYLE_PATH = Path(__file__).parent / "assets" / "styles.css"


# Streamlit recommends st.cache_data for serializable computational results,
# including NumPy transformations. Source:
# https://docs.streamlit.io/develop/concepts/architecture/caching
@st.cache_data(show_spinner=False)
def cached_decomposition(matrix: np.ndarray) -> SVDResult:
    """Reuse an image decomposition when a widget reruns the app."""
    return decompose(matrix)


def choose_rank(rank: int) -> None:
    """Apply a rank preset before Streamlit recreates the slider."""
    st.session_state.rank = rank


def main() -> None:
    """Render the first complete SVD exploration flow."""
    st.set_page_config(
        page_title=PRODUCT_NAME,
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.html(STYLE_PATH)
    st.markdown(
        '<a class="skip-link" href="#choose-an-image">Skip to image controls</a>',
        unsafe_allow_html=True,
    )

    st.caption("INTERACTIVE MATRIX LAB")
    st.title(PRODUCT_NAME)
    st.markdown(
        "See singular value decomposition rebuild an image, one rank at a time. "
        "Choose an experiment or bring your own image, then watch structure return as error falls."
    )
    st.header("Choose an image", divider="gray")

    # The widget limit complements the stricter byte and decoded-pixel checks in
    # svd_lab.images. Source:
    # https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
    samples = {sample.name: sample for sample in available_samples()}
    sample_column, upload_column = st.columns([1, 1.35], gap="large")
    with sample_column:
        sample_name = st.selectbox(
            "Sample image",
            options=tuple(samples),
            help="Each sample emphasizes a different singular-value pattern.",
        )
    with upload_column:
        upload = st.file_uploader(
            "Upload your own image",
            type=["png", "jpg", "jpeg"],
            max_upload_size=10,
            help=(
                "PNG or JPEG, up to 10 MiB. Images are processed in memory, resized to "
                "at most 512 px on the longest side, and not saved."
            ),
        )

    if upload is None:
        sample = samples[sample_name]
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

    st.header("Set the rank", divider="gray")
    st.caption("Use a preset or tune the retained components one at a time.")
    initial_rank = min(12, decomposition.max_rank)
    if "rank" not in st.session_state:
        st.session_state.rank = initial_rank
    elif st.session_state.rank > decomposition.max_rank:
        st.session_state.rank = decomposition.max_rank

    presets = rank_presets(decomposition.max_rank)
    preset_columns = st.columns(len(presets), gap="small")
    for column, preset in zip(preset_columns, presets, strict=True):
        column.button(
            f"{preset.label} · rank {preset.rank}",
            on_click=choose_rank,
            args=(preset.rank,),
            use_container_width=True,
        )

    rank = st.slider(
        "Retained rank",
        min_value=1,
        max_value=decomposition.max_rank,
        help="The number of singular components used to rebuild the image.",
        key="rank",
    )
    reconstruction = reconstruct(decomposition, rank)
    metrics = metrics_for(decomposition, rank)
    insight = reconstruction_insight(metrics)

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

    st.header("Compare", divider="gray")
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
            caption=reconstruction_caption(rank),
            width="stretch",
        )

    show_difference = st.toggle(
        "Show difference map",
        help="Reveal where the current reconstruction differs from the processed original.",
    )
    if show_difference:
        difference = np.abs(matrix - reconstruction)
        peak_difference = float(np.max(difference))
        visible_difference = difference / peak_difference if peak_difference > 0.0 else difference
        st.subheader("Absolute difference")
        st.image(
            to_display_image(visible_difference),
            caption=(
                "Brighter pixels differ more. Values are scaled to the current maximum "
                f"absolute difference of {peak_difference:.3f}."
            ),
            width="stretch",
        )

    st.header("Read the spectrum", divider="gray")
    st.caption(
        "Tall singular values carry the strongest matrix patterns. The highlighted prefix "
        "is included in the current reconstruction."
    )
    st.plotly_chart(
        singular_value_figure(decomposition, rank),
        width="stretch",
        config={"displayModeBar": False},
        key="singular-value-spectrum",
    )

    st.subheader(insight.title)
    if metrics.retained_energy is None:
        st.info(insight.body)
    else:
        st.write(insight.body)
        st.markdown(
            f"At **rank {rank}**, the reconstruction retains "
            f"**{metrics.retained_energy:.1%}** of the matrix energy while using "
            f"**{metrics.factorized_value_count:,}** factor values instead of "
            f"**{metrics.original_value_count:,}** original matrix values."
        )

    with st.expander("How the reconstruction works"):
        st.latex(r"A_k = U_k\,\Sigma_k\,V_k^T")
        st.markdown(
            "SVD orders image patterns by their singular values. Keeping the first "
            "**k** patterns gives the best rank-**k** approximation under the Frobenius norm."
        )

    st.divider()
    st.caption("Built with NumPy, Plotly, and Streamlit · Images are processed in memory.")


if __name__ == "__main__":
    main()
