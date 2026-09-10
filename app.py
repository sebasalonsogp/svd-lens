"""Streamlit entry point for SVD Lens.

Keep framework-specific rendering, widget state, and caching in this module.
The reusable numerical and presentation logic belongs in ``svd_lab``.
"""

from __future__ import annotations

import streamlit as st

from svd_lab import PRODUCT_NAME


def main() -> None:
    """Render the application shell."""
    st.set_page_config(
        page_title=PRODUCT_NAME,
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title(PRODUCT_NAME)
    st.caption("See how singular value decomposition rebuilds an image.")

    st.info(
        "The project scaffold is ready. The first vertical slice will add a "
        "sample image, upload control, rank reconstruction, metrics, and a "
        "singular-value plot."
    )

    with st.expander("Planned interaction"):
        st.markdown(
            """
            1. Start with a curated image or upload a PNG or JPEG.
            2. Move the rank slider to rebuild the image.
            3. Compare the result with the original.
            4. Inspect retained energy, reconstruction error, and estimated
               representation size.
            """
        )


if __name__ == "__main__":
    main()
