"""Integration tests for the primary Streamlit interaction."""

from io import BytesIO
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

from svd_lab.explanations import reconstruction_caption

APP_PATH = Path(__file__).parents[1] / "app.py"


def png_bytes(size: tuple[int, int] = (3, 2)) -> bytes:
    image = Image.new("RGB", size, (30, 100, 200))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_app_renders_default_sample_analysis() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()

    assert not app.exception
    assert app.title[0].value == "SVD Lens"
    assert app.file_uploader[0].label == "Upload your own image"
    assert app.file_uploader[0].allowed_type == [".png", ".jpg", ".jpeg"]
    assert "512 px" in app.file_uploader[0].help
    assert app.selectbox[0].label == "Sample image"
    assert app.selectbox[0].options == ["Geometric study", "Soft bands", "Woven detail"]
    assert app.slider[0].label == "Retained rank"
    assert app.slider[0].min == 1
    assert app.slider[0].max == 180
    assert app.slider[0].value == 12
    assert [metric.label for metric in app.metric] == [
        "Rank",
        "Retained energy",
        "Relative error",
        "Representation ratio",
    ]
    assert app.metric[0].value == "12 / 180"
    assert len(app.get("plotly_chart")) == 1
    assert [heading.value for heading in app.subheader[:2]] == [
        "Processed original",
        "Rank-12 reconstruction",
    ]


def test_rank_slider_updates_the_analysis() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    initial_energy = app.metric[1].value
    initial_error = app.metric[2].value

    app.slider[0].set_value(1).run()

    assert not app.exception
    assert app.metric[0].value == "1 / 180"
    assert app.metric[1].value != initial_energy
    assert app.metric[2].value != initial_error
    assert app.subheader[1].value == "Rank-1 reconstruction"
    assert reconstruction_caption(1) == "Rebuilt from the leading singular component."


def test_valid_upload_replaces_the_default_sample() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()

    app.file_uploader[0].set_value(("portrait.png", png_bytes(), "image/png")).run()

    assert not app.exception
    assert app.slider[0].max == 2
    assert app.slider[0].value == 2
    assert any("PNG · 3 × 2 px" in caption.value for caption in app.caption)


def test_sample_selector_changes_the_source_image() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()

    app.selectbox[0].select("Soft bands").run()

    assert not app.exception
    assert app.selectbox[0].value == "Soft bands"
    assert any("Sample: Soft bands" in caption.value for caption in app.caption)


def test_invalid_upload_shows_a_safe_error_state() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()

    app.file_uploader[0].set_value(("broken.png", b"not an image", "image/png")).run()

    assert not app.exception
    assert app.error[0].value == "Upload must be a valid PNG or JPEG image."
    assert not app.slider
