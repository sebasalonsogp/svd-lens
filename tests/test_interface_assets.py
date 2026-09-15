from pathlib import Path
from tomllib import load

ROOT = Path(__file__).parents[1]


def test_theme_uses_the_project_visual_system() -> None:
    with (ROOT / ".streamlit" / "config.toml").open("rb") as config_file:
        theme = load(config_file)["theme"]

    assert theme["base"] == "dark"
    assert theme["primaryColor"] == "#2EC4A6"
    assert theme["linkColor"] == "#63E6CE"
    assert theme["borderColor"] == "#344840"
    assert theme["showWidgetBorder"] is True
    assert theme["baseRadius"] == "small"
    assert theme["buttonRadius"] == "small"
    assert theme["font"].startswith("'IBM Plex Sans':")
    assert theme["headingFont"].startswith("Newsreader:")
    assert theme["codeFont"].startswith("'IBM Plex Mono':")
    assert theme["baseFontSize"] == 16
    assert theme["baseFontWeight"] == 400
    assert theme["headingFontWeights"] == [600, 600, 600, 600, 600, 600]


def test_interface_styles_include_keyboard_and_mobile_rules() -> None:
    styles = (ROOT / "assets" / "styles.css").read_text(encoding="utf-8")

    assert ".skip-link:focus" in styles
    assert "min-height: 2.75rem" in styles
    assert "max-width: 72ch" in styles
    assert "text-wrap: balance" in styles
    assert "font-variant-numeric: tabular-nums" in styles
    assert "::selection" in styles
    assert "@media (max-width: 640px)" in styles
