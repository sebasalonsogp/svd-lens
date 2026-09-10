"""Tests for boundaries that keep the domain package UI-independent."""

from __future__ import annotations

import ast
from pathlib import Path

DOMAIN_PACKAGE = Path(__file__).parents[1] / "svd_lab"
FORBIDDEN_UI_PACKAGES = {"gradio", "streamlit"}


def imported_package_roots(path: Path) -> set[str]:
    """Return top-level package names imported by a Python source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.partition(".")[0])

    return roots


def test_domain_package_has_no_ui_framework_imports() -> None:
    violations: dict[str, set[str]] = {}

    for path in DOMAIN_PACKAGE.glob("*.py"):
        forbidden = imported_package_roots(path) & FORBIDDEN_UI_PACKAGES
        if forbidden:
            violations[path.name] = forbidden

    assert not violations, f"UI framework imports found in domain package: {violations}"
