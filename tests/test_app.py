"""Smoke test: every Streamlit page renders without an exception and shows the required caveats."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(scope="module")
def app():
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    return at


def _all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [w.value for w in at.warning] + [i.value for i in at.info] + [c.value for c in at.caption]
    return "\n".join(str(p) for p in parts)


def test_pages_render_without_errors(app):
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page


def test_capital_page_shows_caveats(app):
    pages = app.sidebar.radio[0].options
    app.sidebar.radio[0].set_value(pages[2]).run()
    text = _all_text(app)
    assert "One-month target" in text
    assert "illustrative" in " ".join(h.value for h in app.header).lower()
    assert "scenario sensitivity, not a calibrated model" in " ".join(h.value for h in app.header)


def test_validation_page_shows_psi_caveat(app):
    pages = app.sidebar.radio[0].options
    app.sidebar.radio[0].set_value(pages[1]).run()
    assert any("not an out-of-time sample" in w.value for w in app.warning)
