"""Analyst Light appearance: presentation only, never security behavior.

These lock the appearance ARCHITECTURE (style vs color scheme) and that Light is
a token override — not a redesign — while Classic stays dark and defaults hold.
"""
from pathlib import Path

from app import app

BASE_CSS = Path('static/css/base.css').read_text()
ORION_JS = Path('static/js/orion.js').read_text()


def _page():
    with app.test_client() as c:
        return c.get('/').data.decode()


# --------------------------- presets & selector ---------------------------- #
def test_appearance_selector_exposes_three_presets_plus_system():
    html = _page()
    block = html.split('id="orion-appearance"')[1].split('</select>')[0]
    for value in ('analyst-dark', 'analyst-light', 'classic', 'analyst-system'):
        assert f'value="{value}"' in block
    for label in ('Analyst Dark', 'Analyst Light', 'Classic CRT'):
        assert label in block
    assert 'Classic Light' not in block            # nonsensical combo not exposed


# --------------------------- no flash on load ------------------------------- #
def test_inline_init_resolves_scheme_before_paint():
    html = _page()
    head = html.split('</head>')[0]
    assert 'data-orion-color-scheme' in head            # set early, in <head>
    assert "localStorage.getItem('orion.colorScheme')" in head
    assert 'prefers-color-scheme: light' in head        # resolves system scheme


# --------------------- color scheme is a token override --------------------- #
def test_light_scheme_overrides_tokens_only():
    assert '[data-orion-color-scheme="light"] {' in BASE_CSS
    light = BASE_CSS.split('[data-orion-color-scheme="light"] {')[1].split('}')[0]
    for token in ('--surface-page', '--text-primary', '--orion-crimson',
                  '--status-unknown', '--action-primary'):
        assert token in light
    assert 'color-scheme: light' in light
    # No pure white as the primary page surface.
    assert '#ffffff' not in light and '#fff;' not in light


def test_dark_remains_default_and_declares_color_scheme():
    root = BASE_CSS.split(':root {')[1].split('\n}')[0]
    assert 'color-scheme: dark' in root
    # Server renders Analyst (dark) by default; backward-compatible.
    assert 'data-orion-appearance="analyst"' in _page()


# ------------------------------ classic intact ------------------------------ #
def test_classic_stays_dark():
    assert 'body[data-orion-appearance="classic"] { --surface-page: #080b0f; }' in BASE_CSS
    # Classic preset maps to dark in the appearance controller.
    assert "appearance = \"classic\"; colorScheme = \"dark\"" in ORION_JS


# ----------------------- tokenized theme-specific colors -------------------- #
def test_offensive_defensive_colors_are_tokenized():
    assert 'border-color: #16407a' not in BASE_CSS          # was dark-only hard-coded
    assert '--border-defensive' in BASE_CSS and '--border-offensive' in BASE_CSS
    assert '.btn-ghost:hover { background: var(--surface-hover); }' in BASE_CSS


# --------------------------- persistence model ------------------------------ #
def test_persistence_separates_appearance_and_color_scheme():
    assert '"orion.appearance"' in ORION_JS and '"orion.colorScheme"' in ORION_JS
    # Backward compatibility: existing classic preference is still honored.
    assert 'read("orion.appearance", "analyst") === "classic"' in ORION_JS
