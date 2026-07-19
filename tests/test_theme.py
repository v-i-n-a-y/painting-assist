# Copyright 2026 Vinay Williams

"""Theme module: pure parts only.

resolve_mode mappings, THEME_MODES contents, and palette construction. QPalette
can be built without a QApplication, so these run headless. Anything needing a
real QGuiApplication (a platform plugin) is out of scope here — see
tests/conftest.py for how the suite treats headless limits.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette

from painting_assist import theme


# ---- THEME_MODES / labels ----
def test_theme_modes_contents():
    assert theme.THEME_MODES == ("system", "light", "dark")
    for mode in theme.THEME_MODES:
        assert mode in theme.THEME_LABELS
        assert isinstance(theme.THEME_LABELS[mode], str)


# ---- resolve_mode ----
def test_resolve_fixed_modes_pass_through():
    # Fixed modes ignore the scheme entirely.
    assert theme.resolve_mode("light", None) == "light"
    assert theme.resolve_mode("dark", None) == "dark"
    assert theme.resolve_mode("light", "dark") == "light"
    assert theme.resolve_mode("dark", "light") == "dark"


def test_resolve_system_from_enum():
    assert theme.resolve_mode("system", Qt.ColorScheme.Dark) == "dark"
    assert theme.resolve_mode("system", Qt.ColorScheme.Light) == "light"
    assert theme.resolve_mode("system", Qt.ColorScheme.Unknown) == "light"


def test_resolve_system_from_string():
    assert theme.resolve_mode("system", "dark") == "dark"
    assert theme.resolve_mode("system", "light") == "light"
    assert theme.resolve_mode("system", "unknown") == "light"
    assert theme.resolve_mode("system", None) == "light"


def test_resolve_unknown_mode_defaults_like_system():
    # Any unexpected mode is treated as system-derived.
    assert theme.resolve_mode("weird", "dark") == "dark"
    assert theme.resolve_mode("weird", "light") == "light"


# ---- palette construction ----
def _has_role(p, group, role, expected_hex):
    return p.color(group, role).name().lower() == expected_hex.lower()


def test_dark_palette_key_colours():
    p = theme.dark_palette()
    assert isinstance(p, QPalette)
    assert _has_role(
        p, QPalette.ColorGroup.Active, QPalette.ColorRole.Window, "#2b2b2b"
    )
    assert _has_role(p, QPalette.ColorGroup.Active, QPalette.ColorRole.Base, "#1e1e1e")
    assert _has_role(
        p, QPalette.ColorGroup.Active, QPalette.ColorRole.Highlight, "#3874f2"
    )
    # Text is light on dark.
    assert p.color(QPalette.ColorRole.Text).lightness() > 180
    # Disabled group is populated (the common miss).
    assert _has_role(
        p, QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, "#6e6e6e"
    )


def test_light_palette_key_colours():
    p = theme.light_palette()
    assert isinstance(p, QPalette)
    assert _has_role(p, QPalette.ColorGroup.Active, QPalette.ColorRole.Base, "#ffffff")
    # Text is dark on light.
    assert p.color(QPalette.ColorRole.Text).lightness() < 80
    assert _has_role(
        p, QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, "#a0a0a0"
    )


def test_build_palette_dispatch():
    assert (
        theme.build_palette("dark").color(QPalette.ColorRole.Window).name().lower()
        == "#2b2b2b"
    )
    assert (
        theme.build_palette("light").color(QPalette.ColorRole.Base).name().lower()
        == "#ffffff"
    )
    # Non-"dark" falls back to light.
    assert (
        theme.build_palette("anything").color(QPalette.ColorRole.Base).name().lower()
        == "#ffffff"
    )


def test_palettes_are_fresh_each_call():
    # Not cached: mutating one must not affect the next construction.
    p1 = theme.dark_palette()
    p1.setColor(QPalette.ColorRole.Window, QPalette().color(QPalette.ColorRole.Window))
    p2 = theme.dark_palette()
    assert p2.color(QPalette.ColorRole.Window).name().lower() == "#2b2b2b"


# ---- dock chrome stylesheet ----
def test_dock_chrome_qss_targets_dock_chrome_only():
    # Both themes style the separators and the dock title, and nothing broader
    # (so the palette-driven look of other widgets is left alone).
    for mode in ("light", "dark"):
        qss = theme.dock_chrome_qss(mode)
        assert isinstance(qss, str) and qss
        assert "QMainWindow::separator" in qss
        assert "QDockWidget::title" in qss


def test_dock_chrome_qss_differs_by_theme():
    # The line/title colours are theme-specific, so the two stylesheets differ.
    assert theme.dock_chrome_qss("light") != theme.dock_chrome_qss("dark")
    # Anything not "dark" falls back to the light chrome.
    assert theme.dock_chrome_qss("anything") == theme.dock_chrome_qss("light")


def test_all_dark_roles_present():
    # Guard against a partial palette: every role we promise is set and the
    # disabled group covers the common-miss roles.
    p = theme.dark_palette()
    roles = [
        QPalette.ColorRole.Window,
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Base,
        QPalette.ColorRole.AlternateBase,
        QPalette.ColorRole.ToolTipBase,
        QPalette.ColorRole.ToolTipText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.PlaceholderText,
        QPalette.ColorRole.Button,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.BrightText,
        QPalette.ColorRole.Highlight,
        QPalette.ColorRole.HighlightedText,
        QPalette.ColorRole.Link,
    ]
    for role in roles:
        assert p.color(QPalette.ColorGroup.Active, role).isValid()
    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Highlight,
    ):
        assert p.color(QPalette.ColorGroup.Disabled, role).isValid()
