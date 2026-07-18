from __future__ import annotations

"""Application theming: system / light / dark.

The palettes are hand-tuned and applied over the Fusion style so the look is
identical across platforms and fully controllable. Everything is idempotent:
calling :func:`apply_theme` again with a different mode rebuilds the palette
from scratch, so switching light <-> dark at runtime fully restores the other
look (we never cache the startup palette).

Live OS-theme following: when the mode is "system" we subscribe to
``QStyleHints.colorSchemeChanged`` so the app tracks the OS flipping between
light and dark. Applying any fixed mode tears that subscription down again. The
connection is held in a module-level slot so there is only ever one.
"""

from typing import Optional, Union

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette

# Public mode identifiers and their human-readable labels.
THEME_MODES = ("system", "light", "dark")

THEME_LABELS = {
    "system": "System",
    "light": "Light",
    "dark": "Dark",
}

# Holds the active (application, slot) pair for the live "system" subscription,
# or None when no fixed/None subscription is active. Kept at module scope so a
# single connection exists regardless of how often apply_theme is called.
_system_connection: Optional[tuple] = None


def resolve_mode(mode: str, scheme: Union["Qt.ColorScheme", str, None]) -> str:
    """Resolve a requested *mode* to a concrete "light" or "dark".

    "light"/"dark" pass straight through. "system" (or anything else) consults
    *scheme*, which may be a ``Qt.ColorScheme`` enum member or the string
    "dark"/"light"/"unknown". Anything that is not clearly dark resolves to
    "light" (unknown -> light).
    """
    if mode == "light":
        return "light"
    if mode == "dark":
        return "dark"

    # "system" (or an unexpected value): derive from the OS colour scheme.
    if scheme == Qt.ColorScheme.Dark or scheme == "dark":
        return "dark"
    return "light"


def system_scheme() -> str:
    """Return the OS colour scheme as "light"/"dark"/"unknown".

    Guards against there being no application instance yet (returns "unknown"),
    which is also what we treat as "not clearly dark" in :func:`resolve_mode`.
    """
    app = QGuiApplication.instance()
    if app is None:
        return "unknown"
    scheme = app.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return "dark"
    if scheme == Qt.ColorScheme.Light:
        return "light"
    return "unknown"


def light_palette() -> QPalette:
    """A cleaned-up light palette (explicit, never the cached startup one)."""
    p = QPalette()

    window = QColor("#f0f0f0")
    base = QColor("#ffffff")
    alt_base = QColor("#e9e9e9")
    text = QColor("#1a1a1a")
    button = QColor("#f0f0f0")
    highlight = QColor("#3874f2")
    disabled_text = QColor("#a0a0a0")

    p.setColor(QPalette.ColorRole.Window, window)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, alt_base)
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor("#ffffdc"))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor("#1a1a1a"))
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor("#8a8a8a"))
    p.setColor(QPalette.ColorRole.Button, button)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.Highlight, highlight)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.Link, QColor("#2a5fd0"))

    # Disabled group (a very common miss).
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor("#c8c8c8"))

    return p


def dark_palette() -> QPalette:
    """A proper dark palette covering every role plus the Disabled group."""
    p = QPalette()

    window = QColor("#2b2b2b")
    base = QColor("#1e1e1e")
    alt_base = QColor("#323232")
    text = QColor("#e6e6e6")
    button = QColor("#3a3a3a")
    highlight = QColor("#3874f2")
    disabled_text = QColor("#6e6e6e")

    p.setColor(QPalette.ColorRole.Window, window)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, alt_base)
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor("#3a3a3a"))
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor("#8a8a8a"))
    p.setColor(QPalette.ColorRole.Button, button)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor("#ff5252"))
    p.setColor(QPalette.ColorRole.Highlight, highlight)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.Link, QColor("#5a9bff"))

    # Disabled group (a very common miss).
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor("#464646"))

    return p


def build_palette(theme: str) -> QPalette:
    """Return the QPalette for a concrete theme ("light" or "dark")."""
    return dark_palette() if theme == "dark" else light_palette()


def _disconnect_system() -> None:
    """Tear down any live "system" colour-scheme subscription."""
    global _system_connection
    if _system_connection is None:
        return
    app, slot = _system_connection
    try:
        app.styleHints().colorSchemeChanged.disconnect(slot)
    except (RuntimeError, TypeError):
        # Already disconnected or the app/hints went away; nothing to do.
        pass
    _system_connection = None


def _connect_system(app: QGuiApplication) -> None:
    """Subscribe to OS colour-scheme changes so "system" mode tracks them.

    Guarded for Qt < 6.5 (no colorSchemeChanged signal). Replaces any existing
    subscription so only one connection is ever live.
    """
    global _system_connection
    _disconnect_system()

    style_hints = app.styleHints()
    signal = getattr(style_hints, "colorSchemeChanged", None)
    if signal is None:
        return

    def _on_scheme_changed(scheme):
        app.setPalette(build_palette(resolve_mode("system", scheme)))

    signal.connect(_on_scheme_changed)
    _system_connection = (app, _on_scheme_changed)


def apply_theme(app: QGuiApplication, mode: str) -> str:
    """Apply *mode* ("system"/"light"/"dark") to *app*; return concrete theme.

    Sets the Fusion style and a hand-tuned palette. Idempotent and safe to call
    repeatedly at runtime to switch looks. When *mode* is "system" the app also
    starts following live OS theme flips; any fixed mode removes that.
    """
    # Fusion gives us a consistent, fully palette-driven look across platforms.
    # Only QApplication/QStyleApplication expose setStyle; guard for QGuiApplication.
    if hasattr(app, "setStyle"):
        app.setStyle("Fusion")

    if mode == "system":
        _connect_system(app)
        theme = resolve_mode("system", system_scheme())
    else:
        _disconnect_system()
        theme = resolve_mode(mode, None)

    app.setPalette(build_palette(theme))
    return theme


def follow_system(app: QGuiApplication):
    """Start following live OS theme changes and return a disconnect callable.

    A thin wrapper over the internal subscription for callers that want to
    manage the connection explicitly. Calling the returned function tears the
    subscription down.
    """
    _connect_system(app)
    return _disconnect_system
