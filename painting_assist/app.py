from __future__ import annotations

"""Application bootstrap.

Importing :mod:`painting_assist.controls` runs the ``@register`` decorators in
each concrete control module, populating the registry *before*
``registry.create_all()`` is invoked inside :class:`MainWindow`.
"""

import os
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from PySide6.QtCore import QSettings

from painting_assist import controls  # noqa: F401  -- runs @register decorators
from painting_assist import theme
from painting_assist.main_window import _SETTINGS_APP, _SETTINGS_ORG, MainWindow
from painting_assist.widgets.settings_dialog import DEFAULT_THEME


def _app_icon() -> QIcon:
    """Load the bundled application icon from ``resources/icon.png``.

    Resolved relative to this file so it works both from a source checkout and a
    hatch-built wheel (``resources/`` ships as package data under
    ``painting_assist``). Returns an empty ``QIcon`` if the file is missing so a
    stray packaging issue never crashes startup.
    """
    path = os.path.join(os.path.dirname(__file__), "resources", "icon.png")
    return QIcon(path) if os.path.exists(path) else QIcon()


def main() -> int:
    """Create the application, show the main window, and run the event loop."""
    app = QApplication(sys.argv)
    app.setApplicationName("Painting Assist")
    app.setApplicationDisplayName("Painting Assist")

    # Apply the persisted theme (system/light/dark) before any window shows,
    # so the first paint is already in the right look.
    settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    mode = str(settings.value("settings/theme", DEFAULT_THEME))
    if mode not in theme.THEME_MODES:
        mode = DEFAULT_THEME
    theme.apply_theme(app, mode)

    icon = _app_icon()
    app.setWindowIcon(icon)
    win = MainWindow()
    win.setWindowIcon(icon)
    win.resize(1280, 800)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
