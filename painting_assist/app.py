from __future__ import annotations

"""Application bootstrap.

Importing :mod:`painting_assist.controls` runs the ``@register`` decorators in
each concrete control module, populating the registry *before*
``registry.create_all()`` is invoked inside :class:`MainWindow`.
"""

import sys

from PySide6.QtWidgets import QApplication

from painting_assist import controls  # noqa: F401  -- runs @register decorators
from painting_assist.main_window import MainWindow


def main() -> int:
    """Create the application, show the main window, and run the event loop."""
    app = QApplication(sys.argv)
    app.setApplicationName("Squint")
    app.setApplicationDisplayName("Squint")
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
