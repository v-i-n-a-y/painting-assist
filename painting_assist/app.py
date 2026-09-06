# Copyright 2026 Vinay Williams

"""Application bootstrap.

Importing :mod:`painting_assist.controls` runs the ``@register`` decorators in
each concrete control module, populating the registry *before*
``registry.create_all()`` is invoked inside :class:`MainWindow`.
"""

from __future__ import annotations

import logging
import os
import platform
import sys

from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from PySide6.QtCore import (
    QSettings,
    QTimer,
    QUrl,
    qInstallMessageHandler,
    QtMsgType,
)

from painting_assist import __version__
from painting_assist import controls  # noqa: F401  -- runs @register decorators
from painting_assist import bug_report, logging_setup, theme
from painting_assist.main_window import (
    _SETTINGS_APP,
    _SETTINGS_ORG,
    _app_data_dir,
    MainWindow,
)
from painting_assist.settings_store import SettingsStore
from painting_assist.widgets.settings_dialog import DEFAULT_THEME

_log = logging.getLogger("painting_assist.app")

# Map Qt's own diagnostic severities onto the stdlib logging levels so framework
# warnings (missing fonts, layout complaints, etc.) land in the same log file.
_QT_LEVELS = {
    QtMsgType.QtDebugMsg: logging.DEBUG,
    QtMsgType.QtInfoMsg: logging.INFO,
    QtMsgType.QtWarningMsg: logging.WARNING,
    QtMsgType.QtCriticalMsg: logging.ERROR,
    QtMsgType.QtFatalMsg: logging.CRITICAL,
}


def _qt_message_handler(mode, context, message: str) -> None:
    """Route Qt's internal messages into the ``qt`` logger."""
    logging.getLogger("qt").log(_QT_LEVELS.get(mode, logging.INFO), "%s", message)


def _error_notifier(msg: str) -> None:
    """Show a non-fatal error dialog after an uncaught exception was logged.

    Called from the logging excepthook, so it defers the dialog onto the Qt
    event loop (``QTimer.singleShot``) rather than constructing it inline, which
    keeps it clear of the failing call stack.
    """

    def show() -> None:
        box = QMessageBox(
            QMessageBox.Critical,
            "Unexpected error",
            "An unexpected error occurred:\n\n"
            f"{msg}\n\n"
            "Details have been written to the log (Help ▸ View Logs…). You can "
            "open a pre-filled GitHub report below — nothing is sent until you "
            "review and submit it.",
        )
        report_btn = box.addButton("Report on GitHub", QMessageBox.ActionRole)
        box.addButton("Close", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is report_btn:
            session = logging_setup.active_session_path()
            log_text = logging_setup.read_log_text(session) if session else ""
            url = bug_report.issue_url(
                __version__,
                log_text,
                title=f"Crash: {msg}",
                intro="Painting Assist reported an uncaught exception.",
            )
            QDesktopServices.openUrl(QUrl(url))

    QTimer.singleShot(0, show)


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

    # Read the persisted theme + log-location override once. On the very first
    # launch after upgrading (no store file yet) fall back to the legacy
    # QSettings theme so an existing preference is honoured on the first paint.
    settings_path = os.path.join(_app_data_dir(), "settings.json")
    configured_log_dir = ""
    if os.path.exists(settings_path):
        store = SettingsStore(settings_path)
        store.load()
        mode = str(store.get("preferences", "theme", DEFAULT_THEME))
        configured_log_dir = str(store.get("preferences", "log_dir", ""))
    else:
        legacy = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        mode = str(legacy.value("settings/theme", DEFAULT_THEME))

    # Configure logging as early as possible so startup problems and any
    # uncaught exception land in a per-session file; then sweep old logs on a
    # background thread and route Qt's own messages into the log too.
    log_dir = logging_setup.resolve_log_dir(configured_log_dir, _app_data_dir())
    session_path = logging_setup.configure(log_dir, notifier=_error_notifier)
    qInstallMessageHandler(_qt_message_handler)
    _log.info(
        "Painting Assist %s starting (Python %s on %s)",
        __version__,
        platform.python_version(),
        platform.platform(),
    )
    _log.info("Logging to %s", session_path)
    logging_setup.cleanup_old_logs_async(log_dir)

    # Surface a degraded colour-mixing engine in the log: without mixbox the app
    # silently falls back to a cruder additive model, which is worth knowing when
    # a user reports poor mix suggestions.
    from painting_assist import mixing

    if not mixing.MIXBOX_AVAILABLE:
        _log.warning(
            "mixbox unavailable — colour mixing is using the additive fallback"
        )

    # Apply the persisted theme before any window shows, so the first paint is
    # already in the right look. MainWindow then imports the legacy value into
    # the store on first run.
    if mode not in theme.THEME_MODES:
        mode = DEFAULT_THEME
    theme.apply_theme(app, mode)

    icon = _app_icon()
    app.setWindowIcon(icon)
    win = MainWindow()
    win.setWindowIcon(icon)
    win.resize(1280, 800)
    win.show()
    _log.info("Main window shown; entering event loop")
    rc = app.exec()
    _shutdown(app, win, rc)
    return rc


def _shutdown(app: QApplication, win: MainWindow, rc: int) -> None:
    """Tear the GUI down in a controlled order, then terminate the process.

    Qt 6.8's Cocoa plugin can segfault while destroying ``QApplication`` at
    interpreter exit: it releases a retained ``NSEvent`` whose ``NSTouch``
    set (from a trackpad gesture earlier in the session) is already gone. That
    destructor normally runs from PySide's ``atexit`` hook, after the session
    has been saved, so the crash costs no data but shows a macOS crash report.

    Everything the user cares about has already happened by now (``closeEvent``
    saved the session and drained the render workers), so we delete the window
    while Python is fully alive, flush the logs, and ``os._exit`` to skip the
    ``QApplication`` destructor and the rest of interpreter finalisation.
    """
    win.deleteLater()
    app.processEvents()
    _log.info("Exiting with status %d", rc)
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    raise SystemExit(main())
