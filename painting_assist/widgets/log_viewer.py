# Copyright 2026 Vinay Williams

"""Dialog for viewing and filtering the application's session log files.

The dialog is a thin presentation layer over :mod:`painting_assist.logging_setup`:
it lists the session log files found in a log directory, lets the user pick which
severity levels to show, and renders the filtered text read-only. All the actual
log discovery, reading and filtering logic lives in the Qt-free ``logging_setup``
module so it stays independently testable; this widget only wires it up to
buttons and a combo box.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFontDatabase, QGuiApplication, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from painting_assist import __version__, bug_report, logging_setup


def _human_size(num_bytes: int) -> str:
    """Return ``num_bytes`` formatted as B, KB or MB with one decimal place."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


class LogViewer(QDialog):
    """View, filter and copy the application's session log files."""

    def __init__(
        self,
        log_dir: str,
        current_session: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Logs")
        self.resize(900, 600)

        self._log_dir = log_dir
        self._current = current_session

        self._session_combo = QComboBox()
        self._session_combo.currentIndexChanged.connect(self._reload)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Session:"))
        top_row.addWidget(self._session_combo, 1)

        self._level_checks: Dict[str, QCheckBox] = {}
        level_row = QHBoxLayout()
        level_row.addWidget(QLabel("Levels:"))
        for name in logging_setup.LEVEL_NAMES:
            check = QCheckBox(name)
            check.setChecked(True)
            check.toggled.connect(self._reload)
            self._level_checks[name] = check
            level_row.addWidget(check)
        level_row.addStretch(1)

        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self._view.selectionChanged.connect(self._update_copy_selection_enabled)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_sessions)
        self._copy_all_btn = QPushButton("Copy All")
        self._copy_all_btn.clicked.connect(self._copy_all)
        self._copy_selection_btn = QPushButton("Copy Selection")
        self._copy_selection_btn.clicked.connect(self._copy_selection)
        self._copy_selection_btn.setEnabled(False)
        open_folder_btn = QPushButton("Open Logs Folder")
        open_folder_btn.clicked.connect(self._open_logs_folder)
        report_btn = QPushButton("Report on GitHub")
        report_btn.setToolTip(
            "Open a pre-filled GitHub issue with this session's log; "
            "nothing is sent until you review and submit it."
        )
        report_btn.clicked.connect(self._report_on_github)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(refresh_btn)
        bottom_row.addWidget(self._copy_all_btn)
        bottom_row.addWidget(self._copy_selection_btn)
        bottom_row.addWidget(open_folder_btn)
        bottom_row.addWidget(report_btn)
        bottom_row.addStretch(1)
        bottom_row.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(top_row)
        layout.addLayout(level_row)
        layout.addWidget(self._view, 1)
        layout.addLayout(bottom_row)

        self._populate_sessions()
        self._reload()

    # ------------------------------------------------------------------ #
    # Session list
    # ------------------------------------------------------------------ #
    def _populate_sessions(self, keep_path: Optional[str] = None) -> None:
        """(Re)populate the session combo, optionally preserving a selection."""
        self._session_combo.blockSignals(True)
        self._session_combo.clear()

        paths = logging_setup.list_session_logs(self._log_dir) if self._log_dir else []
        if not paths:
            self._session_combo.addItem("No logs found", None)
            self._session_combo.setEnabled(False)
            self._session_combo.blockSignals(False)
            return

        self._session_combo.setEnabled(True)
        select_index = 0
        keep_index: Optional[int] = None
        for i, path in enumerate(paths):
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            label = f"{os.path.basename(path)}  ({_human_size(size)})"
            if path == self._current:
                label += "  — current session"
                select_index = i
            self._session_combo.addItem(label, path)
            if keep_path is not None and path == keep_path:
                keep_index = i

        self._session_combo.setCurrentIndex(
            keep_index if keep_index is not None else select_index
        )
        self._session_combo.blockSignals(False)

    def _refresh_sessions(self) -> None:
        """Re-scan the log directory, keeping the current selection if possible."""
        current_path = self._session_combo.currentData()
        self._populate_sessions(keep_path=current_path)
        self._reload()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _reload(self) -> None:
        """Re-read the selected session log and render it filtered by level."""
        path = self._session_combo.currentData()
        if path is None:
            self._view.setPlainText("")
            return
        text = logging_setup.read_log_text(path)
        levels = [
            name for name, check in self._level_checks.items() if check.isChecked()
        ]
        self._view.setPlainText(logging_setup.filter_log_text(text, levels))
        self._view.moveCursor(QTextCursor.End)

    # ------------------------------------------------------------------ #
    # Buttons
    # ------------------------------------------------------------------ #
    def _copy_all(self) -> None:
        QGuiApplication.clipboard().setText(self._view.toPlainText())
        original = self._copy_all_btn.text()
        self._copy_all_btn.setText("Copied!")
        QTimer.singleShot(1200, lambda: self._restore_copy_all_text(original))

    def _restore_copy_all_text(self, original: str) -> None:
        try:
            self._copy_all_btn.setText(original)
        except RuntimeError:
            # The dialog (and its buttons) may already have been destroyed.
            pass

    def _copy_selection(self) -> None:
        selected = self._view.textCursor().selectedText()
        if not selected:
            return
        QGuiApplication.clipboard().setText(selected.replace(" ", "\n"))

    def _update_copy_selection_enabled(self) -> None:
        self._copy_selection_btn.setEnabled(self._view.textCursor().hasSelection())

    def _open_logs_folder(self) -> None:
        if self._log_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._log_dir))

    def _report_on_github(self) -> None:
        """Open a pre-filled GitHub issue carrying the selected session's log."""
        path = self._session_combo.currentData()
        log_text = logging_setup.read_log_text(path) if path else ""
        url = bug_report.issue_url(
            __version__,
            log_text,
            title=f"Bug report — Painting Assist {__version__}",
        )
        QDesktopServices.openUrl(QUrl(url))
