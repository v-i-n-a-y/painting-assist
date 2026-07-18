from __future__ import annotations

"""Application settings dialog (theme + automatic update checking).

The dialog edits plain values and reports them back through :meth:`values`;
persistence (QSettings) and applying the choices live in ``MainWindow``, so
this widget stays a dumb, testable form. Interval choices are expressed in
hours (0 = never check automatically).
"""

from typing import Dict, Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

# (label, hours between automatic update checks); 0 disables automatic checks.
UPDATE_INTERVALS = [
    ("Every launch", 0.001),
    ("Every 6 hours", 6.0),
    ("Daily", 24.0),
    ("Weekly", 168.0),
    ("Never", 0.0),
]

THEME_CHOICES = [
    ("Match system", "system"),
    ("Light", "light"),
    ("Dark", "dark"),
]

DEFAULT_THEME = "system"
DEFAULT_UPDATE_HOURS = 24.0


def nearest_interval_index(hours: float) -> int:
    """Return the UPDATE_INTERVALS index whose hours value best matches ``hours``.

    Persisted settings may hold values no longer in the list (or corrupted
    ones); snapping to the nearest option keeps the dialog consistent.
    """
    best, best_dist = 0, float("inf")
    for i, (_label, h) in enumerate(UPDATE_INTERVALS):
        dist = abs(h - hours)
        if dist < best_dist:
            best, best_dist = i, dist
    return best


def theme_index(mode: str) -> int:
    """Return the THEME_CHOICES index for ``mode``, defaulting to system."""
    for i, (_label, value) in enumerate(THEME_CHOICES):
        if value == mode:
            return i
    return 0


class SettingsDialog(QDialog):
    """Modal settings form: theme mode and automatic-update interval."""

    def __init__(
        self,
        theme: str = DEFAULT_THEME,
        update_hours: float = DEFAULT_UPDATE_HOURS,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)

        form = QFormLayout()

        self._theme_combo = QComboBox()
        for label, _value in THEME_CHOICES:
            self._theme_combo.addItem(label)
        self._theme_combo.setCurrentIndex(theme_index(theme))
        form.addRow("Theme", self._theme_combo)

        self._interval_combo = QComboBox()
        for label, _hours in UPDATE_INTERVALS:
            self._interval_combo.addItem(label)
        self._interval_combo.setCurrentIndex(nearest_interval_index(update_hours))
        form.addRow("Check for updates", self._interval_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self) -> Dict[str, object]:
        """Return the chosen settings: {"theme": mode, "update_hours": hours}."""
        return {
            "theme": THEME_CHOICES[self._theme_combo.currentIndex()][1],
            "update_hours": UPDATE_INTERVALS[self._interval_combo.currentIndex()][1],
        }
