# Copyright 2026 Vinay Williams

"""Custom editor widget for :class:`GridControl`.

Shows the grid's generic parameter rows (reusing the panel's ``ParamWidget``)
plus a read-only summary of where the gridlines fall on the canvas, so the
painter can read off the exact positions to mark on their surface. The positions
text is pushed in by the window (it depends on the canvas calibration, which
lives outside the control); the widget only reads from the control otherwise.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


class GridEditor(QWidget):
    """Generic grid param rows plus a canvas-position readout.

    Emits :attr:`paramChanged` ``(name, value)`` on every user change (forwarded
    straight from the per-param :class:`ParamWidget`) and :attr:`interaction`
    ``(bool)`` around live slider drags, matching the custom-editor contract the
    control panel expects.
    """

    paramChanged = Signal(str, object)
    interaction = Signal(bool)

    def __init__(self, control, parent=None):
        """Build the param rows for ``control`` and the positions readout."""
        super().__init__(parent)
        self._control = control
        self._param_widgets = {}

        # Lazy import to keep Qt/UI deps of the control module light and avoid
        # any import-order concerns with the panel module.
        from painting_assist.widgets.control_panel import build_param_widget

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        for spec in control.params():
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            label = QLabel(spec.label)
            label.setMinimumWidth(70)
            if spec.tooltip:
                label.setToolTip(spec.tooltip)
            row.addWidget(label, 0)
            widget = build_param_widget(spec, control.get(spec.name))
            widget.valueChanged.connect(self.paramChanged)
            widget.interaction.connect(self.interaction)
            self._param_widgets[spec.name] = widget
            row.addWidget(widget, 1)
            root.addLayout(row)

        # Read-only canvas-position summary, filled in by the window.
        self._positions_label = QLabel("")
        self._positions_label.setWordWrap(True)
        self._positions_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        smaller = QFont(self._positions_label.font())
        if smaller.pointSizeF() > 0:
            smaller.setPointSizeF(smaller.pointSizeF() * 0.92)
        self._positions_label.setFont(smaller)
        root.addWidget(self._positions_label)

    def set_positions_text(self, text: str) -> None:
        """Set the read-only gridline-position summary shown under the params."""
        self._positions_label.setText(text)

    def refresh(self) -> None:
        """Re-read every param value from the control into its widget.

        Does not emit :attr:`paramChanged` (``ParamWidget.set_value`` is a silent
        programmatic set), matching how the panel refreshes on undo/reset/session
        restore. The positions summary is refreshed separately by the window.
        """
        for name, widget in self._param_widgets.items():
            widget.set_value(self._control.get(name))
