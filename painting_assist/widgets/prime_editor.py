# Copyright 2026 Vinay Williams

"""Custom editor widget for :class:`PrimeControl`.

Shows the priming technique and tint-strength rows (reusing the panel's
``ParamWidget``) plus a readout of the recommended ground colour: a large
swatch with its hex/RGB values, the detected majority colour, a one-line
rationale for the technique, and a copy-to-clipboard button. The recommendation
itself is computed by the window from the rendered image (it depends on the
image, which lives outside the control) and pushed in via
:meth:`set_result`; the widget only reads from the control otherwise.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from painting_assist.priming import DESCRIPTIONS, PrimeResult

# Placeholder swatch shown before an image has produced a recommendation.
_NO_RESULT_RGB = (128, 128, 128)


def _swatch_style(rgb) -> str:
    """A stylesheet painting a label as a bordered colour swatch."""
    return "background-color: #{:02X}{:02X}{:02X}; border: 1px solid #808080;".format(
        *rgb
    )


class PrimeEditor(QWidget):
    """Technique/strength rows plus the recommended ground-colour readout.

    Emits :attr:`paramChanged` ``(name, value)`` on every user change (forwarded
    straight from the per-param :class:`ParamWidget`) and :attr:`interaction`
    ``(bool)`` around live slider drags, matching the custom-editor contract the
    control panel expects.
    """

    paramChanged = Signal(str, object)
    interaction = Signal(bool)

    def __init__(self, control, parent=None):
        """Build the param rows for ``control`` and the swatch readout."""
        super().__init__(parent)
        self._control = control
        self._param_widgets = {}
        self._result: Optional[PrimeResult] = None

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

        # Recommended ground: large swatch + hex + RGB.
        self._swatch = QLabel("")
        self._swatch.setFixedSize(44, 44)
        self._swatch.setToolTip("Recommended ground colour")
        self._hex_label = QLabel("")
        hex_font = QFont(self._hex_label.font())
        hex_font.setBold(True)
        self._hex_label.setFont(hex_font)
        self._rgb_label = QLabel("")
        values_box = QVBoxLayout()
        values_box.setContentsMargins(0, 0, 0, 0)
        values_box.setSpacing(0)
        values_box.addWidget(self._hex_label)
        values_box.addWidget(self._rgb_label)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self._swatch, 0)
        top.addLayout(values_box)
        top.addStretch(1)
        root.addLayout(top)

        # Detected majority colour: small swatch + hex.
        self._majority_swatch = QLabel("")
        self._majority_swatch.setFixedSize(16, 16)
        self._majority_swatch.setToolTip("The reference's majority colour")
        self._majority_label = QLabel("")
        mid = QHBoxLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        mid.addWidget(self._majority_swatch, 0)
        mid.addSpacing(4)
        mid.addWidget(self._majority_label, 0)
        mid.addStretch(1)
        root.addLayout(mid)

        # One-line rationale for the selected technique.
        self._description_label = QLabel("")
        self._description_label.setWordWrap(True)
        self._description_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        smaller = QFont(self._description_label.font())
        if smaller.pointSizeF() > 0:
            smaller.setPointSizeF(smaller.pointSizeF() * 0.92)
        self._description_label.setFont(smaller)
        root.addWidget(self._description_label)

        self._copy_btn = QPushButton("Copy hex")
        self._copy_btn.setToolTip("Copy the ground colour's hex code to the clipboard")
        self._copy_btn.clicked.connect(self._on_copy)
        root.addWidget(self._copy_btn, 0, Qt.AlignLeft)

        self._apply_no_result()

    # ------------------------------------------------------------------ #
    # Readout
    # ------------------------------------------------------------------ #
    def set_result(self, result: Optional[PrimeResult]) -> None:
        """Show the recommended ground colour (or the no-image placeholder)."""
        self._result = result
        if result is None:
            self._apply_no_result()
            return
        self._swatch.setStyleSheet(_swatch_style(result.rgb))
        self._hex_label.setText(result.hex)
        self._rgb_label.setText(
            "{}, {}, {}".format(*result.rgb)
        )
        self._majority_swatch.setStyleSheet(_swatch_style(result.majority))
        self._majority_label.setText("Majority " + result.majority_hex)
        self._description_label.setText(
            DESCRIPTIONS.get(result.technique, "")
        )
        self._copy_btn.setEnabled(True)

    def _apply_no_result(self) -> None:
        """Grey out the readout until an image produces a recommendation."""
        self._swatch.setStyleSheet(_swatch_style(_NO_RESULT_RGB))
        self._hex_label.setText("—")
        self._rgb_label.setText("Load an image to get a recommendation")
        self._majority_swatch.setStyleSheet(_swatch_style(_NO_RESULT_RGB))
        self._majority_label.setText("")
        self._description_label.setText(DESCRIPTIONS.get("midtone", ""))
        self._copy_btn.setEnabled(False)

    def _on_copy(self) -> None:
        """Copy the ground colour's hex code to the clipboard."""
        if self._result is None:
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._result.hex)

    # ------------------------------------------------------------------ #
    # Editor contract
    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        """Re-read every param value from the control into its widget.

        Does not emit :attr:`paramChanged` (``ParamWidget.set_value`` is a silent
        programmatic set), matching how the panel refreshes on undo/reset/session
        restore. The swatch readout is refreshed separately by the window.
        """
        for name, widget in self._param_widgets.items():
            widget.set_value(self._control.get(name))
