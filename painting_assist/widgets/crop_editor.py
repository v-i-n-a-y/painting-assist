# Copyright 2026 Vinay Williams

"""Custom editor widget for :class:`CropControl` (the "Canvas & Crop" tool).

This widget edits the canvas dimensions / unit / aspect-lock and drives the
interactive crop tool in the viewport. It only ever *reads* from the control;
all changes are reported back via Qt signals so the surrounding application can
apply them. The classic anti-feedback pattern is used: a ``self._suppress``
flag is raised while programmatically loading values into the widgets, and every
user-callback bails out early while it is set.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class CropEditor(QWidget):
    """Editor for canvas size, aspect lock and the interactive crop region.

    Signals
    -------
    paramChanged(str, object)
        Emitted with ``(param_name, value)`` when the user edits a parameter.
        Covers ``"canvas_w"``/``"canvas_h"`` (float), ``"unit"`` (str),
        ``"lock_ratio"`` (bool) and the crop-clearing rectangle params
        (``"rx"``, ``"ry"``, ``"rw"``, ``"rh"``).
    interaction(bool)
        Declared for contract uniformity with slider-based editors. This
        widget has no sliders and never emits it.
    editRequested(bool)
        Emitted ``True`` to ask the viewport to begin interactive crop editing
        and ``False`` to finish/apply it. The main window connects directly.
    """

    paramChanged = Signal(str, object)
    interaction = Signal(bool)
    editRequested = Signal(bool)

    _ADJUST_TEXT = "Adjust crop region…"
    _APPLY_TEXT = "Apply crop"

    def __init__(self, control, parent=None) -> None:
        """Build the widgets, wire callbacks and load current control values."""
        super().__init__(parent)
        self._control = control
        self._suppress = False
        self._editing = False
        # Last width/height seen, so a linked edit can preserve the pre-change
        # aspect ratio (see :meth:`_linked`). Kept in sync in refresh() and
        # after every handled dimension change.
        self._prev_w = 0.0
        self._prev_h = 0.0

        root = QVBoxLayout(self)

        # --- Canvas size row -------------------------------------------- #
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Canvas size:"))

        self.width = QDoubleSpinBox()
        self.width.setRange(0.1, 100000.0)
        self.width.setDecimals(1)
        size_row.addWidget(self.width)

        size_row.addWidget(QLabel("×"))

        self.height = QDoubleSpinBox()
        self.height.setRange(0.1, 100000.0)
        self.height.setDecimals(1)
        size_row.addWidget(self.height)

        self.unit = QComboBox()
        for label, value in (
            ("cm", "cm"),
            ("inch", "in"),
            ("mm", "mm"),
            ("px", "px"),
            ("ratio", "ratio"),
        ):
            self.unit.addItem(label, value)
        size_row.addWidget(self.unit)
        size_row.addStretch(1)
        root.addLayout(size_row)

        # --- Lock to canvas ratio --------------------------------------- #
        self.lock = QCheckBox("Lock to canvas ratio")
        root.addWidget(self.lock)

        # --- Edit / Clear buttons --------------------------------------- #
        btn_row = QHBoxLayout()
        self.edit_btn = QPushButton(self._ADJUST_TEXT)
        self.edit_btn.setCheckable(True)
        btn_row.addWidget(self.edit_btn)

        self.clear_btn = QPushButton("Clear crop")
        btn_row.addWidget(self.clear_btn)
        root.addLayout(btn_row)

        # --- Info label ------------------------------------------------- #
        self.info = QLabel()
        self.info.setWordWrap(True)
        root.addWidget(self.info)

        # --- Wiring ----------------------------------------------------- #
        self.width.valueChanged.connect(self._on_width)
        self.height.valueChanged.connect(self._on_height)
        self.unit.currentIndexChanged.connect(self._on_unit)
        self.lock.toggled.connect(self._on_lock)
        self.edit_btn.clicked.connect(self._on_edit_clicked)
        self.clear_btn.clicked.connect(self._on_clear)

        self.refresh()

    # ------------------------------------------------------------------ #
    # User callbacks (guarded by the anti-feedback flag)
    # ------------------------------------------------------------------ #
    def _on_width(self, value: float) -> None:
        """Report a change to the canvas width.

        Once a crop has been applied with the ratio locked, the aspect ratio is
        fixed by that crop, so the height is scaled to match — this keeps the
        physical canvas size proportional for the gridline/measure tools.
        """
        if self._suppress:
            return
        value = float(value)
        self.paramChanged.emit("canvas_w", value)
        if self._linked() and self._prev_h > 0:
            ratio = self._prev_w / self._prev_h
            if ratio > 0:
                new_h = self._clamped(self.height, value / ratio)
                self._set_spin(self.height, new_h)
                self.paramChanged.emit("canvas_h", new_h)
        self._prev_w = float(self.width.value())
        self._prev_h = float(self.height.value())
        self._update_info()

    def _on_height(self, value: float) -> None:
        """Report a change to the canvas height (linked to width once cropped)."""
        if self._suppress:
            return
        value = float(value)
        self.paramChanged.emit("canvas_h", value)
        if self._linked() and self._prev_h > 0:
            ratio = self._prev_w / self._prev_h
            if ratio > 0:
                new_w = self._clamped(self.width, value * ratio)
                self._set_spin(self.width, new_w)
                self.paramChanged.emit("canvas_w", new_w)
        self._prev_w = float(self.width.value())
        self._prev_h = float(self.height.value())
        self._update_info()

    def _on_unit(self, _index: int) -> None:
        """Report a change to the display unit."""
        if self._suppress:
            return
        self.paramChanged.emit("unit", str(self.unit.currentData()))

    def _on_lock(self, checked: bool) -> None:
        """Report a change to the aspect-lock flag."""
        if self._suppress:
            return
        self.paramChanged.emit("lock_ratio", bool(checked))
        self._update_info()

    def _on_edit_clicked(self) -> None:
        """Toggle interactive crop editing and notify the viewport."""
        if self._suppress:
            return
        if not self._editing:
            self._editing = True
            self.edit_btn.setChecked(True)
            self.edit_btn.setText(self._APPLY_TEXT)
            self._update_lock_enabled()
            self.editRequested.emit(True)
        else:
            self._editing = False
            self.edit_btn.setChecked(False)
            self.edit_btn.setText(self._ADJUST_TEXT)
            self._update_lock_enabled()
            self.editRequested.emit(False)

    def _on_clear(self) -> None:
        """Reset the crop rect to the full frame and finish any editing."""
        if self._suppress:
            return
        self.paramChanged.emit("rx", 0.0)
        self.paramChanged.emit("ry", 0.0)
        self.paramChanged.emit("rw", 1.0)
        self.paramChanged.emit("rh", 1.0)
        if self._editing:
            self.editRequested.emit(False)
            self.set_editing(False)

    # ------------------------------------------------------------------ #
    # Aspect-linking helpers
    # ------------------------------------------------------------------ #
    def _linked(self) -> bool:
        """Whether the width/height spinboxes should move together.

        True once a crop has been applied and we are NOT mid-edit: the crop
        fixes the aspect ratio (freeform or locked alike), so resizing the
        canvas for gridlines must preserve it. While cropping, the dimensions
        stay free so the user can define/reshape the ratio.
        """
        if self._editing:
            return False
        return bool(getattr(self._control, "has_crop", lambda: False)())

    def _update_lock_enabled(self) -> None:
        """Lock is only meaningful while adjusting a crop.

        Freeform-vs-locked decides how the crop box behaves as you drag it.
        Once a crop is applied its aspect is baked in, so the toggle is greyed
        out to stop the canvas ratio being knocked out of step with the crop.
        """
        self.lock.setEnabled(self._editing)
        self.lock.setToolTip(
            "On: crop box is locked to the canvas ratio. Off: freeform crop."
            if self._editing
            else "Available while adjusting a crop; the applied crop fixes the ratio."
        )

    @staticmethod
    def _clamped(spin, value: float) -> float:
        """Clamp ``value`` to a spinbox's valid range."""
        return max(spin.minimum(), min(spin.maximum(), float(value)))

    def _set_spin(self, spin, value: float) -> None:
        """Set a spinbox value without re-triggering its user callback."""
        was = self._suppress
        self._suppress = True
        try:
            spin.setValue(float(value))
        finally:
            self._suppress = was

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        """Re-read all values from the control into the widgets.

        Performed under the ``self._suppress`` flag so the resulting widget
        signals do NOT emit :attr:`paramChanged`. Also refreshes the info
        label and the edit button's text.
        """
        self._suppress = True
        try:
            self.width.setValue(float(self._control.get("canvas_w")))
            self.height.setValue(float(self._control.get("canvas_h")))

            unit = str(self._control.get("unit"))
            idx = self.unit.findData(unit)
            if idx >= 0:
                self.unit.setCurrentIndex(idx)

            self.lock.setChecked(bool(self._control.get("lock_ratio")))
            self._prev_w = float(self.width.value())
            self._prev_h = float(self.height.value())
        finally:
            self._suppress = False

        self.edit_btn.setText(self._APPLY_TEXT if self._editing else self._ADJUST_TEXT)
        self.edit_btn.setChecked(self._editing)
        self._update_lock_enabled()
        self._update_info()

    def set_editing(self, editing: bool) -> None:
        """Sync the toggle button to ``editing`` WITHOUT emitting a signal.

        Called by the main window when editing ends for external reasons
        (e.g. a new image is opened or Reset is pressed).
        """
        self._editing = bool(editing)
        self._suppress = True
        try:
            self.edit_btn.setChecked(self._editing)
            self.edit_btn.setText(
                self._APPLY_TEXT if self._editing else self._ADJUST_TEXT
            )
        finally:
            self._suppress = False
        self._update_lock_enabled()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _update_info(self) -> None:
        """Update the word-wrapped summary of the current crop state."""
        if not bool(self.lock.isChecked()):
            self.info.setText("Freeform crop")
            return

        w = float(self.width.value())
        h = float(self.height.value())
        if w <= 0 or h <= 0:
            self.info.setText("Freeform crop")
            return

        ratio = w / h
        self.info.setText(
            "Ratio {} ({:.2f}) — drag the box in the image".format(
                self._ratio_text(w, h), ratio
            )
        )

    @staticmethod
    def _ratio_text(w: float, h: float) -> str:
        """Return a simplified ``a:b`` ratio if w/h are near-integers, else decimal."""
        if abs(w - round(w)) < 1e-6 and abs(h - round(h)) < 1e-6:
            wi, hi = int(round(w)), int(round(h))
            divisor = math.gcd(wi, hi)
            if divisor > 0:
                return "{}:{}".format(wi // divisor, hi // divisor)
        return "{:.2f}".format(w / h)
