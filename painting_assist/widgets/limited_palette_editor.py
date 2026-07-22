# Copyright 2026 Vinay Williams

"""Custom editor for :class:`LimitedPaletteControl`.

Shows the palette-source and preset pickers (generic combos) plus a live strip of
the palette swatches actually being mixed from. When the source is *Sampled*, the
strip's colours can be added (from a colour dialog or by sampling the image via
the eyedropper) and removed, and the effective list is written back to the
control's ``samples_json`` param through the standard ``paramChanged`` signal.

The My Paints inventory is injected into the control's ``paints_json`` by the
window (the worker cannot read live state), so for the *My Paints* source this
editor simply reflects whatever the control currently resolves.
"""

from __future__ import annotations

import json
from typing import List, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

RGB = Tuple[int, int, int]
_SWATCH_COLS = 8


def _hex(rgb: RGB) -> str:
    return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


class LimitedPaletteEditor(QWidget):
    """Source/preset pickers plus an editable strip of palette swatches.

    Emits :attr:`paramChanged` ``(name, value)`` and :attr:`interaction`
    ``(bool)`` per the custom-editor contract, plus :attr:`sampleRequested` when
    the painter asks to pick a palette colour off the image (the window arms the
    eyedropper and feeds the result back via :meth:`add_sampled_rgb`).
    """

    paramChanged = Signal(str, object)
    interaction = Signal(bool)
    sampleRequested = Signal()

    def __init__(self, control, parent=None):
        super().__init__(parent)
        self._control = control
        self._param_widgets = {}

        from painting_assist.widgets.control_panel import build_param_widget

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # Source + preset pickers (the two injected JSON blobs are managed here,
        # not shown as raw text boxes).
        for spec in control.params():
            if spec.name in ("paints_json", "samples_json"):
                continue
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            label = QLabel(spec.label)
            label.setMinimumWidth(70)
            if spec.tooltip:
                label.setToolTip(spec.tooltip)
            row.addWidget(label, 0)
            widget = build_param_widget(spec, control.get(spec.name))
            widget.valueChanged.connect(self._on_param)
            widget.interaction.connect(self.interaction)
            self._param_widgets[spec.name] = widget
            row.addWidget(widget, 1)
            root.addLayout(row)

        # Live swatch strip of the palette being mixed from.
        self._swatch_host = QWidget()
        self._swatch_grid = QGridLayout(self._swatch_host)
        self._swatch_grid.setContentsMargins(0, 2, 0, 2)
        self._swatch_grid.setSpacing(3)
        root.addWidget(self._swatch_host)

        # Sampled-source controls (add / sample / clear), hidden otherwise.
        self._sampled_row = QWidget()
        srow = QHBoxLayout(self._sampled_row)
        srow.setContentsMargins(0, 0, 0, 0)
        add_btn = QPushButton("Add colour…")
        add_btn.setToolTip("Add a colour to the palette from the colour dialog")
        add_btn.clicked.connect(self._on_add_colour)
        sample_btn = QPushButton("Sample from image")
        sample_btn.setToolTip("Click a colour on the image to add it to the palette")
        sample_btn.clicked.connect(self.sampleRequested)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._on_clear_samples)
        srow.addStretch(1)
        for b in (add_btn, sample_btn, clear_btn):
            srow.addWidget(b)
        root.addWidget(self._sampled_row)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)

        self._rebuild()

    # ------------------------------------------------------------------ #
    def _on_param(self, name: str, value: object) -> None:
        """Forward a combo change, then rebuild the swatch strip for the new source."""
        self.paramChanged.emit(name, value)
        self._rebuild()

    def _samples(self) -> List[str]:
        """Current sampled colours as a list of hex strings (robust to junk)."""
        try:
            data = json.loads(self._control.get("samples_json") or "[]")
        except (TypeError, ValueError):
            return []
        return [s for s in data if isinstance(s, str)] if isinstance(data, list) else []

    def _set_samples(self, hexes: List[str]) -> None:
        text = json.dumps(hexes)
        self._control.set("samples_json", text)
        self.paramChanged.emit("samples_json", text)
        self._rebuild()

    def add_sampled_rgb(self, rgb: RGB) -> None:
        """Append a colour (e.g. from the eyedropper) to the sampled palette."""
        self._set_samples(self._samples() + [_hex(rgb)])

    def _on_add_colour(self) -> None:
        colour = QColorDialog.getColor(
            QColor(200, 200, 200), self, "Add palette colour"
        )
        if colour.isValid():
            self.add_sampled_rgb((colour.red(), colour.green(), colour.blue()))

    def _on_clear_samples(self) -> None:
        self._set_samples([])

    def _clear_grid(self) -> None:
        while self._swatch_grid.count():
            item = self._swatch_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _rebuild(self) -> None:
        """Repopulate the swatch strip and sampled controls from the control state."""
        for name, widget in self._param_widgets.items():
            widget.set_value(self._control.get(name))

        source = self._control.get("source")
        sampled = source == "sampled"
        self._sampled_row.setVisible(sampled)

        tubes = self._control.tubes()
        self._clear_grid()
        for i, rgb in enumerate(tubes):
            swatch = QLabel()
            swatch.setFixedSize(20, 20)
            swatch.setStyleSheet(
                "background: %s; border: 1px solid rgba(0,0,0,90);" % _hex(rgb)
            )
            tip = _hex(rgb) + ("  (click to remove)" if sampled else "")
            swatch.setToolTip(tip)
            if sampled:
                swatch.setCursor(Qt.PointingHandCursor)
                swatch.mousePressEvent = lambda _e, idx=i: self._remove_sample(idx)
            self._swatch_grid.addWidget(swatch, i // _SWATCH_COLS, i % _SWATCH_COLS)

        if not tubes:
            if sampled:
                self._hint.setText("Add or sample colours to build a palette.")
            elif source == "my_paints":
                self._hint.setText("Add tubes in File > My Paints to use them here.")
            else:
                self._hint.setText("")
        else:
            self._hint.setText("%d colour palette." % len(tubes))

    def _remove_sample(self, index: int) -> None:
        hexes = self._samples()
        if 0 <= index < len(hexes):
            del hexes[index]
            self._set_samples(hexes)

    # ------------------------------------------------------------------ #
    def set_paints(self, _paints) -> None:
        """Refresh after the injected My Paints inventory changed (colours come
        from the control's ``paints_json``, so just rebuild the strip)."""
        self._rebuild()

    def refresh(self) -> None:
        """Re-sync the combos and swatch strip from the control (no signals)."""
        self._rebuild()
