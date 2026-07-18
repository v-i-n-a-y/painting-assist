# Copyright 2026 Vinay Williams

"""The palette side panel — a strip of flat swatches from the quantize centroids.

The colour maths lives in module-level pure functions (``rgb_to_hex``,
``lab_readout``, ``colour_readout``) so it is unit-testable headlessly without a
Qt event loop. The widgets are thin: they lay out swatches and show the readout
for whichever colour was last clicked or sampled.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

RGB = Tuple[int, int, int]

# Swatch geometry (matches the compact, flat look of the control panel).
SWATCH_HEIGHT = 40
SWATCH_MIN_WIDTH = 12


# ---------------------------------------------------------------------------
# Pure colour maths (Qt-free, unit-testable)
# ---------------------------------------------------------------------------
def rgb_to_hex(rgb: Sequence[int]) -> str:
    """Return the ``#RRGGBB`` hex string for an (r, g, b) uint8 triple."""
    r, g, b = (int(c) & 0xFF for c in rgb[:3])
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def rgb_to_lab(rgb: Sequence[int]) -> Tuple[float, float, float]:
    """Convert an (r, g, b) uint8 triple to OpenCV's uint8 Lab encoding.

    Returns floats in OpenCV's 0..255 Lab convention (L scaled to 255, a/b
    centred on 128), matching how the quantize control clusters colour.
    """
    import cv2  # local import keeps this module importable without cv2 for pure use

    px = np.array(
        [[[int(rgb[0]) & 0xFF, int(rgb[1]) & 0xFF, int(rgb[2]) & 0xFF]]], dtype=np.uint8
    )
    lab = cv2.cvtColor(px, cv2.COLOR_RGB2Lab)[0, 0]
    return float(lab[0]), float(lab[1]), float(lab[2])


def lab_readout(lab: Sequence[float]) -> Dict[str, float]:
    """Derive value %, hue angle and chroma from an OpenCV uint8 Lab triple.

    * value % = ``L * 100 / 255`` (undo the uint8 L scaling to get 0..100).
    * hue angle (degrees) = ``atan2(b - 128, a - 128)`` in [-180, 180].
    * chroma = ``hypot(a - 128, b - 128)``.
    """
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    return {
        "value_pct": L * 100.0 / 255.0,
        "hue_deg": math.degrees(math.atan2(b - 128.0, a - 128.0)),
        "chroma": math.hypot(a - 128.0, b - 128.0),
    }


def colour_readout(rgb: Sequence[int]) -> Dict[str, object]:
    """Return a full readout dict (hex, rgb, value_pct, hue_deg, chroma) for a colour."""
    r, g, b = (int(c) & 0xFF for c in rgb[:3])
    out: Dict[str, object] = {"hex": rgb_to_hex((r, g, b)), "rgb": (r, g, b)}
    out.update(lab_readout(rgb_to_lab((r, g, b))))
    return out


def format_readout(rgb: Sequence[int]) -> str:
    """Format the one-line readout string shown in the panel for a colour."""
    d = colour_readout(rgb)
    r, g, b = d["rgb"]  # type: ignore[misc]
    return "{}   RGB {} {} {}   value {:.0f}%   hue {:.0f} deg   chroma {:.0f}".format(
        d["hex"], r, g, b, d["value_pct"], d["hue_deg"], d["chroma"]
    )


def render_palette_strip(
    colours: Sequence[Sequence[int]], swatch: int = 64, height: int = 64
) -> "object":
    """Draw ``colours`` as side-by-side swatches into an RGB Pillow image.

    Pure Pillow/numpy so it is importable and callable without a running
    QApplication (the lead uses it to save a PNG). Each colour occupies a
    ``swatch`` wide by ``height`` tall block; the returned image is
    ``swatch * len(colours)`` wide. An empty palette yields a 1x``height``
    black image so callers always get a valid image back.
    """
    from PIL import Image  # local import: keep module usable without Pillow

    swatch = max(1, int(swatch))
    height = max(1, int(height))
    triples = [
        (int(c[0]) & 0xFF, int(c[1]) & 0xFF, int(c[2]) & 0xFF) for c in (colours or [])
    ]
    if not triples:
        return Image.new("RGB", (1, height), (0, 0, 0))

    arr = np.zeros((height, swatch * len(triples), 3), dtype=np.uint8)
    for i, triple in enumerate(triples):
        arr[:, i * swatch : (i + 1) * swatch] = triple
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------
class _Swatch(QFrame):
    """A single flat colour block; clicking it emits ``clicked(rgb)``."""

    clicked = Signal(tuple)

    def __init__(self, rgb: RGB, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(SWATCH_HEIGHT)
        self.setMinimumWidth(SWATCH_MIN_WIDTH)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setFrameShape(QFrame.NoFrame)
        self.set_rgb(rgb)

    def set_rgb(self, rgb: RGB) -> None:
        """Recolour the swatch (used by the reusable sampled-colour chip)."""
        self._rgb: RGB = rgb
        self.setToolTip(rgb_to_hex(rgb))
        r, g, b = rgb
        self.setStyleSheet("background-color: rgb({}, {}, {});".format(r, g, b))

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._rgb)
        super().mousePressEvent(event)


class PalettePanel(QWidget):
    """Horizontal strip of colour swatches plus a readout for the picked colour.

    Fed by :meth:`set_colours` with the quantize palette (already sorted). A
    click copies the swatch's hex to the clipboard, emits ``swatchClicked(rgb)``
    and updates the in-panel readout. :meth:`set_sample` shows the same readout
    for an arbitrary colour (for a future eyedropper); :meth:`clear` empties it.
    """

    swatchClicked = Signal(tuple)  # (r, g, b)
    sampleSizeChanged = Signal(int)  # odd sampling radius in pixels

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._swatches: List[_Swatch] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        self._strip = QHBoxLayout()
        self._strip.setContentsMargins(0, 0, 0, 0)
        self._strip.setSpacing(2)
        outer.addLayout(self._strip)

        # Eyedropper sampling radius: how many pixels to region-average when
        # picking a colour. Odd values keep the sample centred on the cursor.
        size_row = QWidget()
        size_layout = QHBoxLayout(size_row)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.setSpacing(8)
        size_label = QLabel("Sample size")
        size_layout.addWidget(size_label)
        self._sample_size = QSpinBox()
        self._sample_size.setRange(1, 51)
        self._sample_size.setSingleStep(2)
        self._sample_size.setValue(1)
        self._sample_size.setSuffix(" px")
        self._sample_size.setToolTip(
            "Region-average radius for the eyedropper (odd, in pixels)."
        )
        self._sample_size.valueChanged.connect(self._on_sample_size_changed)
        size_layout.addWidget(self._sample_size)
        size_layout.addStretch(1)
        outer.addWidget(size_row)

        # Dedicated "sampled colour" slot, kept clearly apart from the palette
        # strip above so a lone sample is not mistaken for a palette entry. A
        # thin divider separates the two; the row hides itself when empty.
        self._sample_divider = QFrame()
        self._sample_divider.setFrameShape(QFrame.HLine)
        self._sample_divider.setFrameShadow(QFrame.Sunken)
        outer.addWidget(self._sample_divider)

        self._sample_row = QWidget()
        sample_layout = QHBoxLayout(self._sample_row)
        sample_layout.setContentsMargins(0, 0, 0, 0)
        sample_layout.setSpacing(8)
        sample_caption = QLabel("Sampled")
        sample_caption.setEnabled(False)
        sample_layout.addWidget(sample_caption)
        self._sample_swatch = _Swatch((0, 0, 0))
        self._sample_swatch.setFixedWidth(SWATCH_HEIGHT)
        self._sample_swatch.clicked.connect(self._on_swatch_clicked)
        sample_layout.addWidget(self._sample_swatch)
        sample_layout.addStretch(1)
        outer.addWidget(self._sample_row)

        self._sample_divider.setVisible(False)
        self._sample_row.setVisible(False)

        self._readout = QLabel("")
        self._readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._readout.setWordWrap(True)
        outer.addWidget(self._readout)

        # A short multi-line mixing recipe for the sampled colour; hidden until
        # set_mixing() supplies text.
        self._mixing = QLabel("")
        self._mixing.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._mixing.setWordWrap(True)
        self._mixing.setVisible(False)
        outer.addWidget(self._mixing)

        self._placeholder = QLabel("Enable Colour groups to see its palette.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setEnabled(False)
        outer.addWidget(self._placeholder)

        outer.addStretch(1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_colours(self, colours: Optional[Sequence[Sequence[int]]]) -> None:
        """Rebuild the swatch strip from a list of (r, g, b) triples."""
        self._clear_swatches()
        colours = list(colours or [])
        self._placeholder.setVisible(not colours)
        for rgb in colours:
            triple: RGB = (int(rgb[0]) & 0xFF, int(rgb[1]) & 0xFF, int(rgb[2]) & 0xFF)
            swatch = _Swatch(triple)
            swatch.clicked.connect(self._on_swatch_clicked)
            self._strip.addWidget(swatch, 1)
            self._swatches.append(swatch)

    def set_sample(self, rgb: Optional[Sequence[int]]) -> None:
        """Show the swatch + readout for a sampled colour (or hide it if None)."""
        if rgb is None:
            self._readout.setText("")
            self._sample_divider.setVisible(False)
            self._sample_row.setVisible(False)
            self.set_mixing(None)
            return
        triple: RGB = (int(rgb[0]) & 0xFF, int(rgb[1]) & 0xFF, int(rgb[2]) & 0xFF)
        self._sample_swatch.set_rgb(triple)
        self._sample_divider.setVisible(True)
        self._sample_row.setVisible(True)
        self._readout.setText(format_readout(triple))

    def set_mixing(self, text: Optional[str]) -> None:
        """Show a short multi-line mixing recipe beneath the readout (hide if None)."""
        if text is None or not str(text).strip():
            self._mixing.setText("")
            self._mixing.setVisible(False)
            return
        self._mixing.setText(str(text))
        self._mixing.setVisible(True)

    def sample_size(self) -> int:
        """Return the current eyedropper region-average radius in pixels."""
        return int(self._sample_size.value())

    def clear(self) -> None:
        """Remove all swatches, the sample slot, mixing text and the readout."""
        self._clear_swatches()
        self.set_sample(None)
        self._placeholder.setVisible(True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _clear_swatches(self) -> None:
        for swatch in self._swatches:
            self._strip.removeWidget(swatch)
            swatch.setParent(None)
            swatch.deleteLater()
        self._swatches = []

    def _on_sample_size_changed(self, value: int) -> None:
        self.sampleSizeChanged.emit(int(value))

    def _on_swatch_clicked(self, rgb: RGB) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(rgb_to_hex(rgb))
        self.set_sample(rgb)
        self.swatchClicked.emit(rgb)
