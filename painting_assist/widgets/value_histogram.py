# Copyright 2026 Vinay Williams

"""Value (Lab-L) distribution histogram: a pure counting function plus a thin,
theme-agnostic QWidget that draws it as bars.

The maths (``value_histogram``, ``value_mass_split``) is Qt-free and unit-
testable headlessly. :class:`ValueHistogram` is a light, paint-only widget the
main window docks and feeds with the last processed RGB frame; every colour it
draws comes from the active QPalette so it tracks the light/dark theme.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainter, QPalette
from PySide6.QtWidgets import QSizePolicy, QWidget

# Lab-L (uint8, 0..255) split into three value masses at even thirds.
_DARK_MAX = 256.0 / 3.0
_LIGHT_MIN = 512.0 / 3.0


# ---------------------------------------------------------------------------
# Pure maths (Qt-free, unit-testable)
# ---------------------------------------------------------------------------
def _as_uint8_rgb(rgb_image) -> np.ndarray:
    """Coerce an image-like into a contiguous HxWx3 uint8 RGB array."""
    arr = np.asarray(rgb_image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("expected an HxWx3 RGB image")
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _lab_l_channel(rgb_image) -> np.ndarray:
    """Convert an RGB image to OpenCV Lab once and return the uint8 L channel.

    Shared by :func:`value_histogram` and :func:`value_mass_split` so a
    widget refresh that needs both only pays for one Lab conversion (see
    :func:`_value_histogram_from_l` / :func:`_value_mass_split_from_l`).
    """
    import cv2  # local import keeps the maths usable without cv2 at import time

    arr = _as_uint8_rgb(rgb_image)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2Lab)[..., 0]


def _value_histogram_from_l(lab_l: np.ndarray, bins: int = 16) -> np.ndarray:
    """Histogram a precomputed Lab-L uint8 array into ``bins`` equal bands."""
    bins = max(1, int(bins))
    if lab_l.size == 0:
        return np.zeros(bins, dtype=np.int64)
    counts, _ = np.histogram(lab_l, bins=bins, range=(0.0, 256.0))
    return counts.astype(np.int64)


def _value_mass_split_from_l(lab_l: np.ndarray) -> Tuple[float, float, float]:
    """Split a precomputed Lab-L uint8 array into (dark, mid, light) fractions."""
    total = lab_l.size
    if total == 0:
        return (0.0, 0.0, 0.0)
    lab_l = lab_l.astype(np.float32)
    dark = int(np.count_nonzero(lab_l < _DARK_MAX))
    light = int(np.count_nonzero(lab_l >= _LIGHT_MIN))
    mid = total - dark - light
    return (dark / total, mid / total, light / total)


def value_histogram(rgb_image, bins: int = 16) -> np.ndarray:
    """Count pixels per Lab-L value band.

    Converts ``rgb_image`` (HxWx3, RGB) to OpenCV's uint8 Lab, then histograms
    the L channel into ``bins`` equal bands across the full 0..255 L range.
    Returns an ``int64`` array of length ``bins`` whose sum is the pixel count;
    black lands in the lowest band, white in the highest. Deterministic.
    """
    return _value_histogram_from_l(_lab_l_channel(rgb_image), bins)


def value_mass_split(rgb_image) -> Tuple[float, float, float]:
    """Return the (dark, mid, light) fraction of pixels by Lab-L thirds.

    Bands: dark ``L < 256/3``, mid in between, light ``L >= 512/3``. The three
    fractions sum to 1 for a non-empty image (``(0, 0, 0)`` for an empty one).
    Computed in float32 (previously float64); results may differ from the
    prior implementation by at most 1e-6.
    """
    return _value_mass_split_from_l(_lab_l_channel(rgb_image))


# ---------------------------------------------------------------------------
# Widget (thin, paint-only, theme-agnostic)
# ---------------------------------------------------------------------------
class ValueHistogram(QWidget):
    """Bars of the value distribution, coloured from the active QPalette.

    Feed it the last processed RGB frame with :meth:`set_image`; :meth:`clear`
    empties it. Bars, background, dividers and labels all come from palette
    roles (Base/Highlight/Mid/WindowText) so the widget tracks the theme.
    """

    def __init__(self, bins: int = 16, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._bins = max(1, int(bins))
        self._counts: Optional[np.ndarray] = None
        self._split: Optional[Tuple[float, float, float]] = None
        self.setMinimumHeight(96)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_image(self, rgb) -> None:
        """Recompute the histogram from an RGB frame (clear if None/empty)."""
        if rgb is None:
            self.clear()
            return
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[2] < 3 or arr.size == 0:
            self.clear()
            return
        lab_l = _lab_l_channel(arr)
        self._counts = _value_histogram_from_l(lab_l, self._bins)
        self._split = _value_mass_split_from_l(lab_l)
        self.update()

    def clear(self) -> None:
        """Drop the current distribution and repaint empty."""
        self._counts = None
        self._split = None
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        pal = self.palette()
        rect = self.rect()
        painter.fillRect(rect, pal.color(QPalette.ColorRole.Base))

        counts = self._counts
        if counts is None or int(counts.sum()) == 0:
            painter.setPen(pal.color(QPalette.ColorRole.PlaceholderText))
            painter.drawText(rect, int(Qt.AlignCenter), "No image")
            painter.end()
            return

        fm = painter.fontMetrics()
        label_h = fm.height() + 4
        margin = 6
        left = margin
        top = margin
        right = rect.width() - margin
        bottom = rect.height() - margin - label_h
        plot_w = max(1, right - left)
        plot_h = max(1, bottom - top)

        n = len(counts)
        peak = max(1, int(counts.max()))

        # Faint thirds dividers behind the bars (dark | mid | light regions).
        painter.setPen(pal.color(QPalette.ColorRole.Mid))
        for frac in (1.0 / 3.0, 2.0 / 3.0):
            x = left + plot_w * frac
            painter.drawLine(QPointF(x, float(top)), QPointF(x, float(bottom)))

        # Bars, from the Highlight accent (reads on both themes).
        painter.setPen(Qt.NoPen)
        painter.setBrush(pal.color(QPalette.ColorRole.Highlight))
        slot = plot_w / n
        bar_w = max(1.0, slot - 1.0)
        for i in range(n):
            bar_h = plot_h * (int(counts[i]) / peak)
            x = left + i * slot
            painter.drawRect(QRectF(x, bottom - bar_h, bar_w, bar_h))

        # Baseline under the bars.
        painter.setPen(pal.color(QPalette.ColorRole.Mid))
        painter.drawLine(
            QPointF(float(left), float(bottom)), QPointF(float(right), float(bottom))
        )

        # Dark / mid / light mass labels below the plot.
        if self._split is not None:
            painter.setPen(pal.color(QPalette.ColorRole.WindowText))
            third = plot_w / 3.0
            for j, name in enumerate(("dark", "mid", "light")):
                seg = QRectF(left + j * third, float(bottom + 2), third, float(label_h))
                painter.drawText(
                    seg,
                    int(Qt.AlignCenter),
                    "{} {:.0f}%".format(name, self._split[j] * 100.0),
                )
        painter.end()
