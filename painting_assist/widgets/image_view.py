from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from painting_assist.utils.image_qt import ndarray_to_qpixmap
from painting_assist.widgets.crop_item import CropItem


class ImageView(QGraphicsView):
    """A zoom/pan/fit image viewport built on ``QGraphicsView``.

    The view owns a single :class:`QGraphicsScene` holding one reused
    :class:`QGraphicsPixmapItem`. Re-rendering the processed image swaps the
    pixmap on that same item rather than rebuilding the scene, so the painter's
    current zoom and pan survive slider-driven re-renders (view-state
    preservation). Wheel events zoom about the cursor; left-drag pans.
    """

    MIN_SCALE = 0.05
    MAX_SCALE = 40.0

    # Emitted while the crop overlay is edited; carries the normalised rect
    # (rx, ry, rw, rh) as fractions 0..1 of the displayed image.
    cropRectChanged = Signal(float, float, float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Create the scene + single pixmap item and configure interaction modes."""
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setScene(self._scene)

        self._crop_item: Optional[CropItem] = None
        self._crop_editing = False

        # Pan with the left mouse button.
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        # Zoom about the point under the cursor.
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        # Smooth scaling for a pleasant reference view.
        self.setRenderHints(
            self.renderHints()
            | QPainter.SmoothPixmapTransform
            | QPainter.Antialiasing
        )
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Whether an image has ever been shown (first image always fits).
        self._has_image = False

    def set_image(self, rgb: np.ndarray, preserve_view: bool = True) -> None:
        """Display ``rgb`` (RGB uint8 HxWx3) by swapping the pixmap on the reused item.

        On the first image (or when ``preserve_view`` is False) the view fits the
        image to the window. Otherwise the current zoom/pan are kept so slider
        drags do not cause the view to jump or refit.
        """
        pixmap = ndarray_to_qpixmap(rgb)
        self._item.setPixmap(pixmap)
        # Keep the scene rectangle tight around the current pixmap.
        self._scene.setSceneRect(self._item.boundingRect())
        if self._crop_item is not None:
            self._crop_item.set_image_rect(self._item.boundingRect())

        first_image = not self._has_image
        self._has_image = True

        if first_image or not preserve_view:
            self.fit_to_window()

    def fit_to_window(self) -> None:
        """Scale the view so the whole image is visible, preserving aspect ratio."""
        if self._item.pixmap().isNull():
            return
        self.fitInView(self._item, Qt.KeepAspectRatio)

    def reset_zoom(self) -> None:
        """Reset the view transform to 1:1 (one image pixel per screen pixel)."""
        self.resetTransform()

    def current_scale(self) -> float:
        """Return the current horizontal scale factor of the view transform."""
        return float(self.transform().m11())

    # ------------------------------------------------------------------ #
    # Interactive crop overlay
    # ------------------------------------------------------------------ #
    def begin_crop(self, rect_norm, aspect: Optional[float]) -> None:
        """Show the crop overlay, seeded from ``rect_norm`` and locked to ``aspect``.

        ``rect_norm`` is ``(rx, ry, rw, rh)`` in 0..1 image fractions; passing the
        full frame ``(0, 0, 1, 1)`` with a locked aspect yields the largest
        centred aspect-correct box. Panning is disabled so the overlay receives
        mouse events; the wheel still zooms.
        """
        if self._item.pixmap().isNull():
            return
        image_rect = self._item.boundingRect()
        if self._crop_item is None:
            self._crop_item = CropItem(image_rect)
            self._crop_item.rectChanged.connect(self._on_crop_rect)
            self._scene.addItem(self._crop_item)
        else:
            self._crop_item.set_image_rect(image_rect)
            self._crop_item.show()

        w = max(1.0, float(self._item.pixmap().width()))
        h = max(1.0, float(self._item.pixmap().height()))
        rx, ry, rw, rh = rect_norm
        self._crop_item.set_rect(QRectF(rx * w, ry * h, rw * w, rh * h))
        self._crop_item.set_aspect(aspect)

        self._crop_editing = True
        self.setDragMode(QGraphicsView.NoDrag)

    def end_crop(self) -> None:
        """Hide the crop overlay and restore normal pan/zoom interaction."""
        self._crop_editing = False
        if self._crop_item is not None:
            self._crop_item.hide()
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_crop_aspect(self, aspect: Optional[float]) -> None:
        """Update the locked aspect ratio of the live crop overlay."""
        if self._crop_item is not None and self._crop_editing:
            self._crop_item.set_aspect(aspect)

    def _on_crop_rect(self, rect: QRectF) -> None:
        """Convert an overlay rect (image px) to normalised fractions and emit."""
        w = max(1.0, float(self._item.pixmap().width()))
        h = max(1.0, float(self._item.pixmap().height()))
        self.cropRectChanged.emit(
            rect.x() / w, rect.y() / h, rect.width() / w, rect.height() / h
        )

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom about the cursor, clamping the cumulative scale to a sane range."""
        if self._item.pixmap().isNull():
            event.ignore()
            return

        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return

        # Smooth, proportional zoom factor.
        factor = 1.25 if delta > 0 else 1.0 / 1.25

        current = self.current_scale()
        target = current * factor
        if target < self.MIN_SCALE:
            factor = self.MIN_SCALE / current
        elif target > self.MAX_SCALE:
            factor = self.MAX_SCALE / current

        if factor != 1.0:
            self.scale(factor, factor)
        event.accept()
