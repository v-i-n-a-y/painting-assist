# Copyright 2026 Vinay Williams

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from painting_assist.utils.image_qt import ndarray_to_qpixmap
from painting_assist.widgets.crop_item import CropItem
from painting_assist.widgets.measure_items import (
    AngleGaugeItem,
    CaliperItem,
    GuidesItem,
)

_EYEDROPPER_CURSOR: Optional[QCursor] = None


def _eyedropper_cursor() -> QCursor:
    """Return a cached eyedropper-shaped cursor (built once, lazily).

    Draws a classic dropper — an angled body ending in a bulb at the top-right
    and a fine tip at the bottom-left — as a 24x24 transparent pixmap, white
    filled with a black outline so it reads against any image. The hotspot is
    the dropper tip (bottom-left), where the sample is taken.
    """
    global _EYEDROPPER_CURSOR
    if _EYEDROPPER_CURSOR is not None:
        return _EYEDROPPER_CURSOR

    size = 24
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    outline = QPen(QColor(0, 0, 0))
    outline.setWidthF(1.2)
    outline.setJoinStyle(Qt.RoundJoin)
    outline.setCapStyle(Qt.RoundCap)
    painter.setPen(outline)
    painter.setBrush(QColor(255, 255, 255))

    # Angled body: a slim parallelogram running from the tip (bottom-left) up to
    # the bulb (top-right), drawn along the pixmap diagonal.
    body = QPolygonF(
        [
            QPointF(3.0, 21.0),  # tip
            QPointF(5.5, 18.5),
            QPointF(16.5, 7.5),
            QPointF(19.0, 10.0),
            QPointF(8.0, 21.0),
            QPointF(5.5, 23.5),
        ]
    )
    painter.drawPolygon(body)

    # Bulb at the top-right end of the body.
    bulb = QPainterPath()
    bulb.addEllipse(QPointF(18.0, 6.0), 4.0, 4.0)
    painter.drawPath(bulb)

    painter.end()

    _EYEDROPPER_CURSOR = QCursor(pixmap, 3, 21)  # hotspot at the tip
    return _EYEDROPPER_CURSOR


class GridOverlayItem(QGraphicsItem):
    """Non-destructive positioning grid drawn over the pixmap in the viewport.

    The lines are computed from a geometry rectangle (the pixmap item's
    ``sceneBoundingRect``) rather than raw pixel counts, so the overlay lands in
    the same place whether the view is showing a full-resolution frame or a
    downscaled interactive preview. Pens are cosmetic (a constant device-pixel
    width) so the lines stay crisp and readable at any zoom instead of scaling
    with the image. The item is hidden unless a spec with ``visible`` true has
    been supplied.
    """

    def __init__(self, parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self._rect = QRectF()
        self._spec: Optional[Dict[str, Any]] = None
        # Above the pixmap; the crop item (z=1000) still sits above this.
        self.setZValue(500)
        self.hide()

    def set_geometry(self, rect: QRectF) -> None:
        """Set the image rectangle (scene coords) the grid divides."""
        self.prepareGeometryChange()
        self._rect = QRectF(rect)
        self.update()

    def set_spec(self, spec: Optional[Dict[str, Any]]) -> None:
        """Apply an overlay spec (see ``GridControl.overlay_spec``) or hide (None)."""
        self._spec = spec
        visible = bool(spec) and bool(spec.get("visible", True))
        self.setVisible(visible)
        self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(self._rect)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        spec = self._spec
        if spec is None or self._rect.isEmpty():
            return

        r, g, b = spec.get("color_rgb", (220, 30, 30))
        opacity = max(0.0, min(1.0, float(spec.get("opacity", 1.0))))
        color = QColor(int(r), int(g), int(b))
        color.setAlphaF(opacity)

        pen = QPen(color)
        pen.setCosmetic(True)  # width is in device pixels, constant across zoom
        pen.setWidth(max(1, int(spec.get("thickness", 1))))
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = self._rect
        cols = max(1, int(spec.get("columns", 1)))
        rows = max(1, int(spec.get("rows", 1)))
        for i in range(1, cols):
            x = rect.left() + rect.width() * i / cols
            painter.drawLine(QLineF(x, rect.top(), x, rect.bottom()))
        for j in range(1, rows):
            y = rect.top() + rect.height() * j / rows
            painter.drawLine(QLineF(rect.left(), y, rect.right(), y))
        if bool(spec.get("diagonals", False)):
            painter.drawLine(QLineF(rect.topLeft(), rect.bottomRight()))
            painter.drawLine(QLineF(rect.topRight(), rect.bottomLeft()))


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

    # Emitted by the eyedropper on click/drag: normalised (x, y) fractions 0..1
    # of the displayed image, so the sample is resolution- and preview-scale-
    # independent (the window samples its full processed frame at these coords).
    colourSampled = Signal(float, float)

    # Emitted by the active measure tool: a human-readable readout string (an
    # angle, a length ratio, or guide positions) for the status bar.
    measureChanged = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Create the scene + single pixmap item and configure interaction modes."""
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setScene(self._scene)

        self._crop_item: Optional[CropItem] = None
        self._crop_editing = False
        self._eyedropper = False

        # Lazily created measure overlays, keyed by mode name; only one is
        # visible at a time (see set_measure_mode).
        self._measure_items: Dict[str, QGraphicsItem] = {}
        self._measure_mode: Optional[str] = None

        # Non-destructive grid overlay, drawn above the pixmap.
        self._grid_item = GridOverlayItem()
        self._scene.addItem(self._grid_item)

        # Pan with the left mouse button.
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        # Zoom about the point under the cursor.
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        # Smooth scaling for a pleasant reference view.
        self.setRenderHints(
            self.renderHints() | QPainter.SmoothPixmapTransform | QPainter.Antialiasing
        )
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Whether an image has ever been shown (first image always fits).
        self._has_image = False

    def set_image(
        self,
        rgb: np.ndarray,
        preserve_view: bool = True,
        display_scale: float = 1.0,
    ) -> None:
        """Display ``rgb`` (RGB uint8 HxWx3) by swapping the pixmap on the reused item.

        On the first image (or when ``preserve_view`` is False) the view fits the
        image to the window. Otherwise the current zoom/pan are kept so slider
        drags do not cause the view to jump or refit.

        ``display_scale`` is the factor the incoming pixels were rendered at
        relative to the full image (1.0 for a full-res frame, e.g. 0.4 for an
        interactive preview). The pixmap item is scaled by ``1 / display_scale``
        and the scene rect kept at full-image size, so a downscaled preview
        occupies the same on-screen geometry as the full frame instead of
        visibly shrinking during slider drags.
        """
        pixmap = ndarray_to_qpixmap(rgb)
        self._item.setPixmap(pixmap)
        self._item.setScale(1.0 / display_scale if display_scale > 0 else 1.0)
        # Keep the scene rectangle at full-image size (item scale accounted for).
        scene_rect = self._item.sceneBoundingRect()
        self._scene.setSceneRect(scene_rect)
        if self._crop_item is not None:
            self._crop_item.set_image_rect(scene_rect)
        # Keep the grid overlay aligned across image swaps and re-renders
        # (including the hold-B before/after swaps, which go through here).
        self._grid_item.set_geometry(scene_rect)
        # Keep any measure overlays aligned to the new frame too.
        for item in self._measure_items.values():
            item.set_image_rect(scene_rect)

        first_image = not self._has_image
        self._has_image = True

        if first_image or not preserve_view:
            self.fit_to_window()

    def set_grid_overlay(self, spec: Optional[Dict[str, Any]]) -> None:
        """Show/update the non-destructive grid overlay, or hide it (``None``).

        ``spec`` is a ``GridControl.overlay_spec()`` dict; passing ``None`` (or a
        spec whose ``visible`` is false) hides the overlay.
        """
        self._grid_item.set_spec(spec)

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

    # ------------------------------------------------------------------ #
    # Interactive measuring overlays
    # ------------------------------------------------------------------ #
    def set_measure_mode(self, mode: Optional[str]) -> None:
        """Activate one measuring tool, or clear all of them (``None``).

        ``mode`` is one of ``"angle"``, ``"caliper"``, ``"guides"`` or ``None``.
        The chosen tool's item is created lazily on first use and reused after.
        While a tool is active, panning is disabled (like the crop overlay) so
        the item receives mouse events; the wheel still zooms. ``None`` hides
        every tool and restores pan (unless the crop overlay is being edited).

        Measure mode and the eyedropper are mutually exclusive; the caller
        enforces that, but ``set_measure_mode(None)`` fully clears measure state.
        """
        for item in self._measure_items.values():
            item.hide()

        if mode is None:
            self._measure_mode = None
            if not self._crop_editing and not self._eyedropper:
                self.setDragMode(QGraphicsView.ScrollHandDrag)
            return

        if self._item.pixmap().isNull():
            return

        item = self._ensure_measure_item(mode)
        item.show()
        self._measure_mode = mode
        self.setDragMode(QGraphicsView.NoDrag)
        self.measureChanged.emit(item.readout())

    def measure_mode(self) -> Optional[str]:
        """Return the active measure tool name, or ``None``."""
        return self._measure_mode

    def _ensure_measure_item(self, mode: str) -> QGraphicsItem:
        """Return the item for ``mode``, creating and wiring it on first use."""
        scene_rect = self._item.sceneBoundingRect()
        existing = self._measure_items.get(mode)
        if existing is not None:
            existing.set_image_rect(scene_rect)
            return existing
        factory = {
            "angle": AngleGaugeItem,
            "caliper": CaliperItem,
            "guides": GuidesItem,
        }[mode]
        item = factory(scene_rect)
        item.changed.connect(self.measureChanged)
        self._scene.addItem(item)
        self._measure_items[mode] = item
        return item

    # ------------------------------------------------------------------ #
    # Eyedropper
    # ------------------------------------------------------------------ #
    def set_eyedropper(self, active: bool) -> None:
        """Toggle eyedropper mode: dropper cursor, click samples a colour.

        While active, panning is disabled so a press samples instead of dragging
        the view (the wheel still zooms). Ignored while the crop overlay is
        being edited — the two press-driven modes must not fight over the mouse.
        """
        if self._crop_editing:
            active = False
        self._eyedropper = active
        if active:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(_eyedropper_cursor())
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().unsetCursor()

    def eyedropper_active(self) -> bool:
        """Whether eyedropper mode is currently on."""
        return self._eyedropper

    def _sample_at(self, view_pos) -> None:
        """Map a viewport position to normalised image coords and emit a sample."""
        if self._item.pixmap().isNull():
            return
        item_pos = self._item.mapFromScene(self.mapToScene(view_pos))
        w = float(self._item.pixmap().width())
        h = float(self._item.pixmap().height())
        if w <= 0 or h <= 0:
            return
        xn = item_pos.x() / w
        yn = item_pos.y() / h
        if 0.0 <= xn < 1.0 and 0.0 <= yn < 1.0:
            self.colourSampled.emit(xn, yn)

    def mousePressEvent(self, event) -> None:
        """In eyedropper mode a left press samples the colour under the cursor."""
        if self._eyedropper and event.button() == Qt.LeftButton:
            self._sample_at(event.position().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """In eyedropper mode dragging with the left button keeps sampling."""
        if self._eyedropper and (event.buttons() & Qt.LeftButton):
            self._sample_at(event.position().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

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
