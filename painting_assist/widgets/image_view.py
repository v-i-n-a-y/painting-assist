# Copyright 2026 Vinay Williams

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPalette,
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

from painting_assist.measure import Calibration
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
    and a fine tip at the bottom-left — over a 24x24 logical area, white filled
    with a black outline so it reads against any image. The pixmap is rendered at
    a 2x device pixel ratio so the antialiased edges stay crisp on HiDPI screens
    instead of looking soft. The hotspot is the dropper tip (bottom-left), where
    the sample is taken.
    """
    global _EYEDROPPER_CURSOR
    if _EYEDROPPER_CURSOR is not None:
        return _EYEDROPPER_CURSOR

    size = 24
    ratio = 2.0  # render at 2x; drawing stays in logical 24x24 coordinates
    pixmap = QPixmap(int(size * ratio), int(size * ratio))
    pixmap.setDevicePixelRatio(ratio)
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
        # Explicit layout fractions when supplied; otherwise fall back to an even
        # columns x rows lattice (keeps older/plain specs working).
        x_fractions = spec.get("x_fractions")
        if x_fractions is None:
            x_fractions = [i / cols for i in range(1, cols)]
        y_fractions = spec.get("y_fractions")
        if y_fractions is None:
            y_fractions = [j / rows for j in range(1, rows)]
        for fx in x_fractions:
            x = rect.left() + rect.width() * float(fx)
            painter.drawLine(QLineF(x, rect.top(), x, rect.bottom()))
        for fy in y_fractions:
            y = rect.top() + rect.height() * float(fy)
            painter.drawLine(QLineF(rect.left(), y, rect.right(), y))
        # Layout-provided diagonal segments (e.g. the armature), as fraction pairs.
        for seg in spec.get("diagonal_lines") or ():
            (fx0, fy0), (fx1, fy1) = seg
            p0 = QPointF(
                rect.left() + rect.width() * float(fx0),
                rect.top() + rect.height() * float(fy0),
            )
            p1 = QPointF(
                rect.left() + rect.width() * float(fx1),
                rect.top() + rect.height() * float(fy1),
            )
            painter.drawLine(QLineF(p0, p1))
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

    # Emitted after any zoom change (wheel, fit-to-window, reset) with the new
    # view scale factor from ``current_scale`` (1.0 == 100%), so the window can
    # show a live zoom percentage.
    zoomChanged = Signal(float)

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
        # Canvas calibration pushed to every measure item for physical readouts.
        self._measure_cal = Calibration()

        # Non-destructive grid overlay, drawn above the pixmap.
        self._grid_item = GridOverlayItem()
        self._scene.addItem(self._grid_item)
        # Last grid spec, kept so the foreground can label gridline positions.
        self._grid_spec: Optional[Dict[str, Any]] = None

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

        # Take keyboard focus so the window's B/V/F shortcuts keep working after
        # a click lands in a side dock; focus is claimed when the first image
        # loads (see set_image), not on construction.
        self.setFocusPolicy(Qt.StrongFocus)

        # Whether an image has ever been shown (first image always fits).
        self._has_image = False

        # Centred placeholder shown over the blank viewport before any image is
        # loaded; the window may override the copy via set_placeholder.
        self._placeholder_lines: list[str] = [
            "Open an image (Ctrl+O)",
            "or drag one here",
        ]

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
            # Claim keyboard focus on an explicit load so the window's key
            # shortcuts route to the viewport; not on preserve-view re-renders,
            # which must not steal focus back from the control panels.
            self.setFocus()

    def set_grid_overlay(self, spec: Optional[Dict[str, Any]]) -> None:
        """Show/update the non-destructive grid overlay, or hide it (``None``).

        ``spec`` is a ``GridControl.overlay_spec()`` dict; passing ``None`` (or a
        spec whose ``visible`` is false) hides the overlay.
        """
        self._grid_item.set_spec(spec)
        self._grid_spec = spec
        self.viewport().update()  # refresh the canvas-position gridline labels

    def set_placeholder(self, lines: list[str]) -> None:
        """Set the centred empty-state copy shown before any image is loaded.

        Each string is one centred line. The overlay is only painted while no
        image has been shown; once an image loads it never reappears.
        """
        self._placeholder_lines = list(lines)
        if not self._has_image:
            self.viewport().update()

    def fit_to_window(self) -> None:
        """Scale the view so the whole image is visible, preserving aspect ratio."""
        if self._item.pixmap().isNull():
            return
        self.fitInView(self._item, Qt.KeepAspectRatio)
        self.zoomChanged.emit(self.current_scale())

    def reset_zoom(self) -> None:
        """Reset the view transform to 1:1 (one image pixel per screen pixel)."""
        self.resetTransform()
        self.zoomChanged.emit(self.current_scale())

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
            self.viewport().update()  # clear the foreground label of the old tool
            if not self._crop_editing and not self._eyedropper:
                self.setDragMode(QGraphicsView.ScrollHandDrag)
            return

        if self._item.pixmap().isNull():
            return

        item = self._ensure_measure_item(mode)
        item.show()
        self._measure_mode = mode
        self.setDragMode(QGraphicsView.NoDrag)
        self.viewport().update()  # draw the tool's label at its start position
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
        item.set_calibration(self._measure_cal)
        item.changed.connect(self.measureChanged)
        # Repaint the whole viewport on any change so the foreground label is
        # redrawn from scratch (never leaving a trail where it used to be).
        item.changed.connect(lambda _text: self.viewport().update())
        self._scene.addItem(item)
        self._measure_items[mode] = item
        return item

    def set_measure_calibration(self, cal: Calibration) -> None:
        """Push a new canvas calibration to every measure tool and refresh.

        Updates the physical-unit readouts live: existing items re-format, the
        foreground labels repaint, and the active tool's status-bar readout is
        re-emitted so the window's status line updates immediately.
        """
        self._measure_cal = cal
        for item in self._measure_items.values():
            item.set_calibration(cal)
        self.viewport().update()
        if self._measure_mode is not None:
            item = self._measure_items.get(self._measure_mode)
            if item is not None:
                self.measureChanged.emit(item.readout())

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
            self.zoomChanged.emit(self.current_scale())
        event.accept()

    # ------------------------------------------------------------------ #
    # Empty-state placeholder
    # ------------------------------------------------------------------ #
    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        """Paint centred placeholder copy over the blank viewport (no image yet).

        Drawn in device (viewport) coordinates with the theme's placeholder-text
        palette colour so it stays legible in light and dark themes, and only
        while ``_has_image`` is false — the first loaded image hides it for good.
        """
        super().drawForeground(painter, rect)
        self._draw_grid_labels(painter)
        self._draw_measure_labels(painter)
        if self._has_image or not self._placeholder_lines:
            return
        painter.save()
        painter.setWorldMatrixEnabled(False)  # draw in viewport (device) coords
        painter.setRenderHint(QPainter.Antialiasing, True)
        font = painter.font()
        pt = font.pointSizeF()
        if pt > 0:
            font.setPointSizeF(pt * 1.15)
        painter.setFont(font)
        painter.setPen(self.palette().color(QPalette.PlaceholderText))
        painter.drawText(
            QRectF(self.viewport().rect()),
            Qt.AlignCenter,
            "\n".join(self._placeholder_lines),
        )
        painter.restore()

    # ------------------------------------------------------------------ #
    # Grid + measure labels (drawn in the foreground, never clipped/trailed)
    # ------------------------------------------------------------------ #
    def _draw_grid_labels(self, painter: QPainter) -> None:
        """Label each gridline with its position on the canvas.

        Only drawn when the grid is visible and a physical canvas size is set
        (a canvas position is meaningless otherwise); labels use the same unit as
        the measure tools, so ``fraction_str`` returns ``""`` -- skipped here --
        whenever no physical reading is available.
        """
        spec = self._grid_spec
        if not spec or not spec.get("visible") or self._item.pixmap().isNull():
            return
        cal = self._measure_cal
        rect = self._item.sceneBoundingRect()
        painter.save()
        painter.setWorldMatrixEnabled(False)
        painter.setRenderHint(QPainter.Antialiasing, True)
        fm = QFontMetrics(painter.font())
        vp_rect = self.viewport().rect()
        for fx in spec.get("x_fractions") or []:
            text = cal.fraction_str(fx, "x")
            if not text:
                continue
            sx = rect.left() + rect.width() * float(fx)
            vp = self.mapFromScene(QPointF(sx, rect.top()))
            self._paint_measure_label(
                painter, fm, vp.x(), vp.y(), text, vp_rect, "below"
            )
        for fy in spec.get("y_fractions") or []:
            text = cal.fraction_str(fy, "y")
            if not text:
                continue
            sy = rect.top() + rect.height() * float(fy)
            vp = self.mapFromScene(QPointF(rect.left(), sy))
            self._paint_measure_label(
                painter, fm, vp.x(), vp.y(), text, vp_rect, "below"
            )
        painter.restore()

    def _draw_measure_labels(self, painter: QPainter) -> None:
        """Paint the active measure tool's labels in viewport coordinates.

        Drawing here (rather than in each item's ``paint``) means labels are not
        clipped to the item's bounding rect, and because the whole viewport is
        repainted on every change they never leave a trail when the tool moves.
        """
        if self._measure_mode is None:
            return
        item = self._measure_items.get(self._measure_mode)
        if item is None or not item.isVisible():
            return
        specs = item.label_specs()
        if not specs:
            return
        painter.save()
        painter.setWorldMatrixEnabled(False)  # device (viewport) coordinates
        painter.setRenderHint(QPainter.Antialiasing, True)
        fm = QFontMetrics(painter.font())
        vp_rect = self.viewport().rect()
        for anchor, text, place in specs:
            vp = self.mapFromScene(item.mapToScene(anchor))
            self._paint_measure_label(
                painter, fm, vp.x(), vp.y(), str(text), vp_rect, place
            )
        painter.restore()

    @staticmethod
    def _paint_measure_label(painter, fm, ax, ay, text, vp_rect, place="above") -> None:
        """Draw one label chip near ``(ax, ay)``, clamped inside ``vp_rect``.

        ``place`` puts the chip ``"above"`` or ``"below"`` the anchor so a point
        can carry a reading above and an edge chip below without overlapping.
        """
        pad = 3
        w = fm.horizontalAdvance(text) + 2 * pad
        h = fm.height() + 2 * pad
        x = ax + 10
        y = ay + 10 if place == "below" else ay - 10 - h
        # Keep the whole chip on-screen even when the anchor is near an edge.
        x = min(
            max(x, vp_rect.left() + 1), max(vp_rect.left() + 1, vp_rect.right() - w)
        )
        y = min(max(y, vp_rect.top() + 1), max(vp_rect.top() + 1, vp_rect.bottom() - h))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRect(QRectF(x, y, w, h))
        painter.setPen(QColor(255, 255, 255, 240))
        painter.drawText(int(x + pad), int(y + pad + fm.ascent()), text)
