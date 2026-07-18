# Copyright 2026 Vinay Williams

"""Interactive crop rectangle drawn over the image in the viewport.

:class:`CropItem` is a :class:`QGraphicsObject` placed at the scene origin (the
scene's coordinate system equals image-pixel space, since the pixmap item also
sits at the origin at 1:1). It paints a dimmed surround, a rule-of-thirds
framed rectangle, and resize handles, and lets the user move/resize that
rectangle with the mouse. When the aspect ratio is locked the rectangle keeps
that ratio and only corner handles are offered; freeform mode adds edge
handles. Geometry is reported via :attr:`rectChanged` in image-pixel
coordinates; the view converts to normalised fractions for the control.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsObject


# Handle identifiers.
_TL, _TR, _BR, _BL = "tl", "tr", "br", "bl"  # corners
_TOP, _RIGHT, _BOTTOM, _LEFT = "top", "right", "bottom", "left"  # edges
_CORNERS = (_TL, _TR, _BR, _BL)
_EDGES = (_TOP, _RIGHT, _BOTTOM, _LEFT)


class CropItem(QGraphicsObject):
    """Movable/resizable crop rectangle with aspect locking."""

    rectChanged = Signal(QRectF)  # current crop rect in image-pixel coords

    HANDLE_PX = 9.0  # on-screen handle half-size (constant regardless of zoom)
    MIN_SIZE = 16.0  # minimum crop size in image pixels

    def __init__(
        self, image_rect: QRectF, parent: Optional[QGraphicsObject] = None
    ) -> None:
        super().__init__(parent)
        self._image_rect = QRectF(image_rect)
        self._rect = QRectF(image_rect)
        self._aspect: Optional[float] = None
        self._mode: Optional[str] = None  # None | "move" | handle id
        self._press_pos = QPointF()
        self._press_rect = QRectF()

        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setZValue(1000)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def set_image_rect(self, image_rect: QRectF) -> None:
        """Update the bounds (a new image was loaded) and clamp the crop."""
        self.prepareGeometryChange()
        self._image_rect = QRectF(image_rect)
        self._rect = self._clamp_rect(self._rect)
        self.update()

    def set_rect(self, rect: QRectF) -> None:
        """Set the crop rectangle (image-pixel coords), clamped to the image."""
        self.prepareGeometryChange()
        self._rect = self._clamp_rect(QRectF(rect))
        if self._aspect is not None:
            self._rect = self._apply_aspect_centered(self._rect)
        self.update()

    def rect(self) -> QRectF:
        """Return the current crop rectangle in image-pixel coords."""
        return QRectF(self._rect)

    def set_aspect(self, aspect: Optional[float]) -> None:
        """Set the locked width/height ratio (``None`` = freeform) and reflow."""
        self.prepareGeometryChange()
        self._aspect = aspect
        if aspect is not None:
            self._rect = self._apply_aspect_centered(self._rect)
        self.update()
        self.rectChanged.emit(QRectF(self._rect))

    # ------------------------------------------------------------------ #
    # QGraphicsItem overrides
    # ------------------------------------------------------------------ #
    def boundingRect(self) -> QRectF:
        m = self.HANDLE_PX / self._scale() + 1.0
        return self._image_rect.adjusted(-m, -m, m, m)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        scale = self._scale()

        # Dim everything outside the crop rectangle.
        outside = QPainterPath()
        outside.addRect(self._image_rect)
        inside = QPainterPath()
        inside.addRect(self._rect)
        painter.fillPath(outside.subtracted(inside), QColor(0, 0, 0, 120))

        # Rule-of-thirds guides.
        thirds_pen = QPen(QColor(255, 255, 255, 90))
        thirds_pen.setWidthF(1.0 / scale)
        painter.setPen(thirds_pen)
        r = self._rect
        for i in (1, 2):
            x = r.left() + r.width() * i / 3.0
            painter.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            y = r.top() + r.height() * i / 3.0
            painter.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))

        # Border.
        border = QPen(QColor(255, 255, 255, 230))
        border.setWidthF(1.6 / scale)
        painter.setPen(border)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(r)

        # Handles.
        hs = self.HANDLE_PX / scale
        painter.setPen(QPen(QColor(40, 40, 40, 230), 1.0 / scale))
        painter.setBrush(QBrush(QColor(255, 255, 255, 240)))
        for _id, c in self._handle_centers().items():
            painter.drawRect(QRectF(c.x() - hs, c.y() - hs, 2 * hs, 2 * hs))

    # ------------------------------------------------------------------ #
    # Mouse interaction
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event) -> None:
        pos = event.pos()
        self._mode = self._zone_at(pos)
        if self._mode is None:
            event.ignore()
            return
        self._press_pos = QPointF(pos)
        self._press_rect = QRectF(self._rect)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._mode is None:
            event.ignore()
            return
        pos = event.pos()
        if self._mode == "move":
            self._do_move(pos)
        elif self._mode in _CORNERS:
            self._do_corner(self._mode, pos)
        else:
            self._do_edge(self._mode, pos)
        self.update()
        self.rectChanged.emit(QRectF(self._rect))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._mode is None:
            event.ignore()
            return
        self._mode = None
        self.rectChanged.emit(QRectF(self._rect))
        event.accept()

    def hoverMoveEvent(self, event) -> None:
        zone = self._zone_at(event.pos())
        self.setCursor(self._cursor_for(zone))
        super().hoverMoveEvent(event)

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def _scale(self) -> float:
        scene = self.scene()
        if scene is not None:
            views = scene.views()
            if views:
                m = float(views[0].transform().m11())
                if m > 1e-6:
                    return m
        return 1.0

    def _active_handles(self):
        ids = list(_CORNERS)
        if self._aspect is None:
            ids += list(_EDGES)
        return ids

    def _handle_centers(self):
        r = self._rect
        cx = (r.left() + r.right()) / 2.0
        cy = (r.top() + r.bottom()) / 2.0
        centers = {
            _TL: QPointF(r.left(), r.top()),
            _TR: QPointF(r.right(), r.top()),
            _BR: QPointF(r.right(), r.bottom()),
            _BL: QPointF(r.left(), r.bottom()),
        }
        if self._aspect is None:
            centers.update(
                {
                    _TOP: QPointF(cx, r.top()),
                    _RIGHT: QPointF(r.right(), cy),
                    _BOTTOM: QPointF(cx, r.bottom()),
                    _LEFT: QPointF(r.left(), cy),
                }
            )
        return centers

    def _zone_at(self, pos: QPointF) -> Optional[str]:
        tol = (self.HANDLE_PX + 3.0) / self._scale()
        centers = self._handle_centers()
        for _id in self._active_handles():
            c = centers[_id]
            if abs(pos.x() - c.x()) <= tol and abs(pos.y() - c.y()) <= tol:
                return _id
        if self._rect.contains(pos):
            return "move"
        return None

    def _do_move(self, pos: QPointF) -> None:
        delta = pos - self._press_pos
        r = QRectF(self._press_rect)
        r.translate(delta)
        img = self._image_rect
        if r.left() < img.left():
            r.moveLeft(img.left())
        if r.top() < img.top():
            r.moveTop(img.top())
        if r.right() > img.right():
            r.moveRight(img.right())
        if r.bottom() > img.bottom():
            r.moveBottom(img.bottom())
        self._rect = r

    def _do_corner(self, handle: str, pos: QPointF) -> None:
        img = self._image_rect
        anchor = self._opposite_corner(handle)
        target = QPointF(
            min(max(pos.x(), img.left()), img.right()),
            min(max(pos.y(), img.top()), img.bottom()),
        )
        dirx = 1.0 if target.x() >= anchor.x() else -1.0
        diry = 1.0 if target.y() >= anchor.y() else -1.0
        avail_w = (img.right() - anchor.x()) if dirx > 0 else (anchor.x() - img.left())
        avail_h = (img.bottom() - anchor.y()) if diry > 0 else (anchor.y() - img.top())

        w = min(abs(target.x() - anchor.x()), avail_w)
        h = min(abs(target.y() - anchor.y()), avail_h)

        if self._aspect is not None:
            # Fit the largest aspect-correct box that stays within both limits.
            if w / self._aspect <= h:
                h = w / self._aspect
            else:
                w = h * self._aspect
            w = min(w, avail_w)
            h = w / self._aspect
            if h > avail_h:
                h = avail_h
                w = h * self._aspect

        w = max(w, self.MIN_SIZE)
        h = max(h, self.MIN_SIZE if self._aspect is None else self.MIN_SIZE / 1.0)
        if self._aspect is not None:
            # keep ratio exact after the min-size bump
            if w < h * self._aspect:
                w = h * self._aspect
            else:
                h = w / self._aspect

        left = anchor.x() if dirx > 0 else anchor.x() - w
        top = anchor.y() if diry > 0 else anchor.y() - h
        self._rect = self._clamp_rect(QRectF(left, top, w, h))

    def _do_edge(self, handle: str, pos: QPointF) -> None:
        img = self._image_rect
        r = QRectF(self._rect)
        if handle == _TOP:
            r.setTop(min(max(pos.y(), img.top()), r.bottom() - self.MIN_SIZE))
        elif handle == _BOTTOM:
            r.setBottom(max(min(pos.y(), img.bottom()), r.top() + self.MIN_SIZE))
        elif handle == _LEFT:
            r.setLeft(min(max(pos.x(), img.left()), r.right() - self.MIN_SIZE))
        elif handle == _RIGHT:
            r.setRight(max(min(pos.x(), img.right()), r.left() + self.MIN_SIZE))
        self._rect = r

    def _opposite_corner(self, handle: str) -> QPointF:
        r = self._press_rect
        return {
            _TL: QPointF(r.right(), r.bottom()),
            _TR: QPointF(r.left(), r.bottom()),
            _BR: QPointF(r.left(), r.top()),
            _BL: QPointF(r.right(), r.top()),
        }[handle]

    def _clamp_rect(self, r: QRectF) -> QRectF:
        """Clamp ``r`` to lie within the image and keep a sane minimum size."""
        img = self._image_rect
        r = r.normalized()
        w = min(max(r.width(), self.MIN_SIZE), img.width())
        h = min(max(r.height(), self.MIN_SIZE), img.height())
        left = min(max(r.left(), img.left()), img.right() - w)
        top = min(max(r.top(), img.top()), img.bottom() - h)
        return QRectF(left, top, w, h)

    def _apply_aspect_centered(self, r: QRectF) -> QRectF:
        """Return the largest aspect-correct rect centred on ``r`` within the image."""
        if self._aspect is None or self._aspect <= 0:
            return self._clamp_rect(r)
        img = self._image_rect
        cx = r.center().x()
        cy = r.center().y()
        w = r.width()
        h = w / self._aspect
        if h > r.height():
            h = r.height()
            w = h * self._aspect
        # Do not exceed the image.
        w = min(w, img.width())
        h = w / self._aspect
        if h > img.height():
            h = img.height()
            w = h * self._aspect
        left = min(max(cx - w / 2.0, img.left()), img.right() - w)
        top = min(max(cy - h / 2.0, img.top()), img.bottom() - h)
        return QRectF(left, top, w, h)

    @staticmethod
    def _cursor_for(zone: Optional[str]):
        if zone in (_TL, _BR):
            return Qt.SizeFDiagCursor
        if zone in (_TR, _BL):
            return Qt.SizeBDiagCursor
        if zone in (_TOP, _BOTTOM):
            return Qt.SizeVerCursor
        if zone in (_LEFT, _RIGHT):
            return Qt.SizeHorCursor
        if zone == "move":
            return Qt.SizeAllCursor
        return Qt.ArrowCursor
