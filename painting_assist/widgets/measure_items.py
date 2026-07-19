# Copyright 2026 Vinay Williams

"""Non-destructive measuring overlays drawn over the image in the viewport.

Three interactive tools, each a :class:`QGraphicsObject` the painter drags on
top of the reference image, all drawn with cosmetic pens so the lines stay a
constant device-pixel width and remain crisp at any zoom (like
``GridOverlayItem``):

* :class:`AngleGaugeItem` -- a pivot and an end handle forming a line; reports
  the line's angle from horizontal (0..180 deg) for sight-size angle checks.
* :class:`CaliperItem` -- two independent segments A and B; reports the ratio
  of their lengths (e.g. ``1 : 1.62``) so one measurement can be compared to
  another.
* :class:`GuidesItem` -- a draggable plumb (vertical) and horizon (horizontal)
  line for checking verticals and horizontals.

The geometry maths lives in the pure module-level helpers :func:`angle_of` and
:func:`ratio_string` so it is unit-testable without a GUI. Each item reports a
human-readable readout string via its ``changed`` signal; the view relays that
as ``ImageView.measureChanged`` for the status bar.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsObject

from painting_assist.measure import Calibration


# --------------------------------------------------------------------------- #
# Pure geometry helpers (unit-tested; no GUI required)
# --------------------------------------------------------------------------- #
def _xy(p) -> Tuple[float, float]:
    """Return ``(x, y)`` from a ``QPointF`` (has ``x()``/``y()``) or a pair."""
    getx = getattr(p, "x", None)
    if callable(getx):
        return float(p.x()), float(p.y())
    return float(p[0]), float(p[1])


def angle_of(p1, p2) -> float:
    """Return the angle of the line ``p1``->``p2`` from horizontal, in degrees.

    The result lies in ``[0, 180)``: a horizontal line is 0, a vertical line is
    90, a line rising at 45 degrees is 45. Because a line and its reverse share
    an orientation the value is direction-agnostic. ``y`` is flipped to the
    usual visual convention (up is positive) so on-screen angles read the way a
    painter expects. A zero-length line reports 0.

    Accepts ``QPointF`` or ``(x, y)`` pairs.
    """
    x1, y1 = _xy(p1)
    x2, y2 = _xy(p2)
    dx = x2 - x1
    dy = y1 - y2  # flip: scene y grows downward, visual angle grows upward
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return math.degrees(math.atan2(dy, dx)) % 180.0


def ratio_string(len_a: float, len_b: float) -> str:
    """Return the proportion of two lengths as ``"1 : X.XX"``.

    The shorter length is normalised to 1, so the string always reads ``1 :``
    followed by the larger-over-smaller ratio to two decimals (equal lengths
    give ``"1 : 1.00"``). If either length is zero the ratio is undefined and
    ``"1 : 0.00"`` is returned as a guard.
    """
    a = abs(float(len_a))
    b = abs(float(len_b))
    lo, hi = (a, b) if a <= b else (b, a)
    if lo <= 0.0:
        return "1 : 0.00"
    return f"1 : {hi / lo:.2f}"


# --------------------------------------------------------------------------- #
# Interactive overlay items
# --------------------------------------------------------------------------- #
class _MeasureItem(QGraphicsObject):
    """Shared base: view-scale helpers, handle hit-testing, label drawing.

    Subclasses live at the scene origin, so item-local coordinates equal scene
    coordinates (the pixmap item also sits at the origin). Geometry is stored in
    scene pixels; ``set_image_rect`` keeps the tool inside the image when a new
    frame is loaded. The ``changed`` signal carries the readout string.
    """

    changed = Signal(str)

    HANDLE_PX = 7.0  # on-screen handle half-size, constant regardless of zoom
    GRAB_PX = 10.0  # hit tolerance in device pixels

    def __init__(
        self, image_rect: QRectF, parent: Optional[QGraphicsObject] = None
    ) -> None:
        super().__init__(parent)
        self._image_rect = QRectF(image_rect)
        self._drag: Optional[str] = None
        # Canvas calibration for physical-unit readouts; default is pixel-only.
        self._cal = Calibration()
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        # Above the grid (z=500), below the crop item (z=1000).
        self.setZValue(600)

    # -- geometry / view helpers ------------------------------------------- #
    def _scale(self) -> float:
        """Return the view's horizontal scale factor (1.0 if not in a view)."""
        scene = self.scene()
        if scene is not None:
            views = scene.views()
            if views:
                m = float(views[0].transform().m11())
                if m > 1e-6:
                    return m
        return 1.0

    def _clamp(self, p: QPointF) -> QPointF:
        """Clamp a point to the image rectangle."""
        r = self._image_rect
        return QPointF(
            min(max(p.x(), r.left()), r.right()),
            min(max(p.y(), r.top()), r.bottom()),
        )

    def set_image_rect(self, image_rect: QRectF) -> None:
        """Update the image bounds (new frame) and clamp the tool into them."""
        self.prepareGeometryChange()
        self._image_rect = QRectF(image_rect)
        self._reclamp()
        self.update()

    def set_calibration(self, cal: Calibration) -> None:
        """Set the canvas calibration used to format readouts, then repaint."""
        self._cal = cal
        self.update()

    # -- subclass hooks ---------------------------------------------------- #
    def _handles(self) -> Dict[str, QPointF]:
        """Return draggable handle points keyed by id."""
        raise NotImplementedError

    def _reclamp(self) -> None:
        """Re-clamp geometry after an image-rect change."""

    def readout(self) -> str:
        """Return the current human-readable measurement string."""
        raise NotImplementedError

    # -- hit-testing ------------------------------------------------------- #
    def _handle_at(self, pos: QPointF) -> Optional[str]:
        tol = self.GRAB_PX / self._scale()
        best_id: Optional[str] = None
        best_d = tol
        for hid, c in self._handles().items():
            d = math.hypot(pos.x() - c.x(), pos.y() - c.y())
            if d <= best_d:
                best_d = d
                best_id = hid
        return best_id

    # -- hover / cursor affordance ----------------------------------------- #
    def hoverMoveEvent(self, event) -> None:
        """Show a grab cursor over a draggable handle, a crosshair otherwise."""
        self.setCursor(self._cursor_for_handle(self._handle_at(event.pos())))
        super().hoverMoveEvent(event)

    def _cursor_for_handle(self, hid: Optional[str]) -> Qt.CursorShape:
        """Return the cursor for hovering handle ``hid`` (field when ``None``).

        Point handles drag freely in any direction, so they use the move-all
        cursor; the open measuring field uses a crosshair to signal precise
        placement. Axis-locked tools override this (see :class:`GuidesItem`).
        """
        if hid is None:
            return Qt.CrossCursor
        return Qt.SizeAllCursor

    # -- mouse ------------------------------------------------------------- #
    def mousePressEvent(self, event) -> None:
        self._drag = self._handle_at(event.pos())
        if self._drag is None:
            event.ignore()
            return
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag is None:
            event.ignore()
            return
        self.prepareGeometryChange()
        self._move_handle(self._drag, self._clamp(event.pos()))
        self.update()
        self.changed.emit(self.readout())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag is None:
            event.ignore()
            return
        self._drag = None
        self.changed.emit(self.readout())
        event.accept()

    def _move_handle(self, hid: str, pos: QPointF) -> None:
        """Apply a drag of handle ``hid`` to the clamped scene point ``pos``."""
        raise NotImplementedError

    # -- drawing helpers --------------------------------------------------- #
    def _cosmetic_pen(self, color: QColor, width: float = 2.0) -> QPen:
        pen = QPen(color)
        pen.setCosmetic(True)  # constant device-pixel width across zoom
        pen.setWidthF(width)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        return pen

    def _draw_handles(
        self, painter: QPainter, color: QColor, points: Dict[str, QPointF]
    ) -> None:
        scale = self._scale()
        hs = self.HANDLE_PX / scale
        painter.setPen(self._cosmetic_pen(QColor(20, 20, 20, 230), 1.0))
        painter.setBrush(color)
        for c in points.values():
            painter.drawRect(QRectF(c.x() - hs, c.y() - hs, 2 * hs, 2 * hs))

    # -- labels (positioned here, painted by the view) --------------------- #
    def _label_anchor(self) -> QPointF:
        """Scene point the primary readout label is anchored near."""
        raise NotImplementedError

    def label_specs(self) -> List[Tuple[QPointF, str, str]]:
        """Return ``(scene_anchor, text, placement)`` labels for the view to draw.

        ``placement`` is ``"above"`` or ``"below"`` — which side of the anchor the
        chip sits, so a point can carry both a reading above and an edge chip
        below without overlap. The view paints these in its foreground layer
        (device coordinates) so a label is never clipped to this item's bounding
        rect. Drawing labels in ``paint()`` used to leave trails when the tool
        moved and clip the text off the top edge; the view repaints the whole
        viewport on any change, so moving is clean and labels stay on-screen.
        """
        return [(self._label_anchor(), self.readout(), "above")]


class AngleGaugeItem(_MeasureItem):
    """A pivot and an end handle; reports the line's angle from horizontal."""

    _COLOR = QColor(255, 150, 40)

    def __init__(
        self, image_rect: QRectF, parent: Optional[QGraphicsObject] = None
    ) -> None:
        super().__init__(image_rect, parent)
        r = image_rect
        cx, cy = r.center().x(), r.center().y()
        span = min(r.width(), r.height()) * 0.3
        self._pivot = QPointF(cx, cy)
        self._end = QPointF(cx + span, cy)

    def _handles(self) -> Dict[str, QPointF]:
        return {"pivot": self._pivot, "end": self._end}

    def _move_handle(self, hid: str, pos: QPointF) -> None:
        if hid == "pivot":
            delta = pos - self._pivot
            self._pivot = pos
            self._end = self._clamp(self._end + delta)
        else:
            self._end = pos

    def _reclamp(self) -> None:
        self._pivot = self._clamp(self._pivot)
        self._end = self._clamp(self._end)

    def readout(self) -> str:
        return f"Angle: {angle_of(self._pivot, self._end):.1f} deg"

    def _label_anchor(self) -> QPointF:
        return self._pivot

    def boundingRect(self) -> QRectF:
        m = (self.HANDLE_PX + self.GRAB_PX) / self._scale() + 2.0
        return QRectF(self._pivot, self._end).normalized().adjusted(-m, -m, m, m)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Faint horizontal reference ray from the pivot (the angle is measured
        # from this baseline).
        ref_len = QLineF(self._pivot, self._end).length()
        ref_end = QPointF(self._pivot.x() + ref_len, self._pivot.y())
        ref_pen = self._cosmetic_pen(QColor(255, 255, 255, 120), 1.0)
        ref_pen.setStyle(Qt.DashLine)
        painter.setPen(ref_pen)
        painter.drawLine(QLineF(self._pivot, ref_end))
        # The measured line.
        painter.setPen(self._cosmetic_pen(self._COLOR, 2.0))
        painter.drawLine(QLineF(self._pivot, self._end))
        self._draw_handles(painter, self._COLOR, self._handles())


class CaliperItem(_MeasureItem):
    """Two independent segments A and B; reports their length ratio."""

    _COLOR_A = QColor(0, 200, 220)
    _COLOR_B = QColor(255, 90, 160)

    def __init__(
        self, image_rect: QRectF, parent: Optional[QGraphicsObject] = None
    ) -> None:
        super().__init__(image_rect, parent)
        r = image_rect
        w, h = r.width(), r.height()
        self._a0 = QPointF(r.left() + w * 0.2, r.top() + h * 0.35)
        self._a1 = QPointF(r.left() + w * 0.5, r.top() + h * 0.35)
        self._b0 = QPointF(r.left() + w * 0.2, r.top() + h * 0.6)
        self._b1 = QPointF(r.left() + w * 0.5, r.top() + h * 0.6)

    def _handles(self) -> Dict[str, QPointF]:
        return {"a0": self._a0, "a1": self._a1, "b0": self._b0, "b1": self._b1}

    def _move_handle(self, hid: str, pos: QPointF) -> None:
        setattr(self, f"_{hid}", pos)

    def _reclamp(self) -> None:
        self._a0 = self._clamp(self._a0)
        self._a1 = self._clamp(self._a1)
        self._b0 = self._clamp(self._b0)
        self._b1 = self._clamp(self._b1)

    def _seg(self, p0: QPointF, p1: QPointF) -> Tuple[str, float]:
        """Return the ``(formatted, numeric)`` length of a segment in the unit."""
        r = self._image_rect
        dx, dy = p1.x() - p0.x(), p1.y() - p0.y()
        return (
            self._cal.length_str(dx, dy, r.width(), r.height()),
            self._cal.length_value(dx, dy, r.width(), r.height()),
        )

    def readout(self) -> str:
        sa, va = self._seg(self._a0, self._a1)
        sb, vb = self._seg(self._b0, self._b1)
        return f"A {sa} : B {sb}  ({ratio_string(va, vb)})"

    def _label_anchor(self) -> QPointF:
        # Main length/ratio label sits at segment A's midpoint so it never
        # collides with the per-endpoint edge chips drawn below each point.
        return QPointF(
            (self._a0.x() + self._a1.x()) / 2.0, (self._a0.y() + self._a1.y()) / 2.0
        )

    def _edge_label(self, p: QPointF) -> str:
        r = self._image_rect
        return self._cal.edge_label(
            p.x(),
            p.y(),
            r.left(),
            r.top(),
            r.right(),
            r.bottom(),
            r.width(),
            r.height(),
        )

    def label_specs(self) -> List[Tuple[QPointF, str, str]]:
        specs = [(self._label_anchor(), self.readout(), "above")]
        if self._cal.show_edges:
            # Each endpoint's distance to its nearest vertical/horizontal edge.
            for p in (self._a0, self._a1, self._b0, self._b1):
                specs.append((p, self._edge_label(p), "below"))
        return specs

    def boundingRect(self) -> QRectF:
        m = (self.HANDLE_PX + self.GRAB_PX) / self._scale() + 2.0
        pts = [self._a0, self._a1, self._b0, self._b1]
        xs = [p.x() for p in pts]
        ys = [p.y() for p in pts]
        return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)).adjusted(
            -m, -m, m, m
        )

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(self._cosmetic_pen(self._COLOR_A, 2.0))
        painter.drawLine(QLineF(self._a0, self._a1))
        painter.setPen(self._cosmetic_pen(self._COLOR_B, 2.0))
        painter.drawLine(QLineF(self._b0, self._b1))
        self._draw_handles(painter, self._COLOR_A, {"a0": self._a0, "a1": self._a1})
        self._draw_handles(painter, self._COLOR_B, {"b0": self._b0, "b1": self._b1})


class GuidesItem(_MeasureItem):
    """A draggable plumb (vertical) and horizon (horizontal) line."""

    _COLOR = QColor(120, 230, 120)

    def __init__(
        self, image_rect: QRectF, parent: Optional[QGraphicsObject] = None
    ) -> None:
        super().__init__(image_rect, parent)
        self._vx = image_rect.center().x()  # plumb x position
        self._hy = image_rect.center().y()  # horizon y position

    def _handles(self) -> Dict[str, QPointF]:
        # Grab targets where the two lines cross the image centre lines.
        r = self._image_rect
        return {
            "plumb": QPointF(self._vx, r.center().y()),
            "horizon": QPointF(r.center().x(), self._hy),
        }

    def _handle_at(self, pos: QPointF) -> Optional[str]:
        # Grab anywhere along a line, not just at a handle box.
        tol = self.GRAB_PX / self._scale()
        d_plumb = abs(pos.x() - self._vx)
        d_horizon = abs(pos.y() - self._hy)
        if d_plumb <= tol and d_plumb <= d_horizon:
            return "plumb"
        if d_horizon <= tol:
            return "horizon"
        return None

    def _cursor_for_handle(self, hid: Optional[str]) -> Qt.CursorShape:
        # The plumb line slides left/right; the horizon slides up/down.
        if hid == "plumb":
            return Qt.SizeHorCursor
        if hid == "horizon":
            return Qt.SizeVerCursor
        return Qt.CrossCursor

    def _move_handle(self, hid: str, pos: QPointF) -> None:
        if hid == "plumb":
            self._vx = pos.x()
        else:
            self._hy = pos.y()

    def _reclamp(self) -> None:
        r = self._image_rect
        self._vx = min(max(self._vx, r.left()), r.right())
        self._hy = min(max(self._hy, r.top()), r.bottom())

    def readout(self) -> str:
        r = self._image_rect
        x = self._cal.axis_str(self._vx - r.left(), "x", r.width(), r.height())
        y = self._cal.axis_str(self._hy - r.top(), "y", r.width(), r.height())
        return f"Plumb {x} from left   Horizon {y} from top"

    def _label_anchor(self) -> QPointF:
        return QPointF(self._vx, self._hy)

    def boundingRect(self) -> QRectF:
        return QRectF(self._image_rect)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        r = self._image_rect
        painter.setPen(self._cosmetic_pen(self._COLOR, 2.0))
        painter.drawLine(QLineF(self._vx, r.top(), self._vx, r.bottom()))
        painter.drawLine(QLineF(r.left(), self._hy, r.right(), self._hy))
