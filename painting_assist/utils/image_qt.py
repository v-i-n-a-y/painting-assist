from __future__ import annotations

"""numpy <-> Qt conversion helpers.

The ``.copy()`` (on the QImage) plus ``np.ascontiguousarray`` are the critical
crash-avoidance details: a ``QImage`` constructed over a numpy buffer does not
own that buffer, so once the numpy array is garbage-collected the QImage would
point at freed memory. Forcing a contiguous buffer and copying makes the QImage
own its own pixels.

Array contract (everywhere in the app): RGB, ``uint8``, shape ``HxWx3``.
"""

import numpy as np
from PySide6.QtGui import QImage, QPixmap


def ndarray_to_qimage(rgb: np.ndarray) -> QImage:
    """Convert an RGB ``uint8`` ``HxWx3`` array to a self-owned ``QImage``.

    Forces the array C-contiguous and ``uint8``, builds a ``Format_RGB888``
    QImage using the real row stride (``rgb.strides[0]``, which equals ``3*w``
    for a contiguous array), then returns ``.copy()`` so the QImage owns its
    buffer and is immune to the source array being collected.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            "ndarray_to_qimage expects an RGB HxWx3 array, got shape "
            f"{rgb.shape!r}"
        )
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
    return qimg.copy()


def ndarray_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    """Convert an RGB ``uint8`` ``HxWx3`` array to a ``QPixmap``."""
    return QPixmap.fromImage(ndarray_to_qimage(rgb))


def qimage_to_ndarray(img: QImage) -> np.ndarray:
    """Convert a ``QImage`` to an RGB ``uint8`` ``HxWx3`` numpy copy.

    The image is first converted to ``Format_RGB888`` so the channel order and
    pixel size are known, then its bits are read row-by-row honouring the
    QImage stride and copied into a fresh contiguous array.
    """
    if img.format() != QImage.Format_RGB888:
        img = img.convertToFormat(QImage.Format_RGB888)
    w = img.width()
    h = img.height()
    bytes_per_line = img.bytesPerLine()
    ptr = img.constBits()
    # constBits() returns a sip.voidptr; size it to the full buffer.
    ptr.setsize(bytes_per_line * h)
    buf = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape((h, bytes_per_line))
    # Trim any row padding and reshape to HxWx3.
    rgb = buf[:, : w * 3].reshape((h, w, 3))
    return np.ascontiguousarray(rgb, dtype=np.uint8)
