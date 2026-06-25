from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal


class ImageModel(QObject):
    """QObject holding ONLY the immutable original RGB array; emits on load."""

    image_loaded = Signal()            # emitted after a new original is set

    def __init__(self) -> None:
        super().__init__()
        self._original: Optional[np.ndarray] = None
        self._path: Optional[str] = None

    def has_image(self) -> bool:
        """True iff an original image is currently loaded."""
        return self._original is not None

    def original(self) -> Optional[np.ndarray]:
        """The unmodified RGB uint8 HxWx3 source. Callers treat as read-only."""
        return self._original

    def path(self) -> Optional[str]:
        """Filesystem path of the loaded image, if known."""
        return self._path

    def set_image(self, rgb: np.ndarray, path: Optional[str] = None) -> None:
        """Store original (made C-contiguous uint8 RGB) + path, then emit image_loaded."""
        self._original = np.ascontiguousarray(rgb, dtype=np.uint8)
        self._path = path
        self.image_loaded.emit()

    def load_path(self, path: str) -> None:
        """Read file via cv2.imread (BGR) -> BGR2RGB -> set_image(rgb, path).

        Falls back to Pillow if cv2 returns None (e.g. odd formats).
        """
        rgb: Optional[np.ndarray] = None

        try:
            import cv2

            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = None

        if rgb is None:
            from PIL import Image

            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

        self.set_image(rgb, path)
