# Copyright 2026 Vinay Williams

"""Flip control — mirror the reference horizontally and/or vertically.

Flipping a reference is a classic painter's check: a mirrored image exposes
drawing errors the eye has grown blind to. This control bakes the flip into the
pixels via :func:`numpy.fliplr` / :func:`numpy.flipud`, non-destructively (a new
array is returned, the input is never mutated).
"""

from __future__ import annotations

from typing import List

import numpy as np

from painting_assist.controls.base import Control, Param, ParamType
from painting_assist.controls.registry import register


@register
class FlipControl(Control):
    """Mirror the image horizontally (default) and/or vertically.

    Active only when enabled *and* at least one axis is selected; otherwise the
    pipeline skips it as a no-op. :meth:`process` returns a freshly flipped copy
    so the source array is never touched.
    """

    id = "flip"
    name = "Flip"
    order = 1  # right after crop (0), before tone (5)

    @classmethod
    def params(cls) -> List[Param]:
        """A horizontal mirror (on by default) plus an optional vertical flip."""
        return [
            Param(
                name="horizontal",
                label="Horizontal",
                ptype=ParamType.BOOL,
                default=True,
                tooltip="Mirror left-to-right (the usual painter's check).",
            ),
            Param(
                name="vertical",
                label="Vertical",
                ptype=ParamType.BOOL,
                default=False,
                tooltip="Flip top-to-bottom.",
            ),
        ]

    def is_active(self) -> bool:
        """Active only when enabled and at least one axis is selected."""
        return self.enabled and (
            bool(self.get("horizontal")) or bool(self.get("vertical"))
        )

    def process(self, img: np.ndarray) -> np.ndarray:
        """Return a new array flipped on the selected axes; input untouched.

        Identity when disabled, mirroring the pipeline's own gate so a direct
        ``process`` call on a disabled control is a no-op.
        """
        if not self.enabled:
            return img
        out = img
        if bool(self.get("horizontal")):
            out = np.fliplr(out)
        if bool(self.get("vertical")):
            out = np.flipud(out)
        # np.flip* return views; materialise a contiguous copy so the caller
        # owns an independent array and the input is never aliased/mutated.
        return np.ascontiguousarray(out) if out is not img else out.copy()
