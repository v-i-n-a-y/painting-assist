from __future__ import annotations

"""Headless smoke test for the control pipeline + BlurControl.

Runs without Qt: builds a ControlPipeline over a synthetic RGB image and
checks the array contract (shape/dtype) plus active vs. passthrough behaviour.
Run with: ``uv run python tests/test_pipeline.py`` — prints PASS or FAIL.
"""

import sys

import numpy as np

from painting_assist.controls import registry  # noqa: F401  -- populate registry
from painting_assist.controls.blur import BlurControl
from painting_assist.pipeline import ControlPipeline


def _make_image() -> np.ndarray:
    """A 256x256x3 uint8 image with structure so blur has something to do."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    # Add a hard edge so neighbouring-pixel blurring is unmistakable.
    img[:, :128] = 30
    img[:, 128:] = 220
    return np.ascontiguousarray(img)


def main() -> int:
    failures = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    img = _make_image()

    # Build a pipeline containing exactly one BlurControl instance.
    blur = BlurControl()
    pipe = ControlPipeline([blur])

    # --- 1. Disabled control => passthrough (identical content). ---
    out_off = pipe.process(img)
    check(out_off.shape == img.shape, "disabled: shape mismatch")
    check(out_off.dtype == np.uint8, "disabled: dtype not uint8")
    check(np.array_equal(out_off, img), "disabled: output should equal input")

    # --- 2. Enabled but radius 0 => still passthrough (is_active False). ---
    pipe.set_enabled("blur", True)
    pipe.set_value("blur", "radius", 0)
    out_zero = pipe.process(img)
    check(np.array_equal(out_zero, img), "radius0: output should equal input")

    # --- 3. Enabled with radius > 0 => real blur, contract preserved. ---
    pipe.set_value("blur", "radius", 60)
    out_blur = pipe.process(img)
    check(out_blur.shape == img.shape, "blurred: shape mismatch")
    check(out_blur.dtype == np.uint8, "blurred: dtype not uint8")
    check(not np.array_equal(out_blur, img), "blurred: output should differ from input")
    # The hard vertical edge must be softened: the column at the seam should no
    # longer be a pure step (intermediate values appear).
    seam = out_blur[128, 120:136, 0].astype(int)
    check(seam.min() > 30 and seam.max() < 220, "blurred: edge not softened")

    # --- 4. Input not mutated by processing. ---
    check(img[0, 0, 0] in (30,), "input mutated (left block changed)")

    # --- 5. Toggle back off => passthrough again (cache correctness). ---
    pipe.set_enabled("blur", False)
    out_off2 = pipe.process(img)
    check(np.array_equal(out_off2, img), "toggle-off: output should equal input")

    # --- 6. via process(states=...) snapshot path. ---
    pipe.set_enabled("blur", True)
    pipe.set_value("blur", "radius", 40)
    states = pipe.snapshot_states()
    out_snap = pipe.process(img, states)
    check(out_snap.shape == img.shape, "snapshot: shape mismatch")
    check(not np.array_equal(out_snap, img), "snapshot: output should differ from input")

    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
