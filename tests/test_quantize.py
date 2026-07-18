# Copyright 2026 Vinay Williams

"""ColourGroupsControl determinism: identical image + colour count must quantize
to byte-identical output (the RNG is seeded from the proxy's own content)."""

from __future__ import annotations

import numpy as np

from painting_assist.controls.quantize import ColourGroupsControl


def _img():
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


def test_quantize_is_deterministic():
    img = _img()
    c = ColourGroupsControl()
    c.set("colours", 6)
    out1 = c.process(img.copy())
    out2 = c.process(img.copy())
    assert out1.shape == img.shape
    assert out1.dtype == np.uint8
    assert np.array_equal(out1, out2)


def test_quantize_does_not_mutate_input():
    img = _img()
    original = img.copy()
    ColourGroupsControl().process(img)
    assert np.array_equal(img, original)
