# Copyright 2026 Vinay Williams

"""ValuesControl: greyscale is neutral, value-steps posterize the lightness
channel, keep_colour preserves hue, and isolate dims out-of-band pixels. All
work is headless (no GUI) and in CIELab, so tolerances allow for the RGB<->Lab
uint8 roundtrip."""

from __future__ import annotations

import cv2
import numpy as np

from painting_assist.controls.values import ValuesControl


def _img():
    """A deterministic, colourful test image spanning the value range."""
    rng = np.random.default_rng(11)
    return rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)


def _lab_L(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2Lab)[:, :, 0]


NEUTRAL = 128


def _reference(img, mode, steps, keep_colour, isolate):
    """Straightforward (pre-LUT) implementation of the value reduction.

    Mirrors the original float-digitize / bincount-means / float-blend
    algorithm so the LUT-based production code can be pinned to it. Kept
    deliberately naive; the point is a trustworthy oracle, not speed.
    """
    lab = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2Lab)
    L = lab[:, :, 0].astype(np.float32)
    edges = np.linspace(0.0, 255.0, steps + 1)
    band = np.clip(np.digitize(L, edges[1:-1]), 0, steps - 1).astype(np.int32)
    if mode == "posterize":
        centres = (edges[:-1] + edges[1:]) * 0.5
        sums = np.bincount(band.ravel(), weights=L.ravel(), minlength=steps)
        counts = np.bincount(band.ravel(), minlength=steps)
        means = np.where(counts > 0, sums / np.maximum(counts, 1), centres)
        new_L = means[band]
        out_lab = lab.copy()
        out_lab[:, :, 0] = np.clip(np.round(new_L), 0, 255).astype(np.uint8)
        if not keep_colour:
            out_lab[:, :, 1] = NEUTRAL
            out_lab[:, :, 2] = NEUTRAL
        out = cv2.cvtColor(out_lab, cv2.COLOR_Lab2RGB)
    else:
        out_lab = lab.copy()
        out_lab[:, :, 1] = NEUTRAL
        out_lab[:, :, 2] = NEUTRAL
        out = cv2.cvtColor(out_lab, cv2.COLOR_Lab2RGB)
    if isolate > 0:
        keep_idx = min(isolate, steps) - 1
        mask = band == keep_idx
        grey = np.full_like(out, NEUTRAL)
        dimmed = np.round(
            0.25 * out.astype(np.float32) + 0.75 * grey.astype(np.float32)
        ).astype(np.uint8)
        out = np.where(mask[:, :, None], out, dimmed)
    return out


def _run(img, mode="grey", steps=3, keep_colour=False, isolate=0):
    c = ValuesControl()
    c.set("mode", mode)
    c.set("steps", steps)
    c.set("keep_colour", keep_colour)
    c.set("isolate", isolate)
    return c.process(img)


def _assert_close(actual, expected, tol=1):
    """Per-channel agreement within ``tol`` (the LUT roundtrip must not move
    a pixel by more than one level relative to the naive reference)."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.uint8
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert int(diff.max()) <= tol


def test_identity_shape_and_dtype():
    img = _img()
    out = ValuesControl().process(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_does_not_mutate_input():
    img = _img()
    original = img.copy()
    c = ValuesControl()
    c.set("mode", "posterize")
    c.process(img)
    assert np.array_equal(img, original)


def test_greyscale_channels_equal():
    img = _img()
    c = ValuesControl()
    c.set("mode", "grey")
    out = c.process(img)
    # A neutral grey has R == G == B (within Lab roundtrip tolerance).
    r = out[:, :, 0].astype(np.int16)
    g = out[:, :, 1].astype(np.int16)
    b = out[:, :, 2].astype(np.int16)
    assert np.max(np.abs(r - g)) <= 2
    assert np.max(np.abs(g - b)) <= 2


def test_posterize_limits_value_count():
    img = _img()
    c = ValuesControl()
    c.set("mode", "posterize")
    c.set("steps", 3)
    out = c.process(img)
    unique_L = np.unique(_lab_L(out))
    # At most `steps` distinct levels, plus a little slack for Lab roundtrip.
    assert len(unique_L) <= 3 + 2


def test_determinism():
    img = _img()
    c = ValuesControl()
    c.set("mode", "posterize")
    c.set("steps", 4)
    out1 = c.process(img.copy())
    out2 = c.process(img.copy())
    assert np.array_equal(out1, out2)


def test_keep_colour_differs_from_neutral_posterize():
    img = _img()
    neutral = ValuesControl()
    neutral.set("mode", "posterize")
    neutral.set("steps", 4)
    neutral.set("keep_colour", False)
    out_neutral = neutral.process(img)

    coloured = ValuesControl()
    coloured.set("mode", "posterize")
    coloured.set("steps", 4)
    coloured.set("keep_colour", True)
    out_colour = coloured.process(img)

    assert not np.array_equal(out_neutral, out_colour)


def test_isolate_dims_out_of_band_pixels():
    img = _img()
    base = ValuesControl()
    base.set("mode", "posterize")
    base.set("steps", 4)
    out_base = base.process(img)

    iso = ValuesControl()
    iso.set("mode", "posterize")
    iso.set("steps", 4)
    iso.set("isolate", 1)  # keep darkest band, dim the rest
    out_iso = iso.process(img)

    # Isolation must change the image (some pixels get dimmed toward grey).
    assert not np.array_equal(out_base, out_iso)

    # Dimmed pixels move closer to flat mid-grey (128) than they were.
    changed = np.any(out_base != out_iso, axis=2)
    assert changed.any()
    before = np.abs(out_base[changed].astype(np.int16) - 128)
    after = np.abs(out_iso[changed].astype(np.int16) - 128)
    assert after.mean() < before.mean()


# --------------------------------------------------------------------------- #
# The LUT-based implementation must match the naive reference within +/-1 per
# channel, so the optimisation is invisible in the output.
# --------------------------------------------------------------------------- #
def test_matches_reference_greyscale():
    img = _img()
    _assert_close(_run(img, mode="grey"), _reference(img, "grey", 3, False, 0))


def test_matches_reference_posterize_steps():
    img = _img()
    for steps in (2, 3, 4, 5, 8):
        _assert_close(
            _run(img, mode="posterize", steps=steps),
            _reference(img, "posterize", steps, False, 0),
        )


def test_matches_reference_keep_colour():
    img = _img()
    _assert_close(
        _run(img, mode="posterize", steps=4, keep_colour=True),
        _reference(img, "posterize", 4, True, 0),
    )


def test_matches_reference_isolate():
    img = _img()
    for iso in (1, 2, 4):
        _assert_close(
            _run(img, mode="posterize", steps=4, isolate=iso),
            _reference(img, "posterize", 4, False, iso),
        )


def test_matches_reference_grey_isolate():
    img = _img()
    _assert_close(
        _run(img, mode="grey", steps=4, isolate=2),
        _reference(img, "grey", 4, False, 2),
    )
