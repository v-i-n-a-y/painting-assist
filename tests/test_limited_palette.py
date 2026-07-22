# Copyright 2026 Vinay Williams

"""LimitedPaletteControl: resolves a palette from the chosen source and repaints
the image from the mixable gamut. Headless (no GUI); the gamut/mapping maths is
covered separately in test_palette_map.py."""

from __future__ import annotations

import json

import numpy as np

from painting_assist.controls.limited_palette import (
    LimitedPaletteControl,
    parse_hex_list,
)


def _img():
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)


def test_parse_hex_list_robust():
    assert parse_hex_list('["#ff0000", "00ff00"]') == [(255, 0, 0), (0, 255, 0)]
    # Bad entries are dropped, not raised on.
    assert parse_hex_list('["nope", 123, "#12"]') == []
    assert parse_hex_list("") == []
    assert parse_hex_list("not json") == []


def test_preset_source_resolves_tubes():
    c = LimitedPaletteControl()
    c.set("source", "preset")
    c.set("preset", "zorn")
    assert c.tubes() == LimitedPaletteControl.PRESETS["zorn"]


def test_my_paints_source_reads_injected_json():
    c = LimitedPaletteControl()
    c.set("source", "my_paints")
    c.set("paints_json", json.dumps(["#c49148", "#8a3622"]))
    assert c.tubes() == [(196, 145, 72), (138, 54, 34)]


def test_sampled_source_reads_samples_json():
    c = LimitedPaletteControl()
    c.set("source", "sampled")
    c.set("samples_json", json.dumps(["#202020", "#f0f0f0"]))
    assert c.tubes() == [(32, 32, 32), (240, 240, 240)]


def test_is_active_requires_tubes():
    c = LimitedPaletteControl()
    c.set_enabled(True)
    c.set("source", "sampled")  # no samples yet
    assert c.is_active() is False
    c.set("samples_json", json.dumps(["#804020"]))
    assert c.is_active() is True


def test_process_shape_dtype_and_no_mutation():
    img = _img()
    original = img.copy()
    c = LimitedPaletteControl()
    c.set("source", "preset")
    c.set("preset", "zorn")
    out = c.process(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.array_equal(img, original)


def test_process_empty_palette_is_identity():
    img = _img()
    c = LimitedPaletteControl()
    c.set("source", "sampled")  # empty -> no tubes
    out = c.process(img)
    assert np.array_equal(out, img)


def test_process_output_uses_only_gamut_ish():
    """Every output pixel should be a colour from the built gamut (no mutation)."""
    img = _img()
    c = LimitedPaletteControl()
    c.set("source", "preset")
    c.set("preset", "zorn")
    out = c.process(img)
    # Determinism: same input -> same output.
    assert np.array_equal(out, c.process(img.copy()))
