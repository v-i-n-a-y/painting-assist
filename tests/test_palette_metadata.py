"""Headless tests for the quantize palette metadata channel and PalettePanel maths."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from painting_assist.controls.quantize import ColourGroupsControl
from painting_assist.pipeline import ControlPipeline
from painting_assist.widgets import palette_panel as pp


def _blocky_image(seed: int = 0) -> np.ndarray:
    """A small RGB image with a handful of distinct flat colour blocks."""
    rng = np.random.default_rng(seed)
    colours = np.array(
        [
            [20, 30, 40],
            [200, 40, 40],
            [40, 200, 40],
            [230, 230, 230],
            [120, 90, 200],
        ],
        dtype=np.uint8,
    )
    tiles = [np.full((16, 16, 3), c, dtype=np.uint8) for c in colours]
    # Lay them out in a row, then add faint noise so k-means has something to do.
    img = np.concatenate(tiles, axis=1)
    noise = rng.integers(-3, 4, size=img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _pipeline_with_quantize(colours: int) -> ControlPipeline:
    pipe = ControlPipeline([ColourGroupsControl()])
    pipe.set_enabled("quantize", True)
    pipe.set_value("quantize", "colours", colours)
    return pipe


def test_palette_present_with_expected_count():
    img = _blocky_image()
    pipe = _pipeline_with_quantize(5)
    meta: dict = {}
    pipe.process(img, pipe.snapshot_states(), token="t", metadata_out=meta)

    assert "palette" in meta
    palette = meta["palette"]
    assert len(palette) == 5
    for entry in palette:
        assert len(entry) == 3
        assert all(isinstance(c, int) and 0 <= c <= 255 for c in entry)


def test_palette_sorted_by_lab_lightness_ascending():
    img = _blocky_image()
    pipe = _pipeline_with_quantize(5)
    meta: dict = {}
    pipe.process(img, pipe.snapshot_states(), token="t", metadata_out=meta)

    palette = meta["palette"]
    lightness = []
    for rgb in palette:
        px = np.array([[list(rgb)]], dtype=np.uint8)
        L = cv2.cvtColor(px, cv2.COLOR_RGB2Lab)[0, 0, 0]
        lightness.append(int(L))
    assert lightness == sorted(lightness)


def test_cached_rerun_still_yields_metadata():
    img = _blocky_image()
    pipe = _pipeline_with_quantize(5)
    states = pipe.snapshot_states()

    first: dict = {}
    pipe.process(img, states, token="t", metadata_out=first)

    # Second identical run should be a cache hit for the quantize stage, yet the
    # palette must still be delivered (metadata cached alongside the image).
    second: dict = {}
    pipe.process(img, states, token="t", metadata_out=second)

    assert "palette" in second
    assert second["palette"] == first["palette"]


def test_no_metadata_when_quantize_disabled():
    img = _blocky_image()
    pipe = ControlPipeline([ColourGroupsControl()])
    meta: dict = {}
    pipe.process(img, pipe.snapshot_states(), token="t", metadata_out=meta)
    assert "palette" not in meta


def test_process_without_metadata_out_still_works():
    img = _blocky_image()
    pipe = _pipeline_with_quantize(4)
    out = pipe.process(img, pipe.snapshot_states(), token="t")
    assert out.shape == img.shape


# --------------------------------------------------------------------------- #
# PalettePanel pure-function maths
# --------------------------------------------------------------------------- #
def test_rgb_to_hex():
    assert pp.rgb_to_hex((0, 0, 0)) == "#000000"
    assert pp.rgb_to_hex((255, 255, 255)) == "#FFFFFF"
    assert pp.rgb_to_hex((18, 52, 86)) == "#123456"


def test_value_pct_black_and_white():
    black = pp.colour_readout((0, 0, 0))
    white = pp.colour_readout((255, 255, 255))
    assert black["value_pct"] == pytest.approx(0.0, abs=0.5)
    assert white["value_pct"] == pytest.approx(100.0, abs=0.5)


def test_neutral_grey_has_low_chroma():
    grey = pp.colour_readout((128, 128, 128))
    assert grey["chroma"] < 3.0


def test_hue_and_chroma_are_finite_for_saturated_colour():
    red = pp.colour_readout((220, 20, 20))
    assert red["chroma"] > 10.0
    assert -180.0 <= red["hue_deg"] <= 180.0


def test_readout_dict_keys_and_rgb_passthrough():
    d = pp.colour_readout((10, 20, 30))
    assert set(d.keys()) == {"hex", "rgb", "value_pct", "hue_deg", "chroma"}
    assert d["rgb"] == (10, 20, 30)
    assert d["hex"] == "#0A141E"


def test_format_readout_contains_hex_and_labels():
    s = pp.format_readout((220, 20, 20))
    assert "#DC1414" in s
    assert "RGB 220 20 20" in s
    assert "value" in s and "hue" in s and "chroma" in s
