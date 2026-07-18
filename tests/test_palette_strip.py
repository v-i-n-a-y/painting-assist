# Copyright 2026 Vinay Williams

"""Headless tests for render_palette_strip (pure Pillow/numpy, no QApplication)."""

from __future__ import annotations

import numpy as np

from painting_assist.widgets.palette_panel import render_palette_strip


def test_size_matches_colours() -> None:
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    img = render_palette_strip(colours, swatch=64, height=48)
    assert img.mode == "RGB"
    assert img.size == (64 * 3, 48)


def test_corner_pixel_colours() -> None:
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    img = render_palette_strip(colours, swatch=10, height=8)
    arr = np.asarray(img)
    assert tuple(arr[0, 0]) == (255, 0, 0)  # first swatch, top-left
    assert tuple(arr[-1, 15]) == (0, 255, 0)  # middle swatch, bottom
    assert tuple(arr[0, -1]) == (0, 0, 255)  # last swatch, top-right


def test_empty_palette_returns_valid_image() -> None:
    img = render_palette_strip([], swatch=32, height=20)
    assert img.mode == "RGB"
    assert img.size == (1, 20)


def test_channels_masked_to_uint8() -> None:
    img = render_palette_strip([(511, -1, 300)], swatch=4, height=4)
    arr = np.asarray(img)
    assert tuple(arr[0, 0]) == (511 & 0xFF, -1 & 0xFF, 300 & 0xFF)
