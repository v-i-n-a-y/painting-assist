# Copyright 2026 Vinay Williams

"""Headless tests for render_palette_strip (pure Pillow/numpy, no QApplication).

The strip frames each swatch with a 1px #808080 border (mirroring the on-screen
swatch border) and, when big enough, labels it with hex + value %. So the exact
corner pixels are border; colour checks sample the swatch interior instead.
"""

from __future__ import annotations

import numpy as np

from painting_assist.widgets.palette_panel import render_palette_strip


def test_size_matches_colours() -> None:
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    img = render_palette_strip(colours, swatch=64, height=48)
    assert img.mode == "RGB"
    assert img.size == (64 * 3, 48)


def test_border_and_annotation_do_not_change_dimensions() -> None:
    # Bordered/annotated and plain strips keep the same swatch*n x height box.
    colours = [(10, 10, 10), (200, 200, 200)]
    annotated = render_palette_strip(colours, swatch=64, height=64, annotate=True)
    plain = render_palette_strip(colours, swatch=64, height=64, annotate=False)
    assert annotated.size == (64 * 2, 64)
    assert plain.size == annotated.size


def test_default_call_still_succeeds() -> None:
    # The old signature (colours only) keeps working; annotate defaults on.
    img = render_palette_strip([(0, 0, 0), (255, 255, 255)])
    assert img.mode == "RGB"
    assert img.size == (64 * 2, 64)


def test_annotated_strip_keeps_swatch_colours() -> None:
    # Text sits at the top of each swatch, so the lower interior stays pure
    # colour: a bordered, annotated swatch still reads as its source colour
    # (including a black swatch, the case that vanishes without a border).
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]
    img = render_palette_strip(colours, swatch=64, height=64, annotate=True)
    arr = np.asarray(img)
    for i, expected in enumerate(colours):
        # bottom-centre of swatch i, just inside the 1px border
        assert tuple(arr[62, i * 64 + 32]) == expected


def test_unannotated_corner_ish_colours() -> None:
    # Annotation off and swatches too small to label: each block is a solid
    # colour framed by a 1px border; sample just inside the top-left corner.
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    img = render_palette_strip(colours, swatch=10, height=8, annotate=False)
    arr = np.asarray(img)
    assert tuple(arr[1, 0 * 10 + 1]) == (255, 0, 0)
    assert tuple(arr[1, 1 * 10 + 1]) == (0, 255, 0)
    assert tuple(arr[1, 2 * 10 + 1]) == (0, 0, 255)


def test_empty_palette_returns_valid_image() -> None:
    img = render_palette_strip([], swatch=32, height=20)
    assert img.mode == "RGB"
    assert img.size == (1, 20)


def test_channels_masked_to_uint8() -> None:
    # Out-of-range channels wrap to uint8; sample the interior (the 1px border
    # overwrites the exact corner).
    img = render_palette_strip([(511, -1, 300)], swatch=6, height=6, annotate=False)
    arr = np.asarray(img)
    assert tuple(arr[3, 3]) == (511 & 0xFF, -1 & 0xFF, 300 & 0xFF)
