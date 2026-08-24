# Copyright 2026 Vinay Williams

"""Tests for painting_assist.measure: unit conversion and canvas calibration.

Pure maths, no Qt — mirrors how the geometry helpers are tested. A Calibration
maps scene (image-pixel) geometry to physical canvas units given the canvas size
and the displayed image's pixel dimensions.
"""

from __future__ import annotations

import pytest

from painting_assist.measure import (
    Calibration,
    convert_canvas_size,
    convert_from_cm,
    format_measure,
)


# ---- pure conversion / formatting ----
def test_convert_from_cm():
    assert convert_from_cm(10.0, "cm") == 10.0
    assert convert_from_cm(10.0, "mm") == 100.0
    assert abs(convert_from_cm(2.54, "in") - 1.0) < 1e-9
    # Unknown unit passes the centimetre value through unchanged.
    assert convert_from_cm(5.0, "px") == 5.0


def test_convert_canvas_size_physical_round_trip():
    w, h = convert_canvas_size(40.0, 30.0, "cm", "in")
    assert abs(w - 40.0 / 2.54) < 1e-9
    assert abs(h - 30.0 / 2.54) < 1e-9
    back_w, back_h = convert_canvas_size(w, h, "in", "cm")
    assert abs(back_w - 40.0) < 1e-9
    assert abs(back_h - 30.0) < 1e-9


def test_convert_canvas_size_mm_and_in():
    w, h = convert_canvas_size(10.0, 5.0, "cm", "mm")
    assert w == pytest.approx(100.0)
    assert h == pytest.approx(50.0)
    w, h = convert_canvas_size(25.4, 10.0, "mm", "in")
    assert w == pytest.approx(1.0)
    assert h == pytest.approx(1.0 / 2.54)  # 10 mm = 1 cm


def test_convert_canvas_size_to_px_adopts_crop_size():
    # A pixel canvas is defined by the crop it sits on.
    assert convert_canvas_size(40.0, 30.0, "cm", "px", crop_px=(600.0, 450.0)) == (
        600.0,
        450.0,
    )


def test_convert_canvas_size_to_px_without_crop_keeps_values():
    assert convert_canvas_size(40.0, 30.0, "cm", "px") == (40.0, 30.0)


def test_convert_canvas_size_to_ratio_keeps_values():
    # The numbers already express the width:height ratio.
    assert convert_canvas_size(40.0, 30.0, "cm", "ratio") == (40.0, 30.0)


def test_convert_canvas_size_from_px_or_ratio_keeps_values():
    # No absolute size is recoverable from px or ratio values.
    assert convert_canvas_size(600.0, 450.0, "px", "cm") == (600.0, 450.0)
    assert convert_canvas_size(4.0, 3.0, "ratio", "cm") == (4.0, 3.0)


def test_convert_canvas_size_same_unit_or_non_positive_noop():
    assert convert_canvas_size(40.0, 30.0, "cm", "cm") == (40.0, 30.0)
    assert convert_canvas_size(0.0, 30.0, "cm", "in") == (0.0, 30.0)
    assert convert_canvas_size(40.0, -1.0, "cm", "in") == (40.0, -1.0)


def test_format_measure():
    assert format_measure(120.4, "px") == "120 px"
    assert format_measure(10.0, "cm") == "10.0 cm"
    assert format_measure(100.0, "mm") == "100 mm"
    assert format_measure(1.0, "in") == "1.00 in"


# ---- calibration: no physical size falls back to pixels ----
def test_default_calibration_is_pixels():
    cal = Calibration()
    assert cal.effective_unit(400, 300) == "px"
    assert not cal.is_physical(400, 300)
    # A 3-4-5 triangle: 30px x 40px -> 50px.
    assert cal.length_str(30, 40, 400, 300) == "50 px"


def test_px_display_unit_overrides_physical_size():
    # A real canvas is set, but the user asked for pixels.
    cal = Calibration(40, 30, "cm", display_unit="px")
    assert cal.length_str(100, 0, 400, 300) == "100 px"


# ---- calibration: physical readings ----
def test_isotropic_length_in_cm():
    # 40 cm across 400 px -> 0.1 cm/px, matching 30 cm / 300 px vertically.
    cal = Calibration(40, 30, "cm", display_unit="cm")
    assert cal.is_physical(400, 300)
    assert cal.length_str(100, 0, 400, 300) == "10.0 cm"
    assert cal.length_str(0, 100, 400, 300) == "10.0 cm"
    # Diagonal 300x400 px -> 30x40 cm -> 50 cm.
    assert cal.length_str(300, 400, 400, 300) == "50.0 cm"


def test_length_in_mm_and_inches():
    cal = Calibration(40, 30, "cm", display_unit="mm")
    assert cal.length_str(100, 0, 400, 300) == "100 mm"
    cal_in = Calibration(40, 30, "cm", display_unit="in")
    # 10 cm = 3.9370... in
    assert cal_in.length_str(100, 0, 400, 300) == "3.94 in"


def test_anisotropic_scales_are_independent():
    # Freeform crop: 40 cm wide over 400 px (0.1 cm/px), 20 cm tall over 400 px
    # (0.05 cm/px). Horizontal and vertical must not share a scale.
    cal = Calibration(40, 20, "cm", display_unit="cm")
    assert cal.length_str(100, 0, 400, 400) == "10.0 cm"
    assert cal.length_str(0, 100, 400, 400) == "5.0 cm"


def test_length_value_matches_string():
    cal = Calibration(40, 30, "cm", display_unit="cm")
    assert abs(cal.length_value(100, 0, 400, 300) - 10.0) < 1e-9
    # Ratio of a horizontal 100px to a vertical 200px (isotropic) is 1:2.
    va = cal.length_value(100, 0, 400, 300)
    vb = cal.length_value(0, 200, 400, 300)
    assert abs(vb / va - 2.0) < 1e-9


def test_axis_str_uses_the_right_scale():
    cal = Calibration(40, 20, "cm", display_unit="cm")
    assert cal.axis_str(100, "x", 400, 400) == "10.0 cm"
    assert cal.axis_str(100, "y", 400, 400) == "5.0 cm"


# ---- calibration: nearest-edge labels ----
def test_edge_label_picks_nearest_sides():
    cal = Calibration(40, 30, "cm", display_unit="cm")
    # Point at (40, 30) px in a 400x300 image: nearer the left (4 cm) than the
    # right (36 cm), and nearer the top (3 cm) than the bottom (27 cm).
    label = cal.edge_label(40, 30, 0, 0, 400, 300, 400, 300)
    assert label == "L 4.0 cm · T 3.0 cm"


def test_edge_label_switches_to_far_sides():
    cal = Calibration(40, 30, "cm", display_unit="cm")
    # Point near the bottom-right: nearer right and bottom.
    label = cal.edge_label(380, 290, 0, 0, 400, 300, 400, 300)
    assert label == "R 2.0 cm · B 1.0 cm"


def test_edge_label_in_pixels_without_canvas():
    cal = Calibration()  # pixel-only
    label = cal.edge_label(40, 30, 0, 0, 400, 300, 400, 300)
    assert label == "L 40 px · T 30 px"


# ---- calibration: gridline positions (fraction of canvas) ----
def test_fraction_str_positions():
    cal = Calibration(40, 30, "cm", display_unit="cm")
    assert cal.fraction_str(0.25, "x") == "10.0 cm"  # 0.25 * 40
    assert cal.fraction_str(1.0 / 3.0, "y") == "10.0 cm"  # (1/3) * 30


def test_fraction_str_empty_without_physical_canvas():
    # No canvas at all, and px display unit, both yield no canvas position.
    assert Calibration().fraction_str(0.5, "x") == ""
    assert Calibration(40, 30, "cm", display_unit="px").fraction_str(0.5, "x") == ""
