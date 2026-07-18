# Copyright 2026 Vinay Williams

"""Pure geometry helpers behind the measuring overlays: angle_of reports a
line's tilt from horizontal in [0, 180) degrees, and ratio_string formats a
length proportion as "1 : X.XX". Headless (no GUI): only the module-level
helpers are exercised, never the QGraphicsItem widgets."""

from __future__ import annotations

import pytest

from painting_assist.widgets.measure_items import angle_of, ratio_string


def test_angle_horizontal_is_zero():
    assert angle_of((0.0, 0.0), (10.0, 0.0)) == pytest.approx(0.0)


def test_angle_vertical_is_ninety():
    # Vertical in either direction reads 90 (orientation is direction-agnostic).
    assert angle_of((0.0, 0.0), (0.0, 10.0)) == pytest.approx(90.0)
    assert angle_of((0.0, 0.0), (0.0, -10.0)) == pytest.approx(90.0)


def test_angle_forty_five_up():
    # Scene y grows downward; rising 45 degrees means a downward y delta.
    assert angle_of((0.0, 10.0), (10.0, 0.0)) == pytest.approx(45.0)


def test_angle_reverse_direction_matches():
    assert angle_of((10.0, 0.0), (0.0, 0.0)) == pytest.approx(0.0)


def test_angle_zero_length_guard():
    assert angle_of((5.0, 5.0), (5.0, 5.0)) == 0.0


def test_ratio_equal_lengths():
    assert ratio_string(50.0, 50.0) == "1 : 1.00"


def test_ratio_golden():
    assert ratio_string(100.0, 161.8) == "1 : 1.62"


def test_ratio_normalises_smaller_to_one():
    # Order-independent: the shorter side is always the "1".
    assert ratio_string(161.8, 100.0) == "1 : 1.62"


def test_ratio_zero_length_guard():
    assert ratio_string(0.0, 50.0) == "1 : 0.00"
    assert ratio_string(0.0, 0.0) == "1 : 0.00"
