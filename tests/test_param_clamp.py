# Copyright 2026 Vinay Williams

"""Param.clamp coercion / range-clamping edge cases."""

from __future__ import annotations

import json
import math

from painting_assist.controls.base import Param, ParamType
from painting_assist.controls.grid import GridControl


def _int(minimum=0, maximum=10, default=3):
    return Param(
        name="n",
        label="N",
        ptype=ParamType.INT,
        default=default,
        minimum=minimum,
        maximum=maximum,
    )


def _float(minimum=0.0, maximum=1.0, default=0.5):
    return Param(
        name="f",
        label="F",
        ptype=ParamType.FLOAT,
        default=default,
        minimum=minimum,
        maximum=maximum,
    )


def _choice():
    return Param(
        name="c",
        label="C",
        ptype=ParamType.CHOICE,
        default="a",
        choices=[("a", "A"), ("b", "B")],
    )


def _text():
    return Param(name="t", label="T", ptype=ParamType.TEXT, default="")


def _bool():
    return Param(name="b", label="B", ptype=ParamType.BOOL, default=False)


# ---- INT ----
def test_int_out_of_range_clamps():
    p = _int()
    assert p.clamp(999) == 10
    assert p.clamp(-5) == 0


def test_int_rounds_float():
    assert _int().clamp(2.6) == 3


def test_int_invalid_returns_default():
    p = _int()
    assert p.clamp("nope") == 3
    assert p.clamp(None) == 3


def test_int_non_finite_returns_default_without_raising():
    # int(round(inf)) raises OverflowError and int(round(nan)) raises ValueError;
    # a non-finite float must fall back to the default rather than escaping.
    p = _int()
    assert p.clamp(float("inf")) == 3
    assert p.clamp(float("-inf")) == 3
    assert p.clamp(float("nan")) == 3


def test_int_json_overflow_literal_returns_default():
    # The real session-restore crash: an out-of-range JSON number parses to inf.
    parsed = json.loads("1e999")
    assert not math.isfinite(parsed)
    assert _int().clamp(parsed) == 3


def test_int_non_finite_default_when_unbounded():
    p = _int(minimum=None, maximum=None)
    assert p.clamp(float("inf")) == 3
    assert p.clamp(float("nan")) == 3


# ---- FLOAT ----
def test_float_out_of_range_clamps():
    p = _float()
    assert p.clamp(5.0) == 1.0
    assert p.clamp(-1.0) == 0.0


def test_float_invalid_returns_default():
    assert _float().clamp("x") == 0.5
    assert _float().clamp(None) == 0.5


def test_float_nan_returns_default():
    # NaN coerces without error, but every NaN comparison is False, so without a
    # guard it would silently pin to the lower bound. clamp guards for it and
    # falls back to the default, bounded or not.
    result = _float(minimum=0.0, maximum=1.0).clamp(float("nan"))
    assert result == 0.5
    assert not math.isnan(result)


def test_float_nan_returns_default_when_unbounded():
    result = _float(minimum=None, maximum=None).clamp(float("nan"))
    assert result == 0.5
    assert not math.isnan(result)


def test_float_inf_returns_default():
    # inf would clamp to the maximum on a bounded param but has no finite value on
    # an unbounded one; reject it uniformly to the default (like NaN).
    assert _float(minimum=0.0, maximum=1.0).clamp(float("inf")) == 0.5
    assert _float(minimum=0.0, maximum=1.0).clamp(float("-inf")) == 0.5
    assert _float(minimum=None, maximum=None).clamp(float("inf")) == 0.5
    assert math.isfinite(_float().clamp(json.loads("1e999")))


# ---- CHOICE ----
def test_choice_valid_and_invalid():
    p = _choice()
    assert p.clamp("b") == "b"
    assert p.clamp("z") == "a"  # invalid -> default
    assert p.clamp(None) == "a"


# ---- TEXT ----
def test_text_coerces_and_handles_none():
    p = _text()
    assert p.clamp(None) == ""
    assert p.clamp(123) == "123"


# ---- BOOL ----
def test_bool_coerces():
    p = _bool()
    assert p.clamp(1) is True
    assert p.clamp(0) is False
    assert p.clamp("") is False


# ---- integration: load_state must survive a non-finite persisted value ----
def test_load_state_with_non_finite_int_does_not_raise():
    # Reproduces the startup crash: a saved session whose INT value round-tripped
    # through JSON as an overflowing literal (-> inf). load_state runs every value
    # through Param.clamp, so it must fall back to the default without raising.
    grid = GridControl()
    grid.load_state({"enabled": True, "values": {"columns": float("inf")}})
    assert grid.get("columns") == 4  # the param default
    assert grid.enabled is True


def test_load_state_non_finite_from_json_payload():
    payload = json.loads('{"enabled": true, "values": {"columns": 1e999}}')
    grid = GridControl()
    grid.load_state(payload)  # must not raise
    assert grid.get("columns") == 4
