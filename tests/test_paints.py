# Copyright 2026 Vinay Williams

"""Tests for the paint catalogue and its JSON serialisation helpers."""

from __future__ import annotations

import json

import pytest

from painting_assist.paints import (
    DEFAULT_CATALOGUE,
    paints_from_json,
    paints_to_json,
)


def test_catalogue_non_empty_and_well_formed():
    assert DEFAULT_CATALOGUE, "catalogue should not be empty"
    assert len(DEFAULT_CATALOGUE) >= 20
    for entry in DEFAULT_CATALOGUE:
        name, rgb = entry
        assert isinstance(name, str) and name
        assert isinstance(rgb, tuple) and len(rgb) == 3
        for component in rgb:
            assert isinstance(component, int)
            assert 0 <= component <= 255


def test_catalogue_names_unique():
    names = [name for name, _ in DEFAULT_CATALOGUE]
    assert len(names) == len(set(names))


def test_catalogue_is_broad():
    assert len(DEFAULT_CATALOGUE) >= 100


def test_catalogue_includes_staples():
    names = {name for name, _ in DEFAULT_CATALOGUE}
    for staple in (
        "Titanium White",
        "Ivory Black",
        "Ultramarine Blue",
        "Cadmium Red",
        "Cadmium Yellow",
        "Yellow Ochre",
        "Burnt Sienna",
        "Raw Umber",
        "Phthalo Blue",
        "Phthalo Green",
        "Alizarin Crimson",
        "Viridian",
    ):
        assert staple in names


def test_roundtrip_preserves_catalogue():
    restored = paints_from_json(paints_to_json(DEFAULT_CATALOGUE))
    assert restored == DEFAULT_CATALOGUE


def test_to_json_emits_valid_json_list():
    data = json.loads(paints_to_json(DEFAULT_CATALOGUE))
    assert isinstance(data, list)
    assert data[0] == {"name": "Titanium White", "rgb": [250, 250, 245]}


def test_from_json_invalid_json_returns_empty():
    assert paints_from_json("not json at all") == []
    assert paints_from_json("") == []
    assert paints_from_json("{unbalanced") == []


def test_from_json_non_list_top_level_returns_empty():
    assert paints_from_json('{"name": "Red", "rgb": [255, 0, 0]}') == []
    assert paints_from_json("42") == []
    assert paints_from_json("null") == []


def test_from_json_drops_bad_entries_without_raising():
    text = json.dumps(
        [
            {"name": "Good", "rgb": [10, 20, 30]},
            {"name": "No colour"},
            {"rgb": [1, 2, 3]},
            {"name": "Short", "rgb": [1, 2]},
            {"name": "Non numeric", "rgb": [1, "x", 3]},
            "not an object",
            42,
            {"name": "", "rgb": [1, 2, 3]},
            {"name": "Also good", "rgb": [40, 50, 60]},
        ]
    )
    result = paints_from_json(text)
    assert result == [("Good", (10, 20, 30)), ("Also good", (40, 50, 60))]


def test_from_json_clamps_and_rounds_components():
    text = json.dumps([{"name": "Clamped", "rgb": [-20, 300, 128.7]}])
    assert paints_from_json(text) == [("Clamped", (0, 255, 129))]


def test_from_json_coerces_non_string_name():
    text = json.dumps([{"name": 7, "rgb": [1, 2, 3]}])
    assert paints_from_json(text) == [("7", (1, 2, 3))]


def test_from_json_accepts_bare_pair_form():
    text = json.dumps([["Red", [255, 0, 0]]])
    assert paints_from_json(text) == [("Red", (255, 0, 0))]


def test_to_json_drops_malformed_entries_without_raising():
    messy = [
        ("Fine", (1, 2, 3)),
        ("Bad rgb", (1, 2)),
        ("Non numeric", (1, None, 3)),
        "junk",
    ]
    restored = paints_from_json(paints_to_json(messy))
    assert restored == [("Fine", (1, 2, 3))]


@pytest.mark.parametrize("text", ["[]", json.dumps([])])
def test_from_json_empty_list_is_empty(text):
    assert paints_from_json(text) == []
