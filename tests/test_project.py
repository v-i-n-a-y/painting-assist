# Copyright 2026 Vinay Williams

"""Tests for the Qt-free ``.paproj`` project-document model."""

from __future__ import annotations

import json

from painting_assist.project import (
    PROJECT_EXTENSION,
    PROJECT_SCHEMA_VERSION,
    ProjectDocument,
    from_json,
    migrate,
)

REALISTIC_CONTROLS = {
    "crop": {"id": "crop", "enabled": True, "values": {"rx": 0.1, "rw": 0.5}},
    "values": {
        "id": "values",
        "enabled": True,
        "values": {"mode": "mono", "mono_hex": "#6e4128"},
    },
}


def test_new_sets_current_version_and_carries_fields():
    measure = {"unit": "cm", "edges": True}
    doc = ProjectDocument.new(
        image_path="/photos/study.jpg",
        controls=REALISTIC_CONTROLS,
        measure=measure,
        app_version="0.8.0",
    )
    assert doc.schema_version == PROJECT_SCHEMA_VERSION
    assert doc.app_version == "0.8.0"
    assert doc.image_path == "/photos/study.jpg"
    assert doc.controls == REALISTIC_CONTROLS
    assert doc.measure == measure


def test_new_deep_copies_inputs():
    controls = {"crop": {"id": "crop", "enabled": True, "values": {"rx": 0.1}}}
    doc = ProjectDocument.new("/a.jpg", controls, {"unit": "cm"})
    controls["crop"]["values"]["rx"] = 0.9
    assert doc.controls["crop"]["values"]["rx"] == 0.1


def test_to_json_from_json_round_trip():
    measure = {"unit": "cm", "edges": True}
    doc = ProjectDocument.new(
        image_path="/photos/étude.jpg",
        controls=REALISTIC_CONTROLS,
        measure=measure,
        app_version="0.8.0",
    )
    restored = from_json(doc.to_json())
    assert restored.image_path == "/photos/étude.jpg"
    assert restored.controls == REALISTIC_CONTROLS
    assert restored.measure == measure
    assert restored.schema_version == PROJECT_SCHEMA_VERSION


def test_to_json_top_level_keys():
    doc = ProjectDocument.new("/a.jpg", {}, {})
    data = json.loads(doc.to_json())
    assert set(data) == {
        "schema_version",
        "app_version",
        "image_path",
        "controls",
        "measure",
    }


def test_from_json_invalid_json_returns_default():
    doc = from_json("{not valid json")
    assert doc.schema_version == PROJECT_SCHEMA_VERSION
    assert doc.image_path == ""
    assert doc.controls == {}
    assert doc.measure == {}
    assert doc.app_version == ""


def test_from_json_non_dict_top_level_returns_default():
    doc = from_json("[1, 2, 3]")
    assert doc.image_path == ""
    assert doc.controls == {}
    assert doc.measure == {}


def test_from_json_missing_controls_defaults_to_empty_dict():
    doc = from_json(json.dumps({"image_path": "/a.jpg", "measure": {"unit": "cm"}}))
    assert doc.controls == {}
    assert doc.image_path == "/a.jpg"
    assert doc.measure == {"unit": "cm"}


def test_from_json_non_dict_controls_coerced_to_empty_dict():
    doc = from_json(json.dumps({"image_path": "/a.jpg", "controls": [1, 2, 3]}))
    assert doc.controls == {}


def test_from_json_non_dict_measure_coerced_to_empty_dict():
    doc = from_json(json.dumps({"image_path": "/a.jpg", "measure": "cm"}))
    assert doc.measure == {}


def test_document_without_schema_version_migrates():
    doc = from_json(json.dumps({"image_path": "/a.jpg", "controls": {}}))
    assert doc.schema_version == PROJECT_SCHEMA_VERSION


def test_migrate_no_op_at_current_version():
    data = {"schema_version": PROJECT_SCHEMA_VERSION, "image_path": "/a.jpg"}
    assert migrate(data) == data


def test_migrate_stamps_missing_version():
    migrated = migrate({"image_path": "/a.jpg"})
    assert migrated["schema_version"] == PROJECT_SCHEMA_VERSION


def test_migrate_does_not_downgrade_newer_document():
    future = PROJECT_SCHEMA_VERSION + 5
    migrated = migrate({"schema_version": future})
    assert migrated["schema_version"] == future


def test_extension_constant():
    assert PROJECT_EXTENSION == ".paproj"
