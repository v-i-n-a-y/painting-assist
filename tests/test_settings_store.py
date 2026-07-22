# Copyright 2026 Vinay Williams

"""Tests for the Qt-free versioned JSON settings store."""

from __future__ import annotations

import json

from painting_assist.settings_store import (
    DEFAULTS,
    SCHEMA_VERSION,
    SettingsStore,
    migrate,
)


def test_load_missing_path_yields_defaults(tmp_path):
    store = SettingsStore(str(tmp_path / "does_not_exist.json"))
    store.load()

    assert store.data == DEFAULTS
    assert store.data["schema_version"] == SCHEMA_VERSION
    # A full copy: every default section is present.
    for section in DEFAULTS:
        assert section in store.data


def test_load_is_deep_copy_not_defaults_alias(tmp_path):
    store = SettingsStore(str(tmp_path / "missing.json"))
    store.load()

    store.data["preferences"]["theme"] = "mutated"
    store.data["paints"].append({"name": "x", "rgb": [1, 2, 3]})

    # The module-level DEFAULTS must be untouched.
    assert DEFAULTS["preferences"]["theme"] == "system"
    assert DEFAULTS["paints"] == []


def test_save_then_load_round_trips(tmp_path):
    path = str(tmp_path / "settings.json")
    store = SettingsStore(path)
    store.load()
    store.set("preferences", "theme", "dark")
    store.set("session", "last_image", "/tmp/pic.png")
    store.data["paints"].append({"name": "Cobalt Blue", "rgb": [30, 70, 150]})
    store.save()

    reloaded = SettingsStore(path)
    reloaded.load()
    assert reloaded.get("preferences", "theme") == "dark"
    assert reloaded.get("session", "last_image") == "/tmp/pic.png"
    assert reloaded.data["paints"] == [{"name": "Cobalt Blue", "rgb": [30, 70, 150]}]


def test_corrupt_json_loads_defaults_without_raising(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{ this is not: valid json ]", encoding="utf-8")

    store = SettingsStore(str(path))
    store.load()  # must not raise
    assert store.data == DEFAULTS


def test_non_dict_top_level_loads_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    store = SettingsStore(str(path))
    store.load()
    assert store.data == DEFAULTS


def test_partial_file_deep_merges(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"preferences": {"theme": "dark"}}), encoding="utf-8")

    store = SettingsStore(str(path))
    store.load()

    prefs = store.data["preferences"]
    # Overridden key comes from the file.
    assert prefs["theme"] == "dark"
    # Sibling preference keys fall back to defaults.
    assert prefs["measure_unit"] == DEFAULTS["preferences"]["measure_unit"]
    assert prefs["tolerance_pct"] == DEFAULTS["preferences"]["tolerance_pct"]
    assert prefs["measure_edges"] == DEFAULTS["preferences"]["measure_edges"]
    # Missing top-level sections are present from defaults.
    assert store.data["paints"] == []
    assert store.data["session"] == DEFAULTS["session"]
    assert "recent_images" in store.data
    assert store.data["schema_version"] == SCHEMA_VERSION


def test_migrate_current_version_is_noop():
    data = {"schema_version": SCHEMA_VERSION, "preferences": {"theme": "dark"}}
    result = migrate(data)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["preferences"]["theme"] == "dark"


def test_migrate_missing_schema_version_upgrades():
    data = {"preferences": {"theme": "dark"}}  # pre-versioning document
    result = migrate(data)
    assert result["schema_version"] == SCHEMA_VERSION
    # Existing data survives the migration.
    assert result["preferences"]["theme"] == "dark"


def test_migrate_never_downgrades_newer_version():
    data = {"schema_version": SCHEMA_VERSION + 5}
    result = migrate(data)
    assert result["schema_version"] == SCHEMA_VERSION + 5


def test_atomic_save_leaves_readable_file(tmp_path):
    path = tmp_path / "settings.json"
    store = SettingsStore(str(path))
    store.load()
    store.set("app_version", "app_version", "1.2.3")  # arbitrary write
    store.set("preferences", "theme", "light")
    store.save()

    # File exists, is valid JSON, re-parses to the saved data, and no temp
    # litter remains beside it.
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["preferences"]["theme"] == "light"
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "settings.json"]
    assert leftovers == []


def test_save_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deeper" / "settings.json"
    store = SettingsStore(str(path))
    store.load()
    store.set("preferences", "theme", "dark")
    store.save()

    assert path.exists()
    reloaded = SettingsStore(str(path))
    reloaded.load()
    assert reloaded.get("preferences", "theme") == "dark"


def test_get_missing_key_returns_default(tmp_path):
    store = SettingsStore(str(tmp_path / "s.json"))
    store.load()
    assert store.get("preferences", "nonexistent", "fallback") == "fallback"
    assert store.get("no_such_section", "key", 42) == 42
    assert store.get("preferences", "theme") == "system"


def test_set_creates_section(tmp_path):
    store = SettingsStore(str(tmp_path / "s.json"))
    store.load()
    store.set("brand_new_section", "flag", True)
    assert store.data["brand_new_section"] == {"flag": True}
    assert store.get("brand_new_section", "flag") is True
