# Copyright 2026 Vinay Williams

"""A versioned JSON settings store for portable, human-meaningful preferences.

The application keeps its portable settings in a single JSON document in the OS
application-data folder, so the format can be migrated forward across releases.
This module owns the schema, the canonical :data:`DEFAULTS`, and the migration
chain; it deliberately stays Qt-free and standard-library only. Where the file
actually lives is decided elsewhere (via ``QStandardPaths``); :class:`SettingsStore`
reads and writes whatever path it is handed.

The store is robust in the same spirit as :mod:`painting_assist.paints`: a
missing, unreadable, or malformed file never raises, it simply falls back to a
deep copy of :data:`DEFAULTS`. A loaded document is migrated to the current
schema and then deep-merged onto the defaults, so a file written by an older or
partial build still yields a complete settings dict with every expected section
and key present.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from typing import Callable

SCHEMA_VERSION = 1

# The canonical default settings document. Every load resolves to a deep copy of
# this structure with the loaded file's values merged over the top, so callers
# can rely on every section and key below always being present.
DEFAULTS: dict = {
    "schema_version": SCHEMA_VERSION,
    "app_version": "",
    "preferences": {
        "theme": "system",
        "update_hours": 24.0,
        "last_update_check": 0.0,
        "tolerance_pct": 25,
        "on_miss": "closest",
        "measure_unit": "cm",
        "measure_edges": True,
    },
    "paints": [],  # list of {"name": str, "rgb": [r, g, b]}
    "mono_hidden": [],  # list of str
    "presets": {},  # {name: {control_id: {...state...}}}
    "recent_images": [],  # list of str paths
    "recent_projects": [],  # list of str paths
    "session": {"last_image": "", "controls": {}},
}


# Migration chain. Each entry maps a from-version ``n`` to a callable that takes a
# document at schema version ``n`` and returns it at version ``n + 1``. Adding a
# migration when the schema changes is a one-liner: bump :data:`SCHEMA_VERSION`
# and register ``_MIGRATIONS[n] = _migrate_n_to_n_plus_1``. There are no prior
# schema versions yet, so the chain is empty.
_MIGRATIONS: dict[int, Callable[[dict], dict]] = {}


def migrate(data: dict) -> dict:
    """Migrate a settings document up to the current :data:`SCHEMA_VERSION`.

    Applies the ``v(n) -> v(n + 1)`` transforms registered in :data:`_MIGRATIONS`
    in order, starting from ``data["schema_version"]``. A missing
    ``schema_version`` is treated as ``0`` (a pre-versioning document). A document
    already at or beyond the current version is returned unchanged, so a newer
    file written by a future build is never downgraded. For a known or older
    version, ``schema_version`` is stamped to :data:`SCHEMA_VERSION` at the end.

    The document is migrated in place and also returned for convenience.
    """
    try:
        version = int(data.get("schema_version", 0))
    except (TypeError, ValueError):
        version = 0

    # Never downgrade a document written by a newer build than we understand.
    if version >= SCHEMA_VERSION:
        return data

    while version < SCHEMA_VERSION:
        transform = _MIGRATIONS.get(version)
        if transform is None:
            # No explicit step for this version; advance without altering data.
            version += 1
            continue
        data = transform(data)
        version += 1

    data["schema_version"] = SCHEMA_VERSION
    return data


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` onto a copy-friendly ``base`` dict.

    ``base`` is mutated in place. For keys present in both where both values are
    dicts, the merge recurses; otherwise the ``overlay`` value wins (lists and
    scalars are taken wholesale from ``overlay``). Keys present only in ``base``
    are left untouched, which is what fills in any section or key a loaded file
    omits. Returns ``base``.
    """
    for key, value in overlay.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge(existing, value)
        else:
            base[key] = value
    return base


class SettingsStore:
    """A read/write wrapper around a single versioned JSON settings file.

    Construction records the path only; call :meth:`load` to populate
    :attr:`data` from disk (or from a deep copy of :data:`DEFAULTS` when the file
    is absent or unreadable), and :meth:`save` to write it back atomically.
    """

    def __init__(self, path: str) -> None:
        """Record the settings file ``path`` without touching the disk.

        :attr:`data` starts as a deep copy of :data:`DEFAULTS` so the store is
        usable before an explicit :meth:`load`.
        """
        self.path = path
        self.data: dict = copy.deepcopy(DEFAULTS)

    def load(self) -> None:
        """Load and normalise the settings document from :attr:`path`.

        On success the file is parsed, migrated to the current schema, and
        deep-merged onto a fresh copy of :data:`DEFAULTS` so any missing section
        or key is filled in. On any failure (file missing, unreadable, invalid
        JSON, or a non-dict top level) :attr:`data` falls back to a deep copy of
        :data:`DEFAULTS`. This method never raises.
        """
        try:
            with open(self.path, encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, ValueError):
            self.data = copy.deepcopy(DEFAULTS)
            return

        if not isinstance(loaded, dict):
            self.data = copy.deepcopy(DEFAULTS)
            return

        migrated = migrate(loaded)
        merged = _deep_merge(copy.deepcopy(DEFAULTS), migrated)
        # The merged document should always report the current schema version.
        merged["schema_version"] = SCHEMA_VERSION
        self.data = merged

    def save(self) -> None:
        """Write :attr:`data` to :attr:`path` as pretty JSON, atomically.

        The document is written to a temporary file in the same directory and
        then moved onto the target with :func:`os.replace`, so a concurrent
        reader never sees a half-written file and a crash mid-write cannot
        corrupt the existing settings. Parent directories are created if needed.
        """
        dirname = os.path.dirname(self.path) or "."
        try:
            os.makedirs(dirname, exist_ok=True)
        except OSError:
            # Directory creation must never raise; fall through and let the write
            # surface any genuine, unavoidable error to the caller instead.
            pass

        text = json.dumps(self.data, indent=2, sort_keys=False, ensure_ascii=False)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=dirname,
            prefix=".settings-",
            suffix=".tmp",
            delete=False,
        )
        tmp_path = handle.name
        try:
            with handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        except OSError:
            # Clean up the temp file on failure so we do not leave litter behind.
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    def get(self, section: str, key: str, default=None):
        """Return ``data[section][key]`` if present, else ``default``.

        Robust to a missing section or a section that is not a dict.
        """
        block = self.data.get(section)
        if not isinstance(block, dict):
            return default
        return block.get(key, default)

    def set(self, section: str, key: str, value) -> None:
        """Set ``data[section][key] = value``, creating the section if absent.

        If ``section`` exists but is not a dict, it is replaced with a fresh dict.
        """
        block = self.data.get(section)
        if not isinstance(block, dict):
            block = {}
            self.data[section] = block
        block[key] = value
