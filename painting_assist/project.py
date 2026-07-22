# Copyright 2026 Vinay Williams

"""A versioned, Qt-free model for the ``.paproj`` project document.

A *project* is the per-painting document a user saves and reopens. It captures
everything needed to restore a working session *except* the source pixels: the
photo is referenced **by path**, not embedded, so a project file stays small and
the original image remains the single source of truth on disk. The trade-off is
that moving or deleting the referenced photo will leave a project unable to find
its image; callers are responsible for resolving or re-pointing a stale path.

Alongside the image path the document carries two opaque JSON-able blobs:

* ``controls`` -- the full image-control capture, of the form
  ``{control_id: {"id": str, "enabled": bool, "values": {name: value}}}`` (what
  the app's ``_capture_state()`` produces). This module never interprets it; it
  is carried faithfully through save and load.
* ``measure`` -- the measure display settings, e.g. ``{"unit": "cm",
  "edges": True}``.

Serialisation mirrors the defensive house style of
:mod:`painting_assist.paints`: :func:`from_json` never raises. Invalid JSON, a
non-dict top level, or missing/mistyped fields all resolve to sensible defaults
at the current schema version rather than an exception. Documents are migrated
forward through :func:`migrate` on load; a newer-than-known document is left at
its own version rather than being downgraded.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Callable

PROJECT_SCHEMA_VERSION = 1
PROJECT_EXTENSION = ".paproj"


@dataclass
class ProjectDocument:
    """An in-memory ``.paproj`` document.

    Fields:

    * ``schema_version`` -- the on-disk schema this document conforms to.
    * ``app_version`` -- the app version that wrote it (informational only).
    * ``image_path`` -- path to the referenced photo, as given; may be absolute.
      The pixels are not embedded.
    * ``controls`` -- opaque image-control capture; see the module docstring.
    * ``measure`` -- measure display settings.
    """

    schema_version: int
    app_version: str
    image_path: str
    controls: dict
    measure: dict

    @classmethod
    def new(
        cls,
        image_path: str,
        controls: dict,
        measure: dict,
        app_version: str = "",
    ) -> ProjectDocument:
        """Build a document at the current schema version.

        The ``controls`` and ``measure`` blobs are deep-copied so later mutation
        of the caller's dicts does not leak into the document.
        """
        return cls(
            schema_version=PROJECT_SCHEMA_VERSION,
            app_version=str(app_version),
            image_path=str(image_path),
            controls=copy.deepcopy(controls),
            measure=copy.deepcopy(measure),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a pretty JSON string.

        Top-level keys are ``schema_version``, ``app_version``, ``image_path``,
        ``controls`` and ``measure``. ``ensure_ascii=False`` keeps non-ASCII
        text (e.g. accented file names) readable in the file.
        """
        data = {
            "schema_version": self.schema_version,
            "app_version": self.app_version,
            "image_path": self.image_path,
            "controls": self.controls,
            "measure": self.measure,
        }
        return json.dumps(data, indent=indent, ensure_ascii=False)


def _default_document() -> ProjectDocument:
    """Return an empty document at the current schema version."""
    return ProjectDocument(
        schema_version=PROJECT_SCHEMA_VERSION,
        app_version="",
        image_path="",
        controls={},
        measure={},
    )


# Migration chain, keyed by from-version -> callable that upgrades a v(n) dict
# to a v(n+1) dict. No prior schema versions exist yet, so this is empty; adding
# a future migration is a one-liner here (e.g. ``1: _migrate_1_to_2``).
_MIGRATIONS: dict[int, Callable[[dict], dict]] = {}


def migrate(data: dict) -> dict:
    """Migrate a raw document ``dict`` forward to :data:`PROJECT_SCHEMA_VERSION`.

    Applies the ``v(n) -> v(n+1)`` chain in :data:`_MIGRATIONS` starting from
    ``data.get("schema_version", 0)``. A document already at (or newer than) the
    current version is never downgraded; a known/older document has its
    ``schema_version`` stamped up to the current version once migrated.
    """
    data = dict(data)
    version = data.get("schema_version", 0)
    if not isinstance(version, int):
        version = 0
    if version >= PROJECT_SCHEMA_VERSION:
        return data
    while version < PROJECT_SCHEMA_VERSION:
        migrator = _MIGRATIONS.get(version)
        if migrator is None:
            break
        data = migrator(data)
        version += 1
    data["schema_version"] = PROJECT_SCHEMA_VERSION
    return data


def from_json(text: str) -> ProjectDocument:
    """Parse a JSON string into a :class:`ProjectDocument`, never raising.

    Robust to malformed input in the house style of
    :mod:`painting_assist.paints`: invalid JSON or a non-dict top level yields a
    default document at the current schema version. Otherwise the raw data is
    migrated forward and its fields are coerced -- ``image_path`` and
    ``app_version`` to strings (default ``""``), ``controls`` and ``measure`` to
    dicts (default ``{}`` when missing or of the wrong type).
    """
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return _default_document()
    if not isinstance(data, dict):
        return _default_document()

    data = migrate(data)

    image_path = data.get("image_path", "")
    image_path = str(image_path) if image_path is not None else ""

    app_version = data.get("app_version", "")
    app_version = str(app_version) if app_version is not None else ""

    controls = data.get("controls")
    if not isinstance(controls, dict):
        controls = {}

    measure = data.get("measure")
    if not isinstance(measure, dict):
        measure = {}

    return ProjectDocument(
        schema_version=PROJECT_SCHEMA_VERSION,
        app_version=app_version,
        image_path=image_path,
        controls=controls,
        measure=measure,
    )
