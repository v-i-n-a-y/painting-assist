# Copyright 2026 Vinay Williams

"""Ensure the repo root is importable so `painting_assist` resolves under
`uv run pytest` regardless of how the editable install exposes itself."""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
