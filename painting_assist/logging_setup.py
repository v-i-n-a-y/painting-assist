# Copyright 2026 Vinay Williams

"""Application-wide logging configuration, retention, and log-viewer helpers.

The application writes a fresh log file per process ("session") into an
app-data ``logs`` folder, tees warnings and above to stderr, captures Python
``warnings.warn`` calls, and installs hooks so uncaught exceptions on the main
thread and on background threads are recorded instead of silently vanishing.
Old session files are pruned on a retention schedule so the log directory does
not grow without bound.

This module is deliberately Qt-free and standard-library only, so it can be
imported and unit tested without a running application or display server. A
log-viewer UI elsewhere in the app is expected to use :func:`list_session_logs`,
:func:`read_log_text`, and :func:`filter_log_text` to browse and filter past
sessions, including multi-line traceback blocks, by level.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable

RETENTION_DAYS: int = 7
LEVEL_NAMES: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_SESSION_GLOB = "session-*.log"

_log = logging.getLogger(__name__)

# Regex matching the start of a new log record, as emitted by LOG_FORMAT with
# the default asctime format (e.g. "2026-07-30 08:38:12,123 | INFO     | ...").
_RECORD_START_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \| (\w+)\s*\|"
)

# Module state recorded by `configure`, used for idempotency, active-file
# lookups, and chaining to whichever hooks were installed before us.
_active_log_dir: str | None = None
_active_session_path: str | None = None
_our_handlers: list[logging.Handler] = []
_prior_excepthook: Callable[..., None] | None = None
_prior_threading_excepthook: Callable[..., None] | None = None
_in_excepthook = False
_in_threading_excepthook = False


def default_log_dir(app_data_dir: str) -> str:
    """Return the conventional ``logs`` subfolder of ``app_data_dir``."""
    return os.path.join(app_data_dir, "logs")


def resolve_log_dir(configured: str | None, app_data_dir: str) -> str:
    """Resolve the log directory to use, honouring an explicit override.

    Returns ``configured`` (stripped of surrounding whitespace) if it is
    truthy after stripping, otherwise falls back to
    :func:`default_log_dir` for ``app_data_dir``.
    """
    if configured:
        stripped = configured.strip()
        if stripped:
            return stripped
    return default_log_dir(app_data_dir)


def session_filename(when: datetime, pid: int) -> str:
    """Return the deterministic session log filename for ``when`` and ``pid``."""
    return f"session-{when:%Y%m%d-%H%M%S}-{pid}.log"


def configure(
    log_dir: str,
    *,
    level: int = logging.DEBUG,
    retention_days: int = RETENTION_DAYS,
    notifier: Callable[[str], None] | None = None,
) -> str:
    """Configure the root logger with a per-session file and a stderr tee.

    Creates ``log_dir`` if needed, then attaches a :class:`logging.FileHandler`
    for a new session file (named via :func:`session_filename`) at ``level``,
    and a :class:`logging.StreamHandler` to stderr at ``WARNING``. Both use
    :data:`LOG_FORMAT`. Python's ``warnings`` module is routed into logging via
    :func:`logging.captureWarnings`, and ``sys.excepthook`` /
    ``threading.excepthook`` are replaced with handlers that log uncaught
    exceptions at ``CRITICAL``, optionally call ``notifier`` with a short
    message, and then chain to whatever hook was previously installed.

    Calling this more than once is safe: handlers added by a previous call are
    removed before new ones are attached, so the root logger never accumulates
    duplicate handlers from this module. Handlers added by other code are left
    untouched. Returns the absolute-or-relative path of the new session file
    (whatever form ``log_dir`` was given in).
    """
    global \
        _active_log_dir, \
        _active_session_path, \
        _prior_excepthook, \
        _prior_threading_excepthook

    os.makedirs(log_dir, exist_ok=True)

    session_path = os.path.join(log_dir, session_filename(datetime.now(), os.getpid()))

    root = logging.getLogger()
    root.setLevel(level)

    # Remove only the handlers this module previously added, so repeated
    # `configure` calls do not duplicate handlers, while handlers installed by
    # other code are left alone.
    for handler in _our_handlers:
        root.removeHandler(handler)
    _our_handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(session_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    _our_handlers.append(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    _our_handlers.append(stream_handler)

    logging.captureWarnings(True)

    _prior_excepthook = sys.excepthook
    sys.excepthook = _make_excepthook(notifier)

    _prior_threading_excepthook = threading.excepthook
    threading.excepthook = _threading_excepthook

    _active_log_dir = log_dir
    _active_session_path = session_path

    return session_path


def _make_excepthook(
    notifier: Callable[[str], None] | None,
) -> Callable[[type[BaseException], BaseException, object], None]:
    """Build the ``sys.excepthook`` replacement installed by :func:`configure`.

    The returned hook logs the exception at CRITICAL, optionally notifies via
    ``notifier``, and then chains to the hook that was active before
    :func:`configure` was called. ``KeyboardInterrupt`` and ``SystemExit`` are
    passed straight through to the prior hook without logging or notifying.
    Recursive invocation (e.g. a failure inside logging itself) is guarded
    against so it cannot loop.
    """

    def _hook(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
        global _in_excepthook

        prior = _prior_excepthook or sys.__excepthook__

        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            prior(exc_type, exc, tb)
            return

        if _in_excepthook:
            prior(exc_type, exc, tb)
            return

        _in_excepthook = True
        try:
            logging.getLogger("painting_assist").critical(
                "Uncaught exception", exc_info=(exc_type, exc, tb)
            )
            if notifier is not None:
                notifier(f"{exc_type.__name__}: {exc}")
        finally:
            _in_excepthook = False

        prior(exc_type, exc, tb)

    return _hook


def _threading_excepthook(args: threading.ExceptHookArgs) -> None:
    """``threading.excepthook`` replacement installed by :func:`configure`.

    Logs the background-thread exception at CRITICAL (skipping if
    ``exc_value`` is ``None``, which threading itself never passes but which
    a caller could conceivably synthesize), then chains to whichever
    ``threading.excepthook`` was active before :func:`configure` ran. Guarded
    against recursive invocation.
    """
    global _in_threading_excepthook

    prior = _prior_threading_excepthook or threading.__excepthook__

    if args.exc_value is None or _in_threading_excepthook:
        prior(args)
        return

    _in_threading_excepthook = True
    try:
        logging.getLogger("painting_assist").critical(
            "Uncaught exception in thread %r",
            args.thread.name if args.thread is not None else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
    finally:
        _in_threading_excepthook = False

    prior(args)


def active_log_dir() -> str | None:
    """Return the log directory passed to the last :func:`configure` call."""
    return _active_log_dir


def active_session_path() -> str | None:
    """Return the session file path from the last :func:`configure` call."""
    return _active_session_path


def cleanup_old_logs(
    log_dir: str,
    retention_days: int = RETENTION_DAYS,
    *,
    now: float | None = None,
) -> list[str]:
    """Delete session log files in ``log_dir`` older than ``retention_days``.

    ``now`` defaults to :func:`time.time`; files with an mtime older than
    ``now - retention_days * 86400`` are removed. The currently active session
    file (per :func:`active_session_path`) is never deleted, even if its mtime
    happens to be old. Returns the list of removed paths (as globbed, i.e.
    joined under ``log_dir``). Per-file deletion failures are swallowed and
    logged; a missing ``log_dir`` yields an empty list rather than an error.
    """
    if not os.path.isdir(log_dir):
        return []

    if now is None:
        now = time.time()
    cutoff = now - retention_days * 86400

    protected = active_session_path()
    removed: list[str] = []

    for path in glob.glob(os.path.join(log_dir, _SESSION_GLOB)):
        if protected is not None and os.path.abspath(path) == os.path.abspath(
            protected
        ):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError as exc:
            _log.warning("Could not stat log file %s: %s", path, exc)
            continue
        if mtime < cutoff:
            try:
                os.remove(path)
            except OSError as exc:
                _log.warning("Could not remove old log file %s: %s", path, exc)
                continue
            removed.append(path)

    return removed


def cleanup_old_logs_async(
    log_dir: str, retention_days: int = RETENTION_DAYS
) -> threading.Thread:
    """Run :func:`cleanup_old_logs` on a background daemon thread and return it.

    The thread is already started when this returns. Any exception raised
    inside the cleanup is caught and logged rather than propagated, since this
    is meant to be fire-and-forget housekeeping at startup.
    """

    def _run() -> None:
        try:
            cleanup_old_logs(log_dir, retention_days)
        except Exception:
            _log.warning("Log cleanup failed", exc_info=True)

    thread = threading.Thread(target=_run, name="log-cleanup", daemon=True)
    thread.start()
    return thread


def list_session_logs(log_dir: str) -> list[str]:
    """Return session log paths in ``log_dir``, newest-first by mtime.

    Returns an empty list if ``log_dir`` does not exist.
    """
    if not os.path.isdir(log_dir):
        return []

    paths = [
        os.path.abspath(p) for p in glob.glob(os.path.join(log_dir, _SESSION_GLOB))
    ]

    def _mtime(path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    paths.sort(key=_mtime, reverse=True)
    return paths


def read_log_text(path: str) -> str:
    """Read a log file as UTF-8 text, returning "" on any read failure."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


@dataclass
class LogEntry:
    """A single parsed log record, including any continuation/traceback lines."""

    level: str
    raw: str


def parse_log_text(text: str) -> list[LogEntry]:
    """Parse ``text`` (as written with :data:`LOG_FORMAT`) into log records.

    A new record starts on any line matching the ``asctime | LEVEL | ...``
    prefix; the level is taken from that line. Lines that do not start a new
    record (continuation lines, including traceback bodies) are appended to
    the current record's :attr:`LogEntry.raw`, joined with ``"\\n"``. Any text
    appearing before the first matching line is kept as a leading record with
    level ``"INFO"`` rather than being discarded. Record order is preserved.
    """
    entries: list[LogEntry] = []
    current_lines: list[str] | None = None
    current_level = "INFO"

    for line in text.splitlines():
        match = _RECORD_START_RE.match(line)
        if match:
            if current_lines is not None:
                entries.append(
                    LogEntry(level=current_level, raw="\n".join(current_lines))
                )
            current_level = match.group(1).strip().upper()
            current_lines = [line]
        else:
            if current_lines is None:
                current_lines = [line]
            else:
                current_lines.append(line)

    if current_lines is not None:
        entries.append(LogEntry(level=current_level, raw="\n".join(current_lines)))

    return entries


def filter_log_text(text: str, levels: Iterable[str]) -> str:
    """Return only the record blocks in ``text`` whose level is in ``levels``.

    ``levels`` is upper-cased for comparison. Matching records (including any
    continuation/traceback lines that belong to them) are joined with
    ``"\\n"`` in their original order. An empty ``levels`` yields ``""``.
    """
    wanted = {level.upper() for level in levels}
    if not wanted:
        return ""

    entries = parse_log_text(text)
    return "\n".join(entry.raw for entry in entries if entry.level in wanted)
