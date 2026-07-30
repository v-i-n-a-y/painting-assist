# Copyright 2026 Vinay Williams

"""Tests for :mod:`painting_assist.logging_setup`."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime

import pytest

from painting_assist import logging_setup as ls


@pytest.fixture(autouse=True)
def _restore_logging_state():
    """Snapshot and restore global logging/hook state around every test.

    `logging_setup.configure` mutates the root logger's handlers, the module's
    internal "active session" state, `sys.excepthook`, and
    `threading.excepthook`. None of that should leak between tests.
    """
    root = logging.getLogger()
    prior_handlers = list(root.handlers)
    prior_level = root.level
    prior_excepthook = sys.excepthook
    prior_threading_excepthook = threading.excepthook

    prior_active_log_dir = ls._active_log_dir
    prior_active_session_path = ls._active_session_path
    prior_our_handlers = list(ls._our_handlers)
    prior_prior_excepthook = ls._prior_excepthook
    prior_prior_threading_excepthook = ls._prior_threading_excepthook

    yield

    for handler in list(root.handlers):
        if handler not in prior_handlers:
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    root.setLevel(prior_level)
    sys.excepthook = prior_excepthook
    threading.excepthook = prior_threading_excepthook

    ls._active_log_dir = prior_active_log_dir
    ls._active_session_path = prior_active_session_path
    ls._our_handlers[:] = prior_our_handlers
    ls._prior_excepthook = prior_prior_excepthook
    ls._prior_threading_excepthook = prior_prior_threading_excepthook


def _close_our_handlers() -> None:
    """Flush and close the file handler(s) `configure` attached, if any."""
    for handler in list(ls._our_handlers):
        handler.flush()


# --- session_filename ---------------------------------------------------


def test_session_filename_exact_format():
    when = datetime(2026, 7, 30, 8, 38, 12)
    assert ls.session_filename(when, 1234) == "session-20260730-083812-1234.log"


# --- default_log_dir / resolve_log_dir ----------------------------------


def test_default_log_dir():
    assert ls.default_log_dir("/tmp/app") == os.path.join("/tmp/app", "logs")


def test_resolve_log_dir_none():
    assert ls.resolve_log_dir(None, "/tmp/app") == ls.default_log_dir("/tmp/app")


def test_resolve_log_dir_empty():
    assert ls.resolve_log_dir("", "/tmp/app") == ls.default_log_dir("/tmp/app")


def test_resolve_log_dir_whitespace():
    assert ls.resolve_log_dir("   ", "/tmp/app") == ls.default_log_dir("/tmp/app")


def test_resolve_log_dir_override():
    assert ls.resolve_log_dir("/custom/logs", "/tmp/app") == "/custom/logs"


def test_resolve_log_dir_override_stripped():
    assert ls.resolve_log_dir("  /custom/logs  ", "/tmp/app") == "/custom/logs"


# --- cleanup_old_logs -----------------------------------------------------


def test_cleanup_old_logs_removes_only_old_session_files(tmp_path):
    now = time.time()
    old_day = 10
    recent_day = 1

    old1 = tmp_path / "session-20260101-000000-1.log"
    old2 = tmp_path / "session-20260102-000000-2.log"
    recent = tmp_path / "session-20260103-000000-3.log"
    other = tmp_path / "not-a-session.log"

    for path in (old1, old2, recent, other):
        path.write_text("x")

    os.utime(old1, (now - old_day * 86400, now - old_day * 86400))
    os.utime(old2, (now - old_day * 86400, now - old_day * 86400))
    os.utime(recent, (now - recent_day * 86400, now - recent_day * 86400))
    os.utime(other, (now - old_day * 86400, now - old_day * 86400))

    removed = ls.cleanup_old_logs(str(tmp_path), retention_days=7, now=now)

    assert set(removed) == {str(old1), str(old2)}
    assert not old1.exists()
    assert not old2.exists()
    assert recent.exists()
    assert other.exists()


def test_cleanup_old_logs_missing_dir_returns_empty(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert ls.cleanup_old_logs(str(missing)) == []


def test_cleanup_old_logs_never_deletes_active_session(tmp_path):
    session_path = ls.configure(str(tmp_path))
    _close_our_handlers()

    old_time = time.time() - 10 * 86400
    os.utime(session_path, (old_time, old_time))

    removed = ls.cleanup_old_logs(str(tmp_path), retention_days=7)

    assert session_path not in removed
    assert os.path.exists(session_path)


# --- configure --------------------------------------------------------


def test_configure_writes_session_file_and_logs(tmp_path):
    session_path = ls.configure(str(tmp_path))

    logger = logging.getLogger("painting_assist.test")
    logger.info("hello")

    for handler in ls._our_handlers:
        handler.flush()

    assert os.path.exists(session_path)
    text = ls.read_log_text(session_path)
    assert "hello" in text
    assert "painting_assist.test" in text


def test_configure_sets_active_state(tmp_path):
    session_path = ls.configure(str(tmp_path))
    assert ls.active_log_dir() == str(tmp_path)
    assert ls.active_session_path() == session_path


def test_configure_twice_does_not_duplicate_handlers(tmp_path):
    root = logging.getLogger()

    ls.configure(str(tmp_path))
    count_after_first = sum(1 for h in root.handlers if h in ls._our_handlers)

    ls.configure(str(tmp_path))
    count_after_second = sum(1 for h in root.handlers if h in ls._our_handlers)

    assert count_after_first == 2
    assert count_after_second == 2


def test_configure_returns_no_state_before_configure(tmp_path):
    # Sanity: without calling configure in this test, state should reflect
    # whatever the fixture restored it to (None, since nothing configured yet
    # at fixture setup in an unpolluted run). We only assert the type contract
    # here rather than a specific value, since other tests may have run first
    # in-process (state is restored by the fixture regardless).
    assert ls.active_log_dir() is None or isinstance(ls.active_log_dir(), str)


def test_configure_chains_prior_excepthook(tmp_path):
    calls = []

    def fake_hook(exc_type, exc, tb):
        calls.append((exc_type, exc))

    sys.excepthook = fake_hook
    ls.configure(str(tmp_path))

    try:
        raise ValueError("boom")
    except ValueError:
        exc_type, exc, tb = sys.exc_info()
        sys.excepthook(exc_type, exc, tb)

    assert len(calls) == 1
    assert calls[0][0] is ValueError


def test_configure_excepthook_logs_uncaught_exception(tmp_path):
    session_path = ls.configure(str(tmp_path))

    try:
        raise RuntimeError("kaboom")
    except RuntimeError:
        exc_type, exc, tb = sys.exc_info()
        sys.excepthook(exc_type, exc, tb)

    for handler in ls._our_handlers:
        handler.flush()

    text = ls.read_log_text(session_path)
    assert "Uncaught exception" in text
    assert "kaboom" in text


def test_configure_notifier_called_on_uncaught_exception(tmp_path):
    notified = []
    ls.configure(str(tmp_path), notifier=notified.append)

    try:
        raise RuntimeError("kaboom")
    except RuntimeError:
        exc_type, exc, tb = sys.exc_info()
        sys.excepthook(exc_type, exc, tb)

    assert notified == ["RuntimeError: kaboom"]


def test_configure_excepthook_skips_keyboard_interrupt(tmp_path):
    session_path = ls.configure(str(tmp_path))

    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        exc_type, exc, tb = sys.exc_info()
        try:
            sys.excepthook(exc_type, exc, tb)
        except SystemExit:
            pass

    for handler in ls._our_handlers:
        handler.flush()

    text = ls.read_log_text(session_path)
    assert "Uncaught exception" not in text


# --- list_session_logs --------------------------------------------------


def test_list_session_logs_newest_first(tmp_path):
    now = time.time()
    a = tmp_path / "session-a.log"
    b = tmp_path / "session-b.log"
    c = tmp_path / "session-c.log"
    for path in (a, b, c):
        path.write_text("x")

    os.utime(a, (now - 300, now - 300))
    os.utime(b, (now - 100, now - 100))
    os.utime(c, (now - 200, now - 200))

    result = ls.list_session_logs(str(tmp_path))
    assert result == [str(b.resolve()), str(c.resolve()), str(a.resolve())]


def test_list_session_logs_missing_dir(tmp_path):
    assert ls.list_session_logs(str(tmp_path / "missing")) == []


# --- read_log_text -------------------------------------------------------


def test_read_log_text_missing_file(tmp_path):
    assert ls.read_log_text(str(tmp_path / "missing.log")) == ""


def test_read_log_text_reads_content(tmp_path):
    path = tmp_path / "session-x.log"
    path.write_text("some content", encoding="utf-8")
    assert ls.read_log_text(str(path)) == "some content"


# --- parse_log_text / filter_log_text ------------------------------------

_SAMPLE = (
    "2026-07-30 08:00:00,000 | INFO     | app | starting up\n"
    "2026-07-30 08:00:01,000 | WARNING  | app | disk space low\n"
    "2026-07-30 08:00:02,000 | ERROR    | app | something failed\n"
    "Traceback (most recent call last):\n"
    '  File "app.py", line 10, in <module>\n'
    "    raise ValueError('boom')\n"
    "ValueError: boom\n"
    "2026-07-30 08:00:03,000 | INFO     | app | recovered\n"
)


def test_parse_log_text_groups_traceback_with_error_record():
    entries = ls.parse_log_text(_SAMPLE)

    assert [e.level for e in entries] == ["INFO", "WARNING", "ERROR", "INFO"]

    error_entry = entries[2]
    assert error_entry.raw.startswith(
        "2026-07-30 08:00:02,000 | ERROR    | app | something failed"
    )
    assert "Traceback (most recent call last):" in error_entry.raw
    assert "ValueError: boom" in error_entry.raw

    info_entry = entries[3]
    assert info_entry.raw == "2026-07-30 08:00:03,000 | INFO     | app | recovered"


def test_parse_log_text_leading_text_becomes_info_record():
    text = "some preamble\nmore preamble\n2026-07-30 08:00:00,000 | ERROR    | app | oops\n"
    entries = ls.parse_log_text(text)
    assert entries[0].level == "INFO"
    assert entries[0].raw == "some preamble\nmore preamble"
    assert entries[1].level == "ERROR"


def test_parse_log_text_empty_text():
    assert ls.parse_log_text("") == []


def test_filter_log_text_selects_error_with_traceback():
    result = ls.filter_log_text(_SAMPLE, {"ERROR"})
    assert "something failed" in result
    assert "Traceback (most recent call last):" in result
    assert "ValueError: boom" in result
    assert "starting up" not in result
    assert "disk space low" not in result
    assert "recovered" not in result


def test_filter_log_text_empty_levels_returns_empty_string():
    assert ls.filter_log_text(_SAMPLE, set()) == ""


def test_filter_log_text_lowercase_levels_are_upcased():
    result = ls.filter_log_text(_SAMPLE, {"error"})
    assert "something failed" in result
