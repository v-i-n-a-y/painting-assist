# Copyright 2026 Vinay Williams

"""Tests for the thread-safe, rate-limited error notifier in :mod:`painting_assist.app`."""

from __future__ import annotations

import pytest

from PySide6.QtWidgets import QApplication

from painting_assist import app as pa_app


@pytest.fixture(autouse=True)
def _restore_notifier_state():
    """Snapshot and restore the module-level dialog rate-limiting state."""
    prior_open = pa_app._error_dialog_open
    prior_message = pa_app._last_error_message
    prior_time = pa_app._last_error_time

    yield

    pa_app._error_dialog_open = prior_open
    pa_app._last_error_message = prior_message
    pa_app._last_error_time = prior_time


@pytest.fixture
def qapp():
    """A QApplication instance, reusing one if pytest-qt or another test made it."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_show_error_dialog_dedupes_identical_message(monkeypatch, qapp):
    calls = []
    monkeypatch.setattr(pa_app, "_run_error_dialog", calls.append)
    pa_app._error_dialog_open = False
    pa_app._last_error_message = None
    pa_app._last_error_time = 0.0

    pa_app._show_error_dialog("RuntimeError: boom")
    pa_app._show_error_dialog("RuntimeError: boom")

    assert calls == ["RuntimeError: boom"]


def test_show_error_dialog_allows_different_message(monkeypatch, qapp):
    calls = []
    monkeypatch.setattr(pa_app, "_run_error_dialog", calls.append)
    pa_app._error_dialog_open = False
    pa_app._last_error_message = None
    pa_app._last_error_time = 0.0

    pa_app._show_error_dialog("RuntimeError: boom")
    pa_app._show_error_dialog("ValueError: other")

    assert calls == ["RuntimeError: boom", "ValueError: other"]


def test_show_error_dialog_suppressed_while_dialog_open(monkeypatch, qapp):
    calls = []

    def _fake_run(msg):
        # Simulate a dialog that is still open (e.g. mid-exec) when a second
        # notification arrives before this one returns.
        assert pa_app._error_dialog_open is True
        calls.append(msg)
        pa_app._show_error_dialog(msg)  # re-entrant call while "open"

    monkeypatch.setattr(pa_app, "_run_error_dialog", _fake_run)
    pa_app._error_dialog_open = False
    pa_app._last_error_message = None
    pa_app._last_error_time = 0.0

    pa_app._show_error_dialog("RuntimeError: boom")

    assert calls == ["RuntimeError: boom"]
