# Copyright 2026 Vinay Williams

"""Worker-thread behaviour of :mod:`painting_assist.updater` and the image loader.

Runs the QRunnables synchronously on the test thread with the network stubbed,
so the tests need a QCoreApplication (for signals) but no display or network.
"""

from __future__ import annotations

import io
import os
import threading

import pytest
from PySide6.QtCore import QCoreApplication

from painting_assist import updater
from painting_assist.image_model import _LoadSignals, _LoadTask
from painting_assist.updater import (
    UpdateChecker,
    _CheckSignals,
    _CheckTask,
    _DownloadSignals,
    _DownloadTask,
)


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


class _FakeResponse(io.BytesIO):
    """Minimal stand-in for a urlopen response (context manager + headers)."""

    def __init__(self, body: bytes, length: int | None = None) -> None:
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body) if length is None else length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _stub_urlopen(monkeypatch, body: bytes):
    monkeypatch.setattr(
        updater.urllib.request, "urlopen", lambda req, timeout=0: _FakeResponse(body)
    )


def test_check_task_malformed_assets_resolves_with_error(qapp, monkeypatch):
    # "assets" is a string rather than a list: parsing must still resolve the
    # check (via ``done``) instead of raising on the pool thread.
    body = b'[{"tag_name": "v9.9.9", "draft": false, "prerelease": false, "assets": "oops"}]'
    _stub_urlopen(monkeypatch, body)
    got = []
    signals = _CheckSignals()
    signals.done.connect(lambda *a: got.append(a))
    _CheckTask("0.1.0", updater.CHANNEL_STABLE, signals).run()
    assert len(got) == 1
    tag, asset, error = got[0]
    assert tag == "v9.9.9" and asset is None and error is None


def test_check_task_non_dict_asset_entry_resolves(qapp, monkeypatch):
    body = (
        b'[{"tag_name": "v9.9.9", "draft": false, "prerelease": false, "assets": [42]}]'
    )
    _stub_urlopen(monkeypatch, body)
    got = []
    signals = _CheckSignals()
    signals.done.connect(lambda *a: got.append(a))
    _CheckTask("0.1.0", updater.CHANNEL_STABLE, signals).run()
    assert len(got) == 1
    assert got[0][0] == "v9.9.9" and got[0][2] is None


def test_download_task_cancel_removes_temp_dir(qapp, monkeypatch):
    _stub_urlopen(monkeypatch, b"x" * (3 * _DownloadTask._CHUNK))
    cancel = threading.Event()
    cancel.set()
    got = []
    signals = _DownloadSignals()
    signals.done.connect(lambda *a: got.append(a))
    made = []
    real_mkdtemp = updater.tempfile.mkdtemp

    def mkdtemp(**kw):
        d = real_mkdtemp(**kw)
        made.append(d)
        return d

    monkeypatch.setattr(updater.tempfile, "mkdtemp", mkdtemp)
    _DownloadTask(
        {"browser_download_url": "http://x/y", "name": "inst"}, signals, cancel
    ).run()
    assert got == [(None, "Download cancelled.")]
    assert made and not os.path.exists(made[0])


def test_download_task_failure_removes_temp_dir(qapp, monkeypatch):
    def boom(req, timeout=0):
        raise OSError("no network")

    monkeypatch.setattr(updater.urllib.request, "urlopen", boom)
    made = []
    real_mkdtemp = updater.tempfile.mkdtemp

    def mkdtemp(**kw):
        d = real_mkdtemp(**kw)
        made.append(d)
        return d

    monkeypatch.setattr(updater.tempfile, "mkdtemp", mkdtemp)
    got = []
    signals = _DownloadSignals()
    signals.done.connect(lambda *a: got.append(a))
    _DownloadTask({"browser_download_url": "http://x/y"}, signals).run()
    assert got[0][0] is None and "no network" in got[0][1]
    assert made and not os.path.exists(made[0])


def test_download_task_success_keeps_file(qapp, monkeypatch):
    _stub_urlopen(monkeypatch, b"payload")
    got = []
    pct = []
    signals = _DownloadSignals()
    signals.done.connect(lambda *a: got.append(a))
    signals.progress.connect(pct.append)
    _DownloadTask({"browser_download_url": "http://x/y", "name": "inst"}, signals).run()
    path, error = got[0]
    assert error is None and os.path.exists(path)
    assert pct[-1] == 100
    updater._remove_tree(os.path.dirname(path))


def test_checker_ignores_second_download_while_one_in_flight(qapp, monkeypatch):
    started = []
    checker = UpdateChecker("0.1.0")
    monkeypatch.setattr(checker._pool, "start", lambda task: started.append(task))
    checker.downloadAndOpen({"browser_download_url": "http://x/a"})
    checker.downloadAndOpen({"browser_download_url": "http://x/b"})
    assert len(started) == 1
    checker._on_download_done(None, "failed")  # completes -> next click allowed
    checker.downloadAndOpen({"browser_download_url": "http://x/c"})
    assert len(started) == 2


def test_checker_shutdown_cancels_and_detaches(qapp, monkeypatch):
    checker = UpdateChecker("0.1.0")
    fired = []
    checker.checkFailed.connect(fired.append)
    checker.upToDate.connect(fired.append)
    checker.shutdown()
    assert checker._dl_cancel.is_set()
    # Results arriving after shutdown go nowhere (direct emit, no queue needed).
    checker._check_signals.done.emit("v0.0.1", None, None)
    checker._dl_signals.done.emit(None, "late")
    qapp.processEvents()
    assert fired == []


def test_emit_tolerates_deleted_holder(qapp):
    signals = _CheckSignals()
    from shiboken6 import delete

    delete(signals)
    updater._emit(signals.done, None, None, "x")  # must not raise


def test_load_task_emit_after_shutdown_is_dropped(qapp, tmp_path):
    from painting_assist.image_model import ImageModel

    model = ImageModel()
    seen = []
    model.load_failed.connect(lambda *a: seen.append(a))
    serial = model._load_serial + 1
    model.shutdown()
    _LoadTask(str(tmp_path / "missing.png"), serial, model._load_signals).run()
    qapp.processEvents()
    assert seen == []


def test_load_signals_survive_deleted_holder(qapp, tmp_path):
    from shiboken6 import delete

    signals = _LoadSignals()
    delete(signals)
    _LoadTask(str(tmp_path / "missing.png"), 1, signals).run()  # must not raise
