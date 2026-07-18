# Copyright 2026 Vinay Williams

"""Update checking and installer download for Painting Assist.

This module owns the "is there a newer release?" question and the mechanics of
fetching the right installer, keeping :class:`~painting_assist.main_window.MainWindow`
thin. It mirrors the threading discipline of
:mod:`~painting_assist.render_controller`: a :class:`QThreadPool` runs the
network work off the GUI thread via a :class:`QRunnable`, and results come back
through a signal holder connected with :data:`Qt.QueuedConnection` so they are
marshalled onto the GUI thread regardless of which worker thread emits them.

Layers, from pure to side-effecting:

1. **Pure helpers** — :func:`parse_version`, :func:`is_newer`, :func:`pick_asset`.
   No network, no Qt; fully unit-testable.
2. **:class:`UpdateChecker`** — asks GitHub for the latest release off-thread and
   emits :attr:`~UpdateChecker.updateAvailable` / :attr:`~UpdateChecker.upToDate`
   / :attr:`~UpdateChecker.checkFailed`. Network failures are turned into
   ``checkFailed`` — they never raise into the GUI.
3. **Download + open** — :meth:`UpdateChecker.downloadAndOpen` fetches an asset
   off-thread (with :attr:`~UpdateChecker.downloadProgress`), then
   :attr:`~UpdateChecker.downloadReady` fires; :func:`open_installer` hands the
   file to the OS.

Scope note: full silent self-replacement (swapping the running binary and
relaunching) is deliberately **out of scope**. The deliverable is to download
the correct installer and open it, so the user completes the install through the
platform's normal installer flow.
"""

from __future__ import annotations

import logging
import os
import platform as _platform_mod
import subprocess
import sys
import tempfile
import urllib.request
from typing import Optional

from PySide6.QtCore import (
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    Signal,
)

from painting_assist import __version__

_log = logging.getLogger(__name__)

RELEASES_API_URL = (
    "https://api.github.com/repos/v-i-n-a-y/painting-assist/releases/latest"
)
_USER_AGENT = "PaintingAssist-Updater"
_NET_TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------- #
# Pure helpers (no network, no Qt)
# ---------------------------------------------------------------------------- #
def parse_version(version: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of ints for ordered comparison.

    Tolerant of a leading ``v``/``V``, missing parts, and non-numeric suffixes:
    ``"v0.3.0"`` and ``"0.3.0"`` both yield ``(0, 3, 0)``; ``"1.2"`` yields
    ``(1, 2)``; a pre-release suffix like ``"1.2.0-rc1"`` yields ``(1, 2, 0)``
    (the ``rc1`` component contributes nothing rather than raising). Leading
    digits within a component are honoured, so ``"2beta"`` reads as ``2``. An
    unparseable or empty string yields ``()``, which sorts below any real
    version.
    """
    if not version:
        return ()
    s = version.strip()
    if s[:1] in ("v", "V"):
        s = s[1:]
    if not s:
        return ()
    parts: list[int] = []
    for chunk in s.split("."):
        parts.append(_leading_int(chunk))
    # Trim trailing zero-only components that came purely from junk so that
    # comparisons stay intuitive is unnecessary here: (0,3,0) vs (0,3) compare
    # correctly under tuple ordering, and callers only ask is_newer.
    return tuple(parts)


def _leading_int(chunk: str) -> int:
    """Return the integer formed by the leading digits of ``chunk`` (0 if none)."""
    digits = ""
    for ch in chunk.strip():
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else 0


def is_newer(remote: str, local: str) -> bool:
    """Return ``True`` when ``remote`` is a strictly newer version than ``local``.

    Comparison is by :func:`parse_version` tuples, zero-padded to equal length
    so that missing trailing components read as zero: ``"0.3.0"`` and ``"0.3"``
    are equal (neither newer), while ``"0.3.1"`` is newer than ``"0.3"``.
    """
    r = parse_version(remote)
    loc = parse_version(local)
    n = max(len(r), len(loc))
    r += (0,) * (n - len(r))
    loc += (0,) * (n - len(loc))
    return r > loc


def pick_asset(
    assets: list[dict],
    platform: str,
    machine: str,
) -> Optional[dict]:
    """Choose the installer asset matching the current platform and architecture.

    ``assets`` is the GitHub releases ``assets`` list; each item is expected to
    have ``"name"`` and ``"browser_download_url"`` keys. ``platform`` is a
    :data:`sys.platform` value (``"darwin"``, ``"win32"``, ``"linux"`` …) and
    ``machine`` a :func:`platform.machine` value (``"arm64"``, ``"x86_64"``,
    ``"AMD64"`` …). Returns the matching asset dict, or ``None`` when no asset
    fits (unknown platform, or the release lacks the expected file).

    Mapping:

    - ``darwin`` + arm64  -> ``*-macos-arm64.dmg``
    - ``darwin`` + x86_64 -> ``*-macos-x86_64.dmg``
    - ``win32``           -> ``*-windows-x64-setup.exe``
    - ``linux``           -> ``*-linux-x86_64.AppImage``
    """
    suffix = _asset_suffix(platform, machine)
    if suffix is None:
        return None
    for asset in assets:
        name = (asset.get("name") or "").lower()
        if name.endswith(suffix):
            return asset
    return None


def _asset_suffix(platform: str, machine: str) -> Optional[str]:
    """Return the lower-cased filename suffix expected for the given platform."""
    mach = _normalise_machine(machine)
    if platform == "darwin":
        if mach == "arm64":
            return "-macos-arm64.dmg"
        if mach == "x86_64":
            return "-macos-x86_64.dmg"
        return None
    if platform == "win32":
        return "-windows-x64-setup.exe"
    if platform.startswith("linux"):
        return "-linux-x86_64.appimage"
    return None


def _normalise_machine(machine: str) -> str:
    """Map assorted architecture spellings onto ``"arm64"`` / ``"x86_64"``."""
    m = (machine or "").lower()
    if m in ("arm64", "aarch64"):
        return "arm64"
    if m in ("x86_64", "amd64", "x64"):
        return "x86_64"
    return m


def running_frozen() -> bool:
    """Return ``True`` when running from a PyInstaller (or similar) bundle.

    The UI only offers to install an update when frozen — a source checkout has
    no self-contained binary to replace — but :meth:`UpdateChecker.check` works
    either way so a developer running from source can still see that a release
    exists.
    """
    return bool(getattr(sys, "frozen", False))


def open_installer(path: str) -> None:
    """Hand a downloaded installer to the OS for the user to complete.

    Per platform:

    - **macOS** — ``open`` the ``.dmg`` so it mounts and Finder shows it.
    - **Windows** — :func:`os.startfile` launches the setup executable.
    - **Linux** — mark the AppImage executable and reveal its folder via
      ``xdg-open`` (AppImages are run by the user, not auto-installed).

    Silent self-replacement is out of scope; this only opens/reveals the file.
    Raises :class:`OSError` (or :class:`FileNotFoundError` for a missing helper)
    if the platform command cannot be run — callers should guard accordingly.
    """
    if sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    elif sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606 - Windows-only
    else:
        # Linux: make the AppImage runnable and reveal the containing folder.
        try:
            mode = os.stat(path).st_mode
            os.chmod(path, mode | 0o111)
        except OSError:
            _log.warning("Could not chmod +x %s", path, exc_info=True)
        folder = os.path.dirname(os.path.abspath(path))
        subprocess.run(["xdg-open", folder], check=False)


# ---------------------------------------------------------------------------- #
# Off-thread task plumbing (mirrors render_controller's discipline)
# ---------------------------------------------------------------------------- #
class _CheckSignals(QObject):
    """Signal holder for a :class:`_CheckTask` (a QRunnable cannot have signals).

    ``done`` carries ``(version|None, asset|None, error|None)``: on success the
    latest tag and (possibly ``None``) matching asset; on failure a human
    message in ``error``. Connected via :data:`Qt.QueuedConnection` so the
    result lands on the GUI thread.
    """

    # (version|None, asset|None, error|None)
    done = Signal(object, object, object)


class _CheckTask(QRunnable):
    """Fetch the latest release JSON off-thread and emit the parsed result.

    Any network/parse error is captured and emitted through ``error`` rather than
    raised, so a failed check never crashes the GUI.
    """

    def __init__(self, local_version: str, signals: _CheckSignals) -> None:
        super().__init__()
        self._local_version = local_version
        self._signals = signals

    def run(self) -> None:  # noqa: D401 - QRunnable entry point
        """Query GitHub, pick the asset, and emit ``done`` (never raises out)."""
        import json

        try:
            req = urllib.request.Request(
                RELEASES_API_URL, headers={"User-Agent": _USER_AGENT}
            )
            with urllib.request.urlopen(req, timeout=_NET_TIMEOUT) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - all failures become checkFailed
            _log.warning("Update check failed", exc_info=True)
            self._signals.done.emit(None, None, str(exc))
            return

        tag = payload.get("tag_name") or payload.get("name") or ""
        assets = payload.get("assets") or []
        asset = pick_asset(assets, sys.platform, _platform_mod.machine())
        self._signals.done.emit(tag, asset, None)


class _DownloadSignals(QObject):
    """Signal holder for a :class:`_DownloadTask`.

    ``progress`` carries a 0..100 int; ``done`` carries ``(path|None, error|None)``.
    Both connected via :data:`Qt.QueuedConnection`.
    """

    progress = Signal(int)
    done = Signal(object, object)  # (path|None, error|None)


class _DownloadTask(QRunnable):
    """Download an asset to a temp directory off-thread, reporting progress."""

    _CHUNK = 64 * 1024

    def __init__(self, asset: dict, signals: _DownloadSignals) -> None:
        super().__init__()
        self._asset = asset
        self._signals = signals

    def run(self) -> None:  # noqa: D401 - QRunnable entry point
        """Stream the asset to a temp file, emitting progress then ``done``."""
        url = self._asset.get("browser_download_url")
        name = self._asset.get("name") or "PaintingAssist-installer"
        if not url:
            self._signals.done.emit(None, "Asset has no download URL")
            return
        try:
            dest_dir = tempfile.mkdtemp(prefix="painting-assist-update-")
            dest = os.path.join(dest_dir, name)
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=_NET_TIMEOUT) as resp:
                total = _content_length(resp)
                read = 0
                last_pct = -1
                with open(dest, "wb") as fh:
                    while True:
                        chunk = resp.read(self._CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        read += len(chunk)
                        if total > 0:
                            pct = int(read * 100 / total)
                            if pct != last_pct:
                                last_pct = pct
                                self._signals.progress.emit(pct)
            if total <= 0:
                # Unknown length: report completion so the UI can settle at 100%.
                self._signals.progress.emit(100)
        except Exception as exc:  # noqa: BLE001 - failures become an error emit
            _log.warning("Update download failed", exc_info=True)
            self._signals.done.emit(None, str(exc))
            return
        self._signals.done.emit(dest, None)


def _content_length(resp) -> int:
    """Return the response ``Content-Length`` as an int, or 0 if absent/invalid."""
    try:
        return int(resp.headers.get("Content-Length", 0))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------- #
# Public controller (GUI thread)
# ---------------------------------------------------------------------------- #
class UpdateChecker(QObject):
    """Check GitHub for a newer release and, on request, download its installer.

    All heavy work runs on :class:`QThreadPool`; every signal below is emitted
    on the GUI thread (queued from the worker). A single checker instance can be
    reused: keep it alive for the lifetime of the window so its queued signals
    are not delivered into a destroyed object.
    """

    # (version: str, asset: dict) — a strictly newer release with a matching asset.
    updateAvailable = Signal(str, dict)
    # (version: str) — the latest release is not newer than what is running.
    upToDate = Signal(str)
    # (message: str) — the check could not complete (network, parse, etc.).
    checkFailed = Signal(str)

    # (percent: int) — 0..100 download progress.
    downloadProgress = Signal(int)
    # (path: str) — the installer finished downloading to this local path.
    downloadReady = Signal(str)

    def __init__(
        self,
        local_version: str = __version__,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._local_version = local_version
        self._pool = QThreadPool.globalInstance()

        self._check_signals = _CheckSignals()
        self._check_signals.done.connect(self._on_check_done, Qt.QueuedConnection)

        self._dl_signals = _DownloadSignals()
        self._dl_signals.progress.connect(self.downloadProgress, Qt.QueuedConnection)
        self._dl_signals.done.connect(self._on_download_done, Qt.QueuedConnection)

    # ------------------------------------------------------------------ #
    # Public API (GUI thread)
    # ------------------------------------------------------------------ #
    def check(self) -> None:
        """Start an off-thread check for a newer release.

        Emits exactly one of :attr:`updateAvailable`, :attr:`upToDate`, or
        :attr:`checkFailed` on the GUI thread once the network call resolves.
        Works whether or not the app is frozen.
        """
        self._pool.start(_CheckTask(self._local_version, self._check_signals))

    def downloadAndOpen(self, asset: dict) -> None:
        """Download ``asset`` off-thread, emitting :attr:`downloadProgress`.

        On success :attr:`downloadReady` fires with the local path; the caller
        then invokes :func:`open_installer` to hand it to the OS (kept as a
        separate step so the UI can prompt before launching). Failures surface
        through :attr:`checkFailed` rather than raising.
        """
        self._pool.start(_DownloadTask(asset, self._dl_signals))

    # ------------------------------------------------------------------ #
    # Internals (GUI thread via QueuedConnection)
    # ------------------------------------------------------------------ #
    def _on_check_done(self, version: object, asset: object, error: object) -> None:
        """Translate a finished check into the right public signal."""
        if error is not None:
            self.checkFailed.emit(str(error))
            return
        tag = str(version or "")
        if tag and is_newer(tag, self._local_version):
            if isinstance(asset, dict):
                self.updateAvailable.emit(tag, asset)
            else:
                # Newer release exists but no installer matches this platform.
                self.checkFailed.emit(
                    "A newer version ({}) is available, but no installer was "
                    "found for this platform.".format(tag)
                )
        else:
            self.upToDate.emit(tag or self._local_version)

    def _on_download_done(self, path: object, error: object) -> None:
        """Emit :attr:`downloadReady` on success, else :attr:`checkFailed`."""
        if error is not None:
            self.checkFailed.emit(str(error))
            return
        self.downloadReady.emit(str(path))
