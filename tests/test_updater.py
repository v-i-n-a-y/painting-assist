# Copyright 2026 Vinay Williams

"""Unit tests for the pure helpers in :mod:`painting_assist.updater`.

These cover version parsing/comparison and per-platform asset selection. They
touch no network and need no QApplication (the pure helpers are Qt-free).
"""

from __future__ import annotations

from painting_assist.updater import is_newer, parse_version, pick_asset


# ---------------------------------------------------------------------------- #
# parse_version
# ---------------------------------------------------------------------------- #
def test_parse_version_strips_leading_v():
    assert parse_version("v0.3.0") == (0, 3, 0)
    assert parse_version("V1.2.3") == (1, 2, 3)


def test_parse_version_without_prefix():
    assert parse_version("0.3.0") == (0, 3, 0)


def test_parse_version_missing_parts():
    assert parse_version("1.2") == (1, 2)
    assert parse_version("1") == (1,)


def test_parse_version_ignores_non_numeric_suffix():
    assert parse_version("1.2.0-rc1") == (1, 2, 0)
    assert parse_version("2.0.0beta") == (2, 0, 0)
    assert parse_version("1.2.3b") == (1, 2, 3)


def test_parse_version_leading_digits_within_component():
    assert parse_version("2beta.1") == (2, 1)


def test_parse_version_empty_or_junk():
    assert parse_version("") == ()
    assert parse_version("   ") == ()
    assert parse_version("vabc") == (0,)


# ---------------------------------------------------------------------------- #
# is_newer
# ---------------------------------------------------------------------------- #
def test_is_newer_true_when_remote_greater():
    assert is_newer("0.4.0", "0.3.0") is True
    assert is_newer("v1.0.0", "0.9.9") is True


def test_is_newer_false_when_equal():
    assert is_newer("0.3.0", "0.3.0") is False
    assert is_newer("v0.3.0", "0.3.0") is False


def test_is_newer_false_when_remote_older():
    assert is_newer("0.2.9", "0.3.0") is False


def test_is_newer_handles_differing_lengths():
    assert is_newer("0.3.1", "0.3") is True
    assert is_newer("0.3", "0.3.1") is False
    assert is_newer("0.3.0", "0.3") is False


# ---------------------------------------------------------------------------- #
# pick_asset
# ---------------------------------------------------------------------------- #
def _assets():
    names = [
        "PaintingAssist-0.4.0-macos-arm64.dmg",
        "PaintingAssist-0.4.0-macos-x86_64.dmg",
        "PaintingAssist-0.4.0-windows-x64-setup.exe",
        "PaintingAssist-0.4.0-linux-x86_64.AppImage",
        "PaintingAssist-0.4.0-linux-x86_64.tar.gz",
    ]
    return [{"name": n, "browser_download_url": "https://example/" + n} for n in names]


def test_pick_asset_macos_arm64():
    a = pick_asset(_assets(), "darwin", "arm64")
    assert a is not None and a["name"].endswith("macos-arm64.dmg")


def test_pick_asset_macos_x86_64():
    a = pick_asset(_assets(), "darwin", "x86_64")
    assert a is not None and a["name"].endswith("macos-x86_64.dmg")


def test_pick_asset_windows():
    a = pick_asset(_assets(), "win32", "AMD64")
    assert a is not None and a["name"].endswith("windows-x64-setup.exe")


def test_pick_asset_linux():
    a = pick_asset(_assets(), "linux", "x86_64")
    assert a is not None and a["name"].endswith("linux-x86_64.AppImage")


def test_pick_asset_linux2_platform_string():
    # sys.platform can be "linux2" on older interpreters; startswith handles it.
    a = pick_asset(_assets(), "linux2", "x86_64")
    assert a is not None and a["name"].endswith("linux-x86_64.AppImage")


def test_pick_asset_unknown_platform_returns_none():
    assert pick_asset(_assets(), "sunos", "sparc") is None


def test_pick_asset_macos_unknown_arch_returns_none():
    assert pick_asset(_assets(), "darwin", "ppc") is None


def test_pick_asset_missing_asset_returns_none():
    # A macOS-only release offers nothing for a Windows client.
    only_mac = [
        {
            "name": "PaintingAssist-0.4.0-macos-arm64.dmg",
            "browser_download_url": "https://example/x",
        }
    ]
    assert pick_asset(only_mac, "win32", "AMD64") is None


def test_pick_asset_normalises_aarch64():
    a = pick_asset(_assets(), "darwin", "aarch64")
    assert a is not None and a["name"].endswith("macos-arm64.dmg")


def test_pick_asset_empty_asset_list():
    assert pick_asset([], "darwin", "arm64") is None
