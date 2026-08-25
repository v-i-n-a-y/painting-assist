# Copyright 2026 Vinay Williams

"""Unit tests for the pure helpers in :mod:`painting_assist.updater`.

These cover version parsing/comparison and per-platform asset selection. They
touch no network and need no QApplication (the pure helpers are Qt-free).
"""

from __future__ import annotations

from painting_assist.updater import (
    CHANNEL_DEVELOPER,
    CHANNEL_STABLE,
    is_newer,
    is_prerelease_tag,
    is_update_candidate,
    parse_version,
    pick_asset,
    pick_latest_release,
)


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


# ---------------------------------------------------------------------------- #
# pick_latest_release (update channels)
# ---------------------------------------------------------------------------- #
def _release(tag, prerelease=False, draft=False):
    return {"tag_name": tag, "prerelease": prerelease, "draft": draft}


def test_pick_latest_release_stable_skips_prereleases():
    releases = [
        _release("v0.14.0-rc1", prerelease=True),
        _release("v0.13.0"),
    ]
    assert pick_latest_release(releases, CHANNEL_STABLE)["tag_name"] == "v0.13.0"


def test_pick_latest_release_developer_takes_prerelease():
    releases = [
        _release("v0.14.0-rc1", prerelease=True),
        _release("v0.13.0"),
    ]
    assert pick_latest_release(releases, CHANNEL_DEVELOPER)["tag_name"] == "v0.14.0-rc1"


def test_pick_latest_release_developer_takes_newer_stable():
    # No prereleases out: the developer channel follows the stable latest.
    releases = [_release("v0.14.0"), _release("v0.13.0")]
    assert pick_latest_release(releases, CHANNEL_DEVELOPER)["tag_name"] == "v0.14.0"


def test_pick_latest_release_skips_drafts_on_both_channels():
    releases = [
        _release("v0.15.0", draft=True),
        _release("v0.14.0-rc1", prerelease=True),
        _release("v0.13.0"),
    ]
    assert pick_latest_release(releases, CHANNEL_STABLE)["tag_name"] == "v0.13.0"
    assert pick_latest_release(releases, CHANNEL_DEVELOPER)["tag_name"] == "v0.14.0-rc1"


def test_pick_latest_release_empty_or_all_draft():
    assert pick_latest_release([], CHANNEL_STABLE) is None
    assert pick_latest_release([], CHANNEL_DEVELOPER) is None
    only_draft = [_release("v0.14.0", draft=True)]
    assert pick_latest_release(only_draft, CHANNEL_DEVELOPER) is None


def test_pick_latest_release_honours_newest_first_order():
    # The API lists newest first; the picker must not sort or look further.
    releases = [_release("v0.13.0"), _release("v0.12.0")]
    assert pick_latest_release(releases, CHANNEL_STABLE)["tag_name"] == "v0.13.0"


# ---------------------------------------------------------------------------- #
# is_update_candidate
# ---------------------------------------------------------------------------- #
def test_candidate_stable_strictly_newer_only():
    assert is_update_candidate("v0.14.0", "0.13.0", CHANNEL_STABLE)
    assert not is_update_candidate("v0.13.0", "0.13.0", CHANNEL_STABLE)
    assert not is_update_candidate("v0.12.0", "0.13.0", CHANNEL_STABLE)


def test_candidate_developer_offers_rc_to_final():
    # Same version, different tag: the final supersedes the rc.
    assert is_update_candidate("v0.14.0", "0.14.0-rc1", CHANNEL_DEVELOPER)
    # ...but the stable channel never offers a same-version re-tag.
    assert not is_update_candidate("v0.14.0", "0.14.0-rc1", CHANNEL_STABLE)


def test_candidate_developer_newer_and_same_tag():
    assert is_update_candidate("v0.15.0-rc1", "0.13.0", CHANNEL_DEVELOPER)
    assert not is_update_candidate("v0.14.0-rc1", "0.14.0-rc1", CHANNEL_DEVELOPER)
    assert not is_update_candidate("v0.12.0", "0.13.0", CHANNEL_DEVELOPER)


def test_candidate_empty_tag_never():
    assert not is_update_candidate("", "0.13.0", CHANNEL_STABLE)
    assert not is_update_candidate("", "0.13.0", CHANNEL_DEVELOPER)


# ---------------------------------------------------------------------------- #
# is_prerelease_tag
# ---------------------------------------------------------------------------- #
def test_prerelease_tag_recognises_suffixes():
    for tag in ("v0.14.0-rc1", "0.14.0-dev2", "v0.14.0-alpha", "0.14.0-beta.1"):
        assert is_prerelease_tag(tag), tag
    for tag in ("v0.14.0", "0.13.0", ""):
        assert not is_prerelease_tag(tag), tag
