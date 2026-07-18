from __future__ import annotations

"""Unit tests for the recent-files list helpers in main_window.

These are pure functions (``update_recent`` is filesystem-free; ``prune_recent``
only calls ``os.path.exists``) so they run headless without a QApplication.
"""

import os

from painting_assist.main_window import _MAX_RECENT, prune_recent, update_recent


def test_update_recent_puts_new_path_first():
    assert update_recent([], "/a.png") == [os.path.abspath("/a.png")]


def test_update_recent_moves_existing_to_front_without_duplicating():
    start = ["/a.png", "/b.png", "/c.png"]
    out = update_recent(start, "/b.png")
    assert out[0] == os.path.abspath("/b.png")
    # /b.png appears exactly once.
    assert out.count(os.path.abspath("/b.png")) == 1
    assert set(map(os.path.abspath, out)) == set(map(os.path.abspath, start))


def test_update_recent_dedupes_by_absolute_path():
    # A relative and absolute reference to the same file collapse to one entry.
    cwd = os.getcwd()
    rel = "somefile.png"
    absolute = os.path.join(cwd, "somefile.png")
    out = update_recent([absolute], rel)
    assert out == [os.path.abspath(rel)]


def test_update_recent_caps_at_limit():
    start = ["/f{}.png".format(i) for i in range(_MAX_RECENT)]
    out = update_recent(start, "/new.png")
    assert len(out) == _MAX_RECENT
    assert out[0] == os.path.abspath("/new.png")
    # The oldest entry fell off the end.
    assert os.path.abspath("/f{}.png".format(_MAX_RECENT - 1)) not in out


def test_update_recent_honours_custom_limit():
    out = update_recent(["/a.png", "/b.png"], "/c.png", limit=2)
    assert out == [os.path.abspath("/c.png"), os.path.abspath("/a.png")]


def test_prune_recent_drops_missing_files(tmp_path):
    present = tmp_path / "here.png"
    present.write_bytes(b"x")
    missing = str(tmp_path / "gone.png")
    out = prune_recent([str(present), missing])
    assert out == [str(present)]


def test_prune_recent_dedupes_preserving_order(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    out = prune_recent([str(a), str(b), str(a)])
    assert out == [str(a), str(b)]


def test_prune_recent_empty():
    assert prune_recent([]) == []
