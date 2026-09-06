# Copyright 2026 Vinay Williams

"""Regression tests for the edge cases fixed in the September 2026 code review."""

from __future__ import annotations

import logging
import sys

import numpy as np

from painting_assist import logging_setup as ls
from painting_assist import palette_map
from painting_assist.colour_mixing import _nnls
from painting_assist.paints import paints_from_json
from painting_assist.print_layout import _axis_boundaries, tile_grid
from painting_assist.updater import CHANNEL_DEVELOPER, is_update_candidate


def test_paints_from_json_infinity_component_is_dropped():
    assert paints_from_json('[{"name": "a", "rgb": [Infinity, 0, 0]}]') == []


def test_palette_map_levels_one_is_clamped():
    img = np.full((2, 2, 3), 120, dtype=np.uint8)
    cands = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    out = palette_map.map_image(img, cands, levels=1)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_palette_map_binning_matches_int64_reference():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(7, 5, 3), dtype=np.uint8)
    cands = rng.integers(0, 256, size=(6, 3), dtype=np.uint8)
    levels = 17
    out = palette_map.map_image(img, cands, levels=levels)
    lut = palette_map._build_lut(cands, levels)
    idx = img.astype(np.int64) * (levels - 1) // 255
    ref = lut[idx[..., 0], idx[..., 1], idx[..., 2]]
    assert np.array_equal(out, ref)


def test_nnls_duplicate_columns_do_not_produce_nan():
    col = np.array([0.2, 0.5, 0.7])
    matrix = np.stack([col, col, np.array([0.9, 0.1, 0.3])], axis=1)
    w = _nnls(matrix, np.array([0.5, 0.4, 0.6]))
    assert np.all(np.isfinite(w)) and np.all(w >= 0)


def test_axis_boundaries_never_repeat():
    # printable ~99.7 mm on a 100 mm canvas: the interior edge rounds to 100.
    b = _axis_boundaries(100, 100.0, 99.7)
    assert b == [0, 100]
    tiles = tile_grid(100, 100, 100.0, 100.0, 109.7, 109.7, 5.0)
    assert all(t.src_w > 0 and t.src_h > 0 for t in tiles)


def test_developer_channel_never_offers_same_version_prerelease():
    assert not is_update_candidate("v0.14.0-rc1", "0.14.0", CHANNEL_DEVELOPER)
    assert is_update_candidate("v0.14.0", "0.14.0-rc1", CHANNEL_DEVELOPER)
    assert is_update_candidate("v0.14.0-rc2", "0.14.0-rc1", CHANNEL_DEVELOPER)


def test_reconfigure_does_not_chain_excepthook_to_itself(tmp_path):
    original = sys.excepthook
    try:
        ls.configure(str(tmp_path / "a"))
        first = sys.excepthook
        ls.configure(str(tmp_path / "b"))
        assert sys.excepthook is not first
        assert ls._prior_excepthook is original
    finally:
        sys.excepthook = original
        for h in list(ls._our_handlers):
            logging.getLogger().removeHandler(h)
            h.close()
        ls._our_handlers.clear()


def test_reconfigure_closes_previous_file_handler(tmp_path):
    original = sys.excepthook
    try:
        ls.configure(str(tmp_path / "a"))
        handler = next(
            h for h in ls._our_handlers if isinstance(h, logging.FileHandler)
        )
        ls.configure(str(tmp_path / "b"))
        assert handler.stream is None  # closed
    finally:
        sys.excepthook = original
        for h in list(ls._our_handlers):
            logging.getLogger().removeHandler(h)
            h.close()
        ls._our_handlers.clear()


def test_pipeline_trim_does_not_evict_downstream_on_next_frame():
    """After a trim, an unchanged frame recomputes only the evicted prefix."""
    from painting_assist.pipeline import MAX_CACHED_STAGES
    from tests.test_cache import AddMulControl, _pipe, _rand_img

    class Counting(AddMulControl):
        calls = 0

        def process(self, img):
            Counting.calls += 1
            return super().process(img)

    n = MAX_CACHED_STAGES + 3
    controls = [Counting(f"s{i}", 2 * i + 1) for i in range(n)]
    pipe = _pipe(controls)
    img = _rand_img()
    first = pipe.process(img, token="T")
    assert Counting.calls == n
    Counting.calls = 0
    second = pipe.process(img, token="T")
    # Only the evicted upstream slots run again; the retained tail must hit.
    assert Counting.calls == n - MAX_CACHED_STAGES
    assert np.array_equal(first, second)
