# Copyright 2026 Vinay Williams

"""Headless-ish GUI smoke test: construct the window, drive the crop-edit flow.

Run with the project's framework Python:  uv run python tests/test_gui_smoke.py
Exits 0 and prints PASS on success. Uses a QTimer to quit the event loop so it
never blocks. Skips cleanly (prints SKIP) if no Qt platform can be initialised.
"""

import os
import sys
import time

import numpy as np


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QTimer
    except Exception as exc:  # pragma: no cover
        print("SKIP (PySide6 import failed): %s" % exc)
        return 0

    app = QApplication(sys.argv or ["squint-test"])

    from painting_assist.main_window import MainWindow

    w = MainWindow()
    w.show()

    # Editors were built by the panel from each control's create_editor().
    crop_ed = w._panel.editor("crop")
    blur_ed = w._panel.editor("blur")
    values_ed = w._panel.editor("values")
    assert crop_ed is not None and type(crop_ed).__name__ == "CropEditor"
    assert blur_ed is not None and type(blur_ed).__name__ == "BlurEditor"
    assert values_ed is not None and type(values_ed).__name__ == "ValuesEditor"

    # Canvas aspect-linking: once a crop is applied with the ratio locked, the
    # crop fixes the aspect ratio, so editing one canvas dimension scales the
    # other to keep the physical size proportional (for gridlines / measuring).
    # The image is a real file on disk: the project round-trip below re-loads
    # it by path, and a missing file would raise a modal warning dialog.
    import tempfile as _tempfile

    import cv2 as _cv2

    _img_dir = _tempfile.mkdtemp(prefix="pa_smoke_")
    _crop_img_path = os.path.join(_img_dir, "crop-test.png")
    _cv2.imwrite(
        _crop_img_path,
        _cv2.cvtColor(np.zeros((300, 400, 3), dtype=np.uint8), _cv2.COLOR_RGB2BGR),
    )
    w._model.set_image(np.zeros((300, 400, 3), dtype=np.uint8), _crop_img_path)
    crop = w._pipeline.control("crop")
    crop.set("rw", 0.5)  # a real crop -> has_crop() True
    crop.set("lock_ratio", True)
    crop.set("canvas_w", 40.0)
    crop.set("canvas_h", 30.0)
    crop_ed.refresh()  # sync spinboxes + remembered ratio (4:3)
    # Lock is only editable while adjusting a crop; a committed crop greys it out
    # so the canvas ratio cannot be knocked out of step with the crop.
    assert crop_ed.lock.isEnabled() is False
    crop_ed.width.setValue(60.0)  # user widens the canvas
    assert abs(crop_ed.height.value() - 45.0) < 1e-6  # height followed 4:3
    assert abs(float(crop.get("canvas_h")) - 45.0) < 1e-6
    # An applied crop keeps the dimensions linked regardless of the (now frozen)
    # lock flag — a freeform crop's baked aspect must be preserved too.
    crop.set("lock_ratio", False)
    crop_ed.refresh()
    crop_ed.width.setValue(30.0)
    assert abs(crop_ed.height.value() - 22.5) < 1e-6  # still 4:3

    # Freeform box drag drives the canvas Height from the box's pixel aspect.
    crop.set("lock_ratio", False)
    crop.set("canvas_w", 40.0)
    img_h, img_w = w._model.original().shape[:2]
    # A box half as wide and full height -> pixel aspect 0.5 * (img_w/img_h).
    w._on_crop_rect_changed(0.0, 0.0, 0.5, 1.0)
    expected_aspect = (0.5 * img_w) / (1.0 * img_h)
    assert abs(float(crop.get("canvas_h")) - 40.0 / expected_aspect) < 1e-6

    # Unit conversion: switching the crop unit re-expresses the canvas size in
    # the new unit (settings-gated), preserving the physical size. The canvas
    # is 40 x 60 cm here (the freeform sync above set the height).
    crop_ed.unit.setCurrentIndex(1)  # cm -> in
    assert abs(float(crop.get("canvas_w")) - 40.0 / 2.54) < 1e-9
    assert abs(float(crop.get("canvas_h")) - 60.0 / 2.54) < 1e-9
    crop_ed.unit.setCurrentIndex(0)  # in -> cm restores the original numbers
    assert abs(float(crop.get("canvas_w")) - 40.0) < 1e-9
    assert abs(float(crop.get("canvas_h")) - 60.0) < 1e-9
    # With conversion disabled the unit change is display-only.
    w._convert_unit = False
    crop_ed.unit.setCurrentIndex(1)  # cm -> in, values untouched
    assert abs(float(crop.get("canvas_w")) - 40.0) < 1e-9
    assert abs(float(crop.get("canvas_h")) - 60.0) < 1e-9
    crop_ed.unit.setCurrentIndex(0)  # back to cm
    w._convert_unit = True

    # Priming: the recommendation tracks the processed frame. Wait for the
    # first full render, then check the swatch readout and the in-place
    # recompute on technique change (the control is an identity stage, so no
    # render is needed for that).
    deadline = time.time() + 10.0
    while w._prime_image is None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert w._prime_image is not None
    prime_ed = w._panel.editor("prime")
    assert prime_ed is not None and prime_ed._result is not None
    assert prime_ed._result.hex.startswith("#")
    assert len(prime_ed._result.hex) == 7
    # Drive the technique combo through the real signal path (index 5 = neutral).
    prime_ed._param_widgets["technique"]._combo.setCurrentIndex(5)
    assert w._pipeline.control("prime").get("technique") == "neutral"
    assert prime_ed._result is not None
    r, g, b = prime_ed._result.rgb
    assert abs(r - g) <= 1 and abs(g - b) <= 1  # a neutral ground is a grey

    # The Values mono-colour picker: saving a colour flows back to My Paints and
    # a custom hex is honoured by the control. Start from a known custom colour.
    n_before = len(w._paints)
    w._pipeline.control("values").set("mono_hex", "#123456")
    values_ed.refresh()  # combo now shows a "Custom (#123456)" entry, selected
    values_ed.paintAdded.emit("Studio umber", (90, 60, 40))
    assert len(w._paints) == n_before + 1
    assert w._paints[-1] == ("Studio umber", (90, 60, 40))

    # Hiding a paint from the mono picker: the window filters it out of the list
    # pushed into the editor, but keeps it in the full inventory for mixing.
    w._mono_hidden = {"Studio umber"}
    visible = w._visible_mono_paints()
    assert ("Studio umber", (90, 60, 40)) not in visible
    assert ("Studio umber", (90, 60, 40)) in w._paints

    # Settings store: portable settings live in a versioned JSON file under the
    # app-data folder; saving a paint persisted it there, and it round-trips.
    from painting_assist.settings_store import SCHEMA_VERSION, SettingsStore

    assert w._settings_path.endswith("settings.json")
    assert w._store.data["schema_version"] == SCHEMA_VERSION
    w._persist_paints()  # writes paints + mono_hidden to the store on disk
    reread = SettingsStore(w._settings_path)
    reread.load()
    saved_names = {p["name"] for p in reread.data["paints"]}
    assert "Studio umber" in saved_names
    assert "Studio umber" in set(reread.data["mono_hidden"])
    # Preferences and session sections are always present and versioned.
    assert "theme" in reread.data["preferences"]
    assert "controls" in reread.data["session"]

    # Project round-trip: saving the current session to a .paproj writes a doc
    # whose controls and measure settings faithfully round-trip through the
    # project module, and applying that doc restores a since-mutated control.
    import tempfile

    from painting_assist import project as project_mod

    tmp_dir = tempfile.mkdtemp(prefix="pa_smoke_")
    tmp_path = os.path.join(tmp_dir, "roundtrip" + project_mod.PROJECT_EXTENSION)
    w._write_project(tmp_path)  # bypass the Save dialog
    assert os.path.exists(tmp_path)
    assert w._current_project_path == tmp_path

    with open(tmp_path, encoding="utf-8") as handle:
        doc = project_mod.from_json(handle.read())
    assert doc.controls, "saved project should carry a non-empty controls blob"
    assert doc.measure["unit"] == w._measure_unit

    # Mutate a control, then applying the saved doc must restore its value.
    w._pipeline.control("blur").set("radius", 17)
    assert w._pipeline.control("blur").get("radius") == 17
    saved_radius = doc.controls["blur"]["values"]["radius"]
    w._apply_project(doc, tmp_path)
    assert w._pipeline.control("blur").get("radius") == saved_radius
    assert w._current_project_path == tmp_path
    base = os.path.basename(tmp_path)[: -len(project_mod.PROJECT_EXTENSION)]
    assert base in w.windowTitle()

    # Each control lives in its own dock, split prep-left / colour-right by
    # default. Dragging a dock to the other side is native Qt and persists via
    # the window's saveState; here we assert the initial arrangement.
    from PySide6.QtCore import Qt

    for cid in ("crop", "flip", "grid", "blur", "quantize", "values"):
        dock = w._panel.dock(cid)
        assert dock is not None, cid
        assert dock.objectName() == "control_dock_%s" % cid
    for cid in ("crop", "flip", "grid"):
        assert w.dockWidgetArea(w._panel.dock(cid)) == Qt.LeftDockWidgetArea, cid
    for cid in ("white_balance", "tone", "blur", "quantize", "values", "temp_map"):
        assert w.dockWidgetArea(w._panel.dock(cid)) == Qt.RightDockWidgetArea, cid

    # Load a synthetic reference.
    img = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    w._model.set_image(img, "synthetic.png")

    # Crop-edit flow: begin -> overlay moves -> apply.
    crop_ed.editRequested.emit(True)
    assert w._crop_editing is True
    assert w._pipeline.control("crop").enabled is True
    w._view.cropRectChanged.emit(0.2, 0.1, 0.5, 0.6)
    c = w._pipeline.control("crop")
    assert abs(c.get("rx") - 0.2) < 1e-6 and abs(c.get("rw") - 0.5) < 1e-6
    crop_ed.editRequested.emit(False)
    assert w._crop_editing is False

    # The committed crop actually trims the image through the pipeline.
    out = w._pipeline.process(img)
    assert out.shape[0] < img.shape[0] and out.shape[1] < img.shape[1], out.shape

    # Blur stepped mode via the pipeline.
    b = w._pipeline.control("blur")
    b.set_enabled(True)
    b.set("mode", "stepped")
    b.set("radius", 80)
    b.set("stage_count", 5)
    b.set("spacing", "even")
    assert b.stage_levels() == [80, 60, 40, 20, 0], b.stage_levels()

    # Before/after compare wipe: the toolbar toggle drives the view's mode.
    w._compare_action.setChecked(True)
    assert w._view._compare_on is True
    w._compare_action.setChecked(False)
    assert w._view._compare_on is False

    # Export gridded reference to PDF: exercise the writer directly (the menu
    # action goes via a file dialog). Both the uncalibrated fit-to-page path and
    # the calibrated true-scale tiled path must produce a non-empty PDF.
    ref = w._model.original()
    pdf_uncal = os.path.join(tmp_dir, "uncal.pdf")
    w._write_gridded_pdf(pdf_uncal, ref, None)
    assert os.path.getsize(pdf_uncal) > 0
    pdf_cal = os.path.join(tmp_dir, "cal.pdf")
    w._write_gridded_pdf(pdf_cal, ref, (300.0, 225.0))
    assert os.path.getsize(pdf_cal) > 0

    # Limited palette simulation: editor present, a preset resolves and repaints,
    # the My Paints inventory is injected, and eyedropper sampling appends a tube.
    import json as _json

    palette_ed = w._panel.editor("limited_palette")
    assert (
        palette_ed is not None and type(palette_ed).__name__ == "LimitedPaletteEditor"
    )
    lp = w._pipeline.control("limited_palette")
    lp.set_enabled(True)
    lp.set("source", "preset")
    lp.set("preset", "zorn")
    assert lp.is_active() is True
    out_lp = lp.process(ref)
    assert out_lp.shape == ref.shape and out_lp.dtype == ref.dtype

    w._sync_palette_sim_paints(render=False)
    assert _json.loads(lp.get("paints_json"))  # inventory injected (non-empty)

    lp.set("source", "sampled")
    w._sampling_for_palette = True
    w._last_image = ref
    w._on_colour_sampled(0.5, 0.5)
    assert _json.loads(lp.get("samples_json"))  # a sampled colour was appended

    # Logging: settings dialog exposes a log-location override that resets to
    # default, and the Help ▸ View Logs viewer renders + level-filters a session.
    from painting_assist import logging_setup
    from painting_assist.widgets.settings_dialog import SettingsDialog
    from painting_assist.widgets.log_viewer import LogViewer

    dlg = SettingsDialog(log_dir="", default_log_dir="/tmp/pa-logs")
    assert dlg.values()["log_dir"] == ""  # empty override => default
    dlg._on_reset_log_dir()
    assert dlg.values()["log_dir"] == ""

    import tempfile as _tempfile
    import logging as _logging

    log_dir = os.path.join(_tempfile.mkdtemp(prefix="pa_logs_"), "logs")
    session = logging_setup.configure(log_dir)
    _logging.getLogger("painting_assist.smoke").warning("smoke warning line")
    _logging.getLogger("painting_assist.smoke").info("smoke info line")
    for _h in _logging.getLogger().handlers:
        _h.flush()
    viewer = LogViewer(log_dir, session)
    viewer._reload()
    assert "smoke warning line" in viewer._view.toPlainText()
    # Unchecking every level but WARNING drops the INFO line from the view.
    for name, box in viewer._level_checks.items():
        box.setChecked(name == "WARNING")
    viewer._reload()
    shown = viewer._view.toPlainText()
    assert "smoke warning line" in shown and "smoke info line" not in shown

    result = {"rc": 1}

    def finish():
        result["rc"] = 0
        app.quit()

    QTimer.singleShot(300, finish)
    app.exec()

    print("PASS" if result["rc"] == 0 else "FAIL")
    return result["rc"]


if __name__ == "__main__":
    raise SystemExit(main())
