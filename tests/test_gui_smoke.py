# Copyright 2026 Vinay Williams

"""Headless-ish GUI smoke test: construct the window, drive the crop-edit flow.

Run with the project's framework Python:  uv run python tests/test_gui_smoke.py
Exits 0 and prints PASS on success. Uses a QTimer to quit the event loop so it
never blocks. Skips cleanly (prints SKIP) if no Qt platform can be initialised.
"""

import sys

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
    assert crop_ed is not None and type(crop_ed).__name__ == "CropEditor"
    assert blur_ed is not None and type(blur_ed).__name__ == "BlurEditor"

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
