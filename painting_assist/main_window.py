from __future__ import annotations

"""Top-level window: builds and wires everything, owns the toolbar.

The control dock is built generically from whatever the registry discovered (via
``ControlPanel``), and the responsiveness mechanics live entirely in
:class:`~painting_assist.render_controller.RenderController`. The one place with
a little control-specific glue is the **crop** tool: an interactive viewport
overlay genuinely needs view-level cooperation that a pure pixel filter does
not. That glue is opt-in and degrades gracefully if no ``crop`` control exists.
"""

import json
import os
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QToolBar,
)

from painting_assist.controls import registry
from painting_assist.image_model import ImageModel
from painting_assist.pipeline import ControlPipeline
from painting_assist.render_controller import RenderController
from painting_assist.widgets.control_panel import ControlPanel
from painting_assist.widgets.image_view import ImageView

APP_NAME = "Painting Assist"

_OPEN_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)"
_SAVE_FILTER = "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif);;All files (*)"

# Image extensions accepted by drag-and-drop, mirroring the Open dialog filter.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# QSettings scope for persisted window geometry/state and the control session.
_SETTINGS_ORG = "Vinay"
_SETTINGS_APP = "Painting Assist"

# Maps a chosen save filter to the extension to append when the user's path has
# none (so a bare "portrait" saves as "portrait.png" rather than a raw file).
_FILTER_EXT = {
    "PNG (*.png)": ".png",
    "JPEG (*.jpg)": ".jpg",
    "TIFF (*.tif)": ".tif",
}


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)

        # ---- core objects ----
        self._model = ImageModel()
        self._pipeline = ControlPipeline(registry.create_all())
        self._view = ImageView()
        self.setCentralWidget(self._view)
        self._panel = ControlPanel(self._pipeline)
        self._renderer = RenderController(self._pipeline, self._model.original)

        # ---- dock ----
        dock = QDockWidget("Controls", self)
        dock.setObjectName("controls_dock")
        dock.setWidget(self._panel)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._dock = dock

        self._slider_down = False
        self._crop_editing = False
        self._fit_next = False

        # Last processed frame shown in the view, kept so the before/after (B-key)
        # toggle can flip to the original and back without forcing a re-render.
        self._last_image: Optional[np.ndarray] = None
        self._last_scale: float = 1.0
        self._showing_before = False

        # Accept image-file drops onto the window.
        self.setAcceptDrops(True)

        # Optional crop tool glue (present only if a "crop" control is registered).
        try:
            self._crop_control = self._pipeline.control("crop")
        except KeyError:
            self._crop_control = None
        self._crop_editor = self._panel.editor("crop")

        self._build_toolbar()

        # ---- the responsiveness signal graph ----
        self._model.image_loaded.connect(self._on_image_loaded)
        self._panel.paramChanged.connect(self._on_param)
        self._panel.enabledChanged.connect(self._on_enabled)
        self._panel.interactionChanged.connect(self._on_interaction)
        self._renderer.rendered.connect(self._on_rendered)

        # ---- crop tool glue ----
        self._view.cropRectChanged.connect(self._on_crop_rect_changed)
        if self._crop_editor is not None:
            self._crop_editor.editRequested.connect(self._on_crop_edit_requested)

        # ---- restore persisted window + control session (built UI first) ----
        self._restore_session()

    # ------------------------------------------------------------------ #
    # Toolbar / menu
    # ------------------------------------------------------------------ #
    def _build_toolbar(self) -> None:
        """Build the Open/Save | Fit/1:1 | Reset toolbar and a View menu."""
        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("main_toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("Open", self)
        open_action.setShortcut(QKeySequence.Open)  # Ctrl+O
        open_action.setStatusTip("Open a reference image")
        open_action.triggered.connect(self._on_open)
        toolbar.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.Save)  # Ctrl+S
        save_action.setStatusTip("Save the processed image you currently see")
        save_action.triggered.connect(self._on_save)
        toolbar.addAction(save_action)

        export_action = QAction("Export Blur Steps", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.setStatusTip(
            "Export one image per blur stage (Blur must be in Stepped mode)"
        )
        export_action.triggered.connect(self._on_export_steps)
        toolbar.addAction(export_action)

        toolbar.addSeparator()

        fit_action = QAction("Fit", self)
        fit_action.setShortcut(QKeySequence("Ctrl+0"))
        fit_action.triggered.connect(self._on_fit)
        toolbar.addAction(fit_action)

        actual_action = QAction("1:1", self)
        actual_action.setShortcut(QKeySequence("Ctrl+1"))
        actual_action.triggered.connect(self._on_actual_size)
        toolbar.addAction(actual_action)

        toolbar.addSeparator()

        reset_action = QAction("Reset", self)
        reset_action.setStatusTip("Reset all controls (keeps the image)")
        reset_action.triggered.connect(self._on_reset)
        toolbar.addAction(reset_action)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self._dock.toggleViewAction())
        view_menu.addSeparator()
        view_menu.addAction(fit_action)
        view_menu.addAction(actual_action)

        self.statusBar().showMessage(
            "Open a reference image to begin. Tip: hold B to compare with the original."
        )

    # ------------------------------------------------------------------ #
    # Toolbar slots
    # ------------------------------------------------------------------ #
    def _on_open(self) -> None:
        """Prompt for an image file and load it into the model."""
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", _OPEN_FILTER)
        if not path:
            return
        try:
            self._model.load_path(path)
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(self, "Open failed", f"Could not open image:\n{exc}")

    def _on_save(self) -> None:
        """Render the current PROCESSED image at full resolution and save it.

        Renders on an *isolated* pipeline (a fresh set of controls loaded with the
        current state) rather than the shared live pipeline, so it neither races a
        worker that may be mid-render nor disturbs the live prefix cache.
        """
        source = self._model.original()
        if source is None:
            QMessageBox.information(self, "Nothing to save", "Open an image first.")
            return
        path, selected = QFileDialog.getSaveFileName(
            self, "Save processed image", "", _SAVE_FILTER
        )
        if not path:
            return
        path = self._ensure_extension(path, selected)
        try:
            save_pipeline = ControlPipeline(registry.create_all())
            for control in self._pipeline.controls():
                save_pipeline.control(control.id).load_state(control.to_state())
            processed = save_pipeline.process(source)
            self._write_image_rgb(path, processed)
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(self, "Save failed", f"Could not save image:\n{exc}")

    @staticmethod
    def _ensure_extension(path: str, selected_filter: str) -> str:
        """Append an extension from the chosen filter when ``path`` lacks one."""
        if os.path.splitext(path)[1]:
            return path
        return path + _FILTER_EXT.get(selected_filter, ".png")

    @staticmethod
    def _write_image_rgb(path: str, rgb: np.ndarray) -> None:
        """Write an RGB uint8 array to ``path``, robust to unicode paths.

        Encodes in-memory with ``cv2.imencode`` (keyed on the file extension) and
        writes the bytes via ``open(..., "wb")``. ``cv2.imwrite`` mangles non-ASCII
        paths on some platforms; encoding then writing ourselves avoids that.
        """
        ext = os.path.splitext(path)[1] or ".png"
        bgr = cv2.cvtColor(
            np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )
        ok, buf = cv2.imencode(ext, bgr)
        if not ok:
            raise IOError("cv2.imencode failed (unsupported extension %r?)" % ext)
        with open(path, "wb") as handle:
            handle.write(buf.tobytes())

    def _on_export_steps(self) -> None:
        """Export one image per blur stage (the coarse->fine progression).

        Runs on an isolated copy of the pipeline so it neither races the live
        renderer nor disturbs the current view: the export pipeline gets a fresh
        set of controls loaded with the current state, blur is forced on and
        stepped through every stage, and each result (crop + blur + grid applied)
        is written to a chosen folder as a numbered PNG.
        """
        source = self._model.original()
        if source is None:
            QMessageBox.information(self, "Nothing to export", "Open an image first.")
            return
        try:
            live_blur = self._pipeline.control("blur")
        except KeyError:
            QMessageBox.information(self, "No blur control", "Blur is unavailable.")
            return
        if str(live_blur.get("mode")) != "stepped":
            QMessageBox.information(
                self,
                "Switch Blur to Stepped",
                "Set the Blur control's Mode to “Stepped” first — that "
                "defines the discrete blur steps to export.",
            )
            return

        directory = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not directory:
            return

        # Build an isolated pipeline mirroring the current control states.
        export_pipeline = ControlPipeline(registry.create_all())
        for control in self._pipeline.controls():
            export_pipeline.control(control.id).load_state(control.to_state())

        blur = export_pipeline.control("blur")
        blur.set_enabled(True)
        count = blur.stage_count()
        levels = blur.stage_levels()

        written = 0
        try:
            for stage in range(1, count + 1):
                blur.set("stage", stage)
                processed = export_pipeline.process(source)
                radius = int(levels[stage - 1]) if stage - 1 < len(levels) else 0
                fname = "blur_step_{:02d}_of_{:02d}_blur{:03d}.png".format(
                    stage, count, radius
                )
                self._write_image_rgb(os.path.join(directory, fname), processed)
                written += 1
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(
                self, "Export failed",
                "Wrote {} of {} steps before an error:\n{}".format(written, count, exc),
            )
            return

        QMessageBox.information(
            self, "Export complete",
            "Wrote {} blur-step images to:\n{}".format(written, directory),
        )

    def _on_reset(self) -> None:
        """Reset every control to defaults (keeps the image loaded) and re-render."""
        if self._crop_editing:
            self._end_crop_edit(render=False)
        self._panel.reset_all()
        self._slider_down = False
        self._renderer.request(interactive=False)

    def _on_fit(self) -> None:
        self._view.fit_to_window()

    def _on_actual_size(self) -> None:
        self._view.reset_zoom()

    # ------------------------------------------------------------------ #
    # Model / panel / renderer slots
    # ------------------------------------------------------------------ #
    def _on_image_loaded(self) -> None:
        """A new original was loaded: show it (fit) and kick a full render."""
        if self._crop_editing:
            self._end_crop_edit(render=False)
        original = self._model.original()
        if original is not None:
            self._view.set_image(original, preserve_view=False)
        self._renderer.request(interactive=False)
        self.statusBar().showMessage("Reference loaded.", 4000)

    def _on_param(self, cid: str, name: str, value: object) -> None:
        """A param value changed: update the pipeline and request a render."""
        self._pipeline.set_value(cid, name, value)
        if self._crop_editing:
            # While cropping we show the untouched original + overlay; just keep
            # the overlay's locked aspect in step with the canvas dimensions.
            if cid == "crop" and name in ("lock_ratio", "canvas_w", "canvas_h"):
                aspect = self._crop_control.aspect() if self._crop_control else None
                self._view.set_crop_aspect(aspect)
            return
        self._renderer.request(interactive=self._slider_down)

    def _on_enabled(self, cid: str, enabled: bool) -> None:
        """A control was toggled on/off: update the pipeline, full-quality render."""
        self._pipeline.set_enabled(cid, enabled)
        if self._crop_editing and cid == "crop" and not enabled:
            self._end_crop_edit(render=True)
            return
        if self._crop_editing:
            return
        self._renderer.request(interactive=False)

    def _on_interaction(self, down: bool) -> None:
        """Slider pressed/released: track drag state; on release do a full pass."""
        self._slider_down = down
        if not down and not self._crop_editing:
            self._renderer.request(interactive=False)

    def _on_rendered(self, image: object, was_full: bool, scale: float) -> None:
        """A processed image arrived: display it (ignored while cropping).

        ``scale`` (1.0 full-res, <1 for an interactive preview) is passed through
        so a downscaled preview is shown at full on-screen size rather than
        shrinking to its pixel dimensions.
        """
        if self._crop_editing:
            return
        # Remember the processed frame so the before/after toggle can restore it
        # without a re-render.
        self._last_image = image
        self._last_scale = scale
        if self._showing_before:
            # The user is holding B: keep showing the original, but the freshly
            # arrived processed frame is now the one we will restore on release.
            return
        preserve = not self._fit_next
        self._view.set_image(image, preserve_view=preserve, display_scale=scale)
        self._fit_next = False

    # ------------------------------------------------------------------ #
    # Crop tool glue
    # ------------------------------------------------------------------ #
    def _on_crop_edit_requested(self, begin: bool) -> None:
        """The crop editor asked to start (True) or finish/apply (False) editing."""
        if begin:
            self._begin_crop_edit()
        else:
            self._end_crop_edit(render=True)

    def _begin_crop_edit(self) -> None:
        """Show the untouched original with the interactive crop overlay."""
        if self._crop_control is None:
            return
        original = self._model.original()
        if original is None:
            QMessageBox.information(self, "No image", "Open an image before cropping.")
            if self._crop_editor is not None:
                self._crop_editor.set_editing(False)
            return

        # Cropping only matters when the control is enabled; turn it on and
        # reflect that in the dock checkbox.
        self._pipeline.set_enabled("crop", True)
        self._panel.refresh_all()

        self._crop_editing = True
        self._view.set_image(original, preserve_view=True)
        self._view.begin_crop(self._crop_control.rect_norm(), self._crop_control.aspect())
        self.statusBar().showMessage(
            "Drag the box to frame your crop; click Apply crop when done.", 6000
        )

    def _end_crop_edit(self, render: bool) -> None:
        """Leave crop-editing mode and (optionally) render the cropped result."""
        self._crop_editing = False
        self._view.end_crop()
        if self._crop_editor is not None:
            self._crop_editor.set_editing(False)
        if render:
            self._fit_next = True
            self._renderer.request(interactive=False)

    def _on_crop_rect_changed(self, rx: float, ry: float, rw: float, rh: float) -> None:
        """The overlay moved/resized: store the normalised rect on the control."""
        if self._crop_control is None:
            return
        self._pipeline.set_value("crop", "rx", rx)
        self._pipeline.set_value("crop", "ry", ry)
        self._pipeline.set_value("crop", "rw", rw)
        self._pipeline.set_value("crop", "rh", rh)

    # ------------------------------------------------------------------ #
    # Drag and drop
    # ------------------------------------------------------------------ #
    @staticmethod
    def _single_dropped_image(event) -> Optional[str]:
        """Return the local path of a single dropped image file, else ``None``.

        Accepts the drop only when the payload is exactly one local file whose
        extension is in :data:`_IMAGE_EXTS` (mirroring the Open dialog filter).
        """
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        urls = mime.urls()
        if len(urls) != 1:
            return None
        url = urls[0]
        if not url.isLocalFile():
            return None
        path = url.toLocalFile()
        if os.path.splitext(path)[1].lower() not in _IMAGE_EXTS:
            return None
        return path

    def dragEnterEvent(self, event) -> None:
        """Accept the drag only for a single local image file."""
        if self._single_dropped_image(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        """Load a dropped image file, mirroring the Open action's error handling."""
        path = self._single_dropped_image(event)
        if path is None:
            event.ignore()
            return
        event.acceptProposedAction()
        try:
            self._model.load_path(path)
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(self, "Open failed", f"Could not open image:\n{exc}")

    # ------------------------------------------------------------------ #
    # Before/after toggle (hold B to view the untouched original)
    # ------------------------------------------------------------------ #
    def keyPressEvent(self, event) -> None:
        """Hold B to show the untouched original; ignores key auto-repeat."""
        if (
            event.key() == Qt.Key_B
            and not event.isAutoRepeat()
            and not self._crop_editing
            and not self._showing_before
        ):
            original = self._model.original()
            if original is not None:
                self._showing_before = True
                self._view.set_image(original, preserve_view=True)
                self.statusBar().showMessage("Showing original (release B to restore).")
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        """Release B to restore the last processed frame without a re-render."""
        if (
            event.key() == Qt.Key_B
            and not event.isAutoRepeat()
            and self._showing_before
        ):
            self._showing_before = False
            if self._last_image is not None:
                self._view.set_image(
                    self._last_image, preserve_view=True, display_scale=self._last_scale
                )
            self.statusBar().clearMessage()
            event.accept()
            return
        super().keyReleaseEvent(event)

    # ------------------------------------------------------------------ #
    # Session persistence (QSettings)
    # ------------------------------------------------------------------ #
    def _restore_session(self) -> None:
        """Restore window geometry/state and the control session from QSettings.

        The image file itself is deliberately NOT auto-reloaded: only the control
        knob states are restored, and a status hint is shown if the last image is
        still on disk. This keeps startup safe (no surprise file I/O) while
        preserving the painter's tuning between runs.
        """
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)

        geometry = settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = settings.value("window/state")
        if state is not None:
            self.restoreState(state)

        raw = settings.value("session/controls")
        if raw:
            try:
                states = json.loads(raw)
            except (ValueError, TypeError):
                states = None
            if isinstance(states, dict):
                for control in self._pipeline.controls():
                    control_state = states.get(control.id)
                    if isinstance(control_state, dict):
                        control.load_state(control_state)
                self._panel.refresh_all()

        last_path = settings.value("session/last_image")
        if last_path and os.path.exists(last_path):
            self.statusBar().showMessage(
                "Last image was {} — open it to resume.".format(last_path), 6000
            )

    def _save_session(self) -> None:
        """Persist window geometry/state and the control session to QSettings.

        Called only from :meth:`closeEvent`, so a mid-session Reset is never
        clobbered by stale saved state (the session is written on close, not live).
        """
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.setValue("window/geometry", self.saveGeometry())
        settings.setValue("window/state", self.saveState())

        states = {
            control.id: control.to_state()
            for control in self._pipeline.controls()
        }
        settings.setValue("session/controls", json.dumps(states))
        settings.setValue("session/last_image", self._model.path() or "")

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:
        """Persist the session, stop the renderer, and drain workers before teardown.

        Without draining, an in-flight worker can emit back into objects that Qt
        has already torn down, which segfaults on exit.
        """
        self._save_session()
        self._renderer.shutdown()
        super().closeEvent(event)
