from __future__ import annotations

"""Top-level window: builds and wires everything, owns the toolbar.

The control dock is built generically from whatever the registry discovered (via
``ControlPanel``), and the responsiveness mechanics live entirely in
:class:`~painting_assist.render_controller.RenderController`. The one place with
a little control-specific glue is the **crop** tool: an interactive viewport
overlay genuinely needs view-level cooperation that a pure pixel filter does
not. That glue is opt-in and degrades gracefully if no ``crop`` control exists.
"""

import os
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
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

APP_NAME = "Squint"

_OPEN_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)"
_SAVE_FILTER = "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif);;All files (*)"


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

        self.statusBar().showMessage("Open a reference image to begin.")

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
        """Render the current PROCESSED image at full resolution and save it."""
        source = self._model.original()
        if source is None:
            QMessageBox.information(self, "Nothing to save", "Open an image first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save processed image", "", _SAVE_FILTER
        )
        if not path:
            return
        try:
            processed = self._pipeline.process(source)
            bgr = cv2.cvtColor(
                np.ascontiguousarray(processed, dtype=np.uint8), cv2.COLOR_RGB2BGR
            )
            if not cv2.imwrite(path, bgr):
                raise IOError("cv2.imwrite returned False (unsupported extension?)")
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(self, "Save failed", f"Could not save image:\n{exc}")

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
                bgr = cv2.cvtColor(
                    np.ascontiguousarray(processed, dtype=np.uint8), cv2.COLOR_RGB2BGR
                )
                radius = int(levels[stage - 1]) if stage - 1 < len(levels) else 0
                fname = "squint_step_{:02d}_of_{:02d}_blur{:03d}.png".format(
                    stage, count, radius
                )
                if not cv2.imwrite(os.path.join(directory, fname), bgr):
                    raise IOError("could not write %s" % fname)
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

    def _on_rendered(self, image: object, was_full: bool) -> None:
        """A processed image arrived: display it (ignored while cropping)."""
        if self._crop_editing:
            return
        preserve = not self._fit_next
        self._view.set_image(image, preserve_view=preserve)
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
    # Shutdown
    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:
        """Stop the renderer and drain worker threads before teardown.

        Without this, an in-flight worker can emit back into objects that Qt has
        already torn down, which segfaults on exit.
        """
        self._renderer.shutdown()
        super().closeEvent(event)
