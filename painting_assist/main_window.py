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

import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSettings, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QToolBar,
)

from painting_assist import __author__, __version__, theme, updater
from painting_assist.controls import registry
from painting_assist.controls.grid import draw_grid
from painting_assist.image_model import ImageModel
from painting_assist.pipeline import ControlPipeline
from painting_assist.render_controller import RenderController
from painting_assist.widgets.control_panel import ControlPanel
from painting_assist.widgets.image_view import ImageView
from painting_assist.widgets.palette_panel import PalettePanel, format_readout
from painting_assist.widgets.settings_dialog import (
    DEFAULT_THEME,
    DEFAULT_UPDATE_HOURS,
    SettingsDialog,
)

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

# How many entries the File ▸ Open Recent list retains.
_MAX_RECENT = 8


def update_recent(paths: list, new_path: str, limit: int = _MAX_RECENT) -> list:
    """Return ``paths`` with ``new_path`` moved to the front, deduped, capped.

    Paths are compared by their absolute form so the same file reached via a
    relative and an absolute path is not duplicated. Order is most-recent-first.
    Pure function (no filesystem access) so it is trivially unit-testable.
    """
    front = os.path.abspath(new_path)
    result = [front]
    seen = {front}
    for p in paths:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        result.append(p)
        if len(result) >= limit:
            break
    return result[:limit]


def prune_recent(paths: list) -> list:
    """Return ``paths`` with duplicates and no-longer-existing files removed.

    Order is preserved. Comparison for dedupe is by absolute path.
    """
    seen = set()
    out = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        if os.path.exists(p):
            out.append(p)
    return out


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

        # ---- palette dock (swatches from the Colour groups control) ----
        self._palette_panel = PalettePanel()
        palette_dock = QDockWidget("Palette", self)
        palette_dock.setObjectName("palette_dock")
        palette_dock.setWidget(self._palette_panel)
        palette_dock.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, palette_dock)
        self._palette_dock = palette_dock

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

        # Optional grid overlay glue (present only if a "grid" control exists).
        try:
            self._grid_control = self._pipeline.control("grid")
        except KeyError:
            self._grid_control = None

        # Recent-files list (most-recent-first), persisted in QSettings.
        self._recent: list = []

        # ---- settings (theme + automatic update checks) ----
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._theme_mode = str(settings.value("settings/theme", DEFAULT_THEME))
        if self._theme_mode not in theme.THEME_MODES:
            self._theme_mode = DEFAULT_THEME
        try:
            self._update_hours = float(
                settings.value("settings/update_hours", DEFAULT_UPDATE_HOURS)
            )
        except (TypeError, ValueError):
            self._update_hours = DEFAULT_UPDATE_HOURS

        # ---- update checker (off-thread; signals land on the GUI thread) ----
        self._updater = updater.UpdateChecker(__version__, self)
        self._updater.updateAvailable.connect(self._on_update_available)
        self._updater.upToDate.connect(self._on_up_to_date)
        self._updater.checkFailed.connect(self._on_update_check_failed)
        self._updater.downloadProgress.connect(self._on_download_progress)
        self._updater.downloadReady.connect(self._on_download_ready)
        # Automatic checks stay silent on failure/up-to-date; manual ones report.
        self._manual_update_check = False
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._auto_check_updates)

        self._build_toolbar()
        self._build_menus()
        self._update_export_enabled()

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

        # ---- eyedropper ----
        self._view.colourSampled.connect(self._on_colour_sampled)

        # ---- restore persisted window + control session (built UI first) ----
        self._restore_session()
        self._update_grid_overlay()
        self._apply_update_schedule()

    # ------------------------------------------------------------------ #
    # Toolbar / menu
    # ------------------------------------------------------------------ #
    def _build_toolbar(self) -> None:
        """Create the shared QActions and the Open/Save | Fit/1:1 | Reset toolbar.

        The actions are stored on ``self`` so :meth:`_build_menus` can reuse the
        very same objects (a menu item and its toolbar button then share enabled
        state, shortcut, and tooltip — e.g. the Export action is disabled in both
        places at once when Blur is not in Stepped mode).
        """
        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("main_toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._open_action = QAction("Open", self)
        self._open_action.setShortcut(QKeySequence.Open)  # Ctrl+O
        self._open_action.setStatusTip("Open a reference image")
        self._open_action.triggered.connect(self._on_open)
        toolbar.addAction(self._open_action)

        self._save_action = QAction("Save", self)
        self._save_action.setShortcut(QKeySequence.Save)  # Ctrl+S
        self._save_action.setStatusTip("Save the processed image you currently see")
        self._save_action.triggered.connect(self._on_save)
        toolbar.addAction(self._save_action)

        self._export_action = QAction("Export Blur Steps", self)
        self._export_action.setShortcut(QKeySequence("Ctrl+E"))
        self._export_action.setStatusTip(
            "Export one image per blur stage (Blur must be in Stepped mode)"
        )
        self._export_action.triggered.connect(self._on_export_steps)
        toolbar.addAction(self._export_action)

        toolbar.addSeparator()

        self._fit_action = QAction("Fit", self)
        self._fit_action.setShortcut(QKeySequence("Ctrl+0"))
        self._fit_action.triggered.connect(self._on_fit)
        toolbar.addAction(self._fit_action)

        self._actual_action = QAction("1:1", self)
        self._actual_action.setShortcut(QKeySequence("Ctrl+1"))
        self._actual_action.triggered.connect(self._on_actual_size)
        toolbar.addAction(self._actual_action)

        toolbar.addSeparator()

        self._eyedropper_action = QAction("Eyedropper", self)
        self._eyedropper_action.setShortcut(QKeySequence("I"))
        self._eyedropper_action.setCheckable(True)
        self._eyedropper_action.setStatusTip(
            "Click the image to read a colour (hex, value, hue, chroma)"
        )
        self._eyedropper_action.toggled.connect(self._on_eyedropper_toggled)
        toolbar.addAction(self._eyedropper_action)

        toolbar.addSeparator()

        self._reset_action = QAction("Reset", self)
        self._reset_action.setStatusTip("Reset all controls (keeps the image)")
        self._reset_action.triggered.connect(self._on_reset)
        toolbar.addAction(self._reset_action)

        # Permanent right-side status-bar label reflecting the renderer's activity.
        self._busy_label = QLabel("")
        self.statusBar().addPermanentWidget(self._busy_label)

        self.statusBar().showMessage(
            "Open a reference image to begin. Tip: hold B to compare with the original."
        )

    def _build_menus(self) -> None:
        """Build the File / View / Help menu bar from the shared actions."""
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self._open_action)

        self._recent_menu = file_menu.addMenu("Open Recent")
        self._recent_menu.aboutToShow.connect(self._rebuild_recent_menu)

        file_menu.addAction(self._save_action)
        file_menu.addAction(self._export_action)
        file_menu.addSeparator()

        settings_action = QAction("Settings…", self)
        # macOS relocates this into the application menu as Preferences.
        settings_action.setMenuRole(QAction.PreferencesRole)
        settings_action.setShortcut(QKeySequence.Preferences)
        settings_action.triggered.connect(self._on_settings)
        file_menu.addAction(settings_action)
        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.setStatusTip("Quit Painting Assist")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self._dock.toggleViewAction())
        view_menu.addAction(self._palette_dock.toggleViewAction())
        view_menu.addSeparator()
        view_menu.addAction(self._eyedropper_action)
        view_menu.addSeparator()
        view_menu.addAction(self._fit_action)
        view_menu.addAction(self._actual_action)

        help_menu = self.menuBar().addMenu("Help")
        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self._on_shortcuts)
        help_menu.addAction(shortcuts_action)

        check_updates_action = QAction("Check for Updates…", self)
        check_updates_action.triggered.connect(self._on_check_updates)
        help_menu.addAction(check_updates_action)

        about_action = QAction("About Painting Assist", self)
        # On macOS Qt relocates this into the application menu automatically.
        about_action.setMenuRole(QAction.AboutRole)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    # ------------------------------------------------------------------ #
    # Recent files (File ▸ Open Recent)
    # ------------------------------------------------------------------ #
    def _rebuild_recent_menu(self) -> None:
        """Repopulate the Open Recent submenu, pruning files that have vanished."""
        pruned = prune_recent(self._recent)
        if pruned != self._recent:
            self._recent = pruned
            self._save_recent()

        menu = self._recent_menu
        menu.clear()
        if not self._recent:
            empty = menu.addAction("(No recent files)")
            empty.setEnabled(False)
            return
        for path in self._recent:
            act = menu.addAction(path)
            act.triggered.connect(
                lambda _checked=False, p=path: self._open_recent(p)
            )
        menu.addSeparator()
        clear = menu.addAction("Clear Recent")
        clear.triggered.connect(self._clear_recent)

    def _add_recent(self, path: str) -> None:
        """Record ``path`` as the most-recently-opened image and persist the list."""
        self._recent = update_recent(self._recent, path)
        self._save_recent()

    def _open_recent(self, path: str) -> None:
        """Open a path chosen from the Open Recent menu (prune it if it's gone)."""
        if not os.path.exists(path):
            QMessageBox.information(
                self, "File missing",
                "That file is no longer available:\n{}".format(path),
            )
            self._recent = prune_recent(self._recent)
            self._save_recent()
            return
        try:
            self._model.load_path(path)
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(self, "Open failed", f"Could not open image:\n{exc}")

    def _clear_recent(self) -> None:
        """Empty the recent-files list."""
        self._recent = []
        self._save_recent()

    def _save_recent(self) -> None:
        """Persist the recent-files list to QSettings as a JSON array."""
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.setValue("session/recent", json.dumps(self._recent))

    # ------------------------------------------------------------------ #
    # Help
    # ------------------------------------------------------------------ #
    def _on_about(self) -> None:
        """Show the About dialog: program name, version and author."""
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<b>{APP_NAME}</b><br>"
            f"Version {__version__}<br><br>"
            f"Author: {__author__}",
        )

    def _on_shortcuts(self) -> None:
        """Show a simple dialog listing the keyboard shortcuts."""
        QMessageBox.information(
            self,
            "Keyboard Shortcuts",
            "Ctrl+O\tOpen image\n"
            "Ctrl+S\tSave processed image\n"
            "Ctrl+E\tExport blur steps\n"
            "Ctrl+0\tFit to window\n"
            "Ctrl+1\tActual size (1:1)\n"
            "Ctrl+Q\tQuit\n"
            "\n"
            "Hold B\tShow the original (before/after)\n"
            "V\tToggle the Values control\n"
            "I\tEyedropper (click the image to read a colour)",
        )

    # ------------------------------------------------------------------ #
    # Render request + busy indicator
    # ------------------------------------------------------------------ #
    def _request_render(self, interactive: bool) -> None:
        """Ask the renderer for a frame and show the busy indicator until it lands."""
        self._renderer.request(interactive=interactive)
        self._busy_label.setText("Rendering…")

    def _update_export_enabled(self) -> None:
        """Enable Export Blur Steps only when a Blur control is in Stepped mode."""
        try:
            stepped = str(self._pipeline.control("blur").get("mode")) == "stepped"
        except KeyError:
            stepped = False
        self._export_action.setEnabled(stepped)
        self._export_action.setToolTip(
            "Export one image per blur stage"
            if stepped
            else "Requires Blur in Stepped mode"
        )

    def _toggle_values(self) -> None:
        """Toggle the Values control's enabled state (V shortcut) and re-render.

        Mirrors exactly what ticking the dock checkbox does: flips the control's
        enabled flag on the pipeline, re-syncs the panel so the checkbox follows,
        and requests a full-quality render. No-ops if there is no Values control.
        """
        try:
            control = self._pipeline.control("values")
        except KeyError:
            return
        self._pipeline.set_enabled("values", not control.enabled)
        self._panel.refresh_all()
        self._request_render(interactive=False)

    # ------------------------------------------------------------------ #
    # Settings (theme + automatic update checks)
    # ------------------------------------------------------------------ #
    def _on_settings(self) -> None:
        """Open the settings dialog; apply and persist any accepted changes."""
        dialog = SettingsDialog(self._theme_mode, self._update_hours, self)
        if not dialog.exec():
            return
        values = dialog.values()
        self._theme_mode = str(values["theme"])
        self._update_hours = float(values["update_hours"])

        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.setValue("settings/theme", self._theme_mode)
        settings.setValue("settings/update_hours", self._update_hours)

        app = QApplication.instance()
        if app is not None:
            theme.apply_theme(app, self._theme_mode)
        self._apply_update_schedule(startup=False)

    # ------------------------------------------------------------------ #
    # Update checking
    # ------------------------------------------------------------------ #
    def _apply_update_schedule(self, startup: bool = True) -> None:
        """Start/stop the automatic update timer per the configured interval.

        Interval semantics: 0 hours = never check automatically; a tiny value
        (the "Every launch" option) = check once at startup with no recurring
        timer; anything larger = a recurring timer at that interval, plus a
        startup check when the last recorded check is older than the interval.
        """
        self._update_timer.stop()
        hours = self._update_hours
        if hours <= 0.0:
            return
        if hours < 1.0:  # "Every launch"
            if startup:
                self._auto_check_updates()
            return
        self._update_timer.start(int(hours * 3600 * 1000))
        if startup:
            settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
            try:
                last = float(settings.value("settings/last_update_check", 0.0))
            except (TypeError, ValueError):
                last = 0.0
            if time.time() - last >= hours * 3600:
                self._auto_check_updates()

    def _auto_check_updates(self) -> None:
        """Timer/startup slot: silent check (only an available update surfaces)."""
        self._manual_update_check = False
        self._record_update_check()
        self._updater.check()

    def _on_check_updates(self) -> None:
        """Help menu slot: explicit check, so every outcome gets a dialog/message."""
        self._manual_update_check = True
        self._record_update_check()
        self.statusBar().showMessage("Checking for updates…", 4000)
        self._updater.check()

    @staticmethod
    def _record_update_check() -> None:
        """Stamp now as the last automatic-check time (throttles startup checks)."""
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.setValue("settings/last_update_check", time.time())

    def _on_update_available(self, version: str, asset: dict) -> None:
        """A newer release exists: offer to download and open its installer."""
        if not updater.running_frozen():
            # Source checkout: installing over it makes no sense; just say so.
            QMessageBox.information(
                self,
                "Update available",
                "Version {} is available (you are on {}).\n"
                "You are running from source — update with git pull, or grab "
                "an installer from the releases page.".format(version, __version__),
            )
            return
        answer = QMessageBox.question(
            self,
            "Update available",
            "Version {} is available (you are on {}).\n"
            "Download and open the installer now?".format(version, __version__),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            self._updater.downloadAndOpen(asset)

    def _on_up_to_date(self, version: str) -> None:
        """No newer release; only worth saying when the user asked explicitly."""
        if self._manual_update_check:
            QMessageBox.information(
                self, "Up to date",
                "You are on the latest version ({}).".format(version),
            )

    def _on_update_check_failed(self, message: str) -> None:
        """Check/download failed: loud for manual checks, silent otherwise."""
        if self._manual_update_check:
            QMessageBox.warning(self, "Update check failed", message)
        self._busy_label.clear()

    def _on_download_progress(self, percent: int) -> None:
        """Show installer download progress in the permanent status label."""
        self._busy_label.setText("Downloading update… {}%".format(percent))

    def _on_download_ready(self, path: str) -> None:
        """The installer downloaded: hand it to the OS and tell the user."""
        self._busy_label.clear()
        try:
            updater.open_installer(path)
        except Exception as exc:  # pragma: no cover - OS/handler error path
            QMessageBox.warning(
                self, "Could not open installer",
                "Downloaded to {} but could not open it:\n{}".format(path, exc),
            )
            return
        QMessageBox.information(
            self,
            "Installer ready",
            "The installer has been opened. Quit Painting Assist and follow "
            "it to finish updating.",
        )

    # ------------------------------------------------------------------ #
    # Grid overlay glue
    # ------------------------------------------------------------------ #
    def _update_grid_overlay(self) -> None:
        """Sync the viewer's non-destructive grid overlay with the grid control."""
        if self._grid_control is None:
            self._view.set_grid_overlay(None)
            return
        self._view.set_grid_overlay(self._grid_control.overlay_spec())

    def _ask_bake_grid(self) -> Optional[dict]:
        """Ask whether files being written should include the grid lines.

        The grid is a viewer overlay and no longer part of the pipeline output,
        so saving would silently drop it. When the overlay is showing, ask once
        (default No) and return the overlay spec to bake with, else ``None``.
        """
        if self._grid_control is None:
            return None
        spec = self._grid_control.overlay_spec()
        if not spec.get("visible"):
            return None
        answer = QMessageBox.question(
            self,
            "Include grid?",
            "The grid overlay is showing. Draw the grid lines into the saved "
            "image as well?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return spec if answer == QMessageBox.Yes else None

    @staticmethod
    def _bake_grid(processed: np.ndarray, spec: Optional[dict]) -> np.ndarray:
        """Draw grid lines per ``spec`` into ``processed`` (no-op when None)."""
        if spec is None:
            return processed
        return draw_grid(
            processed,
            spec["columns"],
            spec["rows"],
            spec["color_rgb"],
            spec["opacity"],
            spec["thickness"],
            spec["diagonals"],
        )

    # ------------------------------------------------------------------ #
    # Eyedropper
    # ------------------------------------------------------------------ #
    def _on_eyedropper_toggled(self, active: bool) -> None:
        """Toolbar/menu toggle: arm or disarm the viewer's eyedropper mode."""
        if active and self._crop_editing:
            # Crop editing owns the mouse; refuse to arm.
            self._eyedropper_action.setChecked(False)
            return
        self._view.set_eyedropper(active)
        if active:
            self.statusBar().showMessage(
                "Eyedropper: click the image to read a colour.", 4000
            )

    def _on_colour_sampled(self, xn: float, yn: float) -> None:
        """The eyedropper picked normalised coords: read the processed pixel.

        Samples the last processed frame (what the painter is actually looking
        at), not the untouched original — during a slider drag that frame is the
        0.4-scale preview, which is fine for reading colour.
        """
        image = self._last_image
        if image is None:
            return
        h, w = image.shape[:2]
        x = min(w - 1, max(0, int(xn * w)))
        y = min(h - 1, max(0, int(yn * h)))
        rgb = tuple(int(c) for c in image[y, x])
        self._palette_panel.set_sample(rgb)
        self.statusBar().showMessage(format_readout(rgb))

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
            processed = self._bake_grid(processed, self._ask_bake_grid())
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

        # Ask once whether the (viewer-overlay) grid should be drawn into the files.
        bake_spec = self._ask_bake_grid()

        written = 0
        try:
            for stage in range(1, count + 1):
                blur.set("stage", stage)
                processed = export_pipeline.process(source)
                processed = self._bake_grid(processed, bake_spec)
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
        self._update_export_enabled()
        self._update_grid_overlay()
        self._request_render(interactive=False)

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
        path = self._model.path()
        if path:
            self._add_recent(path)
        self._request_render(interactive=False)
        self.statusBar().showMessage("Reference loaded.", 4000)

    def _on_param(self, cid: str, name: str, value: object) -> None:
        """A param value changed: update the pipeline and request a render."""
        self._pipeline.set_value(cid, name, value)
        if cid == "grid":
            # The grid is a viewer overlay, not a pipeline stage — repaint the
            # overlay directly; no render needed.
            self._update_grid_overlay()
            return
        if self._crop_editing:
            # While cropping we show the untouched original + overlay; just keep
            # the overlay's locked aspect in step with the canvas dimensions.
            if cid == "crop" and name in ("lock_ratio", "canvas_w", "canvas_h"):
                aspect = self._crop_control.aspect() if self._crop_control else None
                self._view.set_crop_aspect(aspect)
            return
        # A blur mode change flips whether Export Blur Steps is available.
        self._update_export_enabled()
        self._request_render(interactive=self._slider_down)

    def _on_enabled(self, cid: str, enabled: bool) -> None:
        """A control was toggled on/off: update the pipeline, full-quality render."""
        self._pipeline.set_enabled(cid, enabled)
        if cid == "grid":
            self._update_grid_overlay()
            return
        if self._crop_editing and cid == "crop" and not enabled:
            self._end_crop_edit(render=True)
            return
        if self._crop_editing:
            return
        self._update_export_enabled()
        self._request_render(interactive=False)

    def _on_interaction(self, down: bool) -> None:
        """Slider pressed/released: track drag state; on release do a full pass."""
        self._slider_down = down
        if not down and not self._crop_editing:
            self._request_render(interactive=False)

    def _on_rendered(
        self, image: object, was_full: bool, scale: float, metadata: object = None
    ) -> None:
        """A processed image arrived: display it (ignored while cropping).

        ``scale`` (1.0 full-res, <1 for an interactive preview) is passed through
        so a downscaled preview is shown at full on-screen size rather than
        shrinking to its pixel dimensions.
        """
        # A full-resolution frame means the renderer has caught up; clear busy.
        if was_full:
            self._busy_label.clear()
        # Feed the palette dock from the frame's side-channel metadata: present
        # while the Colour groups control is active, absent (-> clear) otherwise.
        palette = metadata.get("palette") if isinstance(metadata, dict) else None
        if palette:
            self._palette_panel.set_colours(palette)
        else:
            self._palette_panel.clear()
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

        # Crop editing owns the mouse; disarm the eyedropper first.
        self._eyedropper_action.setChecked(False)

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
            self._request_render(interactive=False)

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
        if (
            event.key() == Qt.Key_V
            and not event.isAutoRepeat()
            and not self._crop_editing
        ):
            self._toggle_values()
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
        """Restore window geometry/state, the control session, and the last image.

        The saved control knob states are applied first, then the last image (if it
        still exists on disk) is reloaded so the painter resumes exactly where they
        left off. The reload is guarded: a missing or unreadable file degrades to a
        status hint rather than blocking startup.
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
        # The restored blur mode determines Export availability.
        self._update_export_enabled()

        self._recent = prune_recent(self._load_recent())

        last_path = settings.value("session/last_image")
        if last_path and os.path.exists(last_path):
            # Reloading emits image_loaded -> _on_image_loaded, which renders with
            # the just-restored control states applied and records the recent entry.
            try:
                self._model.load_path(last_path)
            except Exception:  # pragma: no cover - GUI/IO error path
                self.statusBar().showMessage(
                    "Couldn't reload last image: {}".format(last_path), 6000
                )

    @staticmethod
    def _load_recent() -> list:
        """Read the persisted recent-files list from QSettings (JSON array)."""
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        raw = settings.value("session/recent")
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return []
        return [p for p in data if isinstance(p, str)] if isinstance(data, list) else []

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
