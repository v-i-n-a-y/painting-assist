# Copyright 2026 Vinay Williams

"""Top-level window: builds and wires everything, owns the toolbar.

The control dock is built generically from whatever the registry discovered (via
``ControlPanel``), and the responsiveness mechanics live entirely in
:class:`~painting_assist.render_controller.RenderController`. The one place with
a little control-specific glue is the **crop** tool: an interactive viewport
overlay genuinely needs view-level cooperation that a pure pixel filter does
not. That glue is opt-in and degrades gracefully if no ``crop`` control exists.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSettings, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QToolBar,
)

from painting_assist import (
    __author__,
    __version__,
    colour_mixing,
    mixing,
    paints,
    theme,
    updater,
)
from painting_assist.controls import registry
from painting_assist.controls.grid import draw_grid
from painting_assist.image_model import ImageModel
from painting_assist.measure import DISPLAY_UNITS, Calibration
from painting_assist.pipeline import ControlPipeline
from painting_assist.render_controller import RenderController
from painting_assist.widgets.control_panel import ControlPanel
from painting_assist.widgets.image_view import ImageView
from painting_assist.widgets.palette_panel import (
    PalettePanel,
    format_readout,
    render_palette_strip,
)
from painting_assist.widgets.value_histogram import ValueHistogram
from painting_assist.widgets.paints_dialog import PaintsDialog
from painting_assist.widgets.settings_dialog import (
    DEFAULT_ON_MISS,
    DEFAULT_THEME,
    DEFAULT_TOLERANCE_PCT,
    DEFAULT_UPDATE_HOURS,
    SettingsDialog,
)

APP_NAME = "Painting Assist"

# Default limited palette used for the eyedropper's mixing suggestions.
_MIX_PALETTE = "zorn"

_OPEN_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)"
_SAVE_FILTER = "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif);;All files (*)"

# Image extensions accepted by drag-and-drop, mirroring the Open dialog filter.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# QSettings scope for persisted window geometry/state and the control session.
_SETTINGS_ORG = "Vinay"
_SETTINGS_APP = "Painting Assist"

# Bumped whenever the dock layout changes shape so a saved arrangement from an
# older version (e.g. the single "Controls" dock) is ignored and the new default
# split is shown instead, rather than restoring a layout that no longer fits.
# v3: nested/inline dock drops enabled, so any earlier all-tabbed arrangement is
# reset once to the clean default split.
_LAYOUT_VERSION = 3

# Maps a chosen save filter to the extension to append when the user's path has
# none (so a bare "portrait" saves as "portrait.png" rather than a raw file).
_FILTER_EXT = {
    "PNG (*.png)": ".png",
    "JPEG (*.jpg)": ".jpg",
    "TIFF (*.tif)": ".tif",
}

# How many entries the File ▸ Open Recent list retains.
_MAX_RECENT = 8

# How many control-state snapshots the undo history keeps.
_UNDO_LIMIT = 50


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

        # ---- per-control docks (each draggable to the left or right) ----
        self._install_control_docks()

        # ---- palette dock (swatches from the Colour groups control) ----
        self._palette_panel = PalettePanel()
        palette_dock = QDockWidget("Palette", self)
        palette_dock.setObjectName("palette_dock")
        palette_dock.setWidget(self._palette_panel)
        palette_dock.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, palette_dock)
        self._palette_dock = palette_dock

        # ---- value histogram dock (value distribution of the current frame) ----
        self._histogram = ValueHistogram()
        hist_dock = QDockWidget("Values histogram", self)
        hist_dock.setObjectName("histogram_dock")
        hist_dock.setWidget(self._histogram)
        hist_dock.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, hist_dock)
        self.tabifyDockWidget(palette_dock, hist_dock)
        palette_dock.raise_()
        self._histogram_dock = hist_dock

        # Latest colour-group palette (for export); kept in step with renders.
        self._current_palette: list = []

        # One-shot grey-point pick: when armed, the next eyedropper sample sets
        # the White balance neutral instead of just reading a colour.
        self._picking_grey = False
        # Session grid-bake choice: None = ask each time, True/False = remembered.
        self._grid_bake_choice: Optional[bool] = None
        # Last colour sampled/clicked, so a settings or inventory change can
        # refresh the mixing suggestion in place.
        self._last_sample_rgb: Optional[tuple] = None

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

        # Undo/redo of the whole control session. Each entry is a
        # {control_id: to_state()} snapshot, the same unit Save and the session
        # use. Applying a snapshot is guarded by ``_applying_state`` so the
        # panel's programmatic refresh does not record itself as a fresh edit.
        self._undo_stack: list = []
        self._redo_stack: list = []
        self._applying_state = False
        self._committed_state: dict = {}

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

        # ---- colour-matching settings + paint inventory ----
        try:
            self._tolerance_pct = int(
                settings.value("settings/tolerance_pct", DEFAULT_TOLERANCE_PCT)
            )
        except (TypeError, ValueError):
            self._tolerance_pct = DEFAULT_TOLERANCE_PCT
        self._on_miss = str(settings.value("settings/on_miss", DEFAULT_ON_MISS))
        self._paints = paints.paints_from_json(
            settings.value("settings/paints") or "[]"
        )

        # ---- measure-tool display settings (remembered across sessions) ----
        self._measure_unit = str(settings.value("settings/measure_unit", "cm"))
        if self._measure_unit not in DISPLAY_UNITS:
            self._measure_unit = "cm"
        self._measure_edges = settings.value("settings/measure_edges", True, type=bool)

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

        # ---- eyedropper + palette ----
        self._view.colourSampled.connect(self._on_colour_sampled)
        self._palette_panel.swatchClicked.connect(self._on_swatch_clicked)

        # ---- measure tools readout ----
        self._view.measureChanged.connect(self._on_measure_changed)

        # ---- viewer feedback + async load ----
        self._view.zoomChanged.connect(self._on_zoom_changed)
        self._model.load_failed.connect(self._on_load_failed)

        # Values (V) and Flip (F) as application-wide shortcuts so they keep
        # working even when a dock widget holds keyboard focus. Hold-B stays in
        # keyPressEvent because it needs the matching key-release.
        self._values_shortcut = QShortcut(QKeySequence("V"), self)
        self._values_shortcut.setContext(Qt.ApplicationShortcut)
        self._values_shortcut.activated.connect(self._toggle_values)
        self._flip_shortcut = QShortcut(QKeySequence("F"), self)
        self._flip_shortcut.setContext(Qt.ApplicationShortcut)
        self._flip_shortcut.activated.connect(self._toggle_flip)

        # A friendly empty state until the first image is opened.
        self._view.set_placeholder(
            [
                "Open an image (Ctrl+O) or drag one here",
                "",
                "Block in the big masses first with Blur,",
                "then reveal detail as your painting builds.",
            ]
        )

        # Minimum size so the docks and viewport can't be crushed.
        self.setMinimumSize(900, 600)

        # ---- restore persisted window + control session (built UI first) ----
        self._restore_session()
        self._update_grid_overlay()
        self._update_measure_calibration()
        self._apply_update_schedule()

        # Baseline for undo/redo: the just-restored state, with empty history.
        self._committed_state = self._capture_state()
        self._update_undo_actions()

    # ------------------------------------------------------------------ #
    # Control docks
    # ------------------------------------------------------------------ #
    def _install_control_docks(self) -> None:
        """Give every control its own dock, split prep-left / colour-right.

        Composition and geometry controls (crop, flip, grid) start on the left;
        the tone and colour controls start on the right. Each dock is freely
        draggable between the two sides, and its placement is remembered across
        sessions via the window's saved state, so this default only applies until
        the painter rearranges the docks (or on a layout-version bump).
        """
        # Allow nested (edge-split) drops and grouped dragging, not just the
        # default tab-on-drop. This is what lets the painter drop a control dock
        # *inline* (stacked above/below another) as well as tabbed onto it: each
        # dock then exposes five drop zones, its four edges split inline and its
        # centre forms a tab group. Without AllowNestedDocks, Qt mostly offers
        # the centre (tab) drop, so docks clump into tab groups instead.
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.GroupedDragging
        )

        left_ids = {"crop", "flip", "grid"}
        self._control_docks: list = []
        # The last dock placed on each side, so the next one stacks beneath it.
        previous = {Qt.LeftDockWidgetArea: None, Qt.RightDockWidgetArea: None}
        for cid, dock in self._panel.docks_in_order():
            area = Qt.LeftDockWidgetArea if cid in left_ids else Qt.RightDockWidgetArea
            prev = previous[area]
            if prev is None:
                self.addDockWidget(area, dock)
            else:
                self.splitDockWidget(prev, dock, Qt.Vertical)
            previous[area] = dock
            self._control_docks.append(dock)

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

        # Export Blur Steps is available from the File menu; keep it off the
        # toolbar since it is wide and disabled unless Blur is in Stepped mode.
        self._export_action = QAction("Export Blur Steps", self)
        self._export_action.setShortcut(QKeySequence("Ctrl+E"))
        self._export_action.setStatusTip(
            "Export one image per blur stage (Blur must be in Stepped mode)"
        )
        self._export_action.triggered.connect(self._on_export_steps)

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

        # Measure tools: mutually exclusive checkable buttons; clicking the
        # active one again returns to no measuring. Kept in a dict so the
        # eyedropper (and each other) can clear them.
        self._measure_actions = {}
        for mode, label, tip in (
            ("angle", "Angle", "Measure the angle of a line from horizontal"),
            ("caliper", "Caliper", "Compare one length against another"),
            ("guides", "Guides", "Drag a plumb and horizon line to check verticals"),
        ):
            act = QAction(label, self)
            act.setCheckable(True)
            act.setStatusTip(tip)
            act.setToolTip(tip)  # hover explains the tool, not just its name
            act.toggled.connect(
                lambda checked, m=mode: self._on_measure_toggled(m, checked)
            )
            toolbar.addAction(act)
            self._measure_actions[mode] = act

        toolbar.addSeparator()

        self._undo_action = QAction("Undo", self)
        self._undo_action.setShortcut(QKeySequence.Undo)  # Ctrl+Z
        self._undo_action.setStatusTip("Undo the last control change")
        self._undo_action.triggered.connect(self._on_undo)
        self._undo_action.setEnabled(False)
        toolbar.addAction(self._undo_action)

        self._redo_action = QAction("Redo", self)
        self._redo_action.setShortcut(QKeySequence.Redo)  # Ctrl+Shift+Z
        self._redo_action.setStatusTip("Redo the last undone control change")
        self._redo_action.triggered.connect(self._on_redo)
        self._redo_action.setEnabled(False)
        toolbar.addAction(self._redo_action)

        toolbar.addSeparator()

        self._reset_action = QAction("Reset", self)
        self._reset_action.setStatusTip("Reset all controls (keeps the image)")
        self._reset_action.triggered.connect(self._on_reset)
        toolbar.addAction(self._reset_action)

        # Measure-tool strip: unit picker and edge-distance toggle, shown only
        # while a measure tool is active (see _set_measure_strip_visible).
        self._measure_strip: list = []
        measure_label = QLabel("Measure:")
        self._measure_unit_combo = QComboBox()
        for unit in DISPLAY_UNITS:
            self._measure_unit_combo.addItem(unit, unit)
        idx = self._measure_unit_combo.findData(self._measure_unit)
        if idx >= 0:
            self._measure_unit_combo.setCurrentIndex(idx)
        self._measure_unit_combo.setToolTip(
            "Units for the caliper/guides readouts (uses the Canvas & Crop size)"
        )
        self._measure_unit_combo.currentIndexChanged.connect(
            self._on_measure_unit_changed
        )
        self._measure_edges_check = QCheckBox("Edge distances")
        self._measure_edges_check.setToolTip(
            "Show each caliper point's distance to the nearest vertical/horizontal edge"
        )
        self._measure_edges_check.setChecked(bool(self._measure_edges))
        self._measure_edges_check.toggled.connect(self._on_measure_edges_changed)
        for widget in (
            measure_label,
            self._measure_unit_combo,
            self._measure_edges_check,
        ):
            self.statusBar().addPermanentWidget(widget)
            widget.setVisible(False)
            self._measure_strip.append(widget)

        # Permanent right-side status-bar labels: render activity and zoom level.
        self._busy_label = QLabel("")
        self.statusBar().addPermanentWidget(self._busy_label)
        self._zoom_label = QLabel("")
        self.statusBar().addPermanentWidget(self._zoom_label)

        self.statusBar().showMessage(
            "Open an image (Ctrl+O) or drag one onto the window."
        )

    def _build_menus(self) -> None:
        """Build the File / View / Help menu bar from the shared actions."""
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self._open_action)

        self._recent_menu = file_menu.addMenu("Open Recent")
        self._recent_menu.aboutToShow.connect(self._rebuild_recent_menu)

        file_menu.addAction(self._save_action)
        file_menu.addAction(self._export_action)

        self._export_palette_action = QAction("Export Palette…", self)
        self._export_palette_action.setStatusTip(
            "Save the current colour-group swatches as a PNG strip"
        )
        self._export_palette_action.setEnabled(False)
        self._export_palette_action.triggered.connect(self._on_export_palette)
        file_menu.addAction(self._export_palette_action)
        file_menu.addSeparator()

        save_preset_action = QAction("Save Preset…", self)
        save_preset_action.setStatusTip(
            "Save the current control settings as a named preset"
        )
        save_preset_action.triggered.connect(self._on_save_preset)
        file_menu.addAction(save_preset_action)

        self._preset_menu = file_menu.addMenu("Apply Preset")
        self._preset_menu.aboutToShow.connect(self._rebuild_preset_menu)

        delete_preset_action = QAction("Delete Preset…", self)
        delete_preset_action.triggered.connect(self._on_delete_preset)
        file_menu.addAction(delete_preset_action)
        file_menu.addSeparator()

        # Keep Settings in the File menu on every platform. macOS would normally
        # relocate a PreferencesRole action into the application menu, but under
        # the frozen bundle that merge doesn't reliably land, leaving no Settings
        # entry anywhere — so we pin it here with NoRole. Cmd+, still works.
        my_paints_action = QAction("My Paints…", self)
        my_paints_action.setStatusTip(
            "Record the paint tubes you own, used for mixing suggestions"
        )
        my_paints_action.triggered.connect(self._on_my_paints)
        file_menu.addAction(my_paints_action)

        self._settings_action = QAction("Settings…", self)
        self._settings_action.setMenuRole(QAction.NoRole)
        self._settings_action.setShortcut(QKeySequence.Preferences)
        self._settings_action.triggered.connect(self._on_settings)
        file_menu.addAction(self._settings_action)
        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.setStatusTip("Quit Painting Assist")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction(self._undo_action)
        edit_menu.addAction(self._redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self._reset_action)

        view_menu = self.menuBar().addMenu("View")
        controls_menu = view_menu.addMenu("Controls")
        for dock in self._control_docks:
            controls_menu.addAction(dock.toggleViewAction())
        view_menu.addAction(self._palette_dock.toggleViewAction())
        view_menu.addAction(self._histogram_dock.toggleViewAction())
        view_menu.addSeparator()
        view_menu.addAction(self._eyedropper_action)

        grey_point_action = QAction("Pick Grey Point…", self)
        grey_point_action.setStatusTip(
            "Click a should-be-neutral patch to correct the colour cast (white balance)"
        )
        grey_point_action.triggered.connect(self._on_pick_grey)
        view_menu.addAction(grey_point_action)

        measure_menu = view_menu.addMenu("Measure")
        for mode in ("angle", "caliper", "guides"):
            measure_menu.addAction(self._measure_actions[mode])
        view_menu.addSeparator()
        view_menu.addAction(self._fit_action)
        view_menu.addAction(self._actual_action)

        help_menu = self.menuBar().addMenu("Help")
        getting_started_action = QAction("Getting Started", self)
        getting_started_action.triggered.connect(self._on_getting_started)
        help_menu.addAction(getting_started_action)

        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self._on_shortcuts)
        help_menu.addAction(shortcuts_action)

        check_updates_action = QAction("Check for Updates…", self)
        check_updates_action.triggered.connect(self._on_check_updates)
        help_menu.addAction(check_updates_action)

        # Pin About in the Help menu too (same reason as Settings above: the
        # macOS app-menu relocation for AboutRole doesn't reliably land in the
        # frozen bundle).
        self._about_action = QAction("About Painting Assist", self)
        self._about_action.setMenuRole(QAction.NoRole)
        self._about_action.triggered.connect(self._on_about)
        help_menu.addAction(self._about_action)

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
            act.triggered.connect(lambda _checked=False, p=path: self._open_recent(p))
        menu.addSeparator()
        clear = menu.addAction("Clear Recent")
        clear.triggered.connect(self._clear_recent)

    def _add_recent(self, path: str) -> None:
        """Record ``path`` as the most-recently-opened image and persist the list."""
        self._recent = update_recent(self._recent, path)
        self._save_recent()

    def _begin_load(self, path: str) -> None:
        """Start an asynchronous image load, showing a busy hint.

        Decoding runs off the GUI thread so a large photo does not freeze the
        window. Success arrives via ``image_loaded`` (:meth:`_on_image_loaded`)
        and failure via ``load_failed`` (:meth:`_on_load_failed`).
        """
        self._busy_label.setText("Opening…")
        self._model.load_path_async(path)

    def _open_recent(self, path: str) -> None:
        """Open a path chosen from the Open Recent menu (prune it if it's gone)."""
        if not os.path.exists(path):
            QMessageBox.information(
                self,
                "File missing",
                "That file is no longer available:\n{}".format(path),
            )
            self._recent = prune_recent(self._recent)
            self._save_recent()
            return
        self._begin_load(path)

    def _clear_recent(self) -> None:
        """Empty the recent-files list."""
        self._recent = []
        self._save_recent()

    def _save_recent(self) -> None:
        """Persist the recent-files list to QSettings as a JSON array."""
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.setValue("session/recent", json.dumps(self._recent))

    # ------------------------------------------------------------------ #
    # Undo / redo (whole control-session snapshots)
    # ------------------------------------------------------------------ #
    def _capture_state(self) -> dict:
        """Return a {control_id: to_state()} snapshot of every control."""
        return {c.id: c.to_state() for c in self._pipeline.controls()}

    def _commit_state(self) -> None:
        """Record the current control state as an undo point, if it changed.

        Called at natural edit boundaries (a discrete param change, a slider
        release, an enable toggle, a crop apply, a reset, a preset apply). The
        pre-change state is what gets pushed, so an undo restores it. No-op while
        a snapshot is being applied, or when nothing actually changed.
        """
        if self._applying_state:
            return
        snapshot = self._capture_state()
        if snapshot == self._committed_state:
            return
        self._undo_stack.append(self._committed_state)
        if len(self._undo_stack) > _UNDO_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._committed_state = snapshot
        self._update_undo_actions()

    def _apply_state(self, state: dict) -> None:
        """Load a control-session snapshot, refresh the panel and re-render.

        Guarded by ``_applying_state`` so the panel's programmatic refresh does
        not feed back through the param/enable slots as a new edit.
        """
        self._applying_state = True
        try:
            for control in self._pipeline.controls():
                control_state = state.get(control.id)
                if isinstance(control_state, dict):
                    try:
                        control.load_state(control_state)
                    except Exception:  # pragma: no cover - defensive
                        # A malformed snapshot for one control degrades to its
                        # current/default state rather than failing the whole apply.
                        pass
            self._panel.refresh_all()
        finally:
            self._applying_state = False
        self._committed_state = self._capture_state()
        self._update_export_enabled()
        self._update_grid_overlay()
        self._request_render(interactive=False)

    def _reset_history(self) -> None:
        """Drop undo/redo history and rebaseline (used when a new image loads)."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._committed_state = self._capture_state()
        self._update_undo_actions()

    def _update_undo_actions(self) -> None:
        """Enable Undo/Redo only when there is something to undo/redo."""
        self._undo_action.setEnabled(bool(self._undo_stack))
        self._redo_action.setEnabled(bool(self._redo_stack))

    def _on_undo(self) -> None:
        """Restore the previous control-session snapshot."""
        if not self._undo_stack:
            return
        if self._crop_editing:
            self._end_crop_edit(render=False)
        self._redo_stack.append(self._committed_state)
        self._apply_state(self._undo_stack.pop())
        self._update_undo_actions()
        self.statusBar().showMessage("Undo", 2000)

    def _on_redo(self) -> None:
        """Re-apply the most recently undone control-session snapshot."""
        if not self._redo_stack:
            return
        self._undo_stack.append(self._committed_state)
        self._apply_state(self._redo_stack.pop())
        self._update_undo_actions()
        self.statusBar().showMessage("Redo", 2000)

    # ------------------------------------------------------------------ #
    # Presets (named control-session snapshots)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_presets() -> dict:
        """Read the persisted presets ({name: snapshot}) from QSettings."""
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        raw = settings.value("settings/presets")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _store_presets(presets: dict) -> None:
        """Persist the presets dict to QSettings as JSON."""
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.setValue("settings/presets", json.dumps(presets))

    def _on_save_preset(self) -> None:
        """Prompt for a name and save the current control settings as a preset."""
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        name = name.strip()
        if not ok or not name:
            return
        presets = self._load_presets()
        if name in presets:
            answer = QMessageBox.question(
                self,
                "Overwrite preset?",
                "A preset named '{}' already exists. Overwrite it?".format(name),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        presets[name] = self._capture_state()
        self._store_presets(presets)
        self.statusBar().showMessage("Saved preset '{}'.".format(name), 4000)

    def _rebuild_preset_menu(self) -> None:
        """Repopulate the Apply Preset submenu from the saved presets."""
        menu = self._preset_menu
        menu.clear()
        presets = self._load_presets()
        if not presets:
            empty = menu.addAction("(No presets saved)")
            empty.setEnabled(False)
            return
        for name in sorted(presets):
            act = menu.addAction(name)
            act.triggered.connect(lambda _checked=False, n=name: self._apply_preset(n))

    def _apply_preset(self, name: str) -> None:
        """Apply a saved preset by name and record it as an undo point."""
        presets = self._load_presets()
        state = presets.get(name)
        if not isinstance(state, dict):
            return
        if self._crop_editing:
            self._end_crop_edit(render=False)
        self._apply_state(state)
        self._commit_state()
        self.statusBar().showMessage("Applied preset '{}'.".format(name), 4000)

    def _on_delete_preset(self) -> None:
        """Choose a preset to delete."""
        presets = self._load_presets()
        if not presets:
            QMessageBox.information(self, "No presets", "There are no saved presets.")
            return
        name, ok = QInputDialog.getItem(
            self, "Delete Preset", "Preset:", sorted(presets), 0, False
        )
        if not ok or name not in presets:
            return
        del presets[name]
        self._store_presets(presets)
        self.statusBar().showMessage("Deleted preset '{}'.".format(name), 4000)

    # ------------------------------------------------------------------ #
    # Help
    # ------------------------------------------------------------------ #
    def _on_about(self) -> None:
        """Show the About dialog: program name, version and author."""
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<b>{APP_NAME}</b><br>Version {__version__}<br><br>Author: {__author__}",
        )

    def _on_getting_started(self) -> None:
        """Show a short in-app primer on the coarse-to-fine workflow."""
        QMessageBox.information(
            self,
            "Getting Started",
            "<b>Painting Assist</b> shows a reference photo the way a painter "
            "sees it, so you can work from a deliberately simplified view.<br><br>"
            "1. Open a photo (Ctrl+O) or drag one onto the window.<br>"
            "2. Turn on <b>Blur</b> and pull it right down to block in the big "
            "value masses first, then ease it back to let detail return as your "
            "painting builds.<br>"
            "3. Use <b>Values</b> to check your notan, <b>Colour groups</b> to "
            "see flat colour masses, and the <b>eyedropper</b> (I) to read and "
            "mix any colour.<br>"
            "4. Nothing changes the original. <b>Save</b> writes out the view you "
            "currently see, and Reset returns to the untouched photo.<br><br>"
            "Hold <b>B</b> at any time to compare with the original.",
        )

    def _on_shortcuts(self) -> None:
        """Show a simple dialog listing the keyboard shortcuts."""
        QMessageBox.information(
            self,
            "Keyboard Shortcuts",
            "Ctrl+O\tOpen image\n"
            "Ctrl+S\tSave processed image\n"
            "Ctrl+E\tExport blur steps\n"
            "Ctrl+,\tSettings\n"
            "Ctrl+0\tFit to window\n"
            "Ctrl+1\tActual size (1:1)\n"
            "Ctrl+Z\tUndo\n"
            "Ctrl+Shift+Z\tRedo\n"
            "Ctrl+Q\tQuit\n"
            "\n"
            "Hold B\tShow the original (before/after)\n"
            "V\tToggle the Values control\n"
            "F\tToggle the Flip control\n"
            "I\tEyedropper (click the image to read a colour)\n"
            "\n"
            "Mouse wheel\tZoom about the cursor\n"
            "Drag\tPan the image",
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
        self._commit_state()
        self._request_render(interactive=False)

    def _toggle_flip(self) -> None:
        """Toggle the Flip control's enabled state (F shortcut) and re-render."""
        try:
            control = self._pipeline.control("flip")
        except KeyError:
            return
        self._pipeline.set_enabled("flip", not control.enabled)
        self._panel.refresh_all()
        self._commit_state()
        self._request_render(interactive=False)

    # ------------------------------------------------------------------ #
    # Settings (theme + automatic update checks)
    # ------------------------------------------------------------------ #
    def _on_settings(self) -> None:
        """Open the settings dialog; apply and persist any accepted changes."""
        dialog = SettingsDialog(
            self._theme_mode,
            self._update_hours,
            self._tolerance_pct,
            self._on_miss,
            self,
        )
        if not dialog.exec():
            return
        values = dialog.values()
        self._theme_mode = str(values["theme"])
        self._update_hours = float(values["update_hours"])
        self._tolerance_pct = int(values["tolerance_pct"])
        self._on_miss = str(values["on_miss"])

        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.setValue("settings/theme", self._theme_mode)
        settings.setValue("settings/update_hours", self._update_hours)
        settings.setValue("settings/tolerance_pct", self._tolerance_pct)
        settings.setValue("settings/on_miss", self._on_miss)

        app = QApplication.instance()
        if app is not None:
            theme.apply_theme(app, self._theme_mode)
        self._apply_update_schedule(startup=False)
        # Refresh any showing mix suggestion with the new tolerance/behaviour.
        if self._last_sample_rgb is not None:
            self._update_mix_display(self._last_sample_rgb)

    def _on_my_paints(self) -> None:
        """Open the paint-inventory manager and persist any changes."""
        dialog = PaintsDialog(self._paints, self)
        if not dialog.exec():
            return
        self._paints = dialog.paints()
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.setValue("settings/paints", paints.paints_to_json(self._paints))
        if self._last_sample_rgb is not None:
            self._update_mix_display(self._last_sample_rgb)

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
        # A corrupt/non-finite persisted interval must not reach int(...*3600*1000).
        if not math.isfinite(hours) or hours <= 0.0:
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
                self,
                "Up to date",
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
                self,
                "Could not open installer",
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
        so saving would silently drop it. When the overlay is showing, ask (with
        a "remember for this session" checkbox) and return the overlay spec to
        bake with, else ``None``. Once answered, the session choice is reused so
        repeated saves are not nagged.
        """
        if self._grid_control is None:
            return None
        spec = self._grid_control.overlay_spec()
        if not spec.get("visible"):
            return None
        if self._grid_bake_choice is not None:
            return spec if self._grid_bake_choice else None
        box = QMessageBox(self)
        box.setWindowTitle("Include grid?")
        box.setText(
            "The grid overlay is showing. Draw the grid lines into the saved "
            "image as well?"
        )
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        remember = QCheckBox("Remember for this session")
        box.setCheckBox(remember)
        include = box.exec() == QMessageBox.Yes
        if remember.isChecked():
            self._grid_bake_choice = include
        return spec if include else None

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
            spec.get("layout", "even"),
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
        if active:
            # Eyedropper and the measure tools both want the mouse; only one.
            self._clear_measure_tools()
        else:
            # Leaving eyedropper mode cancels any pending grey-point pick.
            self._picking_grey = False
        self._view.set_eyedropper(active)
        if active and not self._picking_grey:
            self.statusBar().showMessage(
                "Eyedropper: click the image to read a colour.", 4000
            )

    def _on_colour_sampled(self, xn: float, yn: float) -> None:
        """The eyedropper picked normalised coords: read the processed colour.

        Samples the last processed frame (what the painter is actually looking
        at), not the untouched original. The palette panel's sample size sets an
        odd window that is averaged, so a noisy pixel does not mislead a mix
        match. During a slider drag the frame is the 0.4-scale preview, which is
        fine for reading colour.
        """
        image = self._last_image
        if image is None:
            return
        h, w = image.shape[:2]
        cx = min(w - 1, max(0, int(xn * w)))
        cy = min(h - 1, max(0, int(yn * h)))
        radius = max(0, self._palette_panel.sample_size() // 2)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        region = image[y0:y1, x0:x1].reshape(-1, image.shape[2])
        rgb = tuple(int(round(v)) for v in region.mean(axis=0))
        if self._picking_grey:
            self._set_grey_point(rgb)
            return
        self._last_sample_rgb = rgb
        self._palette_panel.set_sample(rgb)
        self._update_mix_display(rgb)
        self.statusBar().showMessage(format_readout(rgb))

    def _set_grey_point(self, rgb: tuple) -> None:
        """Use a sampled patch as the White balance neutral (grey-point)."""
        self._picking_grey = False
        self._eyedropper_action.setChecked(False)
        try:
            self._pipeline.control("white_balance")
        except KeyError:
            QMessageBox.information(
                self, "No white balance", "The White balance control is unavailable."
            )
            return
        r, g, b = (int(c) for c in rgb)
        self._pipeline.set_value("white_balance", "neutral_r", r)
        self._pipeline.set_value("white_balance", "neutral_g", g)
        self._pipeline.set_value("white_balance", "neutral_b", b)
        self._pipeline.set_enabled("white_balance", True)
        self._panel.refresh_all()
        self._commit_state()
        self._request_render(interactive=False)
        self.statusBar().showMessage(
            "Grey point set from {}; White balance is on.".format(format_readout(rgb)),
            5000,
        )

    def _on_pick_grey(self) -> None:
        """Arm a one-shot grey-point pick: the next eyedropper click sets it."""
        if self._crop_editing:
            return
        self._picking_grey = True
        self._clear_measure_tools()
        self._eyedropper_action.setChecked(True)
        self.statusBar().showMessage(
            "Click something that should be neutral grey to correct the colour cast.",
            6000,
        )

    def _on_swatch_clicked(self, rgb: tuple) -> None:
        """A palette swatch was clicked: show a mixing suggestion for it."""
        rgb = tuple(int(c) for c in rgb)
        self._last_sample_rgb = rgb
        self._palette_panel.set_sample(rgb)
        self._update_mix_display(rgb)

    def _update_mix_display(self, rgb: tuple) -> None:
        """Update the palette panel's mixing text and the achieved-mix swatch.

        The readout line describes the colour (value, temperature, hue). The
        mixing line is a suggestion from the painter's own tubes when they have
        set any up (mixbox pigment mixing, honouring the tolerance and
        closest-versus-buy preference), otherwise from a built-in limited palette
        as a rough guide. When tubes are set, the mix swatch shows the colour the
        recipe actually produces, side by side with the sampled colour.
        """
        info = colour_mixing.describe_colour(rgb)
        head = "{value:.0f}% value · {temperature} {hue_name} · {modifier}".format(
            **info
        )
        if self._paints:
            suggestion = mixing.suggest(
                rgb,
                self._paints,
                self._tolerance_pct,
                self._on_miss,
                paints.DEFAULT_CATALOGUE,
            )
            if suggestion.recipe:
                self._palette_panel.set_mix(suggestion.mixed_rgb)
                recipe = ", ".join(
                    "{} {:.0f}%".format(name, share * 100)
                    for name, share in suggestion.recipe
                )
                self._palette_panel.set_mixing(
                    "{}\n{}\n{}".format(head, recipe, suggestion.message)
                )
            else:
                self._palette_panel.set_mix(None)
                self._palette_panel.set_mixing(
                    "{}\n{}".format(head, suggestion.message)
                )
            return
        # No tubes recorded yet: fall back to the built-in limited palette.
        self._palette_panel.set_mix(None)
        parts = colour_mixing.suggest_mix(rgb, _MIX_PALETTE)
        label = colour_mixing.BASE_PALETTES[_MIX_PALETTE]["label"]
        recipe = ", ".join(
            "{} {:.0f}%".format(name, share * 100) for name, share in parts
        )
        self._palette_panel.set_mixing(
            "{}\n{}: {}  (set up My Paints for your tubes)".format(head, label, recipe)
        )

    # ------------------------------------------------------------------ #
    # Measure tools
    # ------------------------------------------------------------------ #
    def _on_measure_toggled(self, mode: str, checked: bool) -> None:
        """A measure-tool button toggled: activate that mode (or clear it)."""
        if not checked:
            # Turning the active tool off returns to no measuring.
            if self._view.measure_mode() == mode:
                self._view.set_measure_mode(None)
            return
        if self._crop_editing:
            self._measure_actions[mode].setChecked(False)
            return
        # One measure tool at a time, and never together with the eyedropper.
        if self._eyedropper_action.isChecked():
            self._eyedropper_action.setChecked(False)
        for other, act in self._measure_actions.items():
            if other != mode and act.isChecked():
                act.blockSignals(True)
                act.setChecked(False)
                act.blockSignals(False)
        self._view.set_measure_mode(mode)
        self._set_measure_strip_visible(True)
        self.statusBar().showMessage(
            "Drag the handles on the image to measure; click the tool again to finish.",
            5000,
        )

    def _clear_measure_tools(self) -> None:
        """Deactivate any measure tool and untick its button."""
        for act in self._measure_actions.values():
            if act.isChecked():
                act.blockSignals(True)
                act.setChecked(False)
                act.blockSignals(False)
        self._view.set_measure_mode(None)
        self._set_measure_strip_visible(False)

    def _on_measure_changed(self, readout: str) -> None:
        """Show the active measure tool's live readout in the status bar."""
        self.statusBar().showMessage(readout)

    # -- measure units / edge distances (status-bar strip) --------------- #
    def _set_measure_strip_visible(self, visible: bool) -> None:
        """Show or hide the unit/edge-distance strip (only while measuring)."""
        for widget in self._measure_strip:
            widget.setVisible(visible)

    def _measure_calibration(self) -> Calibration:
        """Build the canvas calibration from the Canvas & Crop size + the strip."""
        canvas_w = canvas_h = 0.0
        canvas_unit = "px"
        if self._crop_control is not None:
            try:
                canvas_w = float(self._crop_control.get("canvas_w"))
                canvas_h = float(self._crop_control.get("canvas_h"))
                canvas_unit = str(self._crop_control.get("unit"))
            except (TypeError, ValueError, KeyError):
                pass
        return Calibration(
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            canvas_unit=canvas_unit,
            display_unit=self._measure_unit,
            show_edges=bool(self._measure_edges),
        )

    def _update_measure_calibration(self) -> None:
        """Push a fresh calibration to the view's measure tools."""
        self._view.set_measure_calibration(self._measure_calibration())

    def _on_measure_unit_changed(self, _index: int) -> None:
        """Persist the chosen measure unit and re-calibrate the tools."""
        self._measure_unit = str(self._measure_unit_combo.currentData())
        QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(
            "settings/measure_unit", self._measure_unit
        )
        self._update_measure_calibration()

    def _on_measure_edges_changed(self, checked: bool) -> None:
        """Persist the edge-distance toggle and re-calibrate the tools."""
        self._measure_edges = bool(checked)
        QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(
            "settings/measure_edges", self._measure_edges
        )
        self._update_measure_calibration()

    # ------------------------------------------------------------------ #
    # Viewer feedback + async load
    # ------------------------------------------------------------------ #
    def _on_zoom_changed(self, scale: float) -> None:
        """Show the current zoom level in the status bar."""
        self._zoom_label.setText("{:.0f}%".format(scale * 100.0))

    def _on_load_failed(self, path: str, message: str) -> None:
        """An asynchronous image load failed: tell the user, clear busy."""
        self._busy_label.clear()
        QMessageBox.critical(
            self, "Open failed", "Could not open image:\n{}".format(message)
        )

    def _on_export_palette(self) -> None:
        """Save the current colour-group swatches as a PNG strip."""
        if not self._current_palette:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export palette", "palette.png", "PNG (*.png)"
        )
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".png"
        try:
            render_palette_strip(self._current_palette).save(path)
        except Exception as exc:  # pragma: no cover - GUI/IO error path
            QMessageBox.critical(
                self, "Export failed", "Could not save palette:\n{}".format(exc)
            )
            return
        self.statusBar().showMessage("Palette saved.", 4000)

    # ------------------------------------------------------------------ #
    # Toolbar slots
    # ------------------------------------------------------------------ #
    def _on_open(self) -> None:
        """Prompt for an image file and load it into the model."""
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", _OPEN_FILTER)
        if not path:
            return
        self._begin_load(path)

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
        # Ask about the grid before the wait cursor so the dialog stays usable.
        bake_spec = self._ask_bake_grid()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._busy_label.setText("Saving…")
        try:
            save_pipeline = ControlPipeline(registry.create_all())
            for control in self._pipeline.controls():
                save_pipeline.control(control.id).load_state(control.to_state())
            processed = save_pipeline.process(source)
            processed = self._bake_grid(processed, bake_spec)
            self._write_image_rgb(path, processed)
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(self, "Save failed", f"Could not save image:\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()
            self._busy_label.clear()

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
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
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

        progress = QProgressDialog("Exporting blur steps…", "Cancel", 0, count, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        written = 0
        try:
            for stage in range(1, count + 1):
                if progress.wasCanceled():
                    break
                progress.setLabelText("Exporting step {} of {}…".format(stage, count))
                progress.setValue(stage - 1)
                blur.set("stage", stage)
                processed = export_pipeline.process(source)
                processed = self._bake_grid(processed, bake_spec)
                radius = int(levels[stage - 1]) if stage - 1 < len(levels) else 0
                fname = "blur_step_{:02d}_of_{:02d}_blur{:03d}.png".format(
                    stage, count, radius
                )
                self._write_image_rgb(os.path.join(directory, fname), processed)
                written += 1
            progress.setValue(count)
        except Exception as exc:  # pragma: no cover - GUI error path
            progress.cancel()
            QMessageBox.critical(
                self,
                "Export failed",
                "Wrote {} of {} steps before an error:\n{}".format(written, count, exc),
            )
            return

        QMessageBox.information(
            self,
            "Export complete",
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
        self._commit_state()
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
        self._busy_label.clear()
        if self._crop_editing:
            self._end_crop_edit(render=False)
        # A previous in-flight render must not paint over the new image.
        self._renderer.invalidate_source()
        original = self._model.original()
        if original is not None:
            self._view.set_image(original, preserve_view=False)
        path = self._model.path()
        if path:
            self._add_recent(path)
        # Undo history is per-reference; a fresh image starts a clean slate.
        self._reset_history()
        self._request_render(interactive=False)
        self.statusBar().showMessage(
            "Reference loaded. Hold B to compare with the original.", 5000
        )

    def _on_param(self, cid: str, name: str, value: object) -> None:
        """A param value changed: update the pipeline and request a render."""
        self._pipeline.set_value(cid, name, value)
        if cid == "crop" and name in ("canvas_w", "canvas_h", "unit"):
            # The canvas size/unit drives the physical measure-tool readouts.
            self._update_measure_calibration()
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
        # A discrete change (combo, checkbox, spin) is its own undo point; a
        # slider drag commits once on release instead of on every tick.
        if not self._slider_down:
            self._commit_state()
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
        self._commit_state()
        self._request_render(interactive=False)

    def _on_interaction(self, down: bool) -> None:
        """Slider pressed/released: track drag state; on release do a full pass."""
        self._slider_down = down
        if not down and not self._crop_editing:
            self._commit_state()
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
            self._current_palette = [tuple(int(c) for c in rgb) for rgb in palette]
            self._palette_panel.set_colours(palette)
        else:
            self._current_palette = []
            self._palette_panel.clear()
        self._export_palette_action.setEnabled(bool(self._current_palette))
        # Keep the value histogram in step with the full-res frame (skip cheap
        # preview frames so the distribution reflects the real image).
        if was_full and image is not None:
            self._histogram.set_image(image)
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

        # Crop editing owns the mouse; disarm the eyedropper and measure tools.
        self._eyedropper_action.setChecked(False)
        self._clear_measure_tools()

        # Cropping only matters when the control is enabled; turn it on and
        # reflect that in the dock checkbox.
        self._pipeline.set_enabled("crop", True)
        self._panel.refresh_all()

        self._crop_editing = True
        self._view.set_image(original, preserve_view=True)
        self._view.begin_crop(
            self._crop_control.rect_norm(), self._crop_control.aspect()
        )
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
            self._commit_state()
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
        self._begin_load(path)

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
        if (
            event.key() == Qt.Key_F
            and not event.isAutoRepeat()
            and not self._crop_editing
        ):
            self._toggle_flip()
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
        # Only restore the saved dock arrangement if it was written by this
        # layout version; otherwise keep the fresh prep-left / colour-right split
        # rather than a stale layout from an older dock shape.
        state = settings.value("window/state")
        try:
            saved_layout = int(settings.value("window/layout_version") or 0)
        except (TypeError, ValueError):
            saved_layout = 0
        if state is not None and saved_layout == _LAYOUT_VERSION:
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
                        try:
                            control.load_state(control_state)
                        except Exception:  # pragma: no cover - defensive
                            # One corrupt control state must not block startup.
                            pass
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
        settings.setValue("window/layout_version", _LAYOUT_VERSION)

        states = {
            control.id: control.to_state() for control in self._pipeline.controls()
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
