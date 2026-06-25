from __future__ import annotations

"""Custom editor widget for :class:`BlurControl` ("Squint").

Presents the blur control's two modes — a reversed continuous "Detail" slider
and a stepped ladder of discrete blur levels — and reports every user change
back to the panel via the ``paramChanged`` signal. The widget only reads from
the control; it never mutates it.
"""

from typing import Callable, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class BlurEditor(QWidget):
    """Continuous / stepped editor for a :class:`BlurControl` instance.

    Emits :attr:`paramChanged` ``(name, value)`` on every user-driven change and
    :attr:`interaction` ``(bool)`` around live slider drags so the renderer can
    drop to a fast preview while dragging.
    """

    paramChanged = Signal(str, object)
    interaction = Signal(bool)

    def __init__(self, control, parent=None):
        """Build the editor for ``control`` and populate it from its values."""
        super().__init__(parent)
        self._control = control
        self._suppress = False
        self._manual_spins: List[QSpinBox] = []

        root = QVBoxLayout(self)

        # ---- Mode selector ------------------------------------------------ #
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Continuous", "continuous")
        self._mode_combo.addItem("Stepped", "stepped")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo, 1)
        root.addLayout(mode_row)

        # ---- Pages -------------------------------------------------------- #
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_continuous_page())
        self._stack.addWidget(self._build_stepped_page())
        root.addWidget(self._stack)

        # Populate everything from the control, then show the right pages.
        self.refresh()

    # ====================================================================== #
    # Page construction
    # ====================================================================== #
    def _build_continuous_page(self) -> QWidget:
        """Continuous page: a reversed Detail slider + spinbox (0..100)."""
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(QLabel("Detail"))
        row = QHBoxLayout()

        # Slider range is 0..100; slider_position = 100 - radius (reversed).
        self._cont_slider = QSlider(Qt.Horizontal)
        self._cont_slider.setRange(0, 100)
        self._cont_slider.valueChanged.connect(self._on_cont_slider)
        self._cont_slider.sliderPressed.connect(self._on_slider_pressed)
        self._cont_slider.sliderReleased.connect(self._on_slider_released)
        row.addWidget(self._cont_slider, 1)

        self._cont_spin = QSpinBox()
        self._cont_spin.setRange(0, 100)
        self._cont_spin.setSuffix(" px")
        self._cont_spin.valueChanged.connect(self._on_cont_spin)
        row.addWidget(self._cont_spin)

        layout.addLayout(row)
        layout.addStretch(1)
        return page

    def _build_stepped_page(self) -> QWidget:
        """Stepped page: stage count, spacing, even/manual area, stage stepper."""
        page = QWidget()
        layout = QVBoxLayout(page)

        # ---- Stages -------------------------------------------------------- #
        stages_row = QHBoxLayout()
        stages_row.addWidget(QLabel("Stages"))
        self._stages_spin = QSpinBox()
        self._stages_spin.setRange(2, 12)
        self._stages_spin.valueChanged.connect(self._on_stage_count_changed)
        stages_row.addWidget(self._stages_spin)
        stages_row.addStretch(1)
        layout.addLayout(stages_row)

        # ---- Spacing ------------------------------------------------------- #
        spacing_row = QHBoxLayout()
        spacing_row.addWidget(QLabel("Spacing"))
        self._spacing_combo = QComboBox()
        self._spacing_combo.addItem("Even", "even")
        self._spacing_combo.addItem("Manual", "manual")
        self._spacing_combo.currentIndexChanged.connect(self._on_spacing_changed)
        spacing_row.addWidget(self._spacing_combo, 1)
        layout.addLayout(spacing_row)

        # ---- Even / Manual sub-area --------------------------------------- #
        self._spacing_stack = QStackedWidget()
        self._spacing_stack.addWidget(self._build_even_area())
        self._spacing_stack.addWidget(self._build_manual_area())
        layout.addWidget(self._spacing_stack)

        # ---- Stage stepper (always visible) ------------------------------- #
        layout.addWidget(self._build_stage_stepper())
        layout.addStretch(1)
        return page

    def _build_even_area(self) -> QWidget:
        """Even sub-area: a (non-reversed) max-blur slider + computed ladder."""
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Max blur (Stage 1)"))
        row = QHBoxLayout()

        self._even_slider = QSlider(Qt.Horizontal)
        self._even_slider.setRange(0, 100)  # not reversed: left = 0, right = 100
        self._even_slider.valueChanged.connect(self._on_even_slider)
        self._even_slider.sliderPressed.connect(self._on_slider_pressed)
        self._even_slider.sliderReleased.connect(self._on_slider_released)
        row.addWidget(self._even_slider, 1)

        self._even_spin = QSpinBox()
        self._even_spin.setRange(0, 100)
        self._even_spin.setSuffix(" px")
        self._even_spin.valueChanged.connect(self._on_even_spin)
        row.addWidget(self._even_spin)

        layout.addLayout(row)

        self._levels_label = QLabel("Levels:")
        layout.addWidget(self._levels_label)
        return area

    def _build_manual_area(self) -> QWidget:
        """Manual sub-area: a rebuildable column of per-stage spinboxes."""
        area = QWidget()
        self._manual_layout = QVBoxLayout(area)
        self._manual_layout.setContentsMargins(0, 0, 0, 0)
        return area

    def _build_stage_stepper(self) -> QWidget:
        """Stage stepper row: Prev / label / Next, plus a quick-jump slider."""
        wrapper = QWidget()
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        self._prev_btn = QPushButton("◀ Prev")
        self._prev_btn.clicked.connect(self._on_prev_stage)
        row.addWidget(self._prev_btn)

        self._stage_label = QLabel("Stage 1 / 2  —  0 px")
        self._stage_label.setAlignment(Qt.AlignCenter)
        self._stage_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        row.addWidget(self._stage_label, 1)

        self._next_btn = QPushButton("Next ▶")
        self._next_btn.clicked.connect(self._on_next_stage)
        row.addWidget(self._next_btn)
        outer.addLayout(row)

        self._stage_slider = QSlider(Qt.Horizontal)
        self._stage_slider.setRange(1, 2)
        self._stage_slider.valueChanged.connect(self._on_stage_slider)
        self._stage_slider.sliderPressed.connect(self._on_slider_pressed)
        self._stage_slider.sliderReleased.connect(self._on_slider_released)
        outer.addWidget(self._stage_slider)
        return wrapper

    # ====================================================================== #
    # Anti-feedback helper
    # ====================================================================== #
    def _set_silently(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` (which mutates widgets) without emitting paramChanged."""
        self._suppress = True
        try:
            fn()
        finally:
            self._suppress = False

    # ====================================================================== #
    # Slider press/release -> interaction signal
    # ====================================================================== #
    def _on_slider_pressed(self) -> None:
        """Tell the renderer a live drag has started."""
        self.interaction.emit(True)

    def _on_slider_released(self) -> None:
        """Tell the renderer a live drag has finished."""
        self.interaction.emit(False)

    # ====================================================================== #
    # Mode / spacing
    # ====================================================================== #
    def _on_mode_changed(self) -> None:
        """Switch the visible page and report the new mode."""
        mode = self._mode_combo.currentData()
        self._stack.setCurrentIndex(1 if mode == "stepped" else 0)
        if self._suppress:
            return
        self.paramChanged.emit("mode", str(mode))

    def _on_spacing_changed(self) -> None:
        """Switch the even/manual sub-area and report the new spacing."""
        spacing = self._spacing_combo.currentData()
        self._spacing_stack.setCurrentIndex(1 if spacing == "manual" else 0)
        if self._suppress:
            return
        self.paramChanged.emit("spacing", str(spacing))

    # ====================================================================== #
    # Continuous page callbacks
    # ====================================================================== #
    def _on_cont_slider(self, position: int) -> None:
        """Reversed slider moved: radius = 100 - position."""
        if self._suppress:
            return
        radius = 100 - int(position)
        self._set_silently(lambda: self._cont_spin.setValue(radius))
        self.paramChanged.emit("radius", radius)

    def _on_cont_spin(self, value: int) -> None:
        """True radius spinbox changed: slider position = 100 - radius."""
        if self._suppress:
            return
        radius = int(value)
        self._set_silently(lambda: self._cont_slider.setValue(100 - radius))
        self.paramChanged.emit("radius", radius)

    # ====================================================================== #
    # Even sub-area callbacks
    # ====================================================================== #
    def _on_even_slider(self, value: int) -> None:
        """Max-blur slider moved (not reversed): keep spin in sync, report."""
        if self._suppress:
            return
        radius = int(value)
        self._set_silently(lambda: self._even_spin.setValue(radius))
        self.paramChanged.emit("radius", radius)
        self._update_levels_label()

    def _on_even_spin(self, value: int) -> None:
        """Max-blur spinbox changed: keep slider in sync, report."""
        if self._suppress:
            return
        radius = int(value)
        self._set_silently(lambda: self._even_slider.setValue(radius))
        self.paramChanged.emit("radius", radius)
        self._update_levels_label()

    def _update_levels_label(self) -> None:
        """Refresh the read-only ladder label from the control."""
        levels = self._control.stage_levels()
        self._levels_label.setText(
            "Levels: " + " · ".join(str(int(v)) for v in levels)
        )

    # ====================================================================== #
    # Stage count
    # ====================================================================== #
    def _on_stage_count_changed(self, value: int) -> None:
        """Stage count changed: report, rebuild manual spins, re-clamp stepper."""
        if self._suppress:
            return
        count = int(value)
        self.paramChanged.emit("stage_count", count)
        self._rebuild_manual_spins()
        self._reclamp_stage_widgets()
        self._update_levels_label()
        self._update_stage_label()

    # ====================================================================== #
    # Manual sub-area
    # ====================================================================== #
    def _rebuild_manual_spins(self) -> None:
        """Recreate the per-stage spinbox column for the current stage count."""
        # Clear any existing rows.
        while self._manual_layout.count():
            item = self._manual_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._manual_spins = []

        levels = self._control.stage_levels()
        count = self._control.stage_count()
        for i in range(count):
            row = QHBoxLayout()
            row.addWidget(QLabel("Stage %d" % (i + 1)))
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setSuffix(" px")
            value = int(levels[i]) if i < len(levels) else 0
            self._set_silently(lambda s=spin, v=value: s.setValue(v))
            spin.valueChanged.connect(self._on_manual_changed)
            row.addWidget(spin, 1)
            self._manual_spins.append(spin)
            self._manual_layout.addLayout(row)

    def _on_manual_changed(self, _value: int) -> None:
        """Any manual spin changed: assemble the CSV string and report it."""
        if self._suppress:
            return
        text = ", ".join(str(s.value()) for s in self._manual_spins)
        self.paramChanged.emit("manual_values", text)

    # ====================================================================== #
    # Stage stepper
    # ====================================================================== #
    def _on_prev_stage(self) -> None:
        """Step to the previous stage (clamped) and report it."""
        if self._suppress:
            return
        count = self._control.stage_count()
        stage = max(1, min(count, self._control.current_stage() - 1))
        self._apply_stage(stage)

    def _on_next_stage(self) -> None:
        """Step to the next stage (clamped) and report it."""
        if self._suppress:
            return
        count = self._control.stage_count()
        stage = max(1, min(count, self._control.current_stage() + 1))
        self._apply_stage(stage)

    def _on_stage_slider(self, value: int) -> None:
        """Quick-jump slider moved: report the new stage."""
        if self._suppress:
            return
        count = self._control.stage_count()
        stage = max(1, min(count, int(value)))
        self._apply_stage(stage)

    def _apply_stage(self, stage: int) -> None:
        """Sync the stepper widgets to ``stage`` and emit paramChanged."""
        self._set_silently(lambda: self._stage_slider.setValue(stage))
        self.paramChanged.emit("stage", int(stage))
        self._update_stage_label(stage)

    def _reclamp_stage_widgets(self) -> None:
        """Keep the stage slider range in step with the stage count."""
        count = self._control.stage_count()
        stage = self._control.current_stage()
        self._set_silently(lambda: self._stage_slider.setRange(1, count))
        self._set_silently(lambda: self._stage_slider.setValue(stage))

    def _update_stage_label(self, stage: int | None = None) -> None:
        """Refresh the centered "Stage N / M  —  R px" label."""
        count = self._control.stage_count()
        if stage is None:
            stage = self._control.current_stage()
        stage = max(1, min(count, int(stage)))
        levels = self._control.stage_levels()
        radius = int(levels[stage - 1]) if 1 <= stage <= len(levels) else 0
        self._stage_label.setText(
            "Stage %d / %d  —  %d px" % (stage, count, radius)
        )

    # ====================================================================== #
    # Refresh
    # ====================================================================== #
    def refresh(self) -> None:
        """Re-read every value from the control and update all widgets.

        Does not emit :attr:`paramChanged` (guarded by ``self._suppress``) and
        rebuilds the manual-stage spinboxes for the current stage count.
        """
        def apply() -> None:
            # Mode.
            mode = str(self._control.get("mode"))
            idx = self._mode_combo.findData(mode)
            if idx >= 0:
                self._mode_combo.setCurrentIndex(idx)
            self._stack.setCurrentIndex(1 if mode == "stepped" else 0)

            # Continuous radius (reversed slider).
            radius = int(self._control.get("radius"))
            radius = max(0, min(100, radius))
            self._cont_spin.setValue(radius)
            self._cont_slider.setValue(100 - radius)

            # Stage count.
            count = self._control.stage_count()
            self._stages_spin.setValue(count)

            # Spacing.
            spacing = str(self._control.get("spacing"))
            sidx = self._spacing_combo.findData(spacing)
            if sidx >= 0:
                self._spacing_combo.setCurrentIndex(sidx)
            self._spacing_stack.setCurrentIndex(1 if spacing == "manual" else 0)

            # Even sub-area (max blur == radius, not reversed).
            self._even_spin.setValue(radius)
            self._even_slider.setValue(radius)

            # Stage stepper range/value.
            stage = self._control.current_stage()
            self._stage_slider.setRange(1, count)
            self._stage_slider.setValue(stage)

        self._set_silently(apply)

        # Rebuild dynamic widgets and labels (these read fresh control state).
        self._rebuild_manual_spins()
        self._update_levels_label()
        self._update_stage_label()
