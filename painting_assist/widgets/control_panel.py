from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from painting_assist.controls.base import Control, Param, ParamType
from painting_assist.pipeline import ControlPipeline


class ParamWidget(QWidget):
    """Editor for ONE :class:`Param`, built generically from its declaration.

    The widget chosen depends on ``spec.ptype``:

    * ``INT``    -> a horizontal ``QSlider`` plus a ``QSpinBox`` readout.
    * ``FLOAT``  -> an int-backed horizontal ``QSlider`` plus a ``QDoubleSpinBox``.
    * ``BOOL``   -> a ``QCheckBox``.
    * ``CHOICE`` -> a ``QComboBox`` (items carry their stored value as userData).

    ``valueChanged(name, value)`` carries the TRUE (un-reversed) param value.
    ``interaction(down)`` fires on slider press/release so the renderer can drop
    to a fast interactive preview while dragging and a full pass on release.
    Programmatic updates via :meth:`set_value` never re-emit ``valueChanged``.
    """

    valueChanged = Signal(str, object)
    interaction = Signal(bool)

    def __init__(
        self,
        spec: Param,
        initial: Any,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._spec = spec
        self._slider: Optional[QSlider] = None
        self._spin: Optional[QWidget] = None  # QSpinBox or QDoubleSpinBox
        self._check: Optional[QCheckBox] = None
        self._combo: Optional[QComboBox] = None
        self._line: Optional[QLineEdit] = None
        # Guard so programmatic updates do not re-emit valueChanged.
        self._suppress = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if spec.tooltip:
            self.setToolTip(spec.tooltip)

        if spec.ptype == ParamType.INT:
            self._build_int(layout, initial)
        elif spec.ptype == ParamType.FLOAT:
            self._build_float(layout, initial)
        elif spec.ptype == ParamType.BOOL:
            self._build_bool(layout, initial)
        elif spec.ptype == ParamType.CHOICE:
            self._build_choice(layout, initial)
        elif spec.ptype == ParamType.TEXT:
            self._build_text(layout, initial)
        else:  # pragma: no cover - defensive; ParamType is closed
            raise ValueError("Unknown ParamType: {!r}".format(spec.ptype))

    # ------------------------------------------------------------------
    # Slider <-> value mapping (handles step granularity and reversed UX)
    # ------------------------------------------------------------------
    def _step(self) -> float:
        """Effective step size for this param (1 for INT, 0.01 for FLOAT defaults)."""
        return float(self._spec.effective_step())

    def _slider_bounds(self) -> int:
        """Return the inclusive maximum slider position (minimum is always 0)."""
        lo = float(self._spec.minimum)
        hi = float(self._spec.maximum)
        step = self._step()
        if step <= 0:
            step = 1.0
        return int(round((hi - lo) / step))

    def _value_to_slider(self, value: float) -> int:
        """Map a true param value to a slider position, honouring ``reversed``."""
        lo = float(self._spec.minimum)
        step = self._step()
        if step <= 0:
            step = 1.0
        pos = int(round((float(value) - lo) / step))
        n = self._slider_bounds()
        if pos < 0:
            pos = 0
        elif pos > n:
            pos = n
        if self._spec.reversed:
            pos = n - pos
        return pos

    def _slider_to_value(self, pos: int) -> float:
        """Map a slider position back to a true param value, honouring ``reversed``."""
        lo = float(self._spec.minimum)
        step = self._step()
        if step <= 0:
            step = 1.0
        n = self._slider_bounds()
        if self._spec.reversed:
            pos = n - pos
        return lo + pos * step

    # ------------------------------------------------------------------
    # Builders per ParamType
    # ------------------------------------------------------------------
    def _build_int(self, layout: QHBoxLayout, initial: Any) -> None:
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(self._slider_bounds())
        slider.setSingleStep(1)
        slider.setPageStep(max(1, self._slider_bounds() // 10))

        spin = QSpinBox()
        spin.setMinimum(int(round(float(self._spec.minimum))))
        spin.setMaximum(int(round(float(self._spec.maximum))))
        spin.setSingleStep(max(1, int(round(self._step()))))
        if self._spec.suffix:
            spin.setSuffix(self._spec.suffix)

        self._slider = slider
        self._spin = spin

        iv = int(round(float(initial)))
        self._suppress = True
        slider.setValue(self._value_to_slider(iv))
        spin.setValue(iv)
        self._suppress = False

        slider.valueChanged.connect(self._on_slider_changed)
        spin.valueChanged.connect(self._on_int_spin_changed)
        slider.sliderPressed.connect(lambda: self.interaction.emit(True))
        slider.sliderReleased.connect(lambda: self.interaction.emit(False))

        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(slider, 1)
        layout.addWidget(spin, 0)

    def _build_float(self, layout: QHBoxLayout, initial: Any) -> None:
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(self._slider_bounds())
        slider.setSingleStep(1)
        slider.setPageStep(max(1, self._slider_bounds() // 10))

        spin = QDoubleSpinBox()
        spin.setMinimum(float(self._spec.minimum))
        spin.setMaximum(float(self._spec.maximum))
        spin.setSingleStep(self._step())
        spin.setDecimals(self._decimals_for_step())
        if self._spec.suffix:
            spin.setSuffix(self._spec.suffix)

        self._slider = slider
        self._spin = spin

        fv = float(initial)
        self._suppress = True
        slider.setValue(self._value_to_slider(fv))
        spin.setValue(fv)
        self._suppress = False

        slider.valueChanged.connect(self._on_slider_changed)
        spin.valueChanged.connect(self._on_float_spin_changed)
        slider.sliderPressed.connect(lambda: self.interaction.emit(True))
        slider.sliderReleased.connect(lambda: self.interaction.emit(False))

        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(slider, 1)
        layout.addWidget(spin, 0)

    def _build_bool(self, layout: QHBoxLayout, initial: Any) -> None:
        check = QCheckBox()
        if self._spec.tooltip:
            check.setToolTip(self._spec.tooltip)
        self._check = check

        self._suppress = True
        check.setChecked(bool(initial))
        self._suppress = False

        check.toggled.connect(self._on_check_toggled)
        layout.addWidget(check, 0)
        layout.addStretch(1)

    def _build_choice(self, layout: QHBoxLayout, initial: Any) -> None:
        combo = QComboBox()
        choices = self._spec.choices or ()
        for stored_value, shown_label in choices:
            combo.addItem(str(shown_label), userData=stored_value)
        self._combo = combo

        self._suppress = True
        idx = self._index_for_value(initial)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        self._suppress = False

        combo.currentIndexChanged.connect(self._on_combo_changed)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(combo, 1)

    def _build_text(self, layout: QHBoxLayout, initial: Any) -> None:
        line = QLineEdit()
        if self._spec.tooltip:
            line.setToolTip(self._spec.tooltip)
        self._line = line

        self._suppress = True
        line.setText("" if initial is None else str(initial))
        self._suppress = False

        # editingFinished (not textChanged) so we do not re-render per keystroke.
        line.editingFinished.connect(self._on_text_edited)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(line, 1)

    def _decimals_for_step(self) -> int:
        """Pick a sensible decimal count from the step size for the spin box."""
        step = self._step()
        decimals = 0
        scaled = step
        # Count fractional digits needed to represent the step (capped at 6).
        while scaled != int(scaled) and decimals < 6:
            scaled *= 10.0
            decimals += 1
        return max(1, decimals)

    def _index_for_value(self, value: Any) -> int:
        """Return the combo index whose stored userData equals ``value`` (-1 if none)."""
        if self._combo is None:
            return -1
        for i in range(self._combo.count()):
            if self._combo.itemData(i) == value:
                return i
        return -1

    # ------------------------------------------------------------------
    # Internal change handlers
    # ------------------------------------------------------------------
    def _on_slider_changed(self, pos: int) -> None:
        if self._suppress:
            return
        value = self._slider_to_value(pos)
        if self._spec.ptype == ParamType.INT:
            value = int(round(value))
        # Reflect into the spin readout without feedback loops.
        self._suppress = True
        if isinstance(self._spin, QSpinBox):
            self._spin.setValue(int(round(value)))
        elif isinstance(self._spin, QDoubleSpinBox):
            self._spin.setValue(float(value))
        self._suppress = False
        self.valueChanged.emit(self._spec.name, value)

    def _on_int_spin_changed(self, value: int) -> None:
        if self._suppress:
            return
        self._suppress = True
        if self._slider is not None:
            self._slider.setValue(self._value_to_slider(int(value)))
        self._suppress = False
        self.valueChanged.emit(self._spec.name, int(value))

    def _on_float_spin_changed(self, value: float) -> None:
        if self._suppress:
            return
        self._suppress = True
        if self._slider is not None:
            self._slider.setValue(self._value_to_slider(float(value)))
        self._suppress = False
        self.valueChanged.emit(self._spec.name, float(value))

    def _on_check_toggled(self, checked: bool) -> None:
        if self._suppress:
            return
        self.valueChanged.emit(self._spec.name, bool(checked))

    def _on_combo_changed(self, _index: int) -> None:
        if self._suppress or self._combo is None:
            return
        self.valueChanged.emit(self._spec.name, self._combo.currentData())

    def _on_text_edited(self) -> None:
        if self._suppress or self._line is None:
            return
        self.valueChanged.emit(self._spec.name, self._line.text())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_value(self, value: Any) -> None:
        """Programmatically set the widget's value WITHOUT emitting ``valueChanged``."""
        self._suppress = True
        try:
            if self._spec.ptype == ParamType.INT:
                iv = int(round(float(value)))
                if self._slider is not None:
                    self._slider.setValue(self._value_to_slider(iv))
                if isinstance(self._spin, QSpinBox):
                    self._spin.setValue(iv)
            elif self._spec.ptype == ParamType.FLOAT:
                fv = float(value)
                if self._slider is not None:
                    self._slider.setValue(self._value_to_slider(fv))
                if isinstance(self._spin, QDoubleSpinBox):
                    self._spin.setValue(fv)
            elif self._spec.ptype == ParamType.BOOL:
                if self._check is not None:
                    self._check.setChecked(bool(value))
            elif self._spec.ptype == ParamType.CHOICE:
                if self._combo is not None:
                    idx = self._index_for_value(value)
                    if idx >= 0:
                        self._combo.setCurrentIndex(idx)
            elif self._spec.ptype == ParamType.TEXT:
                if self._line is not None:
                    self._line.setText("" if value is None else str(value))
        finally:
            self._suppress = False


def build_param_widget(spec: Param, initial: Any) -> ParamWidget:
    """Factory for a :class:`ParamWidget` from a :class:`Param` and its initial value."""
    return ParamWidget(spec, initial)


class ControlSection(QGroupBox):
    """One checkable group box per :class:`Control`.

    The header checkbox mirrors ``control.enabled``; the body holds one labelled
    :class:`ParamWidget` per entry in ``control.params()``. Signals bubble the
    control id alongside the param name/value so the panel and main window stay
    free of any control-specific knowledge.
    """

    paramChanged = Signal(str, str, object)  # (control_id, name, value)
    enabledChanged = Signal(str, bool)  # (control_id, enabled)
    interactionChanged = Signal(bool)  # forwarded slider pressed/released

    def __init__(self, control: Control, parent: Optional[QWidget] = None) -> None:
        super().__init__(control.name, parent)
        self._control = control
        self._param_widgets: Dict[str, ParamWidget] = {}
        self._custom_editor: Optional[QWidget] = None

        self.setCheckable(True)
        self.setChecked(bool(control.enabled))

        body = QVBoxLayout(self)
        body.setContentsMargins(8, 4, 8, 8)
        body.setSpacing(4)

        editor = control.create_editor()
        if editor is not None:
            # The control supplies its own editor; honour its signal contract and
            # skip the generic Param rows entirely.
            self._custom_editor = editor
            editor.paramChanged.connect(self._on_param_changed)
            if hasattr(editor, "interaction"):
                editor.interaction.connect(self.interactionChanged)
            body.addWidget(editor)
        else:
            for spec in control.params():
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)

                label = QLabel(spec.label)
                label.setMinimumWidth(70)
                if spec.tooltip:
                    label.setToolTip(spec.tooltip)
                row.addWidget(label, 0)

                widget = build_param_widget(spec, control.get(spec.name))
                widget.valueChanged.connect(self._on_param_changed)
                widget.interaction.connect(self.interactionChanged)
                self._param_widgets[spec.name] = widget
                row.addWidget(widget, 1)

                body.addLayout(row)

        self.toggled.connect(self._on_toggled)

    @property
    def custom_editor(self) -> Optional[QWidget]:
        """The control's custom editor widget, or ``None`` if it uses generic UI."""
        return self._custom_editor

    def _on_param_changed(self, name: str, value: object) -> None:
        self.paramChanged.emit(self._control.id, name, value)

    def _on_toggled(self, checked: bool) -> None:
        self.enabledChanged.emit(self._control.id, bool(checked))

    def refresh_from_control(self) -> None:
        """Push the control's current enabled + values back into the widgets."""
        blocked = self.blockSignals(True)
        self.setChecked(bool(self._control.enabled))
        self.blockSignals(blocked)
        if self._custom_editor is not None:
            if hasattr(self._custom_editor, "refresh"):
                self._custom_editor.refresh()
            return
        for name, widget in self._param_widgets.items():
            widget.set_value(self._control.get(name))


class ControlPanel(QWidget):
    """Dock contents: one :class:`ControlSection` per pipeline control, in order.

    The build loop is the extensibility proof: it never inspects a concrete
    control type, only iterates ``pipeline.controls()`` and wires up bubbled
    signals. Adding a control therefore needs no edits here.
    """

    paramChanged = Signal(str, str, object)  # (control_id, name, value)
    enabledChanged = Signal(str, bool)  # (control_id, enabled)
    interactionChanged = Signal(bool)  # bubbled slider pressed/released

    def __init__(
        self,
        pipeline: ControlPipeline,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._sections: List[ControlSection] = []
        self._sections_by_id: Dict[str, ControlSection] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        for control in pipeline.controls():
            section = ControlSection(control)
            section.paramChanged.connect(self.paramChanged)
            section.enabledChanged.connect(self.enabledChanged)
            section.interactionChanged.connect(self.interactionChanged)
            self._sections.append(section)
            self._sections_by_id[control.id] = section
            layout.addWidget(section)

        layout.addStretch(1)
        scroll.setWidget(container)

    def editor(self, control_id: str) -> Optional[QWidget]:
        """Return the custom editor widget for ``control_id`` (``None`` if generic/absent).

        Lets the window wire up control-specific interactions (e.g. the crop
        tool's interactive overlay) without the panel knowing any control type.
        """
        section = self._sections_by_id.get(control_id)
        return section.custom_editor if section is not None else None

    def reset_all(self) -> None:
        """Reset every control via the pipeline, then refresh all sections' widgets."""
        self._pipeline.reset()
        self.refresh_all()

    def refresh_all(self) -> None:
        """Re-sync every section's widgets from its control's current state."""
        for section in self._sections:
            section.refresh_from_control()
