# Copyright 2026 Vinay Williams

"""Custom editor widget for :class:`ValuesControl`.

Renders the Values control's generic parameter rows (reusing the panel's
``ParamWidget``) but replaces the Monochromatic *mono colour* row with a proper
colour chooser: a dropdown of built-in preset pigments plus the painter's own
"My Paints" tubes, a system colour picker for one-off custom hues, and a button
to save a picked colour into My Paints so it is remembered across sessions.

The effective colour is stored back into the control as a ``#rrggbb`` string via
the standard ``paramChanged`` signal, so the worker snapshot stays self-contained
(no registry lookup on the worker thread). Saving to My Paints is surfaced to the
window via :attr:`paintsChanged`; the window owns the inventory and its
persistence, matching how the grid editor's canvas positions are pushed in.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from painting_assist.controls.values import ValuesControl

RGB = Tuple[int, int, int]


def _swatch_icon(rgb: RGB, size: int = 18) -> QIcon:
    """Return a small solid-colour icon for a colour swatch."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return QIcon(pixmap)


class ValuesEditor(QWidget):
    """Values param rows with a My-Paints-aware colour chooser for mono mode.

    Emits :attr:`paramChanged` ``(name, value)`` and :attr:`interaction`
    ``(bool)`` per the custom-editor contract, plus :attr:`paintsChanged`
    ``(list)`` when the painter saves a picked colour into their paint inventory
    (the window persists it and pushes the updated list back via
    :meth:`set_paints`).
    """

    paramChanged = Signal(str, object)
    interaction = Signal(bool)
    paintsChanged = Signal(object)  # list[(name, (r, g, b))]

    def __init__(self, control, parent=None):
        super().__init__(parent)
        self._control = control
        self._param_widgets = {}
        self._paints: List[Tuple[str, RGB]] = []
        self._rebuilding = False

        from painting_assist.widgets.control_panel import build_param_widget

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        for spec in control.params():
            if spec.name == "mono_hex":
                root.addLayout(self._build_colour_row())
                continue
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            label = QLabel(spec.label)
            label.setMinimumWidth(70)
            if spec.tooltip:
                label.setToolTip(spec.tooltip)
            row.addWidget(label, 0)
            widget = build_param_widget(spec, control.get(spec.name))
            widget.valueChanged.connect(self.paramChanged)
            widget.interaction.connect(self.interaction)
            self._param_widgets[spec.name] = widget
            row.addWidget(widget, 1)
            root.addLayout(row)

        self._rebuild_colour_combo()

    # ------------------------------------------------------------------ #
    # Colour row
    # ------------------------------------------------------------------ #
    def _build_colour_row(self) -> QVBoxLayout:
        """Build the mono-colour chooser (label + combo, then the two buttons)."""
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        label = QLabel("Mono colour")
        label.setMinimumWidth(70)
        label.setToolTip(
            "Monochromatic mode: the single pigment hue the value study is "
            "stained with (a preset, one of your paints, or a custom colour)."
        )
        top.addWidget(label, 0)

        self._combo = QComboBox()
        self._combo.setIconSize(QSize(18, 18))
        self._combo.activated.connect(self._on_combo_activated)
        top.addWidget(self._combo, 1)
        col.addLayout(top)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        custom_btn = QPushButton("Custom…")
        custom_btn.setToolTip("Pick any colour from the system colour dialog")
        custom_btn.clicked.connect(self._on_pick_custom)
        save_btn = QPushButton("Save to My Paints")
        save_btn.setToolTip("Remember the current colour as one of your paint tubes")
        save_btn.clicked.connect(self._on_save_to_paints)
        buttons.addStretch(1)
        buttons.addWidget(custom_btn)
        buttons.addWidget(save_btn)
        col.addLayout(buttons)
        return col

    def _current_hex(self) -> str:
        """Current mono colour as a normalised ``#rrggbb`` string."""
        return ValuesControl.rgb_to_hex(
            ValuesControl.parse_hex(self._control.get("mono_hex"))
        )

    def _add_header(self, text: str) -> None:
        """Append a non-selectable section header to the combo."""
        self._combo.addItem(text)
        item = self._combo.model().item(self._combo.count() - 1)
        item.setEnabled(False)

    def _add_colour(self, name: str, rgb: RGB) -> None:
        """Append a selectable colour entry carrying its hex in UserRole."""
        self._combo.addItem(_swatch_icon(rgb), name)
        self._combo.setItemData(
            self._combo.count() - 1, ValuesControl.rgb_to_hex(rgb), Qt.UserRole
        )

    def _rebuild_colour_combo(self) -> None:
        """Repopulate presets + My Paints (+ a custom entry) and select the current hex.

        Purely programmatic: guarded by ``_rebuilding`` so it never emits
        ``paramChanged`` while syncing the widget to the control's state.
        """
        self._rebuilding = True
        try:
            self._combo.clear()
            current = self._current_hex()
            select_index = -1

            self._add_header("Presets")
            for name, rgb in ValuesControl.MONO_PRESETS:
                self._add_colour(name, rgb)
                if (
                    self._combo.itemData(self._combo.count() - 1, Qt.UserRole)
                    == current
                ):
                    select_index = self._combo.count() - 1

            if self._paints:
                self._add_header("My Paints")
                for name, rgb in self._paints:
                    self._add_colour(name, rgb)
                    data = self._combo.itemData(self._combo.count() - 1, Qt.UserRole)
                    if select_index < 0 and data == current:
                        select_index = self._combo.count() - 1

            # A one-off custom colour that matches no preset/tube gets its own entry.
            if select_index < 0:
                self._add_header("Custom")
                self._add_colour(
                    "Custom (%s)" % current, ValuesControl.parse_hex(current)
                )
                select_index = self._combo.count() - 1

            self._combo.setCurrentIndex(select_index)
        finally:
            self._rebuilding = False

    def _set_colour(self, rgb: RGB) -> None:
        """Push a chosen colour to the control (hex) and rebuild the combo."""
        hex_value = ValuesControl.rgb_to_hex(rgb)
        self._control.set("mono_hex", hex_value)
        self._rebuild_colour_combo()
        self.paramChanged.emit("mono_hex", hex_value)

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _on_combo_activated(self, index: int) -> None:
        if self._rebuilding:
            return
        hex_value = self._combo.itemData(index, Qt.UserRole)
        if not hex_value:  # a header row (shouldn't fire, headers are disabled)
            return
        self._control.set("mono_hex", hex_value)
        # Rebuild so any stale one-off "Custom" entry drops once a preset/tube is
        # chosen; the guard stops this from re-emitting paramChanged.
        self._rebuild_colour_combo()
        self.paramChanged.emit("mono_hex", hex_value)

    def _on_pick_custom(self) -> None:
        initial = ValuesControl.parse_hex(self._control.get("mono_hex"))
        colour = QColorDialog.getColor(
            QColor(*initial), self, "Choose the monochrome colour"
        )
        if colour.isValid():
            self._set_colour((colour.red(), colour.green(), colour.blue()))

    def _on_save_to_paints(self) -> None:
        rgb = ValuesControl.parse_hex(self._control.get("mono_hex"))
        name, ok = QInputDialog.getText(self, "Save to My Paints", "Paint name:")
        name = name.strip()
        if not ok or not name:
            return
        self._paints = list(self._paints) + [(name, rgb)]
        self._rebuild_colour_combo()  # so it now shows under My Paints, selected
        self.paintsChanged.emit(self._paints)

    # ------------------------------------------------------------------ #
    # Window-driven updates
    # ------------------------------------------------------------------ #
    def set_paints(self, paints: Optional[List[Tuple[str, RGB]]]) -> None:
        """Set the My Paints inventory shown in the colour dropdown."""
        self._paints = [
            (str(name), (int(r), int(g), int(b))) for name, (r, g, b) in (paints or [])
        ]
        self._rebuild_colour_combo()

    def refresh(self) -> None:
        """Re-read every param from the control into its widget (no signals emitted)."""
        for name, widget in self._param_widgets.items():
            widget.set_value(self._control.get(name))
        self._rebuild_colour_combo()
