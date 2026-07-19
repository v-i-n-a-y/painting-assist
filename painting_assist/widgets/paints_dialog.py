# Copyright 2026 Vinay Williams

"""Dialog for managing the painter's paint-tube inventory ("My Paints").

The inventory is a plain list of ``(name, (r, g, b))`` tubes. This dialog only
edits that list and hands it back via :meth:`paints`; persistence (QSettings)
lives in the main window, keeping this widget a self-contained editor. Tubes can
be entered by hand (name plus a colour picked from the system colour dialog) or
chosen from a built-in catalogue of common artist pigments.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from painting_assist.paints import DEFAULT_CATALOGUE

RGB = Tuple[int, int, int]


def _swatch_icon(rgb: RGB) -> QIcon:
    """Return a small solid-colour icon for a paint swatch."""
    pixmap = QPixmap(24, 24)
    pixmap.fill(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return QIcon(pixmap)


class PaintsDialog(QDialog):
    """Add, edit and remove the paint tubes the painter owns."""

    def __init__(
        self,
        paints: Optional[List[Tuple[str, RGB]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("My Paints")
        self.setModal(True)
        self.resize(360, 420)

        self._list = QListWidget()
        self._list.setIconSize(QSize(24, 24))
        for name, rgb in paints or []:
            self._add_item(str(name), tuple(int(c) for c in rgb))
        self._list.itemDoubleClicked.connect(lambda _item: self._edit_selected())

        add_btn = QPushButton("Add…")
        add_btn.clicked.connect(self._add_custom)
        catalogue_btn = QPushButton("Add from catalogue…")
        catalogue_btn.clicked.connect(self._add_from_catalogue)
        edit_btn = QPushButton("Edit…")
        edit_btn.clicked.connect(self._edit_selected)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_selected)

        button_col = QVBoxLayout()
        for widget in (add_btn, catalogue_btn, edit_btn, remove_btn):
            button_col.addWidget(widget)
        button_col.addStretch(1)

        row = QHBoxLayout()
        row.addWidget(self._list, 1)
        row.addLayout(button_col)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(row, 1)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------ #
    # List helpers
    # ------------------------------------------------------------------ #
    def _add_item(self, name: str, rgb: RGB) -> None:
        item = QListWidgetItem(_swatch_icon(rgb), name)
        item.setData(Qt.UserRole, rgb)
        self._list.addItem(item)

    def _pick_colour(self, initial: RGB) -> Optional[RGB]:
        colour = QColorDialog.getColor(
            QColor(*initial), self, "Choose the paint's colour"
        )
        if not colour.isValid():
            return None
        return (colour.red(), colour.green(), colour.blue())

    def _add_custom(self) -> None:
        name, ok = QInputDialog.getText(self, "Add paint", "Paint name:")
        name = name.strip()
        if not ok or not name:
            return
        rgb = self._pick_colour((200, 200, 200))
        if rgb is None:
            return
        self._add_item(name, rgb)

    def _add_from_catalogue(self) -> None:
        names = [name for name, _rgb in DEFAULT_CATALOGUE]
        name, ok = QInputDialog.getItem(
            self, "Add from catalogue", "Pigment:", names, 0, False
        )
        if not ok or not name:
            return
        for cat_name, rgb in DEFAULT_CATALOGUE:
            if cat_name == name:
                self._add_item(cat_name, tuple(int(c) for c in rgb))
                return

    def _edit_selected(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        name, ok = QInputDialog.getText(
            self, "Edit paint", "Paint name:", text=item.text()
        )
        name = name.strip()
        if not ok or not name:
            return
        rgb = self._pick_colour(item.data(Qt.UserRole))
        if rgb is None:
            return
        item.setText(name)
        item.setIcon(_swatch_icon(rgb))
        item.setData(Qt.UserRole, rgb)

    def _remove_selected(self) -> None:
        row = self._list.currentRow()
        if row >= 0:
            self._list.takeItem(row)

    # ------------------------------------------------------------------ #
    # Result
    # ------------------------------------------------------------------ #
    def paints(self) -> List[Tuple[str, RGB]]:
        """Return the current tube list as ``(name, (r, g, b))`` tuples."""
        out: List[Tuple[str, RGB]] = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            rgb = item.data(Qt.UserRole)
            out.append((item.text(), (int(rgb[0]), int(rgb[1]), int(rgb[2]))))
        return out
