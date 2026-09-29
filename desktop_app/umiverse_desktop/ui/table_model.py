from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QFontMetrics


class DataFrameTableModel(QAbstractTableModel):
    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        super().__init__()
        self.frame = frame if frame is not None else pd.DataFrame()

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self.frame = frame.copy()
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        value = self.frame.iat[index.row(), index.column()]
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self.frame.columns[section]) if section < len(self.frame.columns) else ""
        return str(section + 1)

    def column_widths(self, font_metrics: QFontMetrics, max_rows: int = 80) -> list[int]:
        widths: list[int] = []
        sample = self.frame.head(max_rows)
        for column in self.frame.columns:
            header_width = font_metrics.horizontalAdvance(str(column)) + 34
            value_width = 0
            if not sample.empty:
                value_width = max(
                    font_metrics.horizontalAdvance(self._format_width_value(value)) + 24
                    for value in sample[column].tolist()
                )
            widths.append(max(96, min(360, max(header_width, value_width))))
        return widths

    @staticmethod
    def _format_width_value(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
