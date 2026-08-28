"""Pandas-backed Qt table model."""
from __future__ import annotations
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
import pandas as pd

class PandasTableModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None):
        super().__init__(); self._df = df if df is not None else pd.DataFrame()
    def set_dataframe(self, df: pd.DataFrame) -> None:
        # Results are immutable after the worker emits them. Keeping that frame
        # avoids a large, synchronous copy of hundreds of report columns in the
        # GUI thread at the end of every run.
        self.beginResetModel(); self._df = df; self.endResetModel()
    def rowCount(self, parent=QModelIndex()): return 0 if parent.isValid() else len(self._df)
    def columnCount(self, parent=QModelIndex()): return 0 if parent.isValid() else len(self._df.columns)
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.EditRole): return None
        value = self._df.iat[index.row(), index.column()]
        return "" if pd.isna(value) else str(value)
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole: return None
        return str(self._df.columns[section]) if orientation == Qt.Horizontal else str(section + 1)
    def sort(self, column, order=Qt.AscendingOrder):
        if self._df.empty: return
        self.layoutAboutToBeChanged.emit(); self._df = self._df.sort_values(self._df.columns[column], ascending=order == Qt.AscendingOrder).reset_index(drop=True); self.layoutChanged.emit()
    @property
    def dataframe(self): return self._df.copy()
