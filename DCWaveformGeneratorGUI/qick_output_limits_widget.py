"""Qt editor for per-generator output limits; hardware logic is Qt-free."""
from copy import deepcopy
from PyQt5 import QtCore, QtWidgets
try:
    from .qick_output_limits import normalize_output_limits
except ImportError:
    from qick_output_limits import normalize_output_limits


class OutputLimitsWidget(QtWidgets.QGroupBox):
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__('AWG output limits', parent)
        self._saved = {}
        self._rows = {}
        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel('After DC/RC compensation, before RF addition. '
                               'Limits include 0 mV for stop/idle. Clipping can alter DC balance.')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.table = QtWidgets.QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(['AWG gen', 'Enable', 'Minimum (mV)', 'Maximum (mV)'])
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setMinimumHeight(125)
        self.table.setMaximumHeight(220)
        layout.addWidget(self.table)

    def _remember(self):
        for channel, (enabled, minimum, maximum) in self._rows.items():
            self._saved[str(channel)] = dict(enabled=enabled.isChecked(),
                min_mv=minimum.value(), max_mv=maximum.value())

    def set_channels(self, channels):
        channels = tuple(dict.fromkeys(int(ch) for ch in channels))
        if channels == tuple(self._rows):
            return
        self._remember()
        self._rows = {}
        self.table.setRowCount(len(channels))
        for row, channel in enumerate(channels):
            value = self._saved.get(str(channel), dict(enabled=False, min_mv=-800.0, max_mv=800.0))
            label = QtWidgets.QTableWidgetItem(str(channel))
            label.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table.setItem(row, 0, label)
            enabled = QtWidgets.QCheckBox()
            enabled.setChecked(value['enabled'])
            self.table.setCellWidget(row, 1, enabled)
            spins = []
            for col, name, low, high in ((2, 'min_mv', -1e6, 0), (3, 'max_mv', 0, 1e6)):
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(low, high)
                spin.setDecimals(6)
                spin.setValue(value[name])
                spin.setEnabled(value['enabled'])
                spin.valueChanged.connect(self.changed.emit)
                enabled.toggled.connect(spin.setEnabled)
                self.table.setCellWidget(row, col, spin)
                spins.append(spin)
            enabled.toggled.connect(self.changed.emit)
            self._rows[channel] = (enabled, *spins)

    def settings(self, channels=None):
        self._remember()
        selected = self._rows if channels is None else channels
        return normalize_output_limits({str(ch): deepcopy(self._saved[str(ch)])
                                        for ch in selected if str(ch) in self._saved})

    def load_settings(self, settings):
        values = normalize_output_limits(settings)
        channels = tuple(self._rows)
        self._rows = {}
        self._saved = values
        self.set_channels(channels)
