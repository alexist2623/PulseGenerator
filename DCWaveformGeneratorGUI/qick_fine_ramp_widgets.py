"""V7 edge-position editor. Creating the dialog performs no hardware I/O."""
from PyQt5 import QtCore, QtWidgets


class FineRampDialog(QtWidgets.QDialog):
    def __init__(self, pulse, row, fabric_mhz, parent=None):
        super().__init__(parent)
        self.setWindowTitle('RAMP v7 — sample positions')
        self.pulse, self.row, self.rate = pulse, row, float(fabric_mhz)
        from clock_time import clock_cycles
        self.capacity = clock_cycles(pulse.t[2*row]-pulse.t[2*row-1], self.rate, units_per_us=1000.) * 16
        self.dt = 1000. / (self.rate * 16)
        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(
            f'1 sample = {self.dt:.9g} ns; reserved RAMP = {self.capacity} samples.\n'
            'Positions are relative to this RAMP. The next SET defines the final voltage.\n'
            'Either edge can rise or fall. The second edge may cross a word boundary.\n'
            'Fine mode: fixed edge positions/voltages; HOLD/RF sweeps remain available.')
        info.setWordWrap(True)
        layout.addWidget(info)
        self.enabled = QtWidgets.QCheckBox('Enable sample-position RAMP (v7)')
        self.enabled.setChecked(row in pulse.fine_ramps)
        layout.addWidget(self.enabled)
        grid = QtWidgets.QGridLayout()
        for c, title in enumerate(('Edge position', 'Sample index', 'Time [ns]')):
            grid.addWidget(QtWidgets.QLabel(title), 0, c)
        self.samples, self.times = {}, {}
        old = pulse.fine_ramps.get(row, {})
        defaults = dict(rise_start=0, rise_end=min(5, self.capacity-1),
                        fall_start=min(10, self.capacity-2), fall_end=self.capacity-1)
        for r, key in enumerate(defaults, 1):
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, min(15 if key == 'rise_start' else 16383 if key == 'fall_start' else (1<<23)-2, self.capacity-1))
            spin.setValue(old.get(key, defaults[key]))
            ns = QtWidgets.QDoubleSpinBox()
            ns.setDecimals(9); ns.setRange(0, spin.maximum()*self.dt)
            ns.setSingleStep(self.dt); ns.setValue(spin.value()*self.dt)
            ns.setKeyboardTracking(False)
            self.samples[key], self.times[key] = spin, ns
            spin.valueChanged.connect(lambda value, target=ns: self._set_time(target, value))
            ns.valueChanged.connect(lambda value, key=key: self._from_time(key, value))
            grid.addWidget(QtWidgets.QLabel(key.replace('_', ' ').title()), r, 0)
            grid.addWidget(spin, r, 1); grid.addWidget(ns, r, 2)
        layout.addLayout(grid)
        self.second = QtWidgets.QCheckBox('Enable second edge')
        self.second.setChecked(old.get('fall_start') is not None)
        layout.addWidget(self.second)
        form = QtWidgets.QFormLayout()
        self.turn = QtWidgets.QDoubleSpinBox()
        self.turn.setDecimals(6); self.turn.setRange(*pulse.v_bounds)
        self.turn.setValue(old.get('turn', pulse.v[2*row]))
        form.addRow('First-edge target [mV]', self.turn)
        layout.addLayout(form)
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.second.toggled.connect(self._enabled_state)
        self.enabled.toggled.connect(self._enabled_state)
        self._enabled_state()

    def _set_time(self, target, sample):
        with QtCore.QSignalBlocker(target):
            target.setValue(sample*self.dt)

    def _from_time(self, key, value):
        self.samples[key].setValue(int(value/self.dt+.5))
        self._set_time(self.times[key], self.samples[key].value())

    def _enabled_state(self):
        for key in self.samples:
            enabled = self.enabled.isChecked() and ('fall' not in key or self.second.isChecked())
            self.samples[key].setEnabled(enabled); self.times[key].setEnabled(enabled)
        self.second.setEnabled(self.enabled.isChecked())
        self.turn.setEnabled(self.enabled.isChecked() and self.second.isChecked())

    def specification(self):
        if not self.enabled.isChecked():
            return None
        value = {key: spin.value() for key, spin in self.samples.items()
                 if 'fall' not in key or self.second.isChecked()}
        if self.second.isChecked():
            value['turn'] = self.turn.value()
        return value

    def accept(self):
        try:
            # Validate a copy; Cancel never mutates the waveform.
            copy = self.pulse.copy()
            copy.set_fine_ramp(self.row, self.specification(), self.rate)
        except (ValueError, TypeError, KeyError) as exc:
            self.error.setText(str(exc))
            return
        super().accept()
