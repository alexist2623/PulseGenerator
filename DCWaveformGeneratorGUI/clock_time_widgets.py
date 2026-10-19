"""Time spinbox: keep imported precision and commit user edits to clock ticks."""
from PyQt5 import QtCore, QtWidgets
try:
    from .clock_time import clock_time
except ImportError:
    from clock_time import clock_time


class ClockTimeSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self, *, rate_mhz, units_per_us=lambda: 1.0, parent=None):
        super().__init__(parent)
        self._rate_mhz = rate_mhz
        self._units_per_us = units_per_us
        self._exact_value = None
        self._display_value = None
        self._clock_commit = False
        self._range_units = float(self._units_per_us())
        self.setDecimals(9)
        self.setKeyboardTracking(False)
        self.valueChanged.connect(self._edited)
        self.editingFinished.connect(self.commit_clock_value)

    def _edited(self, _value):
        if not self._clock_commit:
            self._exact_value = None

    def setValue(self, value):
        previous = self.value()
        with QtCore.QSignalBlocker(self):
            # A unit change must scale the allowed physical range as well.
            # Otherwise a large us value is silently clamped when shown in ns.
            units = float(self._units_per_us())
            if units != self._range_units:
                ratio = units / self._range_units
                super().setRange(self.minimum() * ratio, self.maximum() * ratio)
                self._range_units = units
            self._exact_value = None
            super().setValue(value)
            self._display_value = super().value()
            self._exact_value = (float(value) if self.minimum() <= value <= self.maximum()
                                 else self._display_value)
        if self.value() != previous:
            self._emit_exact_value()

    def _emit_exact_value(self):
        was_committing = self._clock_commit
        self._clock_commit = True
        try:
            self.valueChanged.emit(self.value())
        finally:
            self._clock_commit = was_committing

    def value(self):
        displayed = super().value()
        if self._exact_value is not None and displayed == self._display_value:
            return self._exact_value
        return displayed

    def refresh_clock_step(self):
        self.setSingleStep(float(self._units_per_us()) / float(self._rate_mhz()))

    def commit_clock_value(self):
        value = clock_time(self.value(), self._rate_mhz(),
                           units_per_us=self._units_per_us(),
                           display_quantum=10.0 ** -self.decimals())
        previous = self.value()
        with QtCore.QSignalBlocker(self):
            self.setValue(value)
        self.refresh_clock_step()
        if self.value() != previous:
            self._emit_exact_value()

    def stepBy(self, steps):
        self.commit_clock_value()
        self.refresh_clock_step()
        value = self.value() + steps * self.singleStep()
        self.setValue(value)
        self.commit_clock_value()
