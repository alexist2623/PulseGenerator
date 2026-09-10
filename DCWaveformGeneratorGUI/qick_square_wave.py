"""Continuous QICK AWG square waves with connector-voltage calibration.

The existing QICK DC_Out gain/frequency/power database supplies the amplitude
scale. Its sine-tone power is converted to peak voltage, using the calibration
dBm reference impedance. The original DAC offset compensation remains separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite, sqrt
from threading import Event
import traceback

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

try:
    from .qick_qcodes_experiment import connect_qick
    from .power_calibration import CalibrationDatabase
except ImportError:
    from qick_qcodes_experiment import connect_qick
    from power_calibration import CalibrationDatabase


@dataclass(frozen=True)
class SquareWaveConfig:
    gen_ch: int = 1
    frequency_hz: float = 40_000.0
    amplitude_mv: float = 10.0
    offset_mv: float = 0.0
    duty_percent: float = 50.0
    calibration_database_path: str = ""
    calibration_run_id: int = 0
    calibration_reference_ohm: float = 50.0
    zero_code: float = 1120.0

    def __post_init__(self):
        if isinstance(self.gen_ch, bool) or not isinstance(self.gen_ch, int) or self.gen_ch < 0:
            raise ValueError("Generator channel must be a nonnegative integer")
        for name, value in asdict(self).items():
            if name == "calibration_database_path":
                if not isinstance(value, str):
                    raise ValueError("Calibration database path must be text")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.frequency_hz <= 0 or self.amplitude_mv <= 0:
            raise ValueError("Frequency and peak amplitude must be positive")
        if not 0 < self.duty_percent < 100:
            raise ValueError("Duty cycle must be between 0 and 100 percent")
        if not isinstance(self.calibration_run_id, int) or self.calibration_run_id < 0:
            raise ValueError("Calibration run ID must be a nonnegative integer")
        if self.calibration_reference_ohm <= 0:
            raise ValueError("Calibration dBm reference impedance must be positive")

    def codes(self, gen_cfg, calibration):
        quantum = 1 << int(gen_cfg.get("dac_invalid_lsb", 0))
        result = []
        for label, voltage in (("High", self.offset_mv + self.amplitude_mv),
                               ("Low", self.offset_mv - self.amplitude_mv)):
            value = voltage * calibration["codes_per_mv"] + self.zero_code
            if not isfinite(value):
                raise ValueError(f"{label} calibrated DAC code is not finite")
            code = int(round(value / quantum)) * quantum
            if not int(gen_cfg["minv"]) <= code <= int(gen_cfg["maxv"]):
                raise ValueError(f"{label} calibrated DAC code {code} exceeds the channel range")
            result.append(code)
        if result[0] == result[1]:
            raise ValueError("The two levels quantize to the same DAC code; increase amplitude")
        return tuple(result)


def normalize_square_wave_settings(settings=None):
    if settings is not None and not isinstance(settings, dict):
        raise TypeError("Square-wave settings must be a JSON object")
    values = dict(settings or {})
    # Discard the obsolete manual slope; loading old settings must not silently
    # bypass the required measured calibration.
    values.pop("codes_per_mv", None)
    return asdict(SquareWaveConfig(**values))


def resolve_square_wave_calibration(config, *, frequency_hz=None):
    """Use the same measured DC_Out run loader as the other QICK GUI tabs."""
    if not config.calibration_database_path.strip():
        raise ValueError("Select a DAC calibration database in this tab or the Calibration tab")
    frequency = config.frequency_hz if frequency_hz is None else float(frequency_hz)
    calibration = CalibrationDatabase(config.calibration_database_path).output_calibration(
        "DC_Out", [frequency / 1e6], run_id=config.calibration_run_id or None)
    # This is the power of the measured calibration SINE, not square-wave RMS
    # power. Convert sine RMS to peak voltage before mapping the SET levels.
    response_dbm = float(calibration.output_power_dbm(
        frequency / 1e6, calibration.reference_gain))
    reference_peak_mv = 1000 * sqrt(
        2 * config.calibration_reference_ohm * 10**((response_dbm - 30) / 10))
    if not isfinite(reference_peak_mv) or reference_peak_mv <= 0:
        raise ValueError("DAC calibration returned an invalid voltage response")
    return dict(database_path=str(calibration.summary.database_path),
                run_id=calibration.summary.run_id, board_type="DC_Out",
                sample_name=calibration.summary.sample_name,
                frequency_hz=frequency, reference_impedance_ohm=config.calibration_reference_ohm,
                response_dbm=response_dbm,
                codes_per_mv=calibration.reference_gain / reference_peak_mv)


def build_square_wave_program(soccfg, config, *, tproc_mhz=None):
    """Build an output-only ASM v1 loop; edge times do not depend on pulse caches."""
    from qick.asm_v1 import QickProgram

    if soccfg["tprocs"][0]["type"] != "axis_tproc64x32_x8":
        raise ValueError("Square-wave output requires QICK tProcessor v1")
    if config.gen_ch >= len(soccfg["gens"]):
        raise ValueError("Selected generator is absent from the loaded firmware")
    gen_cfg = soccfg["gens"][config.gen_ch]
    if gen_cfg.get("type") != "axis_awg_tuning_v1":
        raise ValueError("Select an axis_awg_tuning_v1 DAC generator")
    clock = float(soccfg["tprocs"][0]["f_time"] if tproc_mhz is None else tproc_mhz)
    if not isfinite(clock) or clock <= 0:
        raise ValueError("tProcessor clock must be positive and finite")
    period = int(round(clock * 1e6 / config.frequency_hz))
    high_ticks = int(round(period * config.duty_percent / 100.0))
    lead = 512
    if period < 512:
        raise ValueError(f"Frequency is too high; maximum is {clock * 1e6 / 512:g} Hz")
    if period + lead >= 2**31:
        raise ValueError("Period exceeds the tProcessor timestamp range")
    fabric_mhz = float(gen_cfg["f_fabric"])
    if min(high_ticks, period - high_ticks) * fabric_mhz / clock < 16:
        raise ValueError("Each level must last at least 16 DAC fabric clocks")
    calibration = resolve_square_wave_calibration(config, frequency_hz=clock * 1e6 / period)
    high_code, low_code = config.codes(gen_cfg, calibration)

    class SquareWaveProgram(QickProgram):
        def __init__(self):
            super().__init__(soccfg)
            # Match the shared Setup clock override used by the other QICK tabs.
            self.tproccfg = dict(self.tproccfg, f_time=clock)
            self.declare_gen(ch=config.gen_ch)
            self.synci(lead)
            self.label("SQUARE_FOREVER")
            # SET holds its value until the next SET. Explicit edge timestamps
            # avoid Python length-cache and runtime memri scheduling mismatches.
            self.set_pulse_registers(ch=config.gen_ch, style="awg_set", value=high_code, duration=1)
            self.pulse(ch=config.gen_ch, t=lead)
            self.set_pulse_registers(ch=config.gen_ch, style="awg_set", value=low_code, duration=1)
            self.pulse(ch=config.gen_ch, t=lead + high_ticks)
            # Bound look-ahead. Short waits also let the v1 stop API's END
            # overwrite be fetched at low frequencies instead of sitting in
            # one long WAIT until after the API reloads the program memory.
            # WAIT shares SET's FIFO, so pace on a port unused by this program.
            wait_port = (int(gen_cfg["tproc_ch"]) + 1) % 8
            self.regwi(0, 1, 0)
            self.safe_regwi(0, 2, period // 256 - 1)
            self.label("SQUARE_PACE")
            self.mathi(0, 1, 1, "+", 256)
            self.wait(0, wait_port, 1)
            self.loopnz(0, 2, "SQUARE_PACE")
            self.waiti(wait_port, period)
            self.synci(period)
            self.condj(0, 0, "==", 0, "SQUARE_FOREVER")
            self.summary = dict(requested_frequency_hz=config.frequency_hz,
                                actual_frequency_hz=clock * 1e6 / period,
                                period_cycles=period, high_cycles=high_ticks,
                                low_cycles=period-high_ticks, tproc_mhz=clock,
                                high_code=high_code, low_code=low_code,
                                actual_duty_percent=100.0*high_ticks/period,
                                calibration=calibration)

    program = SquareWaveProgram()
    program.compile()
    return program


class SquareWaveWorker(QtCore.QObject):
    """Own the Pyro connection on one thread until a confirmed stop."""
    started = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)
    stop_failed = QtCore.pyqtSignal(str)

    def __init__(self, connection, config, *, tproc_mhz, connector=None, program_factory=None):
        super().__init__()
        self.connection, self.config = connection, config
        self.tproc_mhz = tproc_mhz
        self.connector = connector or connect_qick
        self.program_factory = program_factory or build_square_wave_program
        self._stop = Event()

    def request_stop(self):
        # Called directly by the GUI: no queued slot while run() is waiting.
        self._stop.set()

    @QtCore.pyqtSlot()
    def run(self):
        soc = None
        touched_hardware = False
        error = None
        try:
            if self._stop.is_set():
                self.finished.emit("Start cancelled")
                return
            soc, soccfg = self.connector(self.connection)
            program = self.program_factory(soccfg, self.config, tproc_mhz=self.tproc_mhz)
            if self._stop.is_set():
                self.finished.emit("Start cancelled")
                return
            touched_hardware = True
            program.config_all(soc, reset=True)
            configure_dc = getattr(soc, "rfb_set_gen_dc", None)
            if callable(configure_dc):
                configure_dc(self.config.gen_ch)
            if not self._stop.is_set():
                soc.start_src("internal")
                soc.start_tproc()
                self.started.emit(program.summary)
            self._stop.wait()
        except Exception:
            error = traceback.format_exc()
        if touched_hardware:
            while True:
                self._stop.clear()
                try:
                    # lazy=True is a no-op for tProcessor v1 and must not be used.
                    soc.stop_tproc()
                    break
                except Exception:
                    self.stop_failed.emit(traceback.format_exc())
                    self._stop.wait()
        if error:
            self.failed.emit(error)
        else:
            self.finished.emit("Stopped: soc.stop_tproc() completed")


class SquareWavePanel(QtWidgets.QWidget):
    start_requested = QtCore.pyqtSignal(object)
    stop_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None, *, calibration_path_provider=None):
        super().__init__(parent)
        self._calibration_path_provider = calibration_path_provider or (lambda: "")
        layout = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        editor = QtWidgets.QVBoxLayout(content)
        editor.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll, 1)
        note = QtWidgets.QLabel("Continuous QICK DAC output. Uses the QICK connection and tProcessor clock in Setup.")
        note.setWordWrap(True)
        editor.addWidget(note)
        self.controls = QtWidgets.QGroupBox("Square wave")
        form = QtWidgets.QFormLayout(self.controls)
        self.gen_ch = QtWidgets.QSpinBox()
        self.gen_ch.setRange(0, 255)
        form.addRow("QICK generator channel", self.gen_ch)
        definitions = (("frequency_hz", "Frequency (Hz)", 0.2, 1e7, 6),
                       ("amplitude_mv", "Peak amplitude (mV)", 0.000001, 1e5, 6),
                       ("offset_mv", "Waveform offset (mV)", -1e5, 1e5, 6),
                       ("duty_percent", "High-level duty (%)", 0.001, 99.999, 3),
                       ("zero_code", "DAC offset compensation (codes)", -1e9, 1e9, 6),
                       ("calibration_reference_ohm", "Calibration dBm reference (ohm)", 0.001, 1e7, 3))
        for name, label, minimum, maximum, decimals in definitions:
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(minimum, maximum)
            widget.setDecimals(decimals)
            widget.setKeyboardTracking(False)
            setattr(self, name, widget)
            form.addRow(label, widget)
        self.calibration_database_path = QtWidgets.QLineEdit()
        self.calibration_database_path.setPlaceholderText("Use database from the Calibration tab")
        self.browse_calibration = QtWidgets.QPushButton("Browse...")
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self.calibration_database_path, 1)
        path_row.addWidget(self.browse_calibration)
        form.addRow("DAC calibration DB", path_row)
        self.calibration_run = QtWidgets.QComboBox()
        self.calibration_run.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.calibration_run.setMinimumContentsLength(14)
        self.calibration_run.addItem("Auto: latest DC_Out run covering frequency", 0)
        self.refresh_calibration = QtWidgets.QPushButton("Load / refresh")
        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(self.calibration_run, 1)
        run_row.addWidget(self.refresh_calibration)
        form.addRow("Measured DAC run", run_row)
        editor.addWidget(self.controls)
        self.calibration_note = QtWidgets.QLabel(
            "Amplitude calibration is loaded from the measured QICK DC_Out data. "
            "Peak amplitude is half the peak-to-peak voltage. The original +1120-code "
            "offset compensation is separate: power measurements do not determine DC offset. "
            "The dBm reference and output load must match the calibration measurement. "
            "Sine-tone gain calibration does not correct square-wave harmonics or settling.")
        self.calibration_note.setWordWrap(True)
        editor.addWidget(self.calibration_note)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Time", units="us")
        self.plot.setLabel("left", "Requested voltage", units="mV")
        self.plot.setMinimumHeight(160)
        self.plot.setMaximumHeight(280)
        self.curve = self.plot.plot(pen=pg.mkPen("#338bd6", width=2))
        editor.addWidget(self.plot, 1)
        row = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start continuous output")
        self.stop_button = QtWidgets.QPushButton("Stop")
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        layout.addLayout(row)
        self.status = QtWidgets.QLabel("Ready")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        stop_note = QtWidgets.QLabel("Stop halts the tProcessor. The DAC may retain its last level after queued commands finish; it is not a zero-voltage command.")
        stop_note.setWordWrap(True)
        layout.addWidget(stop_note)
        self.load_settings({})
        for name in ("gen_ch", "frequency_hz", "amplitude_mv", "offset_mv", "duty_percent",
                     "zero_code", "calibration_reference_ohm"):
            getattr(self, name).valueChanged.connect(self.update_preview)
            getattr(self, name).valueChanged.connect(self._mark_calibration_stale)
        self.calibration_database_path.textChanged.connect(self._database_changed)
        self.calibration_run.currentIndexChanged.connect(self._apply_calibration)
        self.browse_calibration.clicked.connect(self._browse_calibration)
        self.refresh_calibration.clicked.connect(self._refresh_calibration)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.set_running(False)

    def settings_dict(self):
        values = {name: getattr(self, name).value() for name in
                  ("gen_ch", "frequency_hz", "amplitude_mv", "offset_mv", "duty_percent",
                   "zero_code", "calibration_reference_ohm")}
        values.update(calibration_database_path=self.calibration_database_path.text().strip(),
                      calibration_run_id=int(self.calibration_run.currentData() or 0))
        return normalize_square_wave_settings(values)

    def resolved_config(self):
        config = SquareWaveConfig(**self.settings_dict())
        return replace(config, calibration_database_path=(config.calibration_database_path
                       or str(self._calibration_path_provider()).strip()))

    def load_settings(self, settings):
        values = normalize_square_wave_settings(settings)
        path = values.pop("calibration_database_path")
        run_id = values.pop("calibration_run_id")
        for name, value in values.items():
            widget = getattr(self, name)
            if not widget.minimum() <= value <= widget.maximum():
                raise ValueError(f"{name} is outside the square-wave control range")
        for name, value in values.items():
            widget = getattr(self, name)
            with QtCore.QSignalBlocker(widget):
                widget.setValue(value)
        with QtCore.QSignalBlocker(self.calibration_database_path):
            self.calibration_database_path.setText(path)
        with QtCore.QSignalBlocker(self.calibration_run):
            self.calibration_run.clear()
            self.calibration_run.addItem("Auto: latest DC_Out run covering frequency", 0)
            if run_id:
                self.calibration_run.addItem(f"Run {run_id} (load to inspect)", run_id)
                self.calibration_run.setCurrentIndex(1)
        self.update_preview()
        self._mark_calibration_stale()

    def _mark_calibration_stale(self, *_args):
        if self.controls.isEnabled():
            self.status.setText("Calibration will be reloaded and validated before output starts.")

    def shared_database_changed(self, *_args):
        if not self.calibration_database_path.text().strip():
            self._database_changed()

    def _database_changed(self):
        with QtCore.QSignalBlocker(self.calibration_run):
            self.calibration_run.clear()
            self.calibration_run.addItem("Auto: latest DC_Out run covering frequency", 0)
        self._mark_calibration_stale()

    def _browse_calibration(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose DAC measurement calibration database",
            self.resolved_config().calibration_database_path,
            "Calibration databases (*.db *.sqlite *.sqlite3);;All files (*)")
        if path:
            self.calibration_database_path.setText(path)
            self._refresh_calibration()

    def _refresh_calibration(self):
        try:
            config = self.resolved_config()
            candidates = CalibrationDatabase(config.calibration_database_path).output_calibration_candidates(
                "DC_Out", [config.frequency_hz / 1e6])
            with QtCore.QSignalBlocker(self.calibration_run):
                self.calibration_run.clear()
                self.calibration_run.addItem("Auto: latest DC_Out run covering frequency", 0)
                for candidate in candidates:
                    record = candidate.summary
                    if record.board_type != "DC_Out":
                        continue
                    label = (f"Run {record.run_id} | {record.sample_name} | "
                             f"{record.frequency_min_mhz:g}..{record.frequency_max_mhz:g} MHz")
                    if not candidate.full_frequency_coverage:
                        label += " | outside frequency range"
                    self.calibration_run.addItem(label, record.run_id)
                    self.calibration_run.setItemData(self.calibration_run.count()-1, label, QtCore.Qt.ToolTipRole)
                index = self.calibration_run.findData(config.calibration_run_id)
                if index < 0:
                    self.calibration_run.addItem(f"Run {config.calibration_run_id} (not a DAC run in this DB)",
                                                 config.calibration_run_id)
                    index = self.calibration_run.count()-1
                self.calibration_run.setCurrentIndex(index)
            self._apply_calibration()
        except Exception as exc:
            self.status.setText(f"Cannot load DAC calibration: {exc}")

    def _apply_calibration(self, *_args):
        try:
            result = resolve_square_wave_calibration(self.resolved_config())
            self.status.setText(f"Loaded DC_Out Run {result['run_id']}: "
                                f"{result['codes_per_mv']:.9g} codes/mV at "
                                f"{result['frequency_hz']:g} Hz (from measured data).")
        except Exception as exc:
            self.status.setText(f"Cannot load DAC calibration: {exc}")

    def update_preview(self):
        period = 1e6 / self.frequency_hz.value()
        high_time = period * self.duty_percent.value() / 100
        high = self.offset_mv.value() + self.amplitude_mv.value()
        low = self.offset_mv.value() - self.amplitude_mv.value()
        self.curve.setData([0, high_time, high_time, period, period,
                            period+high_time, period+high_time, 2*period],
                           [high, high, low, low, high, high, low, low])

    def _start(self):
        try:
            self.start_requested.emit(self.resolved_config())
        except (TypeError, ValueError) as exc:
            self.status.setText(str(exc))

    def set_running(self, running, message=None):
        self.controls.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if message is not None:
            self.status.setText(message)
