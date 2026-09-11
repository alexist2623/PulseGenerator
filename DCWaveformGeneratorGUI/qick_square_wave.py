"""Continuous QICK AWG square waves using a configured output voltage range."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from threading import Event
import traceback

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

try:
    from .qick_qcodes_experiment import connect_qick
    from .dc_waveform_core import DEFAULT_QICK_FULL_SCALE_MV
    from .qick_fine_tune_sweep import normalized_to_dac
except ImportError:
    from qick_qcodes_experiment import connect_qick
    from dc_waveform_core import DEFAULT_QICK_FULL_SCALE_MV
    from qick_fine_tune_sweep import normalized_to_dac


@dataclass(frozen=True)
class SquareWaveConfig:
    gen_ch: int = 1
    frequency_hz: float = 40_000.0
    amplitude_mv: float = 10.0
    offset_mv: float = 0.0
    duty_percent: float = 50.0
    full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV
    zero_code: float = 1120.0

    def __post_init__(self):
        if isinstance(self.gen_ch, bool) or not isinstance(self.gen_ch, int) or self.gen_ch < 0:
            raise ValueError("Generator channel must be a nonnegative integer")
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.frequency_hz <= 0 or self.amplitude_mv <= 0:
            raise ValueError("Frequency and peak amplitude must be positive")
        if not 0 < self.duty_percent < 100:
            raise ValueError("Duty cycle must be between 0 and 100 percent")
        if self.full_scale_mv <= 0:
            raise ValueError("Output full scale must be positive")

    def codes(self, gen_cfg):
        invalid_lsb = int(gen_cfg.get("dac_invalid_lsb", 0))
        quantum = 1 << invalid_lsb
        min_code, max_code = int(gen_cfg["minv"]), int(gen_cfg["maxv"])
        full_scale_code = max(abs(min_code), max_code + quantum)
        result = []
        for label, voltage in (("High", self.offset_mv + self.amplitude_mv),
                               ("Low", self.offset_mv - self.amplitude_mv)):
            if abs(voltage) > self.full_scale_mv:
                raise ValueError(f"{label} voltage exceeds +/-{self.full_scale_mv:g} mV output full scale")
            # Use the AWG Tuning conversion, adding offset before quantization.
            amplitude = voltage / self.full_scale_mv + self.zero_code / full_scale_code
            if not isfinite(amplitude) or not -1.0 <= amplitude <= 1.0:
                raise ValueError(f"{label} DAC level including offset compensation exceeds the channel range")
            code = normalized_to_dac(amplitude, min_code=min_code,
                                     max_code=max_code, invalid_lsb=invalid_lsb)
            result.append(code)
        if result[0] == result[1]:
            raise ValueError("The two levels quantize to the same DAC code; increase amplitude")
        return tuple(result)


def normalize_square_wave_settings(settings=None):
    if settings is not None and not isinstance(settings, dict):
        raise TypeError("Square-wave settings must be a JSON object")
    values = dict(settings or {})
    # Older square-wave settings used measured power or a manual slope.
    # Preserve waveform values while migrating to the explicit output range.
    for obsolete in ("codes_per_mv", "calibration_database_path",
                     "calibration_run_id", "calibration_reference_ohm"):
        values.pop(obsolete, None)
    return asdict(SquareWaveConfig(**values))


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
    high_code, low_code = config.codes(gen_cfg)

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
                                full_scale_mv=config.full_scale_mv)

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

    def __init__(self, parent=None):
        super().__init__(parent)
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
                       ("full_scale_mv", "Maximum output (+/- mV)", 1.0, 1e6, 6))
        for name, label, minimum, maximum, decimals in definitions:
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(minimum, maximum)
            widget.setDecimals(decimals)
            widget.setKeyboardTracking(False)
            setattr(self, name, widget)
            form.addRow(label, widget)
        editor.addWidget(self.controls)
        self.output_note = QtWidgets.QLabel(
            "Maximum output sets the +/- voltage range, as in AWG Tuning. "
            "Peak amplitude is half the peak-to-peak voltage; High/Low = waveform "
            "offset +/- peak amplitude. DAC offset compensation is added separately. "
            "This tab uses its own maximum output setting.")
        self.output_note.setWordWrap(True)
        editor.addWidget(self.output_note)
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
                     "zero_code", "full_scale_mv"):
            getattr(self, name).valueChanged.connect(self.update_preview)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.set_running(False)

    def settings_dict(self):
        values = {name: getattr(self, name).value() for name in
                  ("gen_ch", "frequency_hz", "amplitude_mv", "offset_mv", "duty_percent",
                   "zero_code", "full_scale_mv")}
        return normalize_square_wave_settings(values)

    def resolved_config(self):
        return SquareWaveConfig(**self.settings_dict())

    def load_settings(self, settings):
        values = normalize_square_wave_settings(settings)
        for name, value in values.items():
            widget = getattr(self, name)
            if not widget.minimum() <= value <= widget.maximum():
                raise ValueError(f"{name} is outside the square-wave control range")
        for name, value in values.items():
            widget = getattr(self, name)
            with QtCore.QSignalBlocker(widget):
                widget.setValue(value)
        self.update_preview()

    def update_preview(self):
        period = 1e6 / self.frequency_hz.value()
        high_time = period * self.duty_percent.value() / 100
        high = self.offset_mv.value() + self.amplitude_mv.value()
        low = self.offset_mv.value() - self.amplitude_mv.value()
        self.curve.setData([0, high_time, high_time, period, period,
                            period+high_time, period+high_time, 2*period],
                           [high, high, low, low, high, high, low, low])
        if self.controls.isEnabled():
            full_scale = self.full_scale_mv.value()
            if max(abs(high), abs(low)) > full_scale:
                self.status.setText(f"High/Low voltage exceeds +/-{full_scale:g} mV maximum output.")
            else:
                self.status.setText(f"Ready: maximum output +/-{full_scale:g} mV; High/Low {high:g} / {low:g} mV.")

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
