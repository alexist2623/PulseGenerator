"""Current-noise analysis for QICK I traces.

The panel mirrors the common laboratory workflow::

    current = input_trace * input_scale / transimpedance_gain
    ASD = sqrt(periodogram(current, scaling="density"))

It accepts the most recent QICK Experiment result or a saved QCoDeS run.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import sqlite3
import traceback
from typing import Any, Mapping, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

try:
    import pyqtgraph as pg
except ImportError:
    pg = None

try:
    from .qick_qcodes_experiment import load_qick_iq_arrays
except ImportError:
    from qick_qcodes_experiment import load_qick_iq_arrays


INPUT_MODES = ("voltage", "adc", "current")
WINDOWS = ("flattop", "hann", "blackmanharris", "boxcar")
DETREND_MODES = ("constant", "linear", "none")
DEFAULT_NOISE_ANALYSIS_SETTINGS = {
    "database_path": str(Path.home() / "qick_experiments.db"),
    "run_id": 0,
    "point_index": 0,
    "repetition_index": 0,
    "input_mode": "voltage",
    "transimpedance_gain_v_per_a": 1.0e8,
    "input_scale_v_per_unit": 1.0,
    "sample_rate_hz": 1.0e6,
    "window": "flattop",
    "detrend": "constant",
    "nfft": 0,
}


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def normalize_noise_analysis_settings(
    values: Optional[Mapping[str, Any]],
) -> dict:
    """Fill defaults and validate JSON-compatible noise settings."""
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise TypeError("noise_analysis must be a JSON object")
    settings = {**DEFAULT_NOISE_ANALYSIS_SETTINGS, **dict(values)}
    settings["database_path"] = str(settings["database_path"]).strip()
    if not settings["database_path"]:
        raise ValueError("noise-analysis database path must not be empty")
    for name in ("run_id", "point_index", "repetition_index", "nfft"):
        settings[name] = _nonnegative_integer(settings[name], name)
    settings["transimpedance_gain_v_per_a"] = _finite_positive(
        settings["transimpedance_gain_v_per_a"],
        "transimpedance_gain_v_per_a",
    )
    settings["input_scale_v_per_unit"] = _finite_positive(
        settings["input_scale_v_per_unit"],
        "input_scale_v_per_unit",
    )
    settings["sample_rate_hz"] = _finite_positive(
        settings["sample_rate_hz"],
        "sample_rate_hz",
    )
    settings["input_mode"] = str(settings["input_mode"]).lower()
    settings["window"] = str(settings["window"]).lower()
    settings["detrend"] = str(settings["detrend"]).lower()
    if settings["input_mode"] not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}")
    if settings["window"] not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}")
    if settings["detrend"] not in DETREND_MODES:
        raise ValueError(f"detrend must be one of {DETREND_MODES}")
    return settings


@dataclass(frozen=True)
class NoiseAnalysisConfig:
    """Numerical conversion and periodogram settings."""

    input_mode: str = "voltage"
    transimpedance_gain_v_per_a: float = 1.0e8
    input_scale_v_per_unit: float = 1.0
    sample_rate_hz: float = 1.0e6
    window: str = "flattop"
    detrend: str = "constant"
    nfft: int = 0

    def __post_init__(self) -> None:
        normalize_noise_analysis_settings({
            **DEFAULT_NOISE_ANALYSIS_SETTINGS,
            "input_mode": self.input_mode,
            "transimpedance_gain_v_per_a": self.transimpedance_gain_v_per_a,
            "input_scale_v_per_unit": self.input_scale_v_per_unit,
            "sample_rate_hz": self.sample_rate_hz,
            "window": self.window,
            "detrend": self.detrend,
            "nfft": self.nfft,
        })


@dataclass(frozen=True)
class NoiseAnalysisResult:
    """Time-domain current and one-sided amplitude spectral density."""

    input_trace: np.ndarray
    time_s: np.ndarray
    current_a: np.ndarray
    frequency_hz: np.ndarray
    asd_a_per_sqrt_hz: np.ndarray
    config: NoiseAnalysisConfig
    source: str = ""
    input_unit: str = ""

    @property
    def mean_current_a(self) -> float:
        return float(np.mean(self.current_a))

    @property
    def rms_current_a(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.current_a))))

    @property
    def std_current_a(self) -> float:
        return float(np.std(self.current_a))


@dataclass(frozen=True)
class NoiseTraceCollection:
    """A selectable set of point/repetition I traces."""

    i_traces: np.ndarray
    sample_rate_hz: float
    unit: str = "ADC units"
    source: str = ""
    database_path: str = ""
    run_id: int = 0

    def __post_init__(self) -> None:
        traces = np.asarray(self.i_traces)
        if traces.ndim != 3:
            raise ValueError("I traces must have shape (point, repetition, sample)")
        if traces.size < 2 or traces.shape[-1] < 2:
            raise ValueError("noise analysis requires at least two I samples")
        if not np.all(np.isfinite(traces)):
            raise ValueError("I traces must contain only finite values")
        _finite_positive(self.sample_rate_hz, "sample_rate_hz")

    @property
    def point_count(self) -> int:
        return int(np.asarray(self.i_traces).shape[0])

    @property
    def repetition_count(self) -> int:
        return int(np.asarray(self.i_traces).shape[1])

    @property
    def sample_count(self) -> int:
        return int(np.asarray(self.i_traces).shape[2])

    def trace(self, point_index: int, repetition_index: int) -> np.ndarray:
        return np.asarray(
            self.i_traces[int(point_index), int(repetition_index)],
            dtype=np.float64,
        )


def analyze_i_trace(
    i_trace: Any,
    config: NoiseAnalysisConfig,
    *,
    source: str = "",
    input_unit: str = "",
) -> NoiseAnalysisResult:
    """Convert one I trace to current and calculate its one-sided ASD."""
    values = np.asarray(i_trace, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("noise analysis requires at least two I samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("I trace must contain only finite values")

    mode = str(config.input_mode).lower()
    if mode == "current":
        current = values.copy()
    elif mode == "voltage":
        current = values / float(config.transimpedance_gain_v_per_a)
    elif mode == "adc":
        current = (
            values
            * float(config.input_scale_v_per_unit)
            / float(config.transimpedance_gain_v_per_a)
        )
    else:
        raise ValueError(f"unsupported input mode {config.input_mode!r}")

    try:
        from scipy import signal
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required for noise analysis; install scipy==1.17.1"
        ) from exc

    nfft = values.size if int(config.nfft) == 0 else max(
        values.size, int(config.nfft)
    )
    detrend = False if config.detrend == "none" else config.detrend
    frequency, density = signal.periodogram(
        current,
        fs=float(config.sample_rate_hz),
        window=str(config.window),
        detrend=detrend,
        nfft=nfft,
        scaling="density",
        return_onesided=True,
    )
    density = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    return NoiseAnalysisResult(
        input_trace=values,
        time_s=np.arange(values.size, dtype=np.float64)
        / float(config.sample_rate_hz),
        current_a=current,
        frequency_hz=np.asarray(frequency, dtype=np.float64),
        asd_a_per_sqrt_hz=np.sqrt(density),
        config=config,
        source=str(source),
        input_unit=str(input_unit),
    )


def _latest_run_id(database_path: Path) -> int:
    connection = sqlite3.connect(str(database_path), timeout=30.0)
    try:
        row = connection.execute("SELECT MAX(run_id) FROM runs").fetchone()
    finally:
        connection.close()
    if row is None or row[0] is None:
        raise ValueError(f"QCoDeS database contains no runs: {database_path}")
    return int(row[0])


def load_noise_trace_collection(
    database_path: Any,
    run_id: int = 0,
) -> NoiseTraceCollection:
    """Load I traces and sample timing from a saved QCoDeS experiment."""
    path = Path(database_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"QCoDeS database does not exist: {path}")
    selected_run = _nonnegative_integer(run_id, "run_id")
    if selected_run == 0:
        selected_run = _latest_run_id(path)
    try:
        from qcodes import initialise_or_create_database_at, load_by_id
    except ImportError as exc:
        raise RuntimeError(
            "QCoDeS==0.58.0 is required to load saved noise traces"
        ) from exc
    initialise_or_create_database_at(str(path))
    dataset = load_by_id(selected_run)
    arrays = load_qick_iq_arrays(dataset)
    metadata = arrays.get("metadata", {})
    layout = metadata.get("measurement_layout", {})
    sample_rate_hz = float(
        layout.get(
            "sample_rate_hz",
            1.0e6 / float(layout.get("sample_period_us", 1.0)),
        )
    )
    return NoiseTraceCollection(
        i_traces=np.asarray(arrays["i"], dtype=np.float64),
        sample_rate_hz=sample_rate_hz,
        unit=str(arrays.get("iq_unit", "ADC units")),
        source=f"QCoDeS Run {selected_run}",
        database_path=str(path),
        run_id=selected_run,
    )


class NoiseTraceLoadWorker(QtCore.QObject):
    """Load a QCoDeS run without blocking the GUI thread."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, database_path: str, run_id: int, parent=None):
        super().__init__(parent)
        self._database_path = str(database_path)
        self._run_id = int(run_id)

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            collection = load_noise_trace_collection(
                self._database_path,
                self._run_id,
            )
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(collection)


if pg is not None:

    class NoiseAnalysisPlotWidget(QtWidgets.QWidget):
        """Time-current and log-log current-ASD plots."""

        MAX_DISPLAY_POINTS = 120_000

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
            self.time_plot = pg.PlotWidget(splitter)
            self.spectrum_plot = pg.PlotWidget(splitter)
            splitter.addWidget(self.time_plot)
            splitter.addWidget(self.spectrum_plot)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            layout.addWidget(splitter)
            self.time_plot.setLabel("bottom", "Time", units="s")
            self.time_plot.setLabel("left", "Current", units="A")
            self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
            self.spectrum_plot.setLabel(
                "left", "Current ASD", units="A/sqrt(Hz)"
            )
            self.spectrum_plot.setLogMode(x=True, y=True)
            for plot in (self.time_plot, self.spectrum_plot):
                plot.showGrid(x=True, y=True, alpha=0.25)
            self._time_curve = self.time_plot.plot(
                pen=pg.mkPen("#2878b5", width=1.4)
            )
            self._spectrum_curve = self.spectrum_plot.plot(
                pen=pg.mkPen("#c43c39", width=1.4)
            )

        @classmethod
        def _linear_display_slice(cls, count: int) -> slice:
            step = max(1, int(np.ceil(count / cls.MAX_DISPLAY_POINTS)))
            return slice(None, None, step)

        @classmethod
        def _log_display_indices(cls, count: int) -> np.ndarray:
            if count <= cls.MAX_DISPLAY_POINTS:
                return np.arange(count, dtype=np.int64)
            return np.unique(
                np.geomspace(1, count, cls.MAX_DISPLAY_POINTS).astype(np.int64) - 1
            )

        def set_result(self, result: NoiseAnalysisResult) -> None:
            time_slice = self._linear_display_slice(result.time_s.size)
            self._time_curve.setData(
                result.time_s[time_slice], result.current_a[time_slice]
            )
            valid = (
                (result.frequency_hz > 0.0)
                & np.isfinite(result.asd_a_per_sqrt_hz)
                & (result.asd_a_per_sqrt_hz > 0.0)
            )
            frequency = result.frequency_hz[valid]
            asd = result.asd_a_per_sqrt_hz[valid]
            indices = self._log_display_indices(frequency.size)
            self._spectrum_curve.setData(frequency[indices], asd[indices])
            suffix = f" - {result.source}" if result.source else ""
            self.time_plot.setTitle(f"Input current vs time{suffix}")
            self.spectrum_plot.setTitle(
                f"Current noise amplitude spectral density{suffix}"
            )
            self.fit_view()

        def clear(self) -> None:
            self._time_curve.setData([], [])
            self._spectrum_curve.setData([], [])

        def fit_view(self) -> None:
            self.time_plot.enableAutoRange(x=True, y=True)
            self.spectrum_plot.enableAutoRange(x=True, y=True)

else:

    class NoiseAnalysisPlotWidget(QtWidgets.QLabel):
        """Dependency error used when PyQtGraph is unavailable."""

        def __init__(self, parent=None):
            super().__init__("pyqtgraph is required for noise-analysis plots", parent)
            self.setAlignment(QtCore.Qt.AlignCenter)

        def set_result(self, _result: NoiseAnalysisResult) -> None:
            return

        def clear(self) -> None:
            return

        def fit_view(self) -> None:
            return


class NoiseAnalysisPanel(QtWidgets.QWidget):
    """Controls for selecting an I trace and calculating current noise."""

    load_requested = QtCore.pyqtSignal(str, int)

    def __init__(self, parent=None, *, default_database_path: str = ""):
        super().__init__(parent)
        self._collection: Optional[NoiseTraceCollection] = None
        self._result: Optional[NoiseAnalysisResult] = None
        outer = QtWidgets.QVBoxLayout(self)

        source_group = QtWidgets.QGroupBox("I Trace Source", self)
        source_form = QtWidgets.QFormLayout(source_group)
        self.source_status = QtWidgets.QLabel("No I trace loaded", source_group)
        self.source_status.setWordWrap(True)
        self.database_path = QtWidgets.QLineEdit(
            str(default_database_path or DEFAULT_NOISE_ANALYSIS_SETTINGS["database_path"])
        )
        browse = QtWidgets.QToolButton(source_group)
        browse.setText("...")
        browse.setToolTip("Select a QCoDeS database")
        database_row = QtWidgets.QHBoxLayout()
        database_row.addWidget(self.database_path, 1)
        database_row.addWidget(browse)
        self.run_id = QtWidgets.QSpinBox(source_group)
        self.run_id.setRange(0, 2_147_483_647)
        self.run_id.setSpecialValueText("Latest")
        self.load_button = QtWidgets.QPushButton("Load QCoDeS I Trace", source_group)
        self.load_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        source_form.addRow("Source:", self.source_status)
        source_form.addRow("QCoDeS DB:", database_row)
        source_form.addRow("Run ID:", self.run_id)
        source_form.addRow(self.load_button)
        outer.addWidget(source_group)

        selector_group = QtWidgets.QGroupBox("Trace Selection", self)
        selector_form = QtWidgets.QFormLayout(selector_group)
        self.point_index = QtWidgets.QSpinBox(selector_group)
        self.repetition_index = QtWidgets.QSpinBox(selector_group)
        for widget in (self.point_index, self.repetition_index):
            # Preserve JSON selections before a run is loaded; set_collection()
            # narrows these bounds to the actual trace dimensions.
            widget.setRange(0, 2_147_483_647)
        selector_form.addRow("Sweep point index:", self.point_index)
        selector_form.addRow("Repetition index:", self.repetition_index)
        outer.addWidget(selector_group)

        conversion_group = QtWidgets.QGroupBox("Current Conversion", self)
        conversion_form = QtWidgets.QFormLayout(conversion_group)
        self.input_mode = QtWidgets.QComboBox(conversion_group)
        self.input_mode.addItem("Voltage trace (V)", "voltage")
        self.input_mode.addItem("ADC units scaled to volts", "adc")
        self.input_mode.addItem("Current trace already in A", "current")
        self.transimpedance_gain = QtWidgets.QDoubleSpinBox(conversion_group)
        self.transimpedance_gain.setRange(1.0e-12, 1.0e20)
        self.transimpedance_gain.setDecimals(6)
        self.transimpedance_gain.setValue(1.0e8)
        self.transimpedance_gain.setSuffix(" V/A")
        self.input_scale = QtWidgets.QDoubleSpinBox(conversion_group)
        self.input_scale.setRange(1.0e-18, 1.0e12)
        self.input_scale.setDecimals(12)
        self.input_scale.setValue(1.0)
        self.input_scale.setSuffix(" V/unit")
        conversion_form.addRow("I trace representation:", self.input_mode)
        conversion_form.addRow("Transimpedance gain:", self.transimpedance_gain)
        conversion_form.addRow("ADC input scale:", self.input_scale)
        outer.addWidget(conversion_group)

        spectrum_group = QtWidgets.QGroupBox("Spectrum", self)
        spectrum_form = QtWidgets.QFormLayout(spectrum_group)
        self.sample_rate = QtWidgets.QDoubleSpinBox(spectrum_group)
        self.sample_rate.setRange(1.0e-9, 1.0e12)
        self.sample_rate.setDecimals(6)
        self.sample_rate.setValue(1.0e6)
        self.sample_rate.setSuffix(" Hz")
        self.window = QtWidgets.QComboBox(spectrum_group)
        self.window.addItems(list(WINDOWS))
        self.detrend = QtWidgets.QComboBox(spectrum_group)
        self.detrend.addItems(list(DETREND_MODES))
        self.nfft = QtWidgets.QSpinBox(spectrum_group)
        self.nfft.setRange(0, 100_000_000)
        self.nfft.setSpecialValueText("Trace length")
        spectrum_form.addRow("Sampling rate:", self.sample_rate)
        spectrum_form.addRow("Window:", self.window)
        spectrum_form.addRow("Detrend:", self.detrend)
        spectrum_form.addRow("NFFT (zero pads only):", self.nfft)
        outer.addWidget(spectrum_group)

        self.analyze_button = QtWidgets.QPushButton("Analyze I Trace", self)
        self.analyze_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        )
        outer.addWidget(self.analyze_button)
        self.statistics = QtWidgets.QLabel("Load an I trace to calculate noise.", self)
        self.statistics.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.statistics.setWordWrap(True)
        outer.addWidget(self.statistics)

        self.plot = NoiseAnalysisPlotWidget(self)
        outer.addWidget(self.plot, 1)

        browse.clicked.connect(self._browse_database)
        self.load_button.clicked.connect(self._request_load)
        self.analyze_button.clicked.connect(self.analyze_selected_trace)
        self.point_index.valueChanged.connect(self._selection_changed)
        self.repetition_index.valueChanged.connect(self._selection_changed)
        self.input_mode.currentIndexChanged.connect(self._update_mode_controls)
        self._update_mode_controls()

    def _browse_database(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select QCoDeS database",
            self.database_path.text(),
            "SQLite database (*.db);;All files (*)",
        )
        if path:
            self.database_path.setText(path)

    def _request_load(self) -> None:
        path = self.database_path.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(
                self, "Cannot load I trace", "QCoDeS database path is empty."
            )
            return
        self.load_requested.emit(path, self.run_id.value())

    def _selection_changed(self, _value: int) -> None:
        if self._collection is not None:
            self.analyze_selected_trace()

    def _update_mode_controls(self, _index: int = 0) -> None:
        mode = self.input_mode.currentData()
        self.transimpedance_gain.setEnabled(mode != "current")
        self.input_scale.setEnabled(mode == "adc")

    def _config(self) -> NoiseAnalysisConfig:
        return NoiseAnalysisConfig(
            input_mode=str(self.input_mode.currentData()),
            transimpedance_gain_v_per_a=self.transimpedance_gain.value(),
            input_scale_v_per_unit=self.input_scale.value(),
            sample_rate_hz=self.sample_rate.value(),
            window=self.window.currentText(),
            detrend=self.detrend.currentText(),
            nfft=self.nfft.value(),
        )

    def _set_input_mode_for_unit(self, unit: str) -> None:
        normalized = str(unit).strip().lower()
        mode = "current" if normalized == "a" else (
            "voltage" if normalized == "v" else "adc"
        )
        index = self.input_mode.findData(mode)
        if index >= 0:
            self.input_mode.setCurrentIndex(index)

    def set_collection(self, collection: NoiseTraceCollection) -> None:
        self._collection = collection
        self.database_path.setText(
            collection.database_path or self.database_path.text()
        )
        if collection.run_id:
            self.run_id.setValue(collection.run_id)
        with QtCore.QSignalBlocker(self.point_index):
            self.point_index.setRange(0, collection.point_count - 1)
            self.point_index.setValue(
                min(self.point_index.value(), collection.point_count - 1)
            )
        with QtCore.QSignalBlocker(self.repetition_index):
            self.repetition_index.setRange(0, collection.repetition_count - 1)
            self.repetition_index.setValue(
                min(
                    self.repetition_index.value(),
                    collection.repetition_count - 1,
                )
            )
        self.sample_rate.setValue(float(collection.sample_rate_hz))
        self._set_input_mode_for_unit(collection.unit)
        self.source_status.setText(
            f"{collection.source or 'I trace'} | {collection.point_count} point(s) x "
            f"{collection.repetition_count} repetition(s) x "
            f"{collection.sample_count} samples | {collection.unit}"
        )
        self.set_loading(False)
        self.analyze_selected_trace()

    def set_experiment_result(self, stored: Any) -> None:
        arrays = load_qick_iq_arrays(stored.dataset)
        metadata = arrays.get("metadata", {})
        layout = metadata.get("measurement_layout", {})
        sample_rate_hz = float(
            layout.get(
                "sample_rate_hz",
                1.0e6 / float(layout.get("sample_period_us", 1.0)),
            )
        )
        self.set_collection(NoiseTraceCollection(
            i_traces=np.asarray(arrays["i"], dtype=np.float64),
            sample_rate_hz=sample_rate_hz,
            unit=str(arrays.get("iq_unit", "ADC units")),
            source=f"Latest Experiment Run {int(stored.run_id)}",
            database_path=str(stored.database_path),
            run_id=int(stored.run_id),
        ))

    def analyze_selected_trace(self) -> Optional[NoiseAnalysisResult]:
        if self._collection is None:
            self.statistics.setText("Load an I trace to calculate noise.")
            return None
        try:
            result = analyze_i_trace(
                self._collection.trace(
                    self.point_index.value(), self.repetition_index.value()
                ),
                self._config(),
                source=(
                    f"{self._collection.source}, point {self.point_index.value()}, "
                    f"rep {self.repetition_index.value()}"
                ),
                input_unit=self._collection.unit,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self.statistics.setText(f"Analysis failed: {exc}")
            return None
        self._result = result
        self.plot.set_result(result)
        self.statistics.setText(
            f"Mean {result.mean_current_a:.6e} A | "
            f"RMS {result.rms_current_a:.6e} A | "
            f"Std {result.std_current_a:.6e} A | "
            f"Duration {result.time_s[-1]:.6g} s | "
            f"Resolution {result.config.sample_rate_hz / max(1, result.frequency_hz.size * 2 - 2):.6g} Hz"
        )
        return result

    def set_loading(self, loading: bool, message: str = "") -> None:
        self.load_button.setEnabled(not loading)
        self.analyze_button.setEnabled(not loading)
        if message:
            self.source_status.setText(str(message))

    def show_load_error(self, message: str) -> None:
        self.set_loading(False, f"Load failed: {message}")

    def detach_plot(self) -> NoiseAnalysisPlotWidget:
        self.layout().removeWidget(self.plot)
        self.plot.setParent(None)
        return self.plot

    def settings_dict(self) -> dict:
        return {
            "database_path": self.database_path.text().strip(),
            "run_id": self.run_id.value(),
            "point_index": self.point_index.value(),
            "repetition_index": self.repetition_index.value(),
            "input_mode": str(self.input_mode.currentData()),
            "transimpedance_gain_v_per_a": self.transimpedance_gain.value(),
            "input_scale_v_per_unit": self.input_scale.value(),
            "sample_rate_hz": self.sample_rate.value(),
            "window": self.window.currentText(),
            "detrend": self.detrend.currentText(),
            "nfft": self.nfft.value(),
        }

    def load_settings(self, values: Mapping[str, Any]) -> None:
        settings = normalize_noise_analysis_settings(values)
        self.database_path.setText(settings["database_path"])
        self.run_id.setValue(settings["run_id"])
        self.point_index.setValue(settings["point_index"])
        self.repetition_index.setValue(settings["repetition_index"])
        mode_index = self.input_mode.findData(settings["input_mode"])
        self.input_mode.setCurrentIndex(mode_index)
        self.transimpedance_gain.setValue(
            settings["transimpedance_gain_v_per_a"]
        )
        self.input_scale.setValue(settings["input_scale_v_per_unit"])
        self.sample_rate.setValue(settings["sample_rate_hz"])
        self.window.setCurrentText(settings["window"])
        self.detrend.setCurrentText(settings["detrend"])
        self.nfft.setValue(settings["nfft"])
        self._update_mode_controls()


__all__ = [
    "DEFAULT_NOISE_ANALYSIS_SETTINGS",
    "NoiseAnalysisConfig",
    "NoiseAnalysisPanel",
    "NoiseAnalysisPlotWidget",
    "NoiseAnalysisResult",
    "NoiseTraceCollection",
    "NoiseTraceLoadWorker",
    "analyze_i_trace",
    "load_noise_trace_collection",
    "normalize_noise_analysis_settings",
]
