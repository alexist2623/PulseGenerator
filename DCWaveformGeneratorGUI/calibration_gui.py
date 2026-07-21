"""GUI controls and workers for QICK power calibration.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import traceback
from typing import Any, Mapping

import numpy as np
from PyQt5 import QtCore, QtWidgets

try:
    from .dc_voltage_calibration import (
        DcVoltageCalibrationConfig,
        MAX_DC_VOLTAGE_SAMPLES_PER_POINT,
        run_dc_voltage_calibration,
    )
    import pyqtgraph as pg
except ImportError:
    from dc_voltage_calibration import (
        DcVoltageCalibrationConfig,
        MAX_DC_VOLTAGE_SAMPLES_PER_POINT,
        run_dc_voltage_calibration,
    )
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
    from matplotlib.figure import Figure

    _USE_PYQTGRAPH = False
else:
    _USE_PYQTGRAPH = True

try:
    from .power_calibration import INPUT_BOARD_TYPES, OUTPUT_BOARD_TYPES
    from .qick_power_calibration import (
        InputPowerCalibrationConfig,
        OscilloscopeConfig,
        OutputPowerCalibrationConfig,
        run_input_power_calibration,
        run_output_power_calibration,
    )
    from .qick_sparameter_sweep import FILTER_TYPES, MAX_RF_OUTPUT_GAIN, POWER_SCALES
    from .sparameter_gui import RfPathCorrectionWidget
except ImportError:
    from power_calibration import INPUT_BOARD_TYPES, OUTPUT_BOARD_TYPES
    from qick_power_calibration import (
        InputPowerCalibrationConfig,
        OscilloscopeConfig,
        OutputPowerCalibrationConfig,
        run_input_power_calibration,
        run_output_power_calibration,
    )
    from qick_sparameter_sweep import FILTER_TYPES, MAX_RF_OUTPUT_GAIN, POWER_SCALES
    from sparameter_gui import RfPathCorrectionWidget


DEFAULT_CALIBRATION_DB_PATH = str(Path.home() / "gain_pwr_calb.db")
MAX_INPUT_CALIBRATION_PLOT_CURVES = 32
CALIBRATION_PATH_MODES = ("output", "input", "dc_voltage")


def _legacy_calibration_paths(
    output: OutputPowerCalibrationConfig,
    input_config: InputPowerCalibrationConfig,
    dc_voltage: DcVoltageCalibrationConfig,
) -> dict[str, dict[str, Any]]:
    """Build independent path defaults from the pre-v22 calibration fields."""
    common = {
        "output_ch": int(input_config.output_ch),
        "readout_ch": int(input_config.readout_ch),
        "output_nqz": int(input_config.nqz),
        "readout_nqz": int(input_config.readout_nqz),
        "output_board_type": str(input_config.output_board_type),
        "input_board_type": str(input_config.input_board_type),
        "output_att1_db": float(input_config.output_att1_db),
        "output_att2_db": float(input_config.output_att2_db),
        "readout_attenuation_db": float(input_config.input_attenuation_db),
        "readout_dc_gain_db": float(input_config.input_dc_gain_db),
        "loss1_db": 0.0,
        "loss2_db": 0.0,
        "amplifier_gain_db": 0.0,
    }
    output_path = {
        **common,
        "output_ch": int(output.output_ch),
        "output_nqz": int(output.nqz),
        "output_board_type": str(output.output_board_type),
        "output_att1_db": float(output.output_att1_db),
        "output_att2_db": float(output.output_att2_db),
    }
    dc_path = {
        **common,
        "output_ch": int(dc_voltage.output_ch),
        "readout_ch": int(dc_voltage.readout_ch),
        "output_nqz": 1,
        "readout_nqz": 1,
        "output_board_type": "DC_Out",
        "input_board_type": "DC_In",
        "output_att1_db": 0.0,
        "output_att2_db": 0.0,
        "readout_attenuation_db": 0.0,
        "readout_dc_gain_db": float(dc_voltage.input_dc_gain_db),
    }
    return {
        "output": output_path,
        "input": dict(common),
        "dc_voltage": dc_path,
    }


def normalize_calibration_paths(
    settings: Mapping[str, Any],
    output: OutputPowerCalibrationConfig,
    input_config: InputPowerCalibrationConfig,
    dc_voltage: DcVoltageCalibrationConfig,
) -> dict[str, dict[str, Any]]:
    """Validate per-subtab paths and migrate the former shared path."""
    defaults = _legacy_calibration_paths(output, input_config, dc_voltage)
    raw_paths = settings.get("paths")
    if raw_paths is None:
        return defaults
    if not isinstance(raw_paths, Mapping):
        raise TypeError("calibration paths must be a JSON object")

    normalized: dict[str, dict[str, Any]] = {}
    for mode in CALIBRATION_PATH_MODES:
        raw_mode = raw_paths.get(mode, {})
        if not isinstance(raw_mode, Mapping):
            raise TypeError(f"calibration {mode} path must be a JSON object")
        values = {**defaults[mode], **dict(raw_mode)}

        for key in ("output_ch", "readout_ch"):
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"calibration {mode} path {key} must be an integer")
            value = int(value)
            if not 0 <= value <= 255:
                raise ValueError(f"calibration {mode} path {key} must be 0..255")
            values[key] = value
        for key in ("output_nqz", "readout_nqz"):
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"calibration {mode} path {key} must be an integer")
            value = int(value)
            if value not in (1, 2):
                raise ValueError(f"calibration {mode} path {key} must be 1 or 2")
            values[key] = value

        output_board = str(values["output_board_type"])
        input_board = str(values["input_board_type"])
        if output_board not in OUTPUT_BOARD_TYPES:
            raise ValueError(
                f"calibration {mode} output board must be one of {OUTPUT_BOARD_TYPES}"
            )
        if input_board not in INPUT_BOARD_TYPES:
            raise ValueError(
                f"calibration {mode} input board must be one of {INPUT_BOARD_TYPES}"
            )
        values["output_board_type"] = output_board
        values["input_board_type"] = input_board

        ranges = {
            "output_att1_db": (0.0, 31.75),
            "output_att2_db": (0.0, 31.75),
            "readout_attenuation_db": (0.0, 31.75),
            "readout_dc_gain_db": (-6.0, 26.0),
            "loss1_db": (0.0, 200.0),
            "loss2_db": (0.0, 200.0),
            "amplifier_gain_db": (-200.0, 200.0),
        }
        for key, (minimum, maximum) in ranges.items():
            value = float(values[key])
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(
                    f"calibration {mode} path {key} must be {minimum:g}..{maximum:g}"
                )
            values[key] = value
        normalized[mode] = values
    return normalized


class _CalibrationPathWidget(RfPathCorrectionWidget):
    """RF path editor bound to exactly one Calibration sub-tab."""

    def __init__(self, owner, mode: str):
        super().__init__(owner, compact=True)
        self._calibration_owner = owner
        self._calibration_mode = mode

    def front_panel_values(self) -> Mapping[str, Any]:
        return self._calibration_owner._front_panel_values_for(
            self._calibration_mode
        )

    def apply_front_panel_settings(self, values: Mapping[str, Any]) -> None:
        self._calibration_owner.apply_path_settings(
            values,
            mode=self._calibration_mode,
        )


def input_calibration_plot_data(
    result: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return frequency, gain, input-power, and ADC arrays for plotting.

    Calibration storage uses dBm and dB ADC values.  The graph uses the
    corresponding positive linear quantities so either axis can use a real
    linear or logarithmic scale without applying a logarithm to negative dBm.
    """
    if not isinstance(result, Mapping):
        raise TypeError("input calibration result must be a mapping")
    required = (
        "frequencies_mhz",
        "gains",
        "input_power_dbm",
        "adc_magnitude_db",
    )
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError(
            "input calibration result is missing " + ", ".join(missing)
        )

    frequencies = np.asarray(result["frequencies_mhz"], dtype=float).reshape(-1)
    gains = np.asarray(result["gains"], dtype=float).reshape(-1)
    input_power_dbm = np.asarray(result["input_power_dbm"], dtype=float)
    adc_magnitude_db = np.asarray(result["adc_magnitude_db"], dtype=float)
    expected_shape = (gains.size, frequencies.size)
    if frequencies.size == 0 or gains.size == 0:
        raise ValueError("input calibration result must not be empty")
    if input_power_dbm.shape != expected_shape:
        raise ValueError(
            "input_power_dbm must have shape (gain, frequency); "
            f"expected {expected_shape}, got {input_power_dbm.shape}"
        )
    if adc_magnitude_db.shape != expected_shape:
        raise ValueError(
            "adc_magnitude_db must have shape (gain, frequency); "
            f"expected {expected_shape}, got {adc_magnitude_db.shape}"
        )
    for name, values in (
        ("frequencies_mhz", frequencies),
        ("gains", gains),
        ("input_power_dbm", input_power_dbm),
        ("adc_magnitude_db", adc_magnitude_db),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")

    input_power_mw = np.power(10.0, input_power_dbm / 10.0)
    adc_magnitude = np.power(10.0, adc_magnitude_db / 20.0)
    if not np.all(np.isfinite(input_power_mw)) or np.any(input_power_mw <= 0.0):
        raise ValueError("input power cannot be represented as positive mW values")
    if not np.all(np.isfinite(adc_magnitude)) or np.any(adc_magnitude <= 0.0):
        raise ValueError("ADC magnitude must be positive for logarithmic plotting")
    return (
        np.ascontiguousarray(frequencies),
        np.ascontiguousarray(gains),
        np.ascontiguousarray(input_power_mw),
        np.ascontiguousarray(adc_magnitude),
    )


def dc_voltage_calibration_plot_data(
    result: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return commanded voltage, measured ADC-I, spread, and fitted ADC-I."""
    if not isinstance(result, Mapping):
        raise TypeError("DC voltage calibration result must be a mapping")
    required = ("voltages_mv", "mean_adc", "std_adc", "calibration")
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError(
            "DC voltage calibration result is missing " + ", ".join(missing)
        )
    voltages_mv = np.asarray(result["voltages_mv"], dtype=float).reshape(-1)
    mean_adc = np.asarray(result["mean_adc"], dtype=float).reshape(-1)
    std_adc = np.asarray(result["std_adc"], dtype=float).reshape(-1)
    if voltages_mv.size < 2:
        raise ValueError("DC voltage calibration plot requires at least two points")
    if mean_adc.shape != voltages_mv.shape or std_adc.shape != voltages_mv.shape:
        raise ValueError(
            "voltages_mv, mean_adc, and std_adc must have equal one-dimensional shapes"
        )
    if not all(
        np.all(np.isfinite(values))
        for values in (voltages_mv, mean_adc, std_adc)
    ):
        raise ValueError("DC voltage calibration plot values must be finite")
    if np.any(std_adc < 0.0):
        raise ValueError("DC voltage calibration standard deviation must be nonnegative")
    calibration = result["calibration"]
    if not isinstance(calibration, Mapping):
        raise TypeError("DC voltage calibration fit must be a mapping")
    try:
        offset_adc = float(calibration["offset_adc"])
        response_adc_per_v = float(calibration["response_adc_per_v"])
    except KeyError as exc:
        raise ValueError(
            f"DC voltage calibration fit is missing {exc.args[0]}"
        ) from exc
    if not np.isfinite(offset_adc) or not np.isfinite(response_adc_per_v):
        raise ValueError("DC voltage calibration fit coefficients must be finite")
    fitted_adc = offset_adc + response_adc_per_v * (voltages_mv / 1000.0)
    return tuple(
        np.ascontiguousarray(values)
        for values in (voltages_mv, mean_adc, std_adc, fitted_adc)
    )


class InputCalibrationPlotWidget(QtWidgets.QWidget):
    """Plot calibrated input power against measured ADC magnitude."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frequencies_mhz = np.empty(0, dtype=float)
        self._gains = np.empty(0, dtype=float)
        self._input_power_mw = np.empty((0, 0), dtype=float)
        self._adc_magnitude = np.empty((0, 0), dtype=float)
        self.displayed_frequency_indices = np.empty(0, dtype=np.int64)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)
        self.frequency_selector = QtWidgets.QComboBox(self)
        self.frequency_selector.addItem("All frequencies", None)
        self.frequency_selector.setEnabled(False)
        self.x_scale = self._scale_combo("Log")
        self.y_scale = self._scale_combo("Log")
        self.status = QtWidgets.QLabel("Run an input calibration to display data", self)
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        controls.addWidget(QtWidgets.QLabel("Frequency:", self))
        controls.addWidget(self.frequency_selector, 1)
        controls.addWidget(QtWidgets.QLabel("X axis:", self))
        controls.addWidget(self.x_scale)
        controls.addWidget(QtWidgets.QLabel("Y axis:", self))
        controls.addWidget(self.y_scale)
        controls.addStretch(1)
        controls.addWidget(self.status)
        layout.addLayout(controls)

        if _USE_PYQTGRAPH:
            self.graph = pg.PlotWidget(self)
            self.graph.setBackground("w")
            self.plot_item = self.graph.getPlotItem()
            self.plot_item.showGrid(x=True, y=True, alpha=0.25)
            self.legend = self.plot_item.addLegend(offset=(10, 10))
            layout.addWidget(self.graph, 1)
        else:
            self.figure = Figure(tight_layout=True)
            self.canvas = Canvas(self.figure)
            self.plot_item = self.figure.subplots(1, 1)
            layout.addWidget(self.canvas, 1)
        self.setMinimumHeight(360)

        self.frequency_selector.currentIndexChanged.connect(self._render)
        self.x_scale.currentIndexChanged.connect(self._apply_axis_scales)
        self.y_scale.currentIndexChanged.connect(self._apply_axis_scales)
        self._apply_axis_scales()

    @staticmethod
    def _scale_combo(default: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItem("Linear", "linear")
        combo.addItem("Log", "log")
        combo.setCurrentText(default)
        return combo

    @property
    def x_scale_mode(self) -> str:
        return str(self.x_scale.currentData())

    @property
    def y_scale_mode(self) -> str:
        return str(self.y_scale.currentData())

    def set_axis_scales(self, x_scale: str, y_scale: str) -> None:
        for name, combo, value in (
            ("X", self.x_scale, x_scale),
            ("Y", self.y_scale, y_scale),
        ):
            index = combo.findData(str(value).lower())
            if index < 0:
                raise ValueError(f"{name} axis scale must be linear or log")
            combo.setCurrentIndex(index)
        self._apply_axis_scales()

    def set_result(self, result: Mapping[str, Any]) -> None:
        (
            self._frequencies_mhz,
            self._gains,
            self._input_power_mw,
            self._adc_magnitude,
        ) = input_calibration_plot_data(result)
        self.frequency_selector.blockSignals(True)
        self.frequency_selector.clear()
        self.frequency_selector.addItem("All frequencies", None)
        for index, frequency in enumerate(self._frequencies_mhz):
            self.frequency_selector.addItem(f"{frequency:.9g} MHz", index)
        self.frequency_selector.setCurrentIndex(0)
        self.frequency_selector.setEnabled(True)
        self.frequency_selector.blockSignals(False)
        self._render()

    def _frequency_indices(self) -> np.ndarray:
        selected = self.frequency_selector.currentData()
        if selected is not None:
            return np.asarray([int(selected)], dtype=np.int64)
        count = self._frequencies_mhz.size
        if count <= MAX_INPUT_CALIBRATION_PLOT_CURVES:
            return np.arange(count, dtype=np.int64)
        return np.unique(
            np.linspace(
                0,
                count - 1,
                MAX_INPUT_CALIBRATION_PLOT_CURVES,
                dtype=np.int64,
            )
        )

    def _render(self, *_args) -> None:
        if self._frequencies_mhz.size == 0:
            return
        indices = self._frequency_indices()
        self.displayed_frequency_indices = indices
        if _USE_PYQTGRAPH:
            self.plot_item.clear()
            self.legend.clear()
            for curve_index, frequency_index in enumerate(indices):
                color = pg.intColor(curve_index, hues=max(1, indices.size), values=1)
                self.plot_item.plot(
                    self._input_power_mw[:, frequency_index],
                    self._adc_magnitude[:, frequency_index],
                    pen=pg.mkPen(color, width=2),
                    symbol="o",
                    symbolSize=7,
                    symbolBrush=color,
                    name=f"{self._frequencies_mhz[frequency_index]:.9g} MHz",
                )
        else:
            self.plot_item.clear()
            for curve_index, frequency_index in enumerate(indices):
                self.plot_item.plot(
                    self._input_power_mw[:, frequency_index],
                    self._adc_magnitude[:, frequency_index],
                    "-o",
                    label=f"{self._frequencies_mhz[frequency_index]:.9g} MHz",
                    color=f"C{curve_index % 10}",
                )
            self.plot_item.grid(True, alpha=0.25)
            self.plot_item.legend(fontsize=8, ncol=2)
        self._apply_axis_scales()
        shown = indices.size
        total = self._frequencies_mhz.size
        limited = " (sampled)" if shown < total else ""
        self.status.setText(
            f"{self._gains.size} gains | {shown}/{total} frequencies{limited}"
        )

    def _apply_axis_scales(self, *_args) -> None:
        x_log = self.x_scale_mode == "log"
        y_log = self.y_scale_mode == "log"
        if _USE_PYQTGRAPH:
            self.plot_item.setLabel("bottom", "Calibrated input power", units="mW")
            self.plot_item.setLabel("left", "ADC magnitude", units="ADC units")
            self.plot_item.setLogMode(x=x_log, y=y_log)
            if self._frequencies_mhz.size:
                self.plot_item.autoRange()
        else:
            self.plot_item.set_xlabel("Calibrated input power [mW]")
            self.plot_item.set_ylabel("ADC magnitude [ADC units]")
            self.plot_item.set_xscale("log" if x_log else "linear")
            self.plot_item.set_yscale("log" if y_log else "linear")
            if self._frequencies_mhz.size:
                self.plot_item.relim()
                self.plot_item.autoscale_view()
            self.canvas.draw_idle()


class DcVoltageCalibrationPlotWidget(QtWidgets.QWidget):
    """Plot measured zero-frequency ADC-I against commanded DC voltage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.voltages_mv = np.empty(0, dtype=float)
        self.mean_adc = np.empty(0, dtype=float)
        self.std_adc = np.empty(0, dtype=float)
        self.fitted_adc = np.empty(0, dtype=float)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self.status = QtWidgets.QLabel(
            "Run a DC voltage calibration to display ADC values",
            self,
        )
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        if _USE_PYQTGRAPH:
            self.graph = pg.PlotWidget(self)
            self.graph.setBackground("w")
            self.plot_item = self.graph.getPlotItem()
            self.plot_item.showGrid(x=True, y=True, alpha=0.25)
            self.legend = self.plot_item.addLegend(offset=(10, 10))
            self.plot_item.setLabel("bottom", "Commanded DC output", units="mV")
            self.plot_item.setLabel("left", "Mean ADC-I", units="ADC units")
            layout.addWidget(self.graph, 1)
        else:
            self.figure = Figure(tight_layout=True)
            self.canvas = Canvas(self.figure)
            self.plot_item = self.figure.subplots(1, 1)
            layout.addWidget(self.canvas, 1)
        self.setMinimumHeight(360)

    def set_result(self, result: Mapping[str, Any]) -> None:
        (
            self.voltages_mv,
            self.mean_adc,
            self.std_adc,
            self.fitted_adc,
        ) = dc_voltage_calibration_plot_data(result)
        calibration = result["calibration"]
        r_squared = float(calibration.get("r_squared", float("nan")))
        response = float(calibration["response_adc_per_v"])
        offset = float(calibration["offset_adc"])
        order = np.argsort(self.voltages_mv)

        if _USE_PYQTGRAPH:
            self.plot_item.clear()
            self.legend.clear()
            self.plot_item.plot(
                self.voltages_mv,
                self.mean_adc,
                pen=pg.mkPen("#2563a6", width=2),
                symbol="o",
                symbolSize=7,
                symbolBrush="#2563a6",
                name="Measured mean ADC-I",
            )
            if np.any(self.std_adc > 0.0):
                error_bars = pg.ErrorBarItem(
                    x=self.voltages_mv,
                    y=self.mean_adc,
                    height=2.0 * self.std_adc,
                    beam=6.0,
                    pen=pg.mkPen("#6b8fb3", width=1),
                )
                self.plot_item.addItem(error_bars)
            self.plot_item.plot(
                self.voltages_mv[order],
                self.fitted_adc[order],
                pen=pg.mkPen("#b33a3a", width=2, style=QtCore.Qt.DashLine),
                name="Linear fit",
            )
            self.plot_item.autoRange()
        else:
            self.plot_item.clear()
            self.plot_item.errorbar(
                self.voltages_mv,
                self.mean_adc,
                yerr=self.std_adc,
                fmt="o-",
                color="C0",
                capsize=3,
                label="Measured mean ADC-I",
            )
            self.plot_item.plot(
                self.voltages_mv[order],
                self.fitted_adc[order],
                "--",
                color="C3",
                label="Linear fit",
            )
            self.plot_item.set_xlabel("Commanded DC output [mV]")
            self.plot_item.set_ylabel("Mean ADC-I [ADC units]")
            self.plot_item.grid(True, alpha=0.25)
            self.plot_item.legend()
            self.canvas.draw_idle()
        self.status.setText(
            f"{self.voltages_mv.size} voltage points | "
            f"response {response:.9g} ADC/V | offset {offset:.9g} ADC | "
            f"R^2 {r_squared:.8f}"
        )


class CalibrationPanel(QtWidgets.QWidget):
    """Output-scope and input-ADC calibration controls."""

    output_requested = QtCore.pyqtSignal()
    input_requested = QtCore.pyqtSignal()
    dc_voltage_requested = QtCore.pyqtSignal()
    dc_application_changed = QtCore.pyqtSignal(bool, str, int)
    path_settings_applied = QtCore.pyqtSignal(object)
    front_panel_requested = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self._path_diagrams: dict[str, RfPathCorrectionWidget] = {}
        self._front_panel_mode = "output"

        database_group = QtWidgets.QGroupBox("Calibration Database")
        database_form = QtWidgets.QFormLayout(database_group)
        self.database_path = QtWidgets.QLineEdit(DEFAULT_CALIBRATION_DB_PATH)
        self.browse_database = QtWidgets.QToolButton()
        self.browse_database.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)
        )
        self.browse_database.setToolTip("Choose gain_pwr_calb.db")
        self.browse_database.clicked.connect(self._browse_database)
        database_row = QtWidgets.QHBoxLayout()
        database_row.addWidget(self.database_path, 1)
        database_row.addWidget(self.browse_database)
        database_form.addRow("QCoDeS DB file:", database_row)
        layout.addWidget(database_group)

        self.tabs = QtWidgets.QTabWidget(self)
        self.tabs.addTab(self._build_output_tab(), "Output / Oscilloscope")
        self.tabs.addTab(self._build_input_tab(), "Input / ADC")
        self.tabs.addTab(self._build_dc_voltage_tab(), "DC Voltage")
        initial_paths = _legacy_calibration_paths(
            OutputPowerCalibrationConfig(
                database_path=DEFAULT_CALIBRATION_DB_PATH
            ),
            InputPowerCalibrationConfig(
                database_path=DEFAULT_CALIBRATION_DB_PATH
            ),
            DcVoltageCalibrationConfig(
                database_path=DEFAULT_CALIBRATION_DB_PATH
            ),
        )
        for mode, values in initial_paths.items():
            self.path_diagram_for(mode).apply_external_settings(values)
        self.tabs.currentChanged.connect(self._select_front_panel_mode)
        layout.addWidget(self.tabs, 1)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        self.status = QtWidgets.QLabel("Ready")
        self.status.setWordWrap(True)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)

    @staticmethod
    def _scroll_form(title: str):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        vertical = QtWidgets.QVBoxLayout(content)
        group = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(group)
        vertical.addWidget(group)
        vertical.addStretch(1)
        scroll.setWidget(content)
        return scroll, form, vertical

    def _new_path_diagram(self, mode: str) -> RfPathCorrectionWidget:
        if mode not in CALIBRATION_PATH_MODES:
            raise ValueError(f"unknown calibration path mode {mode!r}")
        diagram = _CalibrationPathWidget(self, mode)
        diagram.settings_applied.connect(
            lambda values, selected=mode: self._apply_local_path_settings(
                selected,
                values,
            )
        )
        diagram.front_panel_requested.connect(
            lambda selected=mode, target=diagram: self._request_front_panel(
                selected,
                target,
            )
        )
        self._path_diagrams[mode] = diagram
        return diagram

    def _select_front_panel_mode(self, index: int) -> None:
        if 0 <= int(index) < len(CALIBRATION_PATH_MODES):
            self._front_panel_mode = CALIBRATION_PATH_MODES[int(index)]

    def _request_front_panel(
        self,
        mode: str,
        target: RfPathCorrectionWidget,
    ) -> None:
        self._front_panel_mode = mode
        self.front_panel_requested.emit(target)

    def path_diagram_for(self, mode: str) -> RfPathCorrectionWidget:
        """Return the independent RF path editor for one calibration sub-tab."""
        try:
            return self._path_diagrams[mode]
        except KeyError as exc:
            raise ValueError(f"unknown calibration path mode {mode!r}") from exc

    @property
    def path_diagram(self) -> RfPathCorrectionWidget:
        """Compatibility alias for the currently selected sub-tab's path."""
        return self.path_diagram_for(self._front_panel_mode)

    @staticmethod
    def _channel(value: int = 0) -> QtWidgets.QSpinBox:
        widget = QtWidgets.QSpinBox()
        widget.setRange(0, 255)
        widget.setValue(value)
        return widget

    @staticmethod
    def _frequency(value: float) -> QtWidgets.QDoubleSpinBox:
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(-100_000.0, 100_000.0)
        widget.setDecimals(9)
        widget.setValue(value)
        widget.setSuffix(" MHz")
        return widget

    @staticmethod
    def _gain(value: int) -> QtWidgets.QSpinBox:
        widget = QtWidgets.QSpinBox()
        widget.setRange(1, MAX_RF_OUTPUT_GAIN)
        widget.setValue(value)
        return widget

    @staticmethod
    def _points(value: int) -> QtWidgets.QSpinBox:
        widget = QtWidgets.QSpinBox()
        widget.setRange(2, 100_000)
        widget.setValue(value)
        return widget

    @staticmethod
    def _attenuation(value: float) -> QtWidgets.QDoubleSpinBox:
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(0.0, 31.75)
        widget.setDecimals(2)
        widget.setSingleStep(0.25)
        widget.setValue(value)
        widget.setSuffix(" dB")
        return widget

    @staticmethod
    def _filter_combo(value: str = "bypass") -> QtWidgets.QComboBox:
        widget = QtWidgets.QComboBox()
        widget.addItems(FILTER_TYPES)
        widget.setCurrentText(value)
        return widget

    @staticmethod
    def _filter_value(value: float) -> QtWidgets.QDoubleSpinBox:
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(0.001, 100.0)
        widget.setDecimals(6)
        widget.setValue(value)
        widget.setSuffix(" GHz")
        return widget

    @staticmethod
    def _scale_combo() -> QtWidgets.QComboBox:
        widget = QtWidgets.QComboBox()
        for value in POWER_SCALES:
            widget.addItem(value.capitalize(), value)
        return widget

    def _build_output_tab(self) -> QtWidgets.QWidget:
        scroll, form, vertical = self._scroll_form("QICK Output Power Calibration")
        self.output_board = QtWidgets.QComboBox()
        self.output_board.addItems(OUTPUT_BOARD_TYPES)
        self.output_board.setCurrentText("RF_Out")
        self.output_ch = self._channel()
        self.output_frequency_start = self._frequency(400.0)
        self.output_frequency_end = self._frequency(500.0)
        self.output_frequency_points = self._points(11)
        self.output_gain_start = self._gain(1000)
        self.output_gain_end = self._gain(MAX_RF_OUTPUT_GAIN)
        self.output_gain_points = self._points(16)
        self.output_gain_scale = self._scale_combo()
        self.output_att1 = self._attenuation(0.0)
        self.output_att2 = self._attenuation(0.0)
        self.output_filter_type = self._filter_combo()
        self.output_filter_cutoff = self._filter_value(2.5)
        self.output_filter_bandwidth = self._filter_value(1.0)
        self.output_experiment_name = QtWidgets.QLineEdit(
            "QICK output power calibration"
        )
        self.output_sample_name = QtWidgets.QLineEdit()
        self.output_sample_name.setPlaceholderText("Auto: RF_Out_<range>MHz")
        form.addRow(
            QtWidgets.QLabel(
                "Board, channel, ATT, filter, and Nyquist settings use this "
                "sub-tab's independent RF path above."
            )
        )
        for label, widget in (
            ("Start frequency:", self.output_frequency_start),
            ("End frequency:", self.output_frequency_end),
            ("Frequency points:", self.output_frequency_points),
            ("Start gain:", self.output_gain_start),
            ("End gain:", self.output_gain_end),
            ("Gain points:", self.output_gain_points),
            ("Gain spacing:", self.output_gain_scale),
            ("Experiment name:", self.output_experiment_name),
            ("Sample name:", self.output_sample_name),
        ):
            form.addRow(label, widget)

        scope_group = QtWidgets.QGroupBox("Oscilloscope FFT Marker")
        scope_form = QtWidgets.QFormLayout(scope_group)
        self.scope_resource = QtWidgets.QLineEdit()
        self.scope_resource.setPlaceholderText(
            "USB0::0x0957::0x1780::<serial>::0::INSTR"
        )
        self.scope_channel = QtWidgets.QSpinBox()
        self.scope_channel.setRange(1, 8)
        self.scope_channel.setValue(2)
        self.scope_function = QtWidgets.QSpinBox()
        self.scope_function.setRange(1, 4)
        self.scope_function.setValue(1)
        self.scope_span = self._frequency(100.0)
        self.scope_averages = QtWidgets.QSpinBox()
        self.scope_averages.setRange(1, 100_000)
        self.scope_averages.setValue(100)
        self.scope_settle = QtWidgets.QDoubleSpinBox()
        self.scope_settle.setRange(0.0, 3600.0)
        self.scope_settle.setDecimals(3)
        self.scope_settle.setValue(2.0)
        self.scope_settle.setSuffix(" s")
        for label, widget in (
            ("VISA resource:", self.scope_resource),
            ("Input channel:", self.scope_channel),
            ("Math function:", self.scope_function),
            ("FFT span:", self.scope_span),
            ("Marker averages:", self.scope_averages),
            ("Settle time:", self.scope_settle),
        ):
            scope_form.addRow(label, widget)
        vertical.insertWidget(1, scope_group)
        self.run_output_button = QtWidgets.QPushButton(
            "Run Oscilloscope Output Calibration"
        )
        self.run_output_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        )
        self.run_output_button.clicked.connect(self.output_requested.emit)
        vertical.insertWidget(2, self.run_output_button)
        self.output_board.currentTextChanged.connect(self._update_board_controls)
        self.output_path_diagram = self._new_path_diagram("output")
        vertical.insertWidget(0, self.output_path_diagram)
        return scroll

    def _build_input_tab(self) -> QtWidgets.QWidget:
        scroll, form, vertical = self._scroll_form("QICK ADC Input Calibration")
        self.input_output_board = QtWidgets.QComboBox()
        self.input_output_board.addItems(OUTPUT_BOARD_TYPES)
        self.input_output_board.setCurrentText("RF_Out")
        self.input_board = QtWidgets.QComboBox()
        self.input_board.addItems(INPUT_BOARD_TYPES)
        self.input_board.setCurrentText("RF_In")
        self.input_output_ch = self._channel()
        self.input_readout_ch = self._channel()
        self.input_frequency_start = self._frequency(400.0)
        self.input_frequency_end = self._frequency(500.0)
        self.input_frequency_points = self._points(11)
        self.input_gain_start = self._gain(1000)
        self.input_gain_end = self._gain(MAX_RF_OUTPUT_GAIN)
        self.input_gain_points = self._points(16)
        self.input_gain_scale = self._scale_combo()
        self.input_scan_time = QtWidgets.QDoubleSpinBox()
        self.input_scan_time.setRange(0.001, 1.0e6)
        self.input_scan_time.setDecimals(6)
        self.input_scan_time.setValue(100.0)
        self.input_scan_time.setSuffix(" us")
        self.input_output_att1 = self._attenuation(0.0)
        self.input_output_att2 = self._attenuation(0.0)
        self.input_attenuation = self._attenuation(0.0)
        self.input_dc_gain = QtWidgets.QDoubleSpinBox()
        self.input_dc_gain.setRange(-6.0, 26.0)
        self.input_dc_gain.setDecimals(1)
        self.input_dc_gain.setSingleStep(1.0)
        self.input_dc_gain.setValue(0.0)
        self.input_dc_gain.setSuffix(" dB")
        self.input_path_loss = QtWidgets.QDoubleSpinBox()
        self.input_path_loss.setRange(-200.0, 200.0)
        self.input_path_loss.setDecimals(6)
        self.input_path_loss.setSuffix(" dB")
        self.input_output_filter = self._filter_combo()
        self.input_output_cutoff = self._filter_value(2.5)
        self.input_output_bandwidth = self._filter_value(1.0)
        self.input_readout_filter = self._filter_combo()
        self.input_readout_cutoff = self._filter_value(2.5)
        self.input_readout_bandwidth = self._filter_value(1.0)
        self.input_trim_low = QtWidgets.QSpinBox()
        self.input_trim_low.setRange(0, 1000)
        self.input_trim_high = QtWidgets.QSpinBox()
        self.input_trim_high.setRange(0, 1000)
        self.input_experiment_name = QtWidgets.QLineEdit(
            "QICK ADC input power calibration"
        )
        self.input_sample_name = QtWidgets.QLineEdit()
        self.input_sample_name.setPlaceholderText("Auto: RF_In_<range>MHz")
        form.addRow(
            QtWidgets.QLabel(
                "Board, channel, ATT, filter, and Nyquist settings use this "
                "sub-tab's independent RF path above."
            )
        )
        for label, widget in (
            ("Start frequency:", self.input_frequency_start),
            ("End frequency:", self.input_frequency_end),
            ("Frequency points:", self.input_frequency_points),
            ("Start gain:", self.input_gain_start),
            ("End gain:", self.input_gain_end),
            ("Gain points:", self.input_gain_points),
            ("Gain spacing:", self.input_gain_scale),
            ("FIR scan time per point:", self.input_scan_time),
            ("External path loss:", self.input_path_loss),
            ("Trim low-gain points:", self.input_trim_low),
            ("Trim high-gain points:", self.input_trim_high),
            ("Experiment name:", self.input_experiment_name),
            ("Sample name:", self.input_sample_name),
        ):
            form.addRow(label, widget)
        self.run_input_button = QtWidgets.QPushButton("Run FIR-DDR Input Calibration")
        self.run_input_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        )
        self.run_input_button.clicked.connect(self.input_requested.emit)
        vertical.insertWidget(1, self.run_input_button)
        plot_group = QtWidgets.QGroupBox("Input Power / ADC Response")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
        self.input_response_plot = InputCalibrationPlotWidget(plot_group)
        plot_layout.addWidget(self.input_response_plot)
        vertical.insertWidget(2, plot_group)
        self.input_output_board.currentTextChanged.connect(self._update_board_controls)
        self.input_board.currentTextChanged.connect(self._update_board_controls)
        self._update_board_controls()
        self.input_path_diagram = self._new_path_diagram("input")
        vertical.insertWidget(0, self.input_path_diagram)
        return scroll

    def _build_dc_voltage_tab(self) -> QtWidgets.QWidget:
        scroll, form, vertical = self._scroll_form(
            "DC_Out to DC_In Voltage Calibration"
        )
        self.dc_application_group = QtWidgets.QGroupBox(
            "Apply DC Input Voltage Calibration"
        )
        self.dc_application_group.setCheckable(True)
        self.dc_application_group.setChecked(False)
        application_form = QtWidgets.QFormLayout(self.dc_application_group)
        self.dc_application_path = QtWidgets.QLineEdit(
            DEFAULT_CALIBRATION_DB_PATH
        )
        self.dc_application_path.setPlaceholderText(
            "QCoDeS DB containing a DC Voltage calibration run"
        )
        self.dc_application_browse = QtWidgets.QToolButton()
        self.dc_application_browse.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        self.dc_application_browse.setToolTip(
            "Choose the DC voltage calibration database to apply"
        )
        application_path_row = QtWidgets.QHBoxLayout()
        application_path_row.addWidget(self.dc_application_path, 1)
        application_path_row.addWidget(self.dc_application_browse)
        self.dc_application_run_id = QtWidgets.QSpinBox()
        self.dc_application_run_id.setRange(0, (1 << 31) - 1)
        self.dc_application_run_id.setSpecialValueText(
            "Latest matching channel/gain"
        )
        application_note = QtWidgets.QLabel(
            "This selection belongs only to the Calibration tab. Select a "
            "calibration independently in AWG Tuning, Stability Diagram, and "
            "Noise Analysis. Run ID 0 selects the latest matching calibration."
        )
        application_note.setWordWrap(True)
        application_form.addRow("Application DB:", application_path_row)
        application_form.addRow("Run ID:", self.dc_application_run_id)
        application_form.addRow(application_note)
        vertical.insertWidget(0, self.dc_application_group)

        self.dc_voltage_output_ch = self._channel(1)
        self.dc_voltage_readout_ch = self._channel(0)
        self.dc_voltage_start_mv = QtWidgets.QDoubleSpinBox()
        self.dc_voltage_start_mv.setRange(-10000.0, 10000.0)
        self.dc_voltage_start_mv.setDecimals(6)
        self.dc_voltage_start_mv.setValue(-800.0)
        self.dc_voltage_start_mv.setSuffix(" mV")
        self.dc_voltage_stop_mv = QtWidgets.QDoubleSpinBox()
        self.dc_voltage_stop_mv.setRange(-10000.0, 10000.0)
        self.dc_voltage_stop_mv.setDecimals(6)
        self.dc_voltage_stop_mv.setValue(800.0)
        self.dc_voltage_stop_mv.setSuffix(" mV")
        self.dc_voltage_points = self._points(33)
        self.dc_voltage_full_scale_mv = QtWidgets.QDoubleSpinBox()
        self.dc_voltage_full_scale_mv.setRange(1.0, 10000.0)
        self.dc_voltage_full_scale_mv.setDecimals(6)
        self.dc_voltage_full_scale_mv.setValue(800.0)
        self.dc_voltage_full_scale_mv.setSuffix(" mV")
        self.dc_voltage_samples = QtWidgets.QSpinBox()
        self.dc_voltage_samples.setRange(
            1,
            MAX_DC_VOLTAGE_SAMPLES_PER_POINT,
        )
        self.dc_voltage_samples.setValue(128)
        self.dc_voltage_repetitions = QtWidgets.QSpinBox()
        self.dc_voltage_repetitions.setRange(1, 100000)
        self.dc_voltage_repetitions.setValue(4)
        self.dc_voltage_input_gain = QtWidgets.QDoubleSpinBox()
        self.dc_voltage_input_gain.setRange(-6.0, 26.0)
        self.dc_voltage_input_gain.setDecimals(1)
        self.dc_voltage_input_gain.setSingleStep(1.0)
        self.dc_voltage_input_gain.setValue(0.0)
        self.dc_voltage_input_gain.setSuffix(" dB")
        self.dc_voltage_settle_us = QtWidgets.QDoubleSpinBox()
        self.dc_voltage_settle_us.setRange(0.0, 1000000.0)
        self.dc_voltage_settle_us.setDecimals(6)
        self.dc_voltage_settle_us.setValue(5.0)
        self.dc_voltage_settle_us.setSuffix(" us")
        self.dc_voltage_margin_samples = QtWidgets.QSpinBox()
        self.dc_voltage_margin_samples.setRange(0, 10000000)
        self.dc_voltage_margin_samples.setValue(1024)
        self.dc_voltage_force_overwrite = QtWidgets.QCheckBox(
            "Allow overwrite of reserved DDR range"
        )
        self.dc_voltage_force_overwrite.setChecked(True)
        self.dc_voltage_experiment_name = QtWidgets.QLineEdit(
            "QICK DC input voltage calibration"
        )
        self.dc_voltage_sample_name = QtWidgets.QLineEdit()
        self.dc_voltage_sample_name.setPlaceholderText(
            "Auto: DC_In_voltage_ch<readout>_gain<dB>"
        )
        frequency_note = QtWidgets.QLabel(
            "This is a DC voltage sweep only. The readout/DDC frequency is "
            "fixed to 0 MHz, and the FIR-DDR path stores 1 MSPS samples."
        )
        frequency_note.setWordWrap(True)
        form.addRow(frequency_note)
        self.dc_voltage_path_note = QtWidgets.QLabel()
        self.dc_voltage_path_note.setWordWrap(True)
        form.addRow("Selected DC path:", self.dc_voltage_path_note)
        for label, widget in (
            ("Start voltage:", self.dc_voltage_start_mv),
            ("Stop voltage:", self.dc_voltage_stop_mv),
            ("Voltage points:", self.dc_voltage_points),
            ("DC output full scale (+/-):", self.dc_voltage_full_scale_mv),
            ("FIR samples / point:", self.dc_voltage_samples),
            ("Repetitions / point:", self.dc_voltage_repetitions),
            ("Settle before trigger:", self.dc_voltage_settle_us),
            ("FIR input margin:", self.dc_voltage_margin_samples),
            ("Experiment name:", self.dc_voltage_experiment_name),
            ("Sample name:", self.dc_voltage_sample_name),
        ):
            form.addRow(label, widget)
        form.addRow(self.dc_voltage_force_overwrite)
        self.run_dc_voltage_button = QtWidgets.QPushButton(
            "Run 0 MHz DC Voltage Calibration"
        )
        self.run_dc_voltage_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        )
        self.run_dc_voltage_button.clicked.connect(
            self.dc_voltage_requested.emit
        )
        vertical.insertWidget(2, self.run_dc_voltage_button)
        dc_plot_group = QtWidgets.QGroupBox("DC Output Voltage / ADC-I Response")
        dc_plot_layout = QtWidgets.QVBoxLayout(dc_plot_group)
        self.dc_voltage_response_plot = DcVoltageCalibrationPlotWidget(
            dc_plot_group
        )
        dc_plot_layout.addWidget(self.dc_voltage_response_plot)
        vertical.insertWidget(3, dc_plot_group)
        self.dc_application_group.toggled.connect(
            self._emit_dc_application_changed
        )
        self.dc_application_path.editingFinished.connect(
            self._emit_dc_application_changed
        )
        self.dc_application_run_id.valueChanged.connect(
            self._emit_dc_application_changed
        )
        self.dc_application_browse.clicked.connect(
            self._browse_dc_application
        )
        self.dc_voltage_path_diagram = self._new_path_diagram("dc_voltage")
        vertical.insertWidget(0, self.dc_voltage_path_diagram)
        self._update_dc_voltage_path_note()
        return scroll

    def _browse_dc_application(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose DC voltage calibration database to apply",
            self.dc_application_path.text().strip(),
            "QCoDeS SQLite database (*.db)",
        )
        if path:
            self.dc_application_path.setText(path)
            self._emit_dc_application_changed()

    def _emit_dc_application_changed(self, *_args) -> None:
        self.dc_application_changed.emit(
            self.dc_application_group.isChecked(),
            self.dc_application_path.text().strip(),
            self.dc_application_run_id.value(),
        )

    def set_dc_application_selection(
        self,
        enabled: bool,
        database_path: str,
        run_id: int,
    ) -> None:
        """Set this Calibration tab's DC calibration selection."""
        with QtCore.QSignalBlocker(self.dc_application_path):
            self.dc_application_path.setText(str(database_path))
        with QtCore.QSignalBlocker(self.dc_application_run_id):
            self.dc_application_run_id.setValue(int(run_id))
        with QtCore.QSignalBlocker(self.dc_application_group):
            self.dc_application_group.setChecked(bool(enabled))
        self._emit_dc_application_changed()

    def dc_application_selection(self) -> Mapping[str, Any]:
        """Return this Calibration tab's DC calibration selection."""
        path = self.dc_application_path.text().strip()
        if self.dc_application_group.isChecked() and not path:
            raise ValueError(
                "DC voltage calibration application DB must not be empty"
            )
        return {
            "enabled": self.dc_application_group.isChecked(),
            "database_path": path,
            "run_id": self.dc_application_run_id.value(),
        }

    def _update_board_controls(self, *_args) -> None:
        output_rf = self.output_board.currentText() == "RF_Out"
        self.output_att1.setEnabled(output_rf)
        self.output_att2.setEnabled(output_rf)
        input_output_rf = self.input_output_board.currentText() == "RF_Out"
        self.input_output_att1.setEnabled(input_output_rf)
        self.input_output_att2.setEnabled(input_output_rf)
        input_rf = self.input_board.currentText() == "RF_In"
        self.input_attenuation.setEnabled(input_rf)
        self.input_dc_gain.setEnabled(not input_rf)

    def _browse_database(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Choose calibration database",
            self.database_path.text().strip() or DEFAULT_CALIBRATION_DB_PATH,
            "QCoDeS SQLite database (*.db)",
        )
        if path:
            path_value = Path(path)
            if path_value.suffix.lower() != ".db":
                path_value = path_value.with_suffix(".db")
            self.database_path.setText(str(path_value))

    def database_path_value(self) -> str:
        value = self.database_path.text().strip()
        if not value:
            raise ValueError("calibration database path must not be empty")
        path = Path(value).expanduser()
        if path.suffix.lower() != ".db":
            path = path.with_suffix(".db")
        return str(path)

    def output_config(self) -> OutputPowerCalibrationConfig:
        return OutputPowerCalibrationConfig(
            database_path=self.database_path_value(),
            output_board_type=self.output_board.currentText(),
            output_ch=self.output_ch.value(),
            frequency_start_mhz=self.output_frequency_start.value(),
            frequency_end_mhz=self.output_frequency_end.value(),
            frequency_points=self.output_frequency_points.value(),
            gain_start=self.output_gain_start.value(),
            gain_end=self.output_gain_end.value(),
            gain_points=self.output_gain_points.value(),
            gain_scale=str(self.output_gain_scale.currentData()),
            output_att1_db=(
                self.output_att1.value()
                if self.output_board.currentText() == "RF_Out"
                else 0.0
            ),
            output_att2_db=(
                self.output_att2.value()
                if self.output_board.currentText() == "RF_Out"
                else 0.0
            ),
            output_filter_type=self.output_filter_type.currentText(),
            output_filter_cutoff_ghz=self.output_filter_cutoff.value(),
            output_filter_bandwidth_ghz=self.output_filter_bandwidth.value(),
            nqz=self.output_path_diagram.applied_values()["output_nqz"],
            experiment_name=self.output_experiment_name.text().strip(),
            sample_name=self.output_sample_name.text().strip(),
            oscilloscope=OscilloscopeConfig(
                visa_resource=self.scope_resource.text().strip(),
                channel=self.scope_channel.value(),
                fft_function=self.scope_function.value(),
                span_mhz=self.scope_span.value(),
                average_count=self.scope_averages.value(),
                settle_seconds=self.scope_settle.value(),
            ),
        )

    def input_config(self) -> InputPowerCalibrationConfig:
        return InputPowerCalibrationConfig(
            database_path=self.database_path_value(),
            output_board_type=self.input_output_board.currentText(),
            input_board_type=self.input_board.currentText(),
            output_ch=self.input_output_ch.value(),
            readout_ch=self.input_readout_ch.value(),
            frequency_start_mhz=self.input_frequency_start.value(),
            frequency_end_mhz=self.input_frequency_end.value(),
            frequency_points=self.input_frequency_points.value(),
            gain_start=self.input_gain_start.value(),
            gain_end=self.input_gain_end.value(),
            gain_points=self.input_gain_points.value(),
            gain_scale=str(self.input_gain_scale.currentData()),
            scan_time_us=self.input_scan_time.value(),
            output_att1_db=(
                self.input_output_att1.value()
                if self.input_output_board.currentText() == "RF_Out"
                else 0.0
            ),
            output_att2_db=(
                self.input_output_att2.value()
                if self.input_output_board.currentText() == "RF_Out"
                else 0.0
            ),
            input_attenuation_db=(
                self.input_attenuation.value()
                if self.input_board.currentText() == "RF_In"
                else 0.0
            ),
            input_dc_gain_db=(
                self.input_dc_gain.value()
                if self.input_board.currentText() == "DC_In"
                else 0.0
            ),
            path_loss_db=self.input_path_loss.value(),
            output_filter_type=self.input_output_filter.currentText(),
            output_filter_cutoff_ghz=self.input_output_cutoff.value(),
            output_filter_bandwidth_ghz=self.input_output_bandwidth.value(),
            readout_filter_type=self.input_readout_filter.currentText(),
            readout_filter_cutoff_ghz=self.input_readout_cutoff.value(),
            readout_filter_bandwidth_ghz=self.input_readout_bandwidth.value(),
            nqz=self.input_path_diagram.applied_values()["output_nqz"],
            readout_nqz=self.input_path_diagram.applied_values()["readout_nqz"],
            fit_trim_low=self.input_trim_low.value(),
            fit_trim_high=self.input_trim_high.value(),
            experiment_name=self.input_experiment_name.text().strip(),
            sample_name=self.input_sample_name.text().strip(),
        )

    def dc_voltage_config(self) -> DcVoltageCalibrationConfig:
        path = self._front_panel_values_for("dc_voltage")
        output_ch = int(path["output_ch"])
        readout_ch = int(path["readout_ch"])
        input_gain_db = float(path["readout_dc_gain_db"])
        self.dc_voltage_output_ch.setValue(output_ch)
        self.dc_voltage_readout_ch.setValue(readout_ch)
        self.dc_voltage_input_gain.setValue(input_gain_db)
        self._update_dc_voltage_path_note()
        return DcVoltageCalibrationConfig(
            database_path=self.database_path_value(),
            output_ch=output_ch,
            readout_ch=readout_ch,
            voltage_start_mv=self.dc_voltage_start_mv.value(),
            voltage_stop_mv=self.dc_voltage_stop_mv.value(),
            voltage_points=self.dc_voltage_points.value(),
            output_full_scale_mv=self.dc_voltage_full_scale_mv.value(),
            samples_per_point=self.dc_voltage_samples.value(),
            repetitions_per_point=self.dc_voltage_repetitions.value(),
            input_dc_gain_db=input_gain_db,
            settle_us=self.dc_voltage_settle_us.value(),
            margin_input_samples=self.dc_voltage_margin_samples.value(),
            force_overwrite=self.dc_voltage_force_overwrite.isChecked(),
            experiment_name=self.dc_voltage_experiment_name.text().strip(),
            sample_name=self.dc_voltage_sample_name.text().strip(),
        )

    def apply_path_settings(
        self,
        values: Mapping[str, Any],
        *,
        mode: str | None = None,
    ) -> None:
        """Apply an RF path only to the selected calibration sub-tab."""
        mode = self._front_panel_mode if mode is None else str(mode)
        diagram = self.path_diagram_for(mode)
        diagram.apply_external_settings(values)
        output_ch = int(values["output_ch"])
        readout_ch = int(values["readout_ch"])
        output_board = str(values["output_board_type"])
        input_board = str(values["input_board_type"])
        att1 = float(values["output_att1_db"])
        att2 = float(values["output_att2_db"])
        input_att = float(values["readout_attenuation_db"])
        input_gain = float(values["readout_dc_gain_db"])

        if mode == "output":
            self.output_ch.setValue(output_ch)
            self.output_board.setCurrentText(output_board)
            self.output_att1.setValue(att1)
            self.output_att2.setValue(att2)
            if "output_filter_type" in values:
                self.output_filter_type.setCurrentText(
                    str(values["output_filter_type"])
                )
            if "output_filter_cutoff_ghz" in values:
                self.output_filter_cutoff.setValue(
                    float(values["output_filter_cutoff_ghz"])
                )
            if "output_filter_bandwidth_ghz" in values:
                self.output_filter_bandwidth.setValue(
                    float(values["output_filter_bandwidth_ghz"])
                )
        elif mode == "input":
            self.input_output_ch.setValue(output_ch)
            self.input_readout_ch.setValue(readout_ch)
            self.input_output_board.setCurrentText(output_board)
            self.input_board.setCurrentText(input_board)
            self.input_output_att1.setValue(att1)
            self.input_output_att2.setValue(att2)
            self.input_attenuation.setValue(input_att)
            self.input_dc_gain.setValue(input_gain)
            if "output_filter_type" in values:
                self.input_output_filter.setCurrentText(
                    str(values["output_filter_type"])
                )
            if "output_filter_cutoff_ghz" in values:
                self.input_output_cutoff.setValue(
                    float(values["output_filter_cutoff_ghz"])
                )
            if "output_filter_bandwidth_ghz" in values:
                self.input_output_bandwidth.setValue(
                    float(values["output_filter_bandwidth_ghz"])
                )
            if "readout_filter_type" in values:
                self.input_readout_filter.setCurrentText(
                    str(values["readout_filter_type"])
                )
            if "readout_filter_cutoff_ghz" in values:
                self.input_readout_cutoff.setValue(
                    float(values["readout_filter_cutoff_ghz"])
                )
            if "readout_filter_bandwidth_ghz" in values:
                self.input_readout_bandwidth.setValue(
                    float(values["readout_filter_bandwidth_ghz"])
                )
        elif mode == "dc_voltage":
            self.dc_voltage_output_ch.setValue(output_ch)
            self.dc_voltage_readout_ch.setValue(readout_ch)
            self.dc_voltage_input_gain.setValue(input_gain)
            self._update_dc_voltage_path_note()
        else:
            raise ValueError(f"unknown calibration path mode {mode!r}")
        self._update_board_controls()

    def _update_dc_voltage_path_note(self) -> None:
        if not hasattr(self, "dc_voltage_path_note"):
            return
        self.dc_voltage_path_note.setText(
            f"DC output generator {self.dc_voltage_output_ch.value()} -> "
            f"DC input readout {self.dc_voltage_readout_ch.value()} | "
            f"input gain {self.dc_voltage_input_gain.value():g} dB. "
            "Change this mapping with the front panel above."
        )

    def _apply_local_path_settings(
        self,
        mode: str,
        values: Mapping[str, Any],
    ) -> None:
        self.apply_path_settings(values, mode=mode)
        applied = dict(values)
        applied["calibration_mode"] = mode
        self.path_settings_applied.emit(applied)

    def front_panel_values(self) -> Mapping[str, Any]:
        """Return the active calibration sub-tab's path for graphical editing."""
        return self._front_panel_values_for(self._front_panel_mode)

    def _front_panel_values_for(self, mode: str) -> Mapping[str, Any]:
        values = self.path_diagram_for(mode)._editor_values()
        if mode == "output":
            values.update(
                {
                    "output_filter_type": self.output_filter_type.currentText(),
                    "output_filter_cutoff_ghz": self.output_filter_cutoff.value(),
                    "output_filter_bandwidth_ghz": self.output_filter_bandwidth.value(),
                }
            )
        elif mode == "input":
            values.update(
                {
                    "output_filter_type": self.input_output_filter.currentText(),
                    "output_filter_cutoff_ghz": self.input_output_cutoff.value(),
                    "output_filter_bandwidth_ghz": self.input_output_bandwidth.value(),
                    "readout_filter_type": self.input_readout_filter.currentText(),
                    "readout_filter_cutoff_ghz": self.input_readout_cutoff.value(),
                    "readout_filter_bandwidth_ghz": self.input_readout_bandwidth.value(),
                }
            )
        return values

    def apply_front_panel_settings(self, values: Mapping[str, Any]) -> None:
        """Apply graphical SMA settings only to this Calibration tab."""
        self.apply_path_settings(values)

    def set_front_panel_configuration(self, configuration) -> None:
        for diagram in self._path_diagrams.values():
            diagram.set_front_panel_configuration(configuration)

    def settings_dict(self) -> Mapping[str, Any]:
        output = asdict(self.output_config())
        output.pop("database_path")
        input_config = asdict(self.input_config())
        input_config.pop("database_path")
        dc_voltage = asdict(self.dc_voltage_config())
        dc_voltage.pop("database_path")
        dc_application = self.dc_application_selection()
        return {
            "database_path": self.database_path_value(),
            "selected_tab": self.tabs.currentIndex(),
            "output": output,
            "input": input_config,
            "dc_voltage": dc_voltage,
            "paths": {
                mode: self.path_diagram_for(mode).applied_values()
                for mode in CALIBRATION_PATH_MODES
            },
            "dc_voltage_application": dict(dc_application),
            "input_plot": {
                "x_scale": self.input_response_plot.x_scale_mode,
                "y_scale": self.input_response_plot.y_scale_mode,
            },
        }

    def load_settings(self, settings: Mapping[str, Any]) -> None:
        settings = dict(settings)
        database_path = str(settings.get("database_path", DEFAULT_CALIBRATION_DB_PATH))
        output_values = dict(settings.get("output", {}))
        input_values = dict(settings.get("input", {}))
        dc_voltage_values = dict(settings.get("dc_voltage", {}))
        dc_application = dict(settings.get("dc_voltage_application", {}))
        scope_values = dict(output_values.pop("oscilloscope", {}))
        output = OutputPowerCalibrationConfig(
            database_path=database_path,
            oscilloscope=OscilloscopeConfig(**scope_values),
            **output_values,
        )
        input_config = InputPowerCalibrationConfig(
            database_path=database_path,
            **input_values,
        )
        dc_voltage = DcVoltageCalibrationConfig(
            database_path=database_path,
            **dc_voltage_values,
        )
        path_values = normalize_calibration_paths(
            settings,
            output,
            input_config,
            dc_voltage,
        )
        self.database_path.setText(database_path)
        assignments = (
            (self.output_ch, output.output_ch),
            (self.output_frequency_start, output.frequency_start_mhz),
            (self.output_frequency_end, output.frequency_end_mhz),
            (self.output_frequency_points, output.frequency_points),
            (self.output_gain_start, output.gain_start),
            (self.output_gain_end, output.gain_end),
            (self.output_gain_points, output.gain_points),
            (self.output_att1, output.output_att1_db),
            (self.output_att2, output.output_att2_db),
            (self.output_filter_cutoff, output.output_filter_cutoff_ghz),
            (self.output_filter_bandwidth, output.output_filter_bandwidth_ghz),
            (self.scope_channel, output.oscilloscope.channel),
            (self.scope_function, output.oscilloscope.fft_function),
            (self.scope_span, output.oscilloscope.span_mhz),
            (self.scope_averages, output.oscilloscope.average_count),
            (self.scope_settle, output.oscilloscope.settle_seconds),
            (self.input_output_ch, input_config.output_ch),
            (self.input_readout_ch, input_config.readout_ch),
            (self.input_frequency_start, input_config.frequency_start_mhz),
            (self.input_frequency_end, input_config.frequency_end_mhz),
            (self.input_frequency_points, input_config.frequency_points),
            (self.input_gain_start, input_config.gain_start),
            (self.input_gain_end, input_config.gain_end),
            (self.input_gain_points, input_config.gain_points),
            (self.input_scan_time, input_config.scan_time_us),
            (self.input_output_att1, input_config.output_att1_db),
            (self.input_output_att2, input_config.output_att2_db),
            (self.input_attenuation, input_config.input_attenuation_db),
            (self.input_dc_gain, input_config.input_dc_gain_db),
            (self.input_path_loss, input_config.path_loss_db),
            (self.input_output_cutoff, input_config.output_filter_cutoff_ghz),
            (self.input_output_bandwidth, input_config.output_filter_bandwidth_ghz),
            (self.input_readout_cutoff, input_config.readout_filter_cutoff_ghz),
            (self.input_readout_bandwidth, input_config.readout_filter_bandwidth_ghz),
            (self.input_trim_low, input_config.fit_trim_low),
            (self.input_trim_high, input_config.fit_trim_high),
            (self.dc_voltage_output_ch, dc_voltage.output_ch),
            (self.dc_voltage_readout_ch, dc_voltage.readout_ch),
            (self.dc_voltage_start_mv, dc_voltage.voltage_start_mv),
            (self.dc_voltage_stop_mv, dc_voltage.voltage_stop_mv),
            (self.dc_voltage_points, dc_voltage.voltage_points),
            (self.dc_voltage_full_scale_mv, dc_voltage.output_full_scale_mv),
            (self.dc_voltage_samples, dc_voltage.samples_per_point),
            (self.dc_voltage_repetitions, dc_voltage.repetitions_per_point),
            (self.dc_voltage_input_gain, dc_voltage.input_dc_gain_db),
            (self.dc_voltage_settle_us, dc_voltage.settle_us),
            (self.dc_voltage_margin_samples, dc_voltage.margin_input_samples),
        )
        for widget, value in assignments:
            widget.setValue(value)
        self.output_board.setCurrentText(output.output_board_type)
        self.output_gain_scale.setCurrentIndex(
            self.output_gain_scale.findData(output.gain_scale)
        )
        self.output_filter_type.setCurrentText(output.output_filter_type)
        self.output_experiment_name.setText(output.experiment_name)
        self.output_sample_name.setText(output.sample_name)
        self.scope_resource.setText(output.oscilloscope.visa_resource)
        self.input_output_board.setCurrentText(input_config.output_board_type)
        self.input_board.setCurrentText(input_config.input_board_type)
        self.input_gain_scale.setCurrentIndex(
            self.input_gain_scale.findData(input_config.gain_scale)
        )
        self.input_output_filter.setCurrentText(input_config.output_filter_type)
        self.input_readout_filter.setCurrentText(input_config.readout_filter_type)
        self.input_experiment_name.setText(input_config.experiment_name)
        self.input_sample_name.setText(input_config.sample_name)
        self.dc_voltage_force_overwrite.setChecked(dc_voltage.force_overwrite)
        self.dc_voltage_experiment_name.setText(dc_voltage.experiment_name)
        self.dc_voltage_sample_name.setText(dc_voltage.sample_name)
        self.set_dc_application_selection(
            bool(dc_application.get("enabled", False)),
            str(dc_application.get("database_path", database_path)),
            int(dc_application.get("run_id", 0)),
        )
        input_plot = dict(settings.get("input_plot", {}))
        self.input_response_plot.set_axis_scales(
            str(input_plot.get("x_scale", "log")),
            str(input_plot.get("y_scale", "log")),
        )
        for mode, values in path_values.items():
            self.apply_path_settings(values, mode=mode)
        self._update_board_controls()
        self.tabs.setCurrentIndex(max(0, min(2, int(settings.get("selected_tab", 0)))))

    def set_running(self, running: bool, message: str) -> None:
        self.run_output_button.setEnabled(not running)
        self.run_input_button.setEnabled(not running)
        self.run_dc_voltage_button.setEnabled(not running)
        self.dc_application_group.setEnabled(not running)
        self.database_path.setEnabled(not running)
        self.browse_database.setEnabled(not running)
        for diagram in self._path_diagrams.values():
            diagram.setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self.progress.setValue(0)
        self.status.setText(message)

    def update_progress(self, percent: int, message: str) -> None:
        percent = max(0, min(100, int(percent)))
        self.progress.setValue(percent)
        self.status.setText(f"{percent}% - {message}")

    def show_result(self, stored) -> None:
        plot_message = ""
        result = getattr(stored, "result", None)
        if isinstance(result, Mapping) and {
            "frequencies_mhz",
            "gains",
            "input_power_dbm",
            "adc_magnitude_db",
        }.issubset(result):
            try:
                self.input_response_plot.set_result(result)
            except (TypeError, ValueError) as exc:
                plot_message = f"\nPlot unavailable: {exc}"
            else:
                self.tabs.setCurrentIndex(1)
        elif isinstance(result, Mapping) and "calibration" in result:
            calibration = result.get("calibration", {})
            try:
                self.dc_voltage_response_plot.set_result(result)
            except (TypeError, ValueError) as exc:
                plot_message = f"\nPlot unavailable: {exc}"
            if isinstance(calibration, Mapping):
                fit_message = (
                    "\n0 MHz DC voltage fit: "
                    f"R^2={float(calibration.get('r_squared', float('nan'))):.8f}, "
                    f"RMSE={float(calibration.get('rmse_adc', float('nan'))):.6g} ADC"
                )
                plot_message = fit_message + plot_message
            self.tabs.setCurrentIndex(2)
            self.set_dc_application_selection(
                True,
                str(stored.database_path),
                int(stored.run_id),
            )
        self.set_running(
            False,
            (
                f"{stored.board_type} calibration Run {stored.run_id}: "
                f"{stored.row_count:,} rows\n{stored.database_path}{plot_message}"
            ),
        )


class CalibrationWorker(QtCore.QObject):
    """Run one blocking calibration outside the GUI thread."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress_changed = QtCore.pyqtSignal(int, str)

    def __init__(self, mode: str, kwargs: Mapping[str, Any], parent=None):
        super().__init__(parent)
        if mode not in ("output", "input", "dc_voltage"):
            raise ValueError("calibration mode must be output, input, or dc_voltage")
        self.mode = mode
        self.kwargs = dict(kwargs)

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            kwargs = dict(self.kwargs)
            kwargs["progress_callback"] = self.progress_changed.emit
            runners = {
                "output": run_output_power_calibration,
                "input": run_input_power_calibration,
                "dc_voltage": run_dc_voltage_calibration,
            }
            stored = runners[self.mode](**kwargs)
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(stored)


def default_calibration_settings() -> Mapping[str, Any]:
    """Return JSON-safe defaults for old settings files without this tab."""
    output = asdict(
        OutputPowerCalibrationConfig(database_path=DEFAULT_CALIBRATION_DB_PATH)
    )
    input_config = asdict(
        InputPowerCalibrationConfig(database_path=DEFAULT_CALIBRATION_DB_PATH)
    )
    dc_voltage = asdict(
        DcVoltageCalibrationConfig(database_path=DEFAULT_CALIBRATION_DB_PATH)
    )
    output.pop("database_path")
    input_config.pop("database_path")
    dc_voltage.pop("database_path")
    paths = _legacy_calibration_paths(
        OutputPowerCalibrationConfig(database_path=DEFAULT_CALIBRATION_DB_PATH),
        InputPowerCalibrationConfig(database_path=DEFAULT_CALIBRATION_DB_PATH),
        DcVoltageCalibrationConfig(database_path=DEFAULT_CALIBRATION_DB_PATH),
    )
    return {
        "database_path": DEFAULT_CALIBRATION_DB_PATH,
        "selected_tab": 0,
        "output": output,
        "input": input_config,
        "dc_voltage": dc_voltage,
        "paths": paths,
        "dc_voltage_application": {
            "enabled": False,
            "database_path": DEFAULT_CALIBRATION_DB_PATH,
            "run_id": 0,
        },
        "input_plot": {
            "x_scale": "log",
            "y_scale": "log",
        },
    }


__all__ = [
    "CALIBRATION_PATH_MODES",
    "CalibrationPanel",
    "CalibrationWorker",
    "DEFAULT_CALIBRATION_DB_PATH",
    "DcVoltageCalibrationPlotWidget",
    "InputCalibrationPlotWidget",
    "dc_voltage_calibration_plot_data",
    "default_calibration_settings",
    "input_calibration_plot_data",
    "normalize_calibration_paths",
]
