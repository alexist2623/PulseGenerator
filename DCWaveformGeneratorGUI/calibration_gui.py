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
    import pyqtgraph as pg
except ImportError:
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


class CalibrationPanel(QtWidgets.QWidget):
    """Output-scope and input-ADC calibration controls."""

    output_requested = QtCore.pyqtSignal()
    input_requested = QtCore.pyqtSignal()
    path_settings_applied = QtCore.pyqtSignal(object)
    front_panel_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        self.path_diagram = RfPathCorrectionWidget(self, compact=True)
        self.path_diagram.settings_applied.connect(self.path_settings_applied.emit)
        self.path_diagram.front_panel_requested.connect(
            self.front_panel_requested.emit
        )
        layout.addWidget(self.path_diagram)

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
                "Board, channel, ATT, filter, and Nyquist settings use the shared "
                "RF path above."
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
                "Board, channel, ATT, filter, and Nyquist settings use the shared "
                "RF path above."
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
        return scroll

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
            nqz=self.path_diagram.applied_values()["output_nqz"],
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
            nqz=self.path_diagram.applied_values()["output_nqz"],
            readout_nqz=self.path_diagram.applied_values()["readout_nqz"],
            fit_trim_low=self.input_trim_low.value(),
            fit_trim_high=self.input_trim_high.value(),
            experiment_name=self.input_experiment_name.text().strip(),
            sample_name=self.input_sample_name.text().strip(),
        )

    def apply_path_settings(self, values: Mapping[str, Any]) -> None:
        """Apply the shared front-panel RF path to both calibration modes."""
        self.path_diagram.apply_external_settings(values)
        output_ch = int(values["output_ch"])
        readout_ch = int(values["readout_ch"])
        output_board = str(values["output_board_type"])
        input_board = str(values["input_board_type"])
        att1 = float(values["output_att1_db"])
        att2 = float(values["output_att2_db"])
        input_att = float(values["readout_attenuation_db"])
        input_gain = float(values["readout_dc_gain_db"])

        self.output_ch.setValue(output_ch)
        self.output_board.setCurrentText(output_board)
        self.output_att1.setValue(att1)
        self.output_att2.setValue(att2)

        self.input_output_ch.setValue(output_ch)
        self.input_readout_ch.setValue(readout_ch)
        self.input_output_board.setCurrentText(output_board)
        self.input_board.setCurrentText(input_board)
        self.input_output_att1.setValue(att1)
        self.input_output_att2.setValue(att2)
        self.input_attenuation.setValue(input_att)
        self.input_dc_gain.setValue(input_gain)
        if "output_filter_type" in values:
            self.output_filter_type.setCurrentText(
                str(values["output_filter_type"])
            )
            self.input_output_filter.setCurrentText(
                str(values["output_filter_type"])
            )
        if "output_filter_cutoff_ghz" in values:
            cutoff = float(values["output_filter_cutoff_ghz"])
            self.output_filter_cutoff.setValue(cutoff)
            self.input_output_cutoff.setValue(cutoff)
        if "output_filter_bandwidth_ghz" in values:
            bandwidth = float(values["output_filter_bandwidth_ghz"])
            self.output_filter_bandwidth.setValue(bandwidth)
            self.input_output_bandwidth.setValue(bandwidth)
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
        self._update_board_controls()

    def set_front_panel_configuration(self, configuration) -> None:
        self.path_diagram.set_front_panel_configuration(configuration)

    def settings_dict(self) -> Mapping[str, Any]:
        output = asdict(self.output_config())
        output.pop("database_path")
        input_config = asdict(self.input_config())
        input_config.pop("database_path")
        return {
            "database_path": self.database_path_value(),
            "selected_tab": self.tabs.currentIndex(),
            "output": output,
            "input": input_config,
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
        input_plot = dict(settings.get("input_plot", {}))
        self.input_response_plot.set_axis_scales(
            str(input_plot.get("x_scale", "log")),
            str(input_plot.get("y_scale", "log")),
        )
        self.path_diagram.apply_external_settings(
            {
                "output_ch": input_config.output_ch,
                "readout_ch": input_config.readout_ch,
                "output_nqz": input_config.nqz,
                "readout_nqz": input_config.readout_nqz,
                "output_board_type": input_config.output_board_type,
                "input_board_type": input_config.input_board_type,
                "output_att1_db": input_config.output_att1_db,
                "output_att2_db": input_config.output_att2_db,
                "readout_attenuation_db": input_config.input_attenuation_db,
                "readout_dc_gain_db": input_config.input_dc_gain_db,
            }
        )
        self._update_board_controls()
        self.tabs.setCurrentIndex(max(0, min(1, int(settings.get("selected_tab", 0)))))

    def set_running(self, running: bool, message: str) -> None:
        self.run_output_button.setEnabled(not running)
        self.run_input_button.setEnabled(not running)
        self.database_path.setEnabled(not running)
        self.browse_database.setEnabled(not running)
        self.path_diagram.setEnabled(not running)
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
        if mode not in ("output", "input"):
            raise ValueError("calibration mode must be output or input")
        self.mode = mode
        self.kwargs = dict(kwargs)

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            kwargs = dict(self.kwargs)
            kwargs["progress_callback"] = self.progress_changed.emit
            stored = (
                run_output_power_calibration(**kwargs)
                if self.mode == "output"
                else run_input_power_calibration(**kwargs)
            )
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
    output.pop("database_path")
    input_config.pop("database_path")
    return {
        "database_path": DEFAULT_CALIBRATION_DB_PATH,
        "selected_tab": 0,
        "output": output,
        "input": input_config,
        "input_plot": {
            "x_scale": "log",
            "y_scale": "log",
        },
    }


__all__ = [
    "CalibrationPanel",
    "CalibrationWorker",
    "DEFAULT_CALIBRATION_DB_PATH",
    "InputCalibrationPlotWidget",
    "default_calibration_settings",
    "input_calibration_plot_data",
]
