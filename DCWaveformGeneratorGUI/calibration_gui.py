"""GUI controls and workers for QICK power calibration.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import traceback
from typing import Any, Mapping

from PyQt5 import QtCore, QtWidgets

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


DEFAULT_CALIBRATION_DB_PATH = str(Path.home() / "gain_pwr_calb.db")


class CalibrationPanel(QtWidgets.QWidget):
    """Output-scope and input-ADC calibration controls."""

    output_requested = QtCore.pyqtSignal()
    input_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

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
        for label, widget in (
            ("Output board:", self.output_board),
            ("QICK generator channel:", self.output_ch),
            ("Start frequency:", self.output_frequency_start),
            ("End frequency:", self.output_frequency_end),
            ("Frequency points:", self.output_frequency_points),
            ("Start gain:", self.output_gain_start),
            ("End gain:", self.output_gain_end),
            ("Gain points:", self.output_gain_points),
            ("Gain spacing:", self.output_gain_scale),
            ("Output ATT1:", self.output_att1),
            ("Output ATT2:", self.output_att2),
            ("Output filter:", self.output_filter_type),
            ("Filter cutoff/center:", self.output_filter_cutoff),
            ("Filter bandwidth:", self.output_filter_bandwidth),
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
        for label, widget in (
            ("Calibrated output board:", self.input_output_board),
            ("Input board:", self.input_board),
            ("QICK generator channel:", self.input_output_ch),
            ("QICK readout channel:", self.input_readout_ch),
            ("Start frequency:", self.input_frequency_start),
            ("End frequency:", self.input_frequency_end),
            ("Frequency points:", self.input_frequency_points),
            ("Start gain:", self.input_gain_start),
            ("End gain:", self.input_gain_end),
            ("Gain points:", self.input_gain_points),
            ("Gain spacing:", self.input_gain_scale),
            ("FIR scan time per point:", self.input_scan_time),
            ("Output ATT1:", self.input_output_att1),
            ("Output ATT2:", self.input_output_att2),
            ("Input attenuation:", self.input_attenuation),
            ("DC input gain:", self.input_dc_gain),
            ("External path loss:", self.input_path_loss),
            ("Output filter:", self.input_output_filter),
            ("Output cutoff/center:", self.input_output_cutoff),
            ("Output bandwidth:", self.input_output_bandwidth),
            ("Input filter:", self.input_readout_filter),
            ("Input cutoff/center:", self.input_readout_cutoff),
            ("Input bandwidth:", self.input_readout_bandwidth),
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
            fit_trim_low=self.input_trim_low.value(),
            fit_trim_high=self.input_trim_high.value(),
            experiment_name=self.input_experiment_name.text().strip(),
            sample_name=self.input_sample_name.text().strip(),
        )

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
        self._update_board_controls()
        self.tabs.setCurrentIndex(max(0, min(1, int(settings.get("selected_tab", 0)))))

    def set_running(self, running: bool, message: str) -> None:
        self.run_output_button.setEnabled(not running)
        self.run_input_button.setEnabled(not running)
        self.database_path.setEnabled(not running)
        self.browse_database.setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self.progress.setValue(0)
        self.status.setText(message)

    def update_progress(self, percent: int, message: str) -> None:
        percent = max(0, min(100, int(percent)))
        self.progress.setValue(percent)
        self.status.setText(f"{percent}% - {message}")

    def show_result(self, stored) -> None:
        self.set_running(
            False,
            (
                f"{stored.board_type} calibration Run {stored.run_id}: "
                f"{stored.row_count:,} rows\n{stored.database_path}"
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
    }


__all__ = [
    "CalibrationPanel",
    "CalibrationWorker",
    "DEFAULT_CALIBRATION_DB_PATH",
    "default_calibration_settings",
]
