"""Independent RF S-parameter sweep controls and result plotting.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import traceback
from typing import Any, Mapping

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
    from .qick_sparameter_sweep import (
        FILTER_TYPES,
        MAX_RF_OUTPUT_GAIN,
        POWER_SCALES,
        SParameterSweepConfig,
        load_sparameter_run,
        run_sparameter_sweep,
    )
    from .power_calibration import INPUT_BOARD_TYPES, OUTPUT_BOARD_TYPES
except ImportError:
    from qick_sparameter_sweep import (
        FILTER_TYPES,
        MAX_RF_OUTPUT_GAIN,
        POWER_SCALES,
        SParameterSweepConfig,
        load_sparameter_run,
        run_sparameter_sweep,
    )
    from power_calibration import INPUT_BOARD_TYPES, OUTPUT_BOARD_TYPES


DEFAULT_SPARAMETER_DB_PATH = str(
    Path.home() / "qick_sparameter_experiments.db"
)


class SParameterSweepPanel(QtWidgets.QWidget):
    """Controls for an RF-only generator/readout hardware frequency sweep."""

    run_requested = QtCore.pyqtSignal()
    load_requested = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget(scroll)
        content_layout = QtWidgets.QVBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        sweep_group = QtWidgets.QGroupBox("Frequency Sweep")
        sweep_form = QtWidgets.QFormLayout(sweep_group)
        self.output_ch = QtWidgets.QSpinBox()
        self.output_ch.setRange(0, 255)
        self.readout_ch = QtWidgets.QSpinBox()
        self.readout_ch.setRange(0, 255)
        self.frequency_start_mhz = self._frequency_spin(10.0)
        self.frequency_end_mhz = self._frequency_spin(100.0)
        self.frequency_points = QtWidgets.QSpinBox()
        self.frequency_points.setRange(2, 1_000_000)
        self.frequency_points.setValue(101)
        self.gain = QtWidgets.QSpinBox()
        self.gain.setRange(0, MAX_RF_OUTPUT_GAIN)
        self.gain.setValue(20000)
        self.gain.setSuffix(f" / {MAX_RF_OUTPUT_GAIN}")
        self.output_power_dbm = self._power_spin(-20.0)
        self.scan_time_us = QtWidgets.QDoubleSpinBox()
        self.scan_time_us.setRange(0.001, 1.0e6)
        self.scan_time_us.setDecimals(6)
        self.scan_time_us.setValue(10.0)
        self.scan_time_us.setSuffix(" us")
        sweep_form.addRow("RF output channel:", self.output_ch)
        sweep_form.addRow("Readout channel:", self.readout_ch)
        sweep_form.addRow("Start frequency:", self.frequency_start_mhz)
        sweep_form.addRow("End frequency:", self.frequency_end_mhz)
        sweep_form.addRow("Frequency points:", self.frequency_points)
        sweep_form.addRow("Single output gain:", self.gain)
        sweep_form.addRow("Single target power:", self.output_power_dbm)
        sweep_form.addRow("Scan time per point:", self.scan_time_us)
        content_layout.addWidget(sweep_group)

        self.power_calibration_enabled = QtWidgets.QGroupBox(
            "Frequency Response Compensation"
        )
        self.power_calibration_enabled.setCheckable(True)
        self.power_calibration_enabled.setChecked(False)
        calibration_form = QtWidgets.QFormLayout(
            self.power_calibration_enabled
        )
        self.calibration_database_path = QtWidgets.QLineEdit()
        self.calibration_database_path.setPlaceholderText(
            "Select gain_pwr_calb.db"
        )
        self.browse_calibration_database = QtWidgets.QToolButton()
        self.browse_calibration_database.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        self.browse_calibration_database.setToolTip(
            "Choose a QCoDeS frequency-response calibration database"
        )
        self.browse_calibration_database.clicked.connect(
            self._browse_calibration_database
        )
        calibration_path_row = QtWidgets.QHBoxLayout()
        calibration_path_row.addWidget(self.calibration_database_path, 1)
        calibration_path_row.addWidget(self.browse_calibration_database)
        self.output_board_type = QtWidgets.QComboBox()
        self.output_board_type.addItems(OUTPUT_BOARD_TYPES)
        self.output_board_type.setCurrentText("RF_Out")
        self.input_board_type = QtWidgets.QComboBox()
        self.input_board_type.addItems(INPUT_BOARD_TYPES)
        self.input_board_type.setCurrentText("DC_In")
        self.calibration_hint = QtWidgets.QLabel(
            "Same-board response; normalized to the weakest frequency in the sweep."
        )
        self.calibration_hint.setWordWrap(True)
        calibration_form.addRow("Calibration DB:", calibration_path_row)
        calibration_form.addRow("Output board:", self.output_board_type)
        calibration_form.addRow("Input board:", self.input_board_type)
        calibration_form.addRow(self.calibration_hint)
        content_layout.addWidget(self.power_calibration_enabled)

        self.power_sweep_enabled = QtWidgets.QGroupBox(
            "Power Sweep (Software)"
        )
        self.power_sweep_enabled.setCheckable(True)
        self.power_sweep_enabled.setChecked(False)
        power_form = QtWidgets.QFormLayout(self.power_sweep_enabled)
        self.power_start_gain = self._gain_spin(1000)
        self.power_end_gain = self._gain_spin(20000)
        self.power_start_dbm = self._power_spin(-30.0)
        self.power_end_dbm = self._power_spin(-10.0)
        self.power_points = QtWidgets.QSpinBox()
        self.power_points.setRange(2, 100_000)
        self.power_points.setValue(5)
        self.power_scale = QtWidgets.QComboBox()
        scale_labels = {"linear": "Linear", "log": "Logarithmic"}
        for scale in POWER_SCALES:
            self.power_scale.addItem(scale_labels[scale], scale)
        power_form.addRow("Start gain code:", self.power_start_gain)
        power_form.addRow("End gain code:", self.power_end_gain)
        power_form.addRow("Start target power:", self.power_start_dbm)
        power_form.addRow("End target power:", self.power_end_dbm)
        power_form.addRow("Power points:", self.power_points)
        power_form.addRow("Spacing:", self.power_scale)
        self.power_sweep_enabled.toggled.connect(
            self._update_power_control_state
        )
        self.power_calibration_enabled.toggled.connect(
            self._update_power_control_state
        )
        content_layout.addWidget(self.power_sweep_enabled)
        self._update_power_control_state(False)

        output_group = QtWidgets.QGroupBox("RF Output Chain")
        output_form = QtWidgets.QFormLayout(output_group)
        self.output_att1_db = self._attenuation_spin(10.0)
        self.output_att2_db = self._attenuation_spin(10.0)
        self.output_filter_type = self._filter_combo()
        self.output_filter_cutoff_ghz = self._filter_spin(2.5)
        self.output_filter_bandwidth_ghz = self._filter_spin(1.0)
        output_form.addRow("ATT1:", self.output_att1_db)
        output_form.addRow("ATT2:", self.output_att2_db)
        output_form.addRow("Filter:", self.output_filter_type)
        output_form.addRow("Cutoff/center:", self.output_filter_cutoff_ghz)
        output_form.addRow("Bandwidth:", self.output_filter_bandwidth_ghz)
        content_layout.addWidget(output_group)

        readout_group = QtWidgets.QGroupBox("RF Readout Chain")
        readout_form = QtWidgets.QFormLayout(readout_group)
        self.readout_attenuation_db = self._attenuation_spin(20.0)
        self.readout_filter_type = self._filter_combo()
        self.readout_filter_cutoff_ghz = self._filter_spin(2.5)
        self.readout_filter_bandwidth_ghz = self._filter_spin(1.0)
        readout_form.addRow("Input attenuation:", self.readout_attenuation_db)
        readout_form.addRow("Filter:", self.readout_filter_type)
        readout_form.addRow("Cutoff/center:", self.readout_filter_cutoff_ghz)
        readout_form.addRow("Bandwidth:", self.readout_filter_bandwidth_ghz)
        content_layout.addWidget(readout_group)

        capture_group = QtWidgets.QGroupBox("FIR DDR Capture")
        capture_form = QtWidgets.QFormLayout(capture_group)
        self.margin_input_samples = QtWidgets.QSpinBox()
        self.margin_input_samples.setRange(0, 10_000_000)
        self.margin_input_samples.setValue(1024)
        self.address = QtWidgets.QSpinBox()
        self.address.setRange(0, 2_147_483_647)
        self.stride_bytes = QtWidgets.QSpinBox()
        self.stride_bytes.setRange(0, 2_147_483_647)
        self.stride_bytes.setSpecialValueText("Automatic")
        self.force_overwrite = QtWidgets.QCheckBox("Allow DDR overwrite")
        capture_form.addRow("FIR input margin:", self.margin_input_samples)
        capture_form.addRow("DDR start address:", self.address)
        capture_form.addRow("Trigger stride (bytes):", self.stride_bytes)
        capture_form.addRow(self.force_overwrite)
        content_layout.addWidget(capture_group)

        storage_group = QtWidgets.QGroupBox("S-Parameter Database")
        storage_form = QtWidgets.QFormLayout(storage_group)
        self.database_path = QtWidgets.QLineEdit(DEFAULT_SPARAMETER_DB_PATH)
        self.browse_database = QtWidgets.QToolButton()
        self.browse_database.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)
        )
        self.browse_database.setToolTip(
            "Choose RF S-parameter QCoDeS SQLite database"
        )
        self.browse_database.clicked.connect(self._browse_database)
        database_row = QtWidgets.QHBoxLayout()
        database_row.addWidget(self.database_path, 1)
        database_row.addWidget(self.browse_database)
        storage_form.addRow("QCoDeS DB file:", database_row)
        content_layout.addWidget(storage_group)
        content_layout.addStretch(1)

        self.run_button = QtWidgets.QPushButton("Run RF S-Parameter Sweep")
        self.run_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        )
        self.run_button.clicked.connect(self.run_requested.emit)
        self.run_id = QtWidgets.QSpinBox()
        self.run_id.setRange(0, 2_147_483_647)
        self.run_id.setSpecialValueText("Latest S-parameter run")
        self.load_button = QtWidgets.QPushButton("Load Saved Run")
        self.load_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        self.load_button.clicked.connect(
            lambda: self.load_requested.emit(self.run_id.value())
        )
        load_row = QtWidgets.QHBoxLayout()
        load_row.addWidget(self.run_id, 1)
        load_row.addWidget(self.load_button)
        outer.addWidget(self.run_button)
        outer.addLayout(load_row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        outer.addWidget(self.progress)
        self.status = QtWidgets.QLabel("Ready")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        self._internal_settings = {
            "nqz": 1,
            "settle_seconds": 0.05,
            "trigger_width_tproc_cycles": 12,
            "recovery_tproc_cycles": 20,
        }

    @staticmethod
    def _frequency_spin(value: float) -> QtWidgets.QDoubleSpinBox:
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(-100_000.0, 100_000.0)
        widget.setDecimals(9)
        widget.setValue(value)
        widget.setSuffix(" MHz")
        return widget

    @staticmethod
    def _gain_spin(value: int) -> QtWidgets.QSpinBox:
        widget = QtWidgets.QSpinBox()
        widget.setRange(0, MAX_RF_OUTPUT_GAIN)
        widget.setValue(value)
        widget.setSuffix(f" / {MAX_RF_OUTPUT_GAIN}")
        return widget

    @staticmethod
    def _power_spin(value: float) -> QtWidgets.QDoubleSpinBox:
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(-200.0, 100.0)
        widget.setDecimals(6)
        widget.setValue(value)
        widget.setSuffix(" dBm")
        return widget

    @staticmethod
    def _attenuation_spin(value: float) -> QtWidgets.QDoubleSpinBox:
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(0.0, 31.75)
        widget.setDecimals(2)
        widget.setSingleStep(0.25)
        widget.setValue(value)
        widget.setSuffix(" dB")
        return widget

    @staticmethod
    def _filter_spin(value: float) -> QtWidgets.QDoubleSpinBox:
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(0.001, 100.0)
        widget.setDecimals(6)
        widget.setValue(value)
        widget.setSuffix(" GHz")
        return widget

    @staticmethod
    def _filter_combo() -> QtWidgets.QComboBox:
        widget = QtWidgets.QComboBox()
        widget.addItems(FILTER_TYPES)
        return widget

    def _update_power_control_state(self, _enabled: bool) -> None:
        sweep_enabled = self.power_sweep_enabled.isChecked()
        calibrated = self.power_calibration_enabled.isChecked()
        self.gain.setEnabled(not sweep_enabled and not calibrated)
        self.output_power_dbm.setEnabled(not sweep_enabled and calibrated)
        for widget in (
            self.power_start_gain,
            self.power_end_gain,
        ):
            widget.setEnabled(sweep_enabled and not calibrated)
        for widget in (self.power_start_dbm, self.power_end_dbm):
            widget.setEnabled(sweep_enabled and calibrated)
        self.power_points.setEnabled(sweep_enabled)
        self.power_scale.setEnabled(sweep_enabled)
        for widget in (
            self.calibration_database_path,
            self.browse_calibration_database,
            self.output_board_type,
            self.input_board_type,
        ):
            widget.setEnabled(calibrated)

    def _browse_calibration_database(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose frequency-response calibration database",
            self.calibration_database_path.text().strip(),
            "QCoDeS SQLite database (*.db)",
        )
        if path:
            self.calibration_database_path.setText(path)

    def _browse_database(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Choose RF S-parameter database",
            self.database_path.text().strip() or DEFAULT_SPARAMETER_DB_PATH,
            "QCoDeS SQLite database (*.db)",
        )
        if path:
            if Path(path).suffix.lower() != ".db":
                path = str(Path(path).with_suffix(".db"))
            self.database_path.setText(path)

    def database_path_value(self) -> str:
        value = self.database_path.text().strip()
        if not value:
            raise ValueError("RF S-parameter database path must not be empty")
        path = Path(value).expanduser()
        if path.suffix.lower() != ".db":
            path = path.with_suffix(".db")
        return str(path)

    def config(self) -> SParameterSweepConfig:
        return SParameterSweepConfig(
            output_ch=self.output_ch.value(),
            readout_ch=self.readout_ch.value(),
            frequency_start_mhz=self.frequency_start_mhz.value(),
            frequency_end_mhz=self.frequency_end_mhz.value(),
            frequency_points=self.frequency_points.value(),
            gain=self.gain.value(),
            power_calibration_enabled=(
                self.power_calibration_enabled.isChecked()
            ),
            calibration_database_path=(
                self.calibration_database_path.text().strip()
            ),
            output_board_type=self.output_board_type.currentText(),
            input_board_type=self.input_board_type.currentText(),
            output_power_dbm=self.output_power_dbm.value(),
            power_sweep_enabled=self.power_sweep_enabled.isChecked(),
            power_start_gain=self.power_start_gain.value(),
            power_end_gain=self.power_end_gain.value(),
            power_start_dbm=self.power_start_dbm.value(),
            power_end_dbm=self.power_end_dbm.value(),
            power_points=self.power_points.value(),
            power_scale=str(self.power_scale.currentData()),
            scan_time_us=self.scan_time_us.value(),
            output_att1_db=self.output_att1_db.value(),
            output_att2_db=self.output_att2_db.value(),
            output_filter_type=self.output_filter_type.currentText(),
            output_filter_cutoff_ghz=self.output_filter_cutoff_ghz.value(),
            output_filter_bandwidth_ghz=(
                self.output_filter_bandwidth_ghz.value()
            ),
            readout_attenuation_db=self.readout_attenuation_db.value(),
            readout_filter_type=self.readout_filter_type.currentText(),
            readout_filter_cutoff_ghz=self.readout_filter_cutoff_ghz.value(),
            readout_filter_bandwidth_ghz=(
                self.readout_filter_bandwidth_ghz.value()
            ),
            margin_input_samples=self.margin_input_samples.value(),
            address=self.address.value(),
            stride_bytes=(
                None if self.stride_bytes.value() == 0 else self.stride_bytes.value()
            ),
            force_overwrite=self.force_overwrite.isChecked(),
            **self._internal_settings,
        )

    def settings_dict(self) -> Mapping[str, Any]:
        return {
            "database_path": self.database_path_value(),
            **asdict(self.config()),
        }

    def load_settings(self, settings: Mapping[str, Any]) -> None:
        values = dict(settings)
        database_path = str(
            values.pop("database_path", DEFAULT_SPARAMETER_DB_PATH)
        ).strip()
        if not database_path:
            raise ValueError("RF S-parameter database path must not be empty")
        config = SParameterSweepConfig(**values)
        widgets = (
            (self.output_ch, config.output_ch),
            (self.readout_ch, config.readout_ch),
            (self.frequency_start_mhz, config.frequency_start_mhz),
            (self.frequency_end_mhz, config.frequency_end_mhz),
            (self.frequency_points, config.frequency_points),
            (self.gain, config.gain),
            (self.output_power_dbm, config.output_power_dbm),
            (self.power_start_gain, config.power_start_gain),
            (self.power_end_gain, config.power_end_gain),
            (self.power_start_dbm, config.power_start_dbm),
            (self.power_end_dbm, config.power_end_dbm),
            (self.power_points, config.power_points),
            (self.scan_time_us, config.scan_time_us),
            (self.output_att1_db, config.output_att1_db),
            (self.output_att2_db, config.output_att2_db),
            (self.output_filter_cutoff_ghz, config.output_filter_cutoff_ghz),
            (
                self.output_filter_bandwidth_ghz,
                config.output_filter_bandwidth_ghz,
            ),
            (self.readout_attenuation_db, config.readout_attenuation_db),
            (self.readout_filter_cutoff_ghz, config.readout_filter_cutoff_ghz),
            (
                self.readout_filter_bandwidth_ghz,
                config.readout_filter_bandwidth_ghz,
            ),
            (self.margin_input_samples, config.margin_input_samples),
            (self.address, config.address),
            (self.stride_bytes, config.stride_bytes or 0),
        )
        for widget, value in widgets:
            widget.setValue(value)
        self.output_filter_type.setCurrentText(config.output_filter_type)
        self.readout_filter_type.setCurrentText(config.readout_filter_type)
        self.calibration_database_path.setText(
            config.calibration_database_path
        )
        self.output_board_type.setCurrentText(config.output_board_type)
        self.input_board_type.setCurrentText(config.input_board_type)
        self.database_path.setText(database_path)
        power_scale_index = self.power_scale.findData(config.power_scale)
        if power_scale_index < 0:
            raise ValueError(f"unsupported power scale {config.power_scale!r}")
        self.power_scale.setCurrentIndex(power_scale_index)
        self.power_calibration_enabled.setChecked(
            config.power_calibration_enabled
        )
        self.power_sweep_enabled.setChecked(config.power_sweep_enabled)
        self._update_power_control_state(config.power_sweep_enabled)
        self.force_overwrite.setChecked(config.force_overwrite)
        self._internal_settings = {
            "nqz": config.nqz,
            "settle_seconds": config.settle_seconds,
            "trigger_width_tproc_cycles": config.trigger_width_tproc_cycles,
            "recovery_tproc_cycles": config.recovery_tproc_cycles,
        }

    def set_running(self, running: bool, message: str) -> None:
        self.run_button.setEnabled(not running)
        self.load_button.setEnabled(not running)
        self.database_path.setEnabled(not running)
        self.browse_database.setEnabled(not running)
        self.power_calibration_enabled.setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self.progress.setValue(0)
        self.status.setText(message)

    def update_progress(self, percent: int, message: str) -> None:
        percent = max(0, min(100, int(percent)))
        self.progress.setValue(percent)
        self.status.setText(f"{percent}% - {message}")

    def show_result(self, stored) -> None:
        result = stored.result
        self.run_id.setValue(stored.run_id)
        power_count = int(getattr(result, "power_count", 1))
        self.set_running(
            False,
            (
                f"Run {stored.run_id}: {power_count} power point(s) x "
                f"{result.frequencies_mhz.size} frequency points, "
                f"{result.sample_count} FIR samples per point\n"
                f"{stored.database_path}"
            ),
        )

    def show_partial_result(self, stored) -> None:
        result = stored.result
        self.run_id.setValue(stored.run_id)
        power_count = int(getattr(result, "power_count", 1))
        self.status.setText(
            f"Run {stored.run_id}: {power_count} power point(s) saved to DB"
        )


if _USE_PYQTGRAPH:
    class SParameterPlotWidget(pg.GraphicsLayoutWidget):
        """Magnitude and unwrapped-phase plots sharing one frequency axis."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setBackground("w")
            self.magnitude_plot = self.addPlot(row=0, col=0)
            self.phase_plot = self.addPlot(row=1, col=0)
            self.phase_plot.setXLink(self.magnitude_plot)
            self.magnitude_plot.showGrid(x=True, y=True, alpha=0.25)
            self.phase_plot.showGrid(x=True, y=True, alpha=0.25)
            self.magnitude_plot.setLabel("left", "Magnitude", units="dB")
            self.phase_plot.setLabel("left", "Unwrapped phase", units="deg")
            self.phase_plot.setLabel("bottom", "RF frequency", units="MHz")
            self._magnitude_legend = self.magnitude_plot.addLegend(
                offset=(10, 10)
            )
            self._phase_legend = self.phase_plot.addLegend(offset=(10, 10))
            self._magnitude_curves = []
            self._phase_curves = []

        def set_result(self, result) -> None:
            frequency = result.frequencies_mhz
            magnitude = result.magnitude_db
            phase = result.phase_unwrapped_deg
            self.magnitude_plot.setLabel(
                "left",
                (
                    "S21 (P input / P output)"
                    if getattr(result, "physical_power_calibrated", False)
                    else "ADC magnitude"
                ),
                units="dB",
            )
            if getattr(magnitude, "ndim", 1) == 1:
                magnitude = [magnitude]
                phase = [phase]
                curve_labels = [
                    None
                    if getattr(result, "output_power_dbm", None) is None
                    else f"{float(result.output_power_dbm):.6g} dBm"
                ]
            else:
                output_powers = getattr(result, "output_powers_dbm", None)
                if output_powers is None:
                    curve_labels = [
                        f"gain {int(gain)}" for gain in result.power_gains
                    ]
                else:
                    curve_labels = [
                        f"{float(power):.6g} dBm" for power in output_powers
                    ]
            for curve in self._magnitude_curves:
                self.magnitude_plot.removeItem(curve)
            for curve in self._phase_curves:
                self.phase_plot.removeItem(curve)
            self._magnitude_curves.clear()
            self._phase_curves.clear()
            self._magnitude_legend.clear()
            self._phase_legend.clear()
            count = len(curve_labels)
            for index, (name, magnitude_values, phase_values) in enumerate(
                zip(curve_labels, magnitude, phase)
            ):
                color = pg.intColor(index, hues=max(1, count), values=1)
                self._magnitude_curves.append(
                    self.magnitude_plot.plot(
                        frequency,
                        magnitude_values,
                        pen=pg.mkPen(color, width=2),
                        symbol="o",
                        symbolSize=5,
                        name=name,
                    )
                )
                self._phase_curves.append(
                    self.phase_plot.plot(
                        frequency,
                        phase_values,
                        pen=pg.mkPen(color, width=2),
                        symbol="o",
                        symbolSize=5,
                        name=name,
                    )
                )
            self.magnitude_plot.autoRange()
            self.phase_plot.autoRange()

else:
    class SParameterPlotWidget(Canvas):
        """Matplotlib fallback for magnitude and unwrapped-phase plots."""

        def __init__(self, parent=None):
            figure = Figure(tight_layout=True)
            super().__init__(figure)
            self.magnitude_plot, self.phase_plot = figure.subplots(
                2, 1, sharex=True
            )

        def set_result(self, result) -> None:
            for axis in (self.magnitude_plot, self.phase_plot):
                axis.clear()
                axis.grid(True, alpha=0.25)
            magnitude = result.magnitude_db
            phase = result.phase_unwrapped_deg
            if getattr(magnitude, "ndim", 1) == 1:
                magnitude = [magnitude]
                phase = [phase]
                curve_labels = [
                    None
                    if getattr(result, "output_power_dbm", None) is None
                    else f"{float(result.output_power_dbm):.6g} dBm"
                ]
            else:
                output_powers = getattr(result, "output_powers_dbm", None)
                if output_powers is None:
                    curve_labels = [
                        f"gain {int(gain)}" for gain in result.power_gains
                    ]
                else:
                    curve_labels = [
                        f"{float(power):.6g} dBm" for power in output_powers
                    ]
            for label, magnitude_values, phase_values in zip(
                curve_labels, magnitude, phase
            ):
                self.magnitude_plot.plot(
                    result.frequencies_mhz,
                    magnitude_values,
                    "-o",
                    label=label,
                )
                self.phase_plot.plot(
                    result.frequencies_mhz,
                    phase_values,
                    "-o",
                    label=label,
                )
            if curve_labels[0] is not None:
                self.magnitude_plot.legend()
                self.phase_plot.legend()
            self.magnitude_plot.set_ylabel(
                (
                    "S21, P input - P output [dB]"
                    if getattr(result, "physical_power_calibrated", False)
                    else "ADC magnitude [dB]"
                )
            )
            self.phase_plot.set_ylabel("Unwrapped phase [deg]")
            self.phase_plot.set_xlabel("RF frequency [MHz]")
            self.draw_idle()


class SParameterSweepWorker(QtCore.QObject):
    """Run blocking QICK RF sweep and QCoDeS operations off the GUI thread."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress_changed = QtCore.pyqtSignal(int, str)
    partial_result = QtCore.pyqtSignal(object)

    def __init__(self, kwargs: Mapping[str, Any], parent=None):
        super().__init__(parent)
        self._kwargs = dict(kwargs)

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            kwargs = dict(self._kwargs)
            kwargs["progress_callback"] = self.progress_changed.emit
            kwargs["partial_callback"] = self.partial_result.emit
            stored = run_sparameter_sweep(**kwargs)
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(stored)


class SParameterLoadWorker(QtCore.QObject):
    """Load a saved RF S-parameter run off the GUI thread."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, database_path: str, run_id: int, parent=None):
        super().__init__(parent)
        self._database_path = database_path
        self._run_id = int(run_id)

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            stored = load_sparameter_run(self._database_path, self._run_id)
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(stored)


__all__ = [
    "DEFAULT_SPARAMETER_DB_PATH",
    "SParameterLoadWorker",
    "SParameterPlotWidget",
    "SParameterSweepPanel",
    "SParameterSweepWorker",
]
