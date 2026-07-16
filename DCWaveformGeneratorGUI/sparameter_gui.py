"""Independent RF S-parameter sweep controls and result plotting.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import traceback
from typing import Any, Mapping

from PyQt5 import QtCore, QtGui, QtWidgets

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


DEFAULT_SPARAMETER_DB_PATH = str(Path.home() / "qick_sparameter_experiments.db")


class _PathComponent(QtWidgets.QFrame):
    """Compact control block placed directly on the RF path diagram."""

    def __init__(self, title: str, widget: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        self.setObjectName("rfPathComponent")
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed,
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 7)
        layout.setSpacing(4)
        self.title = QtWidgets.QLabel(title)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        layout.addWidget(self.title)
        layout.addWidget(widget)
        self.setStyleSheet(
            "QFrame#rfPathComponent {"
            "  background: #ffffff;"
            "  border: 1px solid #aeb7c2;"
            "  border-radius: 4px;"
            "}"
            "QFrame#rfPathComponent QLabel {"
            "  color: #20252b; border: none; background: transparent;"
            "}"
        )

    def set_title(self, title: str) -> None:
        self.title.setText(title)


class RfPathCorrectionWidget(QtWidgets.QGroupBox):
    """U-shaped output-to-DUT-to-input path with board-aware controls."""

    settings_applied = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__("RF Path and DUT De-embedding", parent)
        self.setMinimumHeight(500)
        self.setStyleSheet(
            "QGroupBox { color: #20252b; font-weight: 600; }"
            "QLabel { color: #20252b; }"
            "QSpinBox, QDoubleSpinBox, QComboBox {"
            "  color: #20252b; background: #ffffff;"
            "}"
        )

        self.output_ch = self._channel_spin()
        self.readout_ch = self._channel_spin()
        self.output_board_type = self._board_combo(OUTPUT_BOARD_TYPES, "RF_Out")
        self.input_board_type = self._board_combo(INPUT_BOARD_TYPES, "DC_In")

        self.output_att1_db = self._attenuation_spin(10.0)
        self.output_att2_db = self._attenuation_spin(10.0)
        self.readout_attenuation_db = self._attenuation_spin(20.0)
        self.readout_dc_gain_db = self._db_spin(0.0, -6.0, 26.0, 1.0)
        self.loss1_db = self._db_spin(0.0, 0.0, 200.0, 0.1)
        self.loss2_db = self._db_spin(0.0, 0.0, 200.0, 0.1)
        self.amplifier_gain_db = self._db_spin(0.0, -200.0, 200.0, 0.1)

        self.input_endpoint = _PathComponent(
            "RF IN",
            self._endpoint_form(
                ("Readout channel", self.readout_ch),
                ("Input board", self.input_board_type),
            ),
            self,
        )
        self.output_endpoint = _PathComponent(
            "RF OUT",
            self._endpoint_form(
                ("Output channel", self.output_ch),
                ("Output board", self.output_board_type),
            ),
            self,
        )
        self.input_condition_stack = QtWidgets.QStackedWidget(self)
        self.input_condition_stack.addWidget(self.readout_attenuation_db)
        self.input_condition_stack.addWidget(self.readout_dc_gain_db)
        self.input_condition_stack.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed,
        )
        self.input_condition_stack.setFixedHeight(
            max(
                self.readout_attenuation_db.sizeHint().height(),
                self.readout_dc_gain_db.sizeHint().height(),
            )
        )
        self.input_condition = _PathComponent(
            "INPUT ATT",
            self.input_condition_stack,
            self,
        )
        self.output_att1_component = _PathComponent("ATT1", self.output_att1_db, self)
        self.output_att2_component = _PathComponent("ATT2", self.output_att2_db, self)
        self.loss1_component = _PathComponent("LOSS1", self.loss1_db, self)
        self.loss2_component = _PathComponent("LOSS2", self.loss2_db, self)
        self.amplifier_component = _PathComponent(
            "AMP GAIN", self.amplifier_gain_db, self
        )
        dut_label = QtWidgets.QLabel("Device under test")
        dut_label.setAlignment(QtCore.Qt.AlignCenter)
        self.dut_component = _PathComponent("DUT", dut_label, self)

        self.summary = QtWidgets.QLabel(self)
        self.summary.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.summary.setWordWrap(True)
        self.summary.setMinimumWidth(210)
        self.summary.setStyleSheet(
            "QLabel { color: #20252b; background: #f3f6f8;"
            " border: 1px solid #aeb7c2; padding: 10px; }"
        )
        self.update_button = QtWidgets.QPushButton("Update", self)
        self.update_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
        )
        self.update_button.setToolTip(
            "Apply the edited RF path values to RF S-parameter and Experiment settings"
        )
        self.apply_status = QtWidgets.QLabel("Applied", self)
        self.apply_status.setAlignment(QtCore.Qt.AlignCenter)
        self.apply_status.setStyleSheet(
            "QLabel { color: #2f6f4e; background: transparent; font-weight: 600; }"
        )
        summary_panel = QtWidgets.QWidget(self)
        summary_layout = QtWidgets.QVBoxLayout(summary_panel)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(8)
        summary_layout.addWidget(self.summary, 1)
        summary_layout.addWidget(self.update_button)
        summary_layout.addWidget(self.apply_status)

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(22, 26, 22, 18)
        layout.setHorizontalSpacing(36)
        layout.setVerticalSpacing(16)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 4)
        layout.setColumnStretch(2, 3)
        layout.addWidget(self.input_endpoint, 0, 0)
        layout.addWidget(summary_panel, 0, 1, 4, 1)
        layout.addWidget(self.output_endpoint, 0, 2)
        layout.addWidget(self.input_condition, 1, 0)
        layout.addWidget(self.output_att1_component, 1, 2)
        layout.addWidget(self.amplifier_component, 2, 0)
        layout.addWidget(self.output_att2_component, 2, 2)
        layout.addWidget(self.loss2_component, 3, 0)
        layout.addWidget(self.loss1_component, 3, 2)
        layout.addWidget(self.dut_component, 4, 1)

        self.output_board_type.currentTextChanged.connect(self._update_board_controls)
        self.input_board_type.currentTextChanged.connect(self._update_board_controls)
        for widget in (
            self.output_ch,
            self.readout_ch,
            self.output_att1_db,
            self.output_att2_db,
            self.readout_attenuation_db,
            self.readout_dc_gain_db,
            self.loss1_db,
            self.loss2_db,
            self.amplifier_gain_db,
        ):
            widget.valueChanged.connect(self._mark_dirty)
        self.output_board_type.currentTextChanged.connect(self._mark_dirty)
        self.input_board_type.currentTextChanged.connect(self._mark_dirty)
        self.update_button.clicked.connect(self.apply_settings)
        self._applied_values = {}
        self._update_board_controls()
        self.apply_settings(emit=False)

    @staticmethod
    def _channel_spin() -> QtWidgets.QSpinBox:
        widget = QtWidgets.QSpinBox()
        widget.setRange(0, 255)
        return widget

    @staticmethod
    def _board_combo(values, selected: str) -> QtWidgets.QComboBox:
        widget = QtWidgets.QComboBox()
        widget.addItems(values)
        widget.setCurrentText(selected)
        return widget

    @staticmethod
    def _attenuation_spin(value: float) -> QtWidgets.QDoubleSpinBox:
        return RfPathCorrectionWidget._db_spin(value, 0.0, 31.75, 0.25)

    @staticmethod
    def _db_spin(
        value: float,
        minimum: float,
        maximum: float,
        step: float,
    ) -> QtWidgets.QDoubleSpinBox:
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(2)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setSuffix(" dB")
        widget.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed,
        )
        return widget

    @staticmethod
    def _endpoint_form(*rows) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for label, control in rows:
            layout.addRow(label, control)
        return widget

    def _update_board_controls(self, *_args) -> None:
        rf_output = self.output_board_type.currentText() == "RF_Out"
        self.output_att1_component.setVisible(rf_output)
        self.output_att2_component.setVisible(rf_output)
        rf_input = self.input_board_type.currentText() == "RF_In"
        self.input_condition_stack.setCurrentIndex(0 if rf_input else 1)
        self.input_condition.set_title("INPUT ATT" if rf_input else "DC INPUT GAIN")
        self._update_summary()
        self.updateGeometry()
        self.update()

    def _editor_values(self) -> dict[str, Any]:
        return {
            "output_ch": self.output_ch.value(),
            "readout_ch": self.readout_ch.value(),
            "output_board_type": self.output_board_type.currentText(),
            "input_board_type": self.input_board_type.currentText(),
            "output_att1_db": self.output_att1_db.value(),
            "output_att2_db": self.output_att2_db.value(),
            "readout_attenuation_db": self.readout_attenuation_db.value(),
            "readout_dc_gain_db": self.readout_dc_gain_db.value(),
            "loss1_db": self.loss1_db.value(),
            "loss2_db": self.loss2_db.value(),
            "amplifier_gain_db": self.amplifier_gain_db.value(),
        }

    def applied_values(self) -> dict[str, Any]:
        """Return the last values committed with the Update button."""
        return dict(self._applied_values)

    def apply_settings(self, _checked=False, *, emit: bool = True) -> None:
        """Commit edited path values and optionally notify linked panels."""
        self._applied_values = self._editor_values()
        self.update_button.setEnabled(False)
        self.apply_status.setText("Applied to RF S-Parameter and Experiment")
        self.apply_status.setStyleSheet(
            "QLabel { color: #2f6f4e; background: transparent; font-weight: 600; }"
        )
        self._update_summary()
        if emit:
            self.settings_applied.emit(self.applied_values())

    def _mark_dirty(self, *_args) -> None:
        if not self._applied_values:
            return
        dirty = self._editor_values() != self._applied_values
        self.update_button.setEnabled(dirty)
        if dirty:
            self.apply_status.setText("Pending changes - click Update")
            self.apply_status.setStyleSheet(
                "QLabel { color: #9a5a16; background: transparent; font-weight: 600; }"
            )
        else:
            self.apply_status.setText("Applied")
            self.apply_status.setStyleSheet(
                "QLabel { color: #2f6f4e; background: transparent; font-weight: 600; }"
            )

    def _update_summary(self, *_args) -> None:
        values = self._applied_values or self._editor_values()
        output_note = (
            "RF_Out: ATT1 and ATT2 active"
            if values["output_board_type"] == "RF_Out"
            else "DC_Out: no onboard ATT1/ATT2"
        )
        input_note = (
            "RF_In: input attenuator active"
            if values["input_board_type"] == "RF_In"
            else "DC_In: no attenuator; LMH6401 gain active"
        )
        correction = (
            values["loss1_db"]
            + values["loss2_db"]
            - values["amplifier_gain_db"]
        )
        self.summary.setText(
            "REFERENCE PLANES\n\n"
            "P_DUT,in = P_RF_OUT - LOSS1\n"
            "P_DUT,out = P_RF_IN + LOSS2 - AMP GAIN\n\n"
            "S21 = P_DUT,out - P_DUT,in\n"
            f"Path correction: {correction:+.2f} dB\n\n"
            f"RF OUT channel {values['output_ch']} -> "
            f"RF IN channel {values['readout_ch']}\n"
            f"{output_note}\n{input_note}\n\n"
            "Physical S21 uses these terms when matching output and input "
            "power calibrations are available."
        )

    @staticmethod
    def _top_center(widget: QtWidgets.QWidget) -> QtCore.QPointF:
        geometry = widget.geometry()
        return QtCore.QPointF(geometry.center().x(), geometry.top())

    @staticmethod
    def _bottom_center(widget: QtWidgets.QWidget) -> QtCore.QPointF:
        geometry = widget.geometry()
        return QtCore.QPointF(geometry.center().x(), geometry.bottom())

    @staticmethod
    def _left_center(widget: QtWidgets.QWidget) -> QtCore.QPointF:
        geometry = widget.geometry()
        return QtCore.QPointF(geometry.left(), geometry.center().y())

    @staticmethod
    def _right_center(widget: QtWidgets.QWidget) -> QtCore.QPointF:
        geometry = widget.geometry()
        return QtCore.QPointF(geometry.right(), geometry.center().y())

    @staticmethod
    def _draw_arrow(
        painter: QtGui.QPainter,
        points,
        color: QtGui.QColor,
    ) -> None:
        if len(points) < 2:
            return
        pen = QtGui.QPen(color, 2.0)
        pen.setJoinStyle(QtCore.Qt.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        path = QtGui.QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)
        painter.drawPath(path)
        start = points[-2]
        end = points[-1]
        direction = end - start
        length = max(1.0, (direction.x() ** 2 + direction.y() ** 2) ** 0.5)
        unit = QtCore.QPointF(direction.x() / length, direction.y() / length)
        normal = QtCore.QPointF(-unit.y(), unit.x())
        arrow = QtGui.QPolygonF(
            [
                end,
                end - unit * 8.0 + normal * 4.0,
                end - unit * 8.0 - normal * 4.0,
            ]
        )
        painter.setBrush(color)
        painter.drawPolygon(arrow)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        output_color = QtGui.QColor("#b85f26")
        input_color = QtGui.QColor("#237a72")

        output_nodes = [self.output_endpoint]
        output_nodes.extend(
            node
            for node in (self.output_att1_component, self.output_att2_component)
            if node.isVisible()
        )
        output_nodes.append(self.loss1_component)
        for first, second in zip(output_nodes, output_nodes[1:]):
            self._draw_arrow(
                painter,
                [self._bottom_center(first), self._top_center(second)],
                output_color,
            )

        loss1_bottom = self._bottom_center(self.loss1_component)
        dut_right = self._right_center(self.dut_component)
        self._draw_arrow(
            painter,
            [
                loss1_bottom,
                QtCore.QPointF(loss1_bottom.x(), dut_right.y()),
                dut_right,
            ],
            output_color,
        )

        dut_left = self._left_center(self.dut_component)
        loss2_bottom = self._bottom_center(self.loss2_component)
        self._draw_arrow(
            painter,
            [
                dut_left,
                QtCore.QPointF(loss2_bottom.x(), dut_left.y()),
                loss2_bottom,
            ],
            input_color,
        )
        input_nodes = [
            self.loss2_component,
            self.amplifier_component,
            self.input_condition,
            self.input_endpoint,
        ]
        for first, second in zip(input_nodes, input_nodes[1:]):
            self._draw_arrow(
                painter,
                [self._top_center(first), self._bottom_center(second)],
                input_color,
            )


class SParameterSweepPanel(QtWidgets.QWidget):
    """Controls for an RF-only generator/readout hardware frequency sweep."""

    run_requested = QtCore.pyqtSignal()
    load_requested = QtCore.pyqtSignal(int)
    path_settings_applied = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget(scroll)
        content_layout = QtWidgets.QVBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        self.path_diagram = RfPathCorrectionWidget(content)
        self.output_ch = self.path_diagram.output_ch
        self.readout_ch = self.path_diagram.readout_ch
        self.output_board_type = self.path_diagram.output_board_type
        self.input_board_type = self.path_diagram.input_board_type
        self.output_att1_db = self.path_diagram.output_att1_db
        self.output_att2_db = self.path_diagram.output_att2_db
        self.readout_attenuation_db = self.path_diagram.readout_attenuation_db
        self.readout_dc_gain_db = self.path_diagram.readout_dc_gain_db
        self.loss1_db = self.path_diagram.loss1_db
        self.loss2_db = self.path_diagram.loss2_db
        self.amplifier_gain_db = self.path_diagram.amplifier_gain_db
        self.path_diagram.settings_applied.connect(self._forward_path_settings)
        content_layout.addWidget(self.path_diagram)

        sweep_group = QtWidgets.QGroupBox("Frequency Sweep")
        sweep_form = QtWidgets.QFormLayout(sweep_group)
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
        calibration_form = QtWidgets.QFormLayout(self.power_calibration_enabled)
        self.calibration_database_path = QtWidgets.QLineEdit()
        self.calibration_database_path.setPlaceholderText("Select gain_pwr_calb.db")
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
        self.calibration_hint = QtWidgets.QLabel(
            "Same-board response; normalized to the weakest frequency in the sweep."
        )
        self.calibration_hint.setWordWrap(True)
        calibration_form.addRow("Calibration DB:", calibration_path_row)
        calibration_form.addRow(self.calibration_hint)
        content_layout.addWidget(self.power_calibration_enabled)

        self.power_sweep_enabled = QtWidgets.QGroupBox("Power Sweep (Software)")
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
        self.power_sweep_enabled.toggled.connect(self._update_power_control_state)
        self.power_calibration_enabled.toggled.connect(self._update_power_control_state)
        content_layout.addWidget(self.power_sweep_enabled)
        self._update_power_control_state(False)

        output_group = QtWidgets.QGroupBox("Output Filter")
        output_form = QtWidgets.QFormLayout(output_group)
        self.output_filter_type = self._filter_combo()
        self.output_filter_cutoff_ghz = self._filter_spin(2.5)
        self.output_filter_bandwidth_ghz = self._filter_spin(1.0)
        output_form.addRow("Filter:", self.output_filter_type)
        output_form.addRow("Cutoff/center:", self.output_filter_cutoff_ghz)
        output_form.addRow("Bandwidth:", self.output_filter_bandwidth_ghz)
        content_layout.addWidget(output_group)

        readout_group = QtWidgets.QGroupBox("Input Filter")
        readout_form = QtWidgets.QFormLayout(readout_group)
        self.readout_filter_type = self._filter_combo()
        self.readout_filter_cutoff_ghz = self._filter_spin(2.5)
        self.readout_filter_bandwidth_ghz = self._filter_spin(1.0)
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
        self.browse_database.setToolTip("Choose RF S-parameter QCoDeS SQLite database")
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

    def _forward_path_settings(self, values: Mapping[str, Any]) -> None:
        """Include filter settings when applying the path to Experiment."""
        linked = dict(values)
        linked.update(
            {
                "output_filter_type": self.output_filter_type.currentText(),
                "output_filter_cutoff_ghz": self.output_filter_cutoff_ghz.value(),
                "output_filter_bandwidth_ghz": (
                    self.output_filter_bandwidth_ghz.value()
                ),
                "readout_filter_type": self.readout_filter_type.currentText(),
                "readout_filter_cutoff_ghz": (
                    self.readout_filter_cutoff_ghz.value()
                ),
                "readout_filter_bandwidth_ghz": (
                    self.readout_filter_bandwidth_ghz.value()
                ),
            }
        )
        self.path_settings_applied.emit(linked)

    def config(self) -> SParameterSweepConfig:
        path = self.path_diagram.applied_values()
        return SParameterSweepConfig(
            output_ch=path["output_ch"],
            readout_ch=path["readout_ch"],
            frequency_start_mhz=self.frequency_start_mhz.value(),
            frequency_end_mhz=self.frequency_end_mhz.value(),
            frequency_points=self.frequency_points.value(),
            gain=self.gain.value(),
            power_calibration_enabled=(self.power_calibration_enabled.isChecked()),
            calibration_database_path=(self.calibration_database_path.text().strip()),
            output_board_type=path["output_board_type"],
            input_board_type=path["input_board_type"],
            output_power_dbm=self.output_power_dbm.value(),
            power_sweep_enabled=self.power_sweep_enabled.isChecked(),
            power_start_gain=self.power_start_gain.value(),
            power_end_gain=self.power_end_gain.value(),
            power_start_dbm=self.power_start_dbm.value(),
            power_end_dbm=self.power_end_dbm.value(),
            power_points=self.power_points.value(),
            power_scale=str(self.power_scale.currentData()),
            scan_time_us=self.scan_time_us.value(),
            output_att1_db=path["output_att1_db"],
            output_att2_db=path["output_att2_db"],
            output_filter_type=self.output_filter_type.currentText(),
            output_filter_cutoff_ghz=self.output_filter_cutoff_ghz.value(),
            output_filter_bandwidth_ghz=(self.output_filter_bandwidth_ghz.value()),
            readout_attenuation_db=path["readout_attenuation_db"],
            readout_dc_gain_db=path["readout_dc_gain_db"],
            readout_filter_type=self.readout_filter_type.currentText(),
            readout_filter_cutoff_ghz=self.readout_filter_cutoff_ghz.value(),
            readout_filter_bandwidth_ghz=(self.readout_filter_bandwidth_ghz.value()),
            loss1_db=path["loss1_db"],
            loss2_db=path["loss2_db"],
            amplifier_gain_db=path["amplifier_gain_db"],
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
            (self.readout_dc_gain_db, config.readout_dc_gain_db),
            (self.readout_filter_cutoff_ghz, config.readout_filter_cutoff_ghz),
            (
                self.readout_filter_bandwidth_ghz,
                config.readout_filter_bandwidth_ghz,
            ),
            (self.loss1_db, config.loss1_db),
            (self.loss2_db, config.loss2_db),
            (self.amplifier_gain_db, config.amplifier_gain_db),
            (self.margin_input_samples, config.margin_input_samples),
            (self.address, config.address),
            (self.stride_bytes, config.stride_bytes or 0),
        )
        for widget, value in widgets:
            widget.setValue(value)
        self.output_filter_type.setCurrentText(config.output_filter_type)
        self.readout_filter_type.setCurrentText(config.readout_filter_type)
        self.calibration_database_path.setText(config.calibration_database_path)
        self.output_board_type.setCurrentText(config.output_board_type)
        self.input_board_type.setCurrentText(config.input_board_type)
        self.path_diagram._update_board_controls()
        self.path_diagram.apply_settings(emit=False)
        self.database_path.setText(database_path)
        power_scale_index = self.power_scale.findData(config.power_scale)
        if power_scale_index < 0:
            raise ValueError(f"unsupported power scale {config.power_scale!r}")
        self.power_scale.setCurrentIndex(power_scale_index)
        self.power_calibration_enabled.setChecked(config.power_calibration_enabled)
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
            self._magnitude_legend = self.magnitude_plot.addLegend(offset=(10, 10))
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
                    (
                        None
                        if getattr(result, "output_power_dbm", None) is None
                        else f"{float(result.output_power_dbm):.6g} dBm"
                    )
                ]
            else:
                output_powers = getattr(result, "output_powers_dbm", None)
                if output_powers is None:
                    curve_labels = [f"gain {int(gain)}" for gain in result.power_gains]
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
            self.magnitude_plot, self.phase_plot = figure.subplots(2, 1, sharex=True)

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
                    (
                        None
                        if getattr(result, "output_power_dbm", None) is None
                        else f"{float(result.output_power_dbm):.6g} dBm"
                    )
                ]
            else:
                output_powers = getattr(result, "output_powers_dbm", None)
                if output_powers is None:
                    curve_labels = [f"gain {int(gain)}" for gain in result.power_gains]
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
