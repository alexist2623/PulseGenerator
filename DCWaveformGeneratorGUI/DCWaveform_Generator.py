#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive DC waveform editor with Keysight QCS and QICK export.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

import json
from dataclasses import asdict, replace
from math import ceil, floor, prod, ulp
from pathlib import Path
import re
import sqlite3
import sys
import traceback
from typing import Tuple, Optional, List, Callable, Mapping, Sequence
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets


class _ValueInputWheelGuard(QtCore.QObject):
    """Prevent wheel gestures from changing value-entry widgets."""

    _VALUE_WIDGET_TYPES = (
        QtWidgets.QAbstractSpinBox,
        QtWidgets.QComboBox,
    )

    @classmethod
    def _value_widget(cls, watched):
        if isinstance(watched, cls._VALUE_WIDGET_TYPES):
            return watched
        if isinstance(watched, QtWidgets.QLineEdit):
            parent = watched.parentWidget()
            if isinstance(parent, cls._VALUE_WIDGET_TYPES):
                return parent
        return None

    @staticmethod
    def _scroll_area(widget):
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QtWidgets.QAbstractScrollArea):
                return parent
            parent = parent.parentWidget()
        return None

    def eventFilter(self, watched, event):
        if event.type() != QtCore.QEvent.Wheel:
            return False

        value_widget = self._value_widget(watched)
        if value_widget is None:
            return False

        scroll_area = self._scroll_area(value_widget)
        if scroll_area is not None:
            viewport = scroll_area.viewport()
            viewport_position = QtCore.QPointF(
                viewport.mapFromGlobal(event.globalPos())
            )
            forwarded = QtGui.QWheelEvent(
                viewport_position,
                event.globalPosF(),
                event.pixelDelta(),
                event.angleDelta(),
                event.buttons(),
                event.modifiers(),
                event.phase(),
                event.inverted(),
                event.source(),
            )
            forwarded.setTimestamp(event.timestamp())
            QtWidgets.QApplication.sendEvent(viewport, forwarded)

        event.accept()
        return True


def _install_value_input_wheel_guard():
    """Install one application-wide guard and retain its QObject lifetime."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        return None
    guard = getattr(app, "_qstl_value_input_wheel_guard", None)
    if guard is None:
        guard = _ValueInputWheelGuard(app)
        app.installEventFilter(guard)
        app._qstl_value_input_wheel_guard = guard
    return guard

try:
    import pyqtgraph  # noqa: F401
except ImportError:
    # Matplotlib remains a compatibility fallback.  The normal optimized path
    # imports no Matplotlib modules during application startup.
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from matplotlib import colors
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    _USE_PYQTGRAPH = False
else:
    # Names below are referenced only by the legacy fallback class bodies.
    Canvas = QtWidgets.QWidget
    Figure = Axes = Line2D = object
    colors = None
    FuncFormatter = MultipleLocator = None
    _USE_PYQTGRAPH = True

try:
    from .dc_waveform_core import (
        BIAS_T_COMPENSATION_MODES,
        BIAS_T_COMPENSATION_TYPES,
        DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
        DEFAULT_BIAS_T_COMPENSATION_FRACTION,
        DEFAULT_BIAS_T_FILTER_TAU_US,
        DEFAULT_DC_MEASURE_GAIN_V_PER_A,
        DEFAULT_INITIAL_VOLTAGE_MV,
        DEFAULT_QCS_FULL_SCALE_V,
        DEFAULT_QICK_FABRIC_MHZ,
        DEFAULT_QICK_TPROC_MHZ,
        DEFAULT_QICK_FULL_SCALE_MV,
        PulseSequence,
        QICK_INPUT_BOARD_TYPES,
        QICK_OUTPUT_BOARD_TYPES,
        QickDdrReadoutSpec,
        QickHoldDurationSweepSpec,
        QickRampRateSweepSpec,
        QickRfPulseSpec,
        QickSweepAxisSpec,
        QickSweepSpec,
        build_qick_sequence,
        generate_qcs_program_code,
        generate_qick_program_code,
        qick_set_segment_names,
        transform_virtual_waveforms,
    )

except ImportError:
    from dc_waveform_core import (
        BIAS_T_COMPENSATION_MODES,
        BIAS_T_COMPENSATION_TYPES,
        DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
        DEFAULT_BIAS_T_COMPENSATION_FRACTION,
        DEFAULT_BIAS_T_FILTER_TAU_US,
        DEFAULT_DC_MEASURE_GAIN_V_PER_A,
        DEFAULT_INITIAL_VOLTAGE_MV,
        DEFAULT_QCS_FULL_SCALE_V,
        DEFAULT_QICK_FABRIC_MHZ,
        DEFAULT_QICK_TPROC_MHZ,
        DEFAULT_QICK_FULL_SCALE_MV,
        PulseSequence,
        QICK_INPUT_BOARD_TYPES,
        QICK_OUTPUT_BOARD_TYPES,
        QickDdrReadoutSpec,
        QickHoldDurationSweepSpec,
        QickRampRateSweepSpec,
        QickRfPulseSpec,
        QickSweepAxisSpec,
        QickSweepSpec,
        build_qick_sequence,
        generate_qcs_program_code,
        generate_qick_program_code,
        qick_set_segment_names,
        transform_virtual_waveforms,
    )

try:
    from .qick_qcodes_experiment import (
        AWG_METADATA_MODE_EXPANDED,
        AWG_METADATA_MODE_PARAMETRIC,
        COMPILE_VALIDATION_BOUNDARY,
        COMPILE_VALIDATION_FULL,
        DEFAULT_AWG_METADATA_MODE,
        IQ_REPETITION_POLICY_COHERENT_AVERAGE,
        IQ_REPETITION_POLICY_PRESERVE,
        QcodesRunConfig,
        QickConnectionConfig,
        build_awg_vertex_record,
        build_awg_vertex_metadata,
        build_awg_waveform_recipe,
        build_qick_program,
        configure_rf_output,
        connect_qick,
        measurement_iq_values,
        normalize_awg_metadata_mode,
        normalize_compile_validation_mode,
        normalize_iq_repetition_policy,
        run_qick_qcodes_experiment,
        write_awg_vertex_metadata_jsonl,
    )
except ImportError:
    from qick_qcodes_experiment import (
        AWG_METADATA_MODE_EXPANDED,
        AWG_METADATA_MODE_PARAMETRIC,
        COMPILE_VALIDATION_BOUNDARY,
        COMPILE_VALIDATION_FULL,
        DEFAULT_AWG_METADATA_MODE,
        IQ_REPETITION_POLICY_COHERENT_AVERAGE,
        IQ_REPETITION_POLICY_PRESERVE,
        QcodesRunConfig,
        QickConnectionConfig,
        build_awg_vertex_record,
        build_awg_vertex_metadata,
        build_awg_waveform_recipe,
        build_qick_program,
        configure_rf_output,
        connect_qick,
        measurement_iq_values,
        normalize_awg_metadata_mode,
        normalize_compile_validation_mode,
        normalize_iq_repetition_policy,
        run_qick_qcodes_experiment,
        write_awg_vertex_metadata_jsonl,
    )

try:
    from .qcs_qcodes_experiment import (
        DEFAULT_QCS_INIT_TIME_S,
        QCS_M5200_INTEGRATION_BLOCK_SAMPLES,
        QCS_M5200_SAMPLE_RATE_HZ,
        QCS_M5301_MAX_RENDERED_SAMPLES,
        QCS_MAX_TOTAL_IQ_AVERAGING_DURATION_S,
        QCS_MAX_TOTAL_IQ_AVERAGING_SAMPLES,
        QCS_NOISE_MAX_RAW_TRACE_SAMPLES,
        QCS_SPARAMETER_MAX_INTEGRATION_SAMPLES,
        QCS_STABILITY_DC_EDGE_PADDING_S,
        QCS_STABILITY_DC_RAMP_S,
        QCS_STABILITY_INTER_SEGMENT_DELAY_S,
        QcsAcquisitionConfig,
        QcsCancellationController,
        QcsConnectionConfig,
        QcsExperimentCancelled,
        QcsM5301CapacityReport,
        QcsNoiseTraceConfig,
        QcsRfPowerCalibrationConfig,
        QcsRfPulseConfig,
        QcsSweepExecutionPreview,
        qcs_m5301_capacity_preview_point_indices,
        qcs_m5301_waveform_capacity_report,
        qcs_sweep_execution_preview,
        qcs_stability_integration_segment_sample_counts,
        plan_qcs_total_iq_averaging,
        quantize_qcs_inter_iteration_delay,
        run_qcs_qcodes_experiment,
        validate_qcs_m5301_waveform_capacity,
    )
except ImportError:
    from qcs_qcodes_experiment import (
        DEFAULT_QCS_INIT_TIME_S,
        QCS_M5200_INTEGRATION_BLOCK_SAMPLES,
        QCS_M5200_SAMPLE_RATE_HZ,
        QCS_M5301_MAX_RENDERED_SAMPLES,
        QCS_MAX_TOTAL_IQ_AVERAGING_DURATION_S,
        QCS_MAX_TOTAL_IQ_AVERAGING_SAMPLES,
        QCS_NOISE_MAX_RAW_TRACE_SAMPLES,
        QCS_SPARAMETER_MAX_INTEGRATION_SAMPLES,
        QCS_STABILITY_DC_EDGE_PADDING_S,
        QCS_STABILITY_DC_RAMP_S,
        QCS_STABILITY_INTER_SEGMENT_DELAY_S,
        QcsAcquisitionConfig,
        QcsCancellationController,
        QcsConnectionConfig,
        QcsExperimentCancelled,
        QcsM5301CapacityReport,
        QcsNoiseTraceConfig,
        QcsRfPowerCalibrationConfig,
        QcsRfPulseConfig,
        QcsSweepExecutionPreview,
        qcs_m5301_capacity_preview_point_indices,
        qcs_m5301_waveform_capacity_report,
        qcs_sweep_execution_preview,
        qcs_stability_integration_segment_sample_counts,
        plan_qcs_total_iq_averaging,
        quantize_qcs_inter_iteration_delay,
        run_qcs_qcodes_experiment,
        validate_qcs_m5301_waveform_capacity,
    )

try:
    from .awg_sweep_map import (
        AwgSweepMapLoadWorker,
        AwgSweepMapPlotWidget,
        AwgSweepRunSelector,
        normalize_awg_sweep_color_ranges,
        normalize_awg_sweep_visible_data,
        reduce_awg_sweep_map,
    )
except ImportError:
    from awg_sweep_map import (
        AwgSweepMapLoadWorker,
        AwgSweepMapPlotWidget,
        AwgSweepRunSelector,
        normalize_awg_sweep_color_ranges,
        normalize_awg_sweep_visible_data,
        reduce_awg_sweep_map,
    )

try:
    from .qick_sparameter_sweep import (
        SParameterSweepConfig,
        SParameterSweepResult,
    )
    from .sparameter_gui import (
        DEFAULT_SPARAMETER_DB_PATH,
        QcsSParameterSweepWorker,
        SParameterLoadWorker,
        SParameterPlotWidget,
        SParameterSweepPanel,
        SParameterSweepWorker,
    )
except ImportError:
    from qick_sparameter_sweep import (
        SParameterSweepConfig,
        SParameterSweepResult,
    )
    from sparameter_gui import (
        DEFAULT_SPARAMETER_DB_PATH,
        QcsSParameterSweepWorker,
        SParameterLoadWorker,
        SParameterPlotWidget,
        SParameterSweepPanel,
        SParameterSweepWorker,
    )

try:
    from .stability_diagram import (
        DEFAULT_STABILITY_DB_PATH,
        QcsStabilityDiagramWorker,
        StabilityDiagramPanel,
        StabilityDiagramWorker,
        StabilityOverlayLoadWorker,
        StabilityOverlaySelector,
        build_stability_hold_sequence,
        normalize_stability_settings,
    )
except ImportError:
    from stability_diagram import (
        DEFAULT_STABILITY_DB_PATH,
        QcsStabilityDiagramWorker,
        StabilityDiagramPanel,
        StabilityDiagramWorker,
        StabilityOverlayLoadWorker,
        StabilityOverlaySelector,
        build_stability_hold_sequence,
        normalize_stability_settings,
    )

try:
    from .noise_analysis import (
        NoiseAcquisitionWorker,
        NoiseAnalysisPanel,
        QcsNoiseAcquisitionWorker,
        NoiseTraceLoadWorker,
        normalize_noise_analysis_settings,
    )
except ImportError:
    from noise_analysis import (
        NoiseAcquisitionWorker,
        NoiseAnalysisPanel,
        QcsNoiseAcquisitionWorker,
        NoiseTraceLoadWorker,
        normalize_noise_analysis_settings,
    )

try:
    from .calibration_gui import (
        CalibrationPanel,
        CalibrationWorker,
        default_calibration_settings,
        normalize_calibration_paths,
        normalize_qcs_calibration_endpoint_settings,
    )
    from .qick_power_calibration import (
        InputPowerCalibrationConfig,
        OscilloscopeConfig,
        OutputPowerCalibrationConfig,
    )
    from .power_calibration import CalibrationDatabase
    from .dc_voltage_calibration import DcVoltageCalibrationConfig
except ImportError:
    from calibration_gui import (
        CalibrationPanel,
        CalibrationWorker,
        default_calibration_settings,
        normalize_calibration_paths,
        normalize_qcs_calibration_endpoint_settings,
    )
    from qick_power_calibration import (
        InputPowerCalibrationConfig,
        OscilloscopeConfig,
        OutputPowerCalibrationConfig,
    )
    from power_calibration import CalibrationDatabase
    from dc_voltage_calibration import DcVoltageCalibrationConfig

try:
    from .qick_front_panel import (
        QickFrontPanelControl,
        QickFrontPanelPreview,
        identify_qick_front_panel,
    )
    from .ddr_memory_usage import (
        calculate_ddr_capture_memory_usage,
        format_binary_bytes,
    )
    from .fir_ddr_profile import format_sample_rate_hz
except ImportError:
    from qick_front_panel import (
        QickFrontPanelControl,
        QickFrontPanelPreview,
        identify_qick_front_panel,
    )
    from ddr_memory_usage import (
        calculate_ddr_capture_memory_usage,
        format_binary_bytes,
    )
    from fir_ddr_profile import format_sample_rate_hz

try:
    from . import qcs_front_panel as _qcs_front_panel_module
    from .qcs_front_panel import (
        QCS_HARDWARE_STATE_DRAFT,
        QCS_HARDWARE_STATE_EXTERNAL,
        QCS_HARDWARE_STATE_IMPORTED,
        QCS_HARDWARE_STATE_IMPORTED_DIRTY,
        QCS_HARDWARE_STATE_SAVED,
        QCS_HARDWARE_STATES,
        QCS_MODULE_MODELS,
        QcsFrontPanelControl,
        QcsFrontPanelPreview,
        identify_qcs_hardware_configuration,
        normalize_qcs_hardware_configuration,
        normalize_qcs_mapper_sha256,
        qcs_hardware_mapper_fingerprint,
        qcs_mapper_file_sha256,
        qcs_role_bindings,
        qcs_workflow_mapper_output_path,
        resize_incomplete_qcs_dc_mappings,
        resize_qcs_dc_mappings,
        synchronize_qcs_hardware_role_names,
        validate_imported_qcs_role_configuration,
    )
except ImportError:
    import qcs_front_panel as _qcs_front_panel_module
    from qcs_front_panel import (
        QCS_HARDWARE_STATE_DRAFT,
        QCS_HARDWARE_STATE_EXTERNAL,
        QCS_HARDWARE_STATE_IMPORTED,
        QCS_HARDWARE_STATE_IMPORTED_DIRTY,
        QCS_HARDWARE_STATE_SAVED,
        QCS_HARDWARE_STATES,
        QCS_MODULE_MODELS,
        QcsFrontPanelControl,
        QcsFrontPanelPreview,
        identify_qcs_hardware_configuration,
        normalize_qcs_hardware_configuration,
        normalize_qcs_mapper_sha256,
        qcs_hardware_mapper_fingerprint,
        qcs_mapper_file_sha256,
        qcs_role_bindings,
        qcs_workflow_mapper_output_path,
        resize_incomplete_qcs_dc_mappings,
        resize_qcs_dc_mappings,
        synchronize_qcs_hardware_role_names,
        validate_imported_qcs_role_configuration,
    )

try:
    from .hardware_front_panel import HardwareFrontPanelPreview
except ImportError:
    from hardware_front_panel import HardwareFrontPanelPreview


DEFAULT_QSTL_AWG_CHANNELS = (1, 3, 5, 7, 8, 9, 10, 11)
DEFAULT_QSTL_RF_CHANNELS = (0, 2, 4, 6, 12, 13, 14, 15)
TIME_UNIT_NS = {"ns": 1.0, "us": 1000.0, "ms": 1_000_000.0}
DEFAULT_TIME_UNIT = "us"
DEFAULT_GUI_DURATION_NS = 1000.0
DEFAULT_GUI_RAMP_NS = 1000.0
DEFAULT_GUI_FLAT_NS = 1000.0
SETTINGS_SCHEMA = "qstl-pulse-generator-gui"
SETTINGS_VERSION = 43
SUPPORTED_SETTINGS_VERSIONS = tuple(range(1, SETTINGS_VERSION + 1))
DEFAULT_GUI_COMPILE_VALIDATION_MODE = COMPILE_VALIDATION_BOUNDARY
EXECUTION_BACKEND_QICK = "qick"
EXECUTION_BACKEND_QCS = "qcs"


def _qcs_rf_calibration_database_has_records(path: str) -> bool:
    """Return whether an existing database contains an M5300 calibration."""

    database_path = Path(str(path).strip()).expanduser()
    if not database_path.is_file():
        return False
    connection = None
    try:
        connection = sqlite3.connect(
            str(database_path),
            timeout=0.1,
        )
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'qcs_rf_power_calibration_runs'"
        ).fetchone()
        if table_exists is None:
            return False
        return connection.execute(
            "SELECT 1 FROM qcs_rf_power_calibration_runs LIMIT 1"
        ).fetchone() is not None
    except sqlite3.Error:
        return False
    finally:
        if connection is not None:
            connection.close()
EXECUTION_BACKENDS = (EXECUTION_BACKEND_QICK, EXECUTION_BACKEND_QCS)
DEFAULT_EXECUTION_BACKEND = EXECUTION_BACKEND_QCS
DEFAULT_QICK_HOST = "192.168.2.99"
DEFAULT_QICK_NS_PORT = 8888
DEFAULT_QICK_PROXY_NAME = "myqick"
DEFAULT_QCS_MAPPER_PATH = "qcs_channel_mapper.qcs"
DEFAULT_QCS_INIT_TIME_US = DEFAULT_QCS_INIT_TIME_S * 1.0e6
DEFAULT_QCS_SAMPLE_RATE_HZ = QCS_M5200_SAMPLE_RATE_HZ
DEFAULT_QCODES_DB_PATH = str(Path.home() / "qick_experiments.db")
DEFAULT_POWER_CALIBRATION_DB_PATH = str(Path.home() / "gain_pwr_calb.db")
DEFAULT_BIAS_T_COMPENSATION_MV = (
    DEFAULT_QICK_FULL_SCALE_MV * DEFAULT_BIAS_T_COMPENSATION_FRACTION
)

DEFAULT_RF_OUTPUT_SETTINGS = {
    "enabled": False,
    "gen_ch": DEFAULT_QSTL_RF_CHANNELS[0],
    "segment_name": "set_0",
    "delay_us": 0.0,
    "duration_us": 1.0,
    "duration_sweep_enabled": False,
    "duration_sweep_start_us": 1.0,
    "duration_sweep_stop_us": 1.0,
    "duration_sweep_count": 1,
    "segment_length_mode": "fixed",
    "frequency_mhz": 50.0,
    "frequency_sweep_enabled": False,
    "frequency_sweep_start_mhz": 50.0,
    "frequency_sweep_stop_mhz": 50.0,
    "frequency_sweep_count": 1,
    "gain": 20000,
    "output_board_type": "RF_Out",
    "att1_db": 0.0,
    "att2_db": 0.0,
    "filter_type": "bypass",
    "filter_cutoff": 2.5,
    "filter_bandwidth": 1.0,
    "phase_degrees": 0.0,
    "nqz": 1,
    "require_within_segment": True,
    "power_calibration_enabled": False,
    "power_calibration_database_path": DEFAULT_POWER_CALIBRATION_DB_PATH,
    "power_calibration_run_id": 0,
    "target_output_power_dbm": -20.0,
    "power_sweep_enabled": False,
    "power_sweep_start_dbm": -20.0,
    "power_sweep_stop_dbm": -20.0,
    "power_sweep_count": 1,
    "qcs_power_calibration_enabled": False,
    "qcs_power_calibration_database_path": DEFAULT_POWER_CALIBRATION_DB_PATH,
    "qcs_power_calibration_run_id": 0,
    "qcs_target_output_power_dbm": -20.0,
}

DEFAULT_RF_READOUT_SETTINGS = {
    "enabled": False,
    "ro_ch": 0,
    "segment_name": "set_0",
    "delay_us": 0.0,
    "samples_per_trigger": 64,
    "qcs_acquisition_duration_s": 64 / QCS_M5200_SAMPLE_RATE_HZ,
    "readout_frequency_mhz": 50.0,
    "margin_input_samples": 1024,
    "fpga_trigger_delay_us": None,
    "force_overwrite": False,
    "post_run_read_delay_seconds": 0.1,
    "input_board_type": "RF_In",
    "attenuation_db": 20.0,
    "dc_gain_db": 0.0,
    "dc_measure_mode": False,
    "dc_measure_gain_v_per_a": DEFAULT_DC_MEASURE_GAIN_V_PER_A,
    "measurement_representation": "adc",
    "dc_voltage_calibration_enabled": False,
    "dc_voltage_calibration_database_path": "",
    "dc_voltage_calibration_run_id": 0,
    "filter_type": "bypass",
    "filter_cutoff": 2.5,
    "filter_bandwidth": 1.0,
    "nqz": 1,
}

DEFAULT_SPARAMETER_SETTINGS = {
    "database_path": DEFAULT_SPARAMETER_DB_PATH,
    **asdict(SParameterSweepConfig()),
    "qcs_amplitude": 0.005,
    # The connected QCS/M5200 path is live-tested with a 1 us flat filter.
    # Keep this GUI default safe while retaining the standalone QICK config
    # dataclass's historical 10 us default for non-GUI callers.
    "scan_time_us": 1.0,
}
DEFAULT_CALIBRATION_SETTINGS = default_calibration_settings()


def _time_from_ns(value_ns: float, unit: str) -> float:
    return float(value_ns) / TIME_UNIT_NS[unit]


def _time_to_ns(value: float, unit: str) -> float:
    return float(value) * TIME_UNIT_NS[unit]


def _remap_set_segment_name(
    segment_name: str,
    operation: str,
    segment_index: int,
) -> Optional[str]:
    """Keep a SET reference attached to the same logical voltage segment."""
    match = re.fullmatch(r"set_(\d+)", str(segment_name))
    if match is None:
        return str(segment_name)
    current_index = int(match.group(1))
    segment_index = int(segment_index)
    if operation == "insert":
        if current_index >= segment_index:
            current_index += 1
    elif operation == "delete":
        if current_index == segment_index:
            return None
        if current_index > segment_index:
            current_index -= 1
    else:
        raise ValueError(f"unsupported segment structure operation {operation!r}")
    return f"set_{current_index}"


def _resolve_stored_set_segment_name(
    segment_name: str,
    valid_names: Sequence[str],
    *,
    label: str,
) -> str:
    """Clamp a stale indexed SET reference while rejecting malformed anchors."""
    segment_name = str(segment_name)
    valid_names = tuple(str(name) for name in valid_names)
    if segment_name in valid_names:
        return segment_name

    match = re.fullmatch(r"set_(\d+)", segment_name)
    indexed_names = []
    for name in valid_names:
        valid_match = re.fullmatch(r"set_(\d+)", name)
        if valid_match is not None:
            indexed_names.append((int(valid_match.group(1)), name))
    if match is not None and indexed_names:
        stored_index = int(match.group(1))
        return min(
            indexed_names,
            key=lambda item: (abs(item[0] - stored_index), item[0]),
        )[1]
    raise ValueError(f"unknown {label} anchor {segment_name!r}")


def _remap_ramp_segment_name(
    segment_name: str,
    operation: str,
    segment_index: int,
) -> Optional[str]:
    """Keep an incoming RAMP reference attached to its destination SET."""
    match = re.fullmatch(r"ramp_(\d+)_to_(\d+)", str(segment_name))
    if match is None:
        return str(segment_name)
    source_index, destination_index = map(int, match.groups())
    if destination_index != source_index + 1:
        return str(segment_name)
    remapped_destination = _remap_set_segment_name(
        f"set_{destination_index}",
        operation,
        segment_index,
    )
    if remapped_destination is None:
        return None
    destination_index = int(remapped_destination.rsplit("_", 1)[1])
    if destination_index <= 0:
        return None
    return f"ramp_{destination_index - 1}_to_{destination_index}"


def _remap_segment_rows(
    rows,
    operation: str,
    segment_index: int,
) -> set:
    """Move table-row state with its original logical SET segment."""
    segment_index = int(segment_index)
    remapped = set()
    for row in rows:
        row = int(row)
        if operation == "insert":
            remapped.add(row + 1 if row >= segment_index else row)
        elif operation == "delete":
            if row == segment_index:
                continue
            remapped.add(row - 1 if row > segment_index else row)
        else:
            raise ValueError(
                f"unsupported segment structure operation {operation!r}"
            )
    return remapped


def rf_pulse_absolute_times_us(
    pulse: PulseSequence,
    spec: QickRfPulseSpec,
) -> Tuple[float, float, float]:
    """Resolve an RF pulse's SET-relative timing onto the GUI sequence axis."""
    try:
        set_index = int(spec.segment_name.rsplit("_", 1)[1])
        start_point, end_point = pulse.flat_segments()[set_index]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"unknown RF SET segment {spec.segment_name!r}") from exc
    set_start_us = float(pulse.t[start_point]) / 1000.0
    set_end_us = float(pulse.t[end_point]) / 1000.0
    pulse_start_us = set_start_us + spec.delay_us
    pulse_end_us = pulse_start_us + spec.duration_us
    if spec.require_within_segment and pulse_end_us > set_end_us + 1.0e-12:
        raise ValueError(
            f"RF pulse ends at {pulse_end_us:.6g} us, after {spec.segment_name} "
            f"ends at {set_end_us:.6g} us"
        )
    return pulse_start_us, pulse_end_us, set_end_us


class _MatplotlibTracePlotWidget(Canvas):
    """Scatter/line plot that traces Pulse-X vs Pulse-Y (voltage-voltage)."""

    hold_edit_requested = QtCore.pyqtSignal(int)
    ramp_edit_requested = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        fig = Figure(tight_layout=False)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.line: Line2D | None = None
        self.x_idx: int | None = None
        self.y_idx: int | None = None
        self.ax.set_xlabel("Pulse-X [mV]")
        self.ax.set_ylabel("Pulse-Y [mV]")
        self.ax.grid(True)
        self._pulse: List[PulseSequence] = []
        self._pan_origin: Optional[Tuple[float, float]] = None
        self._time_unit = "us"
        self._stability_result = None
        self._stability_quantity = "magnitude"
        self._stability_bounds: Optional[
            Tuple[float, float, float, float]
        ] = None
        self._trace_edit_targets = {}
        self._output_names: Tuple[str, ...] = ()

        self.mpl_connect("motion_notify_event",  self._on_move)
        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("scroll_event",         self._on_scroll)
        self.mpl_connect("pick_event", self._on_trace_label_pick)

    @property
    def has_selection(self) -> bool:
        return (
            self.x_idx is not None
            and self.y_idx is not None
            and self.x_idx < len(self._pulse)
            and self.y_idx < len(self._pulse)
        )

    def uses_port(self, index: int) -> bool:
        return index == self.x_idx or index == self.y_idx

    def set_time_unit(self, unit: str) -> None:
        if unit not in TIME_UNIT_NS:
            raise ValueError(f"unsupported trace time unit {unit!r}")
        self._time_unit = unit
        if self._pulse:
            self.refresh_trace(self._pulse)

    def set_output_names(self, names: Sequence[str]) -> None:
        names = tuple(str(name) for name in names)
        if self._pulse and len(names) != len(self._pulse):
            raise ValueError("trace output-name count must match pulse count")
        self._output_names = names
        if self._pulse:
            self.refresh_trace(self._pulse)

    def _output_label(self, index: int) -> str:
        if index < len(self._output_names):
            return self._output_names[index]
        return f"awg_{index}"

    def set_stability_overlay(self, result, quantity: str = "magnitude") -> None:
        if quantity not in {"i", "q", "magnitude", "phase"}:
            raise ValueError(f"unsupported Stability overlay data {quantity!r}")
        self._stability_result = result
        self._stability_quantity = quantity
        if self._pulse:
            self.refresh_trace(self._pulse)

    def _stability_values(self, result) -> Tuple[np.ndarray, str, str]:
        if self._stability_quantity == "i":
            return np.asarray(result.i_mean, dtype=float), "I", result.value_unit
        if self._stability_quantity == "q":
            return np.asarray(result.q_mean, dtype=float), "Q", result.value_unit
        if self._stability_quantity == "phase":
            return (
                np.asarray(result.phase_deg, dtype=float),
                "Phase",
                "deg",
            )
        return (
            np.asarray(result.magnitude, dtype=float),
            "Magnitude",
            result.value_unit,
        )

    def _duration_pair_text(
        self,
        x_ns: Optional[float],
        y_ns: Optional[float],
    ) -> str:
        if x_ns is None or y_ns is None:
            return "initial"
        x_value = _time_from_ns(x_ns, self._time_unit)
        y_value = _time_from_ns(y_ns, self._time_unit)
        if np.isclose(x_value, y_value, rtol=0.0, atol=1.0e-12):
            return f"{x_value:.6g} {self._time_unit}"
        return (
            f"X {x_value:.6g} / Y {y_value:.6g} "
            f"{self._time_unit}"
        )

    @staticmethod
    def _axis_edges(values: np.ndarray) -> Tuple[float, float]:
        values = np.asarray(values, dtype=float)
        if values.size == 1:
            return float(values[0] - 0.5), float(values[0] + 0.5)
        step = float(np.median(np.diff(values)))
        return float(values[0] - step / 2.0), float(values[-1] + step / 2.0)

    def _draw_stability_overlay(self) -> None:
        self._stability_bounds = None
        result = self._stability_result
        if result is None or self.x_idx is None or self.y_idx is None:
            return
        trace_axes = (f"awg_{self.x_idx}", f"awg_{self.y_idx}")
        result_axes = (str(result.x_axis_label), str(result.y_axis_label))
        overlay, quantity_label, quantity_unit = self._stability_values(result)
        if result_axes == trace_axes:
            x_values = np.asarray(result.x_voltage_mv, dtype=float)
            y_values = np.asarray(result.y_voltage_mv, dtype=float)
        elif result_axes == trace_axes[::-1]:
            x_values = np.asarray(result.y_voltage_mv, dtype=float)
            y_values = np.asarray(result.x_voltage_mv, dtype=float)
            overlay = overlay.T
        else:
            self.ax.set_title(
                "Last stability scan axes do not match selected trace"
            )
            return
        x_low, x_high = self._axis_edges(x_values)
        y_low, y_high = self._axis_edges(y_values)
        self.ax.imshow(
            overlay,
            extent=(x_low, x_high, y_low, y_high),
            origin="lower",
            aspect="auto",
            alpha=0.52,
            cmap="viridis",
            zorder=0,
        )
        self._stability_bounds = (x_low, x_high, y_low, y_high)
        source_label = (
            str(getattr(result, "source_label", "")).strip()
            or f"Stability scan {result.iteration}"
        )
        self.ax.set_title(
            f"Trace over {source_label} {quantity_label} [{quantity_unit}]"
        )

    def _draw_point_timing(
        self,
        pulse_x: PulseSequence,
        pulse_y: PulseSequence,
    ) -> None:
        point_count = min(pulse_x.set_count, pulse_y.set_count)
        for point_index in range(point_count):
            flat_index = 2 * point_index
            x_name = pulse_x.segment_name(point_index)
            y_name = pulse_y.segment_name(point_index)
            point_name = (
                x_name if x_name == y_name else f"X {x_name} / Y {y_name}"
            )
            hold_text = self._duration_pair_text(
                pulse_x.t[flat_index + 1] - pulse_x.t[flat_index],
                pulse_y.t[flat_index + 1] - pulse_y.t[flat_index],
            )
            lines = [point_name, f"Hold {hold_text}"]
            hold_annotation = self.ax.annotate(
                "\n".join(lines),
                (
                    float(pulse_x.v[flat_index]),
                    float(pulse_y.v[flat_index]),
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                zorder=3,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "0.3",
                    "alpha": 0.78,
                },
                picker=True,
            )
            self._trace_edit_targets[hold_annotation] = (
                "hold",
                point_index,
            )
            if point_index:
                ramp_text = self._duration_pair_text(
                    pulse_x.t[flat_index] - pulse_x.t[flat_index - 1],
                    pulse_y.t[flat_index] - pulse_y.t[flat_index - 1],
                )
                previous_flat_index = flat_index - 2
                ramp_annotation = self.ax.annotate(
                    f"Ramp {ramp_text}",
                    (
                        0.5 * (
                            float(pulse_x.v[previous_flat_index])
                            + float(pulse_x.v[flat_index])
                        ),
                        0.5 * (
                            float(pulse_y.v[previous_flat_index])
                            + float(pulse_y.v[flat_index])
                        ),
                    ),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    zorder=4,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "#fff8cd",
                        "edgecolor": "#7d5a00",
                        "alpha": 0.82,
                    },
                    picker=True,
                )
                self._trace_edit_targets[ramp_annotation] = (
                    "ramp",
                    point_index,
                )

    def refresh_trace(self, pulses: list[PulseSequence]):
        """Refresh the trace plot with the selected pulses."""
        self._pulse = pulses
        if (
            self.x_idx is None or
            self.y_idx is None or
            self.x_idx >= len(self._pulse) or
            self.y_idx >= len(self._pulse)
        ):
            self.ax.cla()
            self.draw_idle()
            return
        px, py = pulses[self.x_idx], pulses[self.y_idx]

        # Build common time grid then interpolate
        t_union = np.unique(np.concatenate((px.t, py.t)))
        vx = np.interp(t_union, px.t, px.v)
        vy = np.interp(t_union, py.t, py.v)

        prev_x_lim = self.ax.get_xlim()
        prev_y_lim = self.ax.get_ylim()
        self.ax.cla()
        self._trace_edit_targets.clear()
        self.ax.set_xlabel(f"{self._output_label(self.x_idx)} [mV]")
        self.ax.set_ylabel(f"{self._output_label(self.y_idx)} [mV]")
        self.ax.grid(True)
        self._draw_stability_overlay()
        self.ax.plot(vx, vy, "-o", color="black", zorder=2)
        self._draw_point_timing(px, py)
        self.ax.set_xlim(prev_x_lim)
        self.ax.set_ylim(prev_y_lim)
        self.draw_idle()

    def fit_view(self):
        """Fit the view to the pulse data."""
        if (
            self.x_idx is None or
            self.y_idx is None or
            self.x_idx >= len(self._pulse) or
            self.y_idx >= len(self._pulse)
        ):
            self.ax.cla()
            self.draw_idle()
            return
        if self._stability_bounds is not None:
            x_low, x_high, y_low, y_high = self._stability_bounds
            self.ax.set_xlim(x_low, x_high)
            self.ax.set_ylim(y_low, y_high)
            self.draw_idle()
            return
        margin_x = max(0.5, 0.03 * float(np.ptp(self._pulse[self.x_idx].v)))
        margin_y = max(0.5, 0.10 * float(np.ptp(self._pulse[self.y_idx].v)))
        x_min = self._pulse[self.x_idx].v.min() - margin_x
        x_max = self._pulse[self.x_idx].v.max() + margin_x
        y_min = self._pulse[self.y_idx].v.min() - margin_y
        y_max = self._pulse[self.y_idx].v.max() + margin_y

        self.ax.set_xlim(
            x_min - margin_x,
            x_max + margin_x
        )
        self.ax.set_ylim(
            y_min - margin_y,
            y_max + margin_y
        )
        self.draw_idle()

    def _on_scroll(self, event):
        factor = -0.1 if event.step > 0 else +0.1
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

        mods = (
            set(event.modifiers) if hasattr(event, "modifiers")
            else {event.key} if event.key else set()
        )

        if "ctrl" in mods:      # Ctrl + wheel : horizontal zoom
            new_w = (xmax - xmin) * factor
            self.ax.set_xlim(xmin - 0.5*new_w, xmax + 0.5*new_w)

        elif "shift" in mods:   # Shift + wheel : vertical zoom
            new_h = (ymax - ymin) * factor
            self.ax.set_ylim(ymin - 0.5*new_h, ymax + 0.5*new_h)

        else:                   # plain wheel : isotropic zoom
            new_w = (xmax - xmin) * factor
            new_h = (ymax - ymin) * factor
            self.ax.set_xlim(xmin - 0.5*new_w, xmax + 0.5*new_w)
            self.ax.set_ylim(ymin - 0.5*new_h, ymax + 0.5*new_h)

        self.draw_idle()

    def _on_press(self, event):
        if (event.button != 1 or not event.inaxes):
            return
        self._pan_origin = (event.xdata, event.ydata)

    def _on_trace_label_pick(self, event) -> None:
        target = self._trace_edit_targets.get(event.artist)
        if target is None:
            return
        kind, segment_index = target
        if kind == "hold":
            self.hold_edit_requested.emit(int(segment_index))
        else:
            self.ramp_edit_requested.emit(int(segment_index))

    def _on_move(self, event):
        if self._pan_origin and event.xdata is not None and event.ydata is not None:
            x0, y0 = self._pan_origin
            dx = x0 - event.xdata
            dy = y0 - event.ydata
            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            self.ax.set_xlim(xmin + dx, xmax + dx)
            self.ax.set_ylim(ymin + dy, ymax + dy)
            self.draw_idle()

    def _on_release(self, _event):
        self._pan_origin = None

class _MatplotlibWaveformPlotWidget(Canvas): # pylint: disable=too-many-instance-attributes
    """Matplotlib widget for displaying and editing a PulseSequence."""
    flat_moved = QtCore.pyqtSignal(int, int, float)   # i0, i1, new_v
    point_moved = QtCore.pyqtSignal(int, int, float)  # i0, i1, new_v

    def __init__(
            self,
            pulse: PulseSequence,
            time_unit: str = DEFAULT_TIME_UNIT,
            parent=None
        ):
        fig                 = Figure(tight_layout=False)
        super().__init__(fig)
        # Pulse and line settings
        self._pulse: List[PulseSequence]    = [pulse]
        self._line: List[Axes]              = []
        self._selected_port_idx             = 0

        # Color highlight settings
        self._default_alpha   = 1.0
        self._dim_alpha       = 0.95
        self._default_lw      = 1.5
        self._highlight_lw    = 2.5
        self._orig_colors     = []
        self._output_names: List[str] = ["awg_0"]

        self.ax             = fig.add_subplot(111)
        self._line,         = [self.ax.plot(
            self._pulse[0].t,
            self._pulse[0].v,
            "-o",
            picker=5,
            label=self._output_names[0],
        )]
        self._orig_colors.append(self._line[0].get_color())
        physical_line, = self.ax.plot(
            self._pulse[0].t,
            self._pulse[0].v,
            "--",
            color=self._orig_colors[0],
            linewidth=1.4,
            zorder=0.5,
        )
        self._physical_line = [physical_line]
        self._physical_time_ns = self._pulse[0].t.copy()
        self._physical_values_mv = np.asarray([self._pulse[0].v.copy()])
        self._voltage_view = "both"
        self._time_unit = time_unit

        # Graph settings
        self.ax.set_xlabel("time [ns]")
        self.ax.set_ylabel("voltage [mV]")
        self.ax.grid(True)
        self.ax.set_autoscale_on(False)
        self.ax.legend(loc="upper right")

        # Dragging and panning settings
        self._drag_flat: Optional[Tuple[int, int]]      = None
        self._drag_point: Optional[Tuple[int, int]]     = None
        self._pan_origin: Optional[Tuple[float, float]] = None
        self._grid_time_ns = 10.0
        self._grid_voltage_mv = 10.0
        self._grid_snap_enabled = False
        self._grid_visible = True
        self._sweep_artists = []
        self._sweep_port_index = None
        self._sweep_time_ns = np.asarray([], dtype=float)
        self._sweep_lower_mv = np.asarray([], dtype=float)
        self._sweep_upper_mv = np.asarray([], dtype=float)
        self._annot         = self.ax.annotate(
            "",
            xy              = (0, 0),
            xytext          = (12, 12),
            textcoords      = "offset points",
            bbox            = dict(boxstyle="round", fc="w"),
            arrowprops      = dict(arrowstyle="->"),
            visible         = False
        )

        self.mpl_connect("motion_notify_event",  self._on_move)
        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("scroll_event",         self._on_scroll)

        self.set_grid(
            time_step_ns=self._grid_time_ns,
            voltage_step_mv=self._grid_voltage_mv,
            snap_enabled=self._grid_snap_enabled,
            visible=self._grid_visible,
        )
        self.fit_view()

    def set_time_unit(self, unit: str) -> None:
        if unit not in TIME_UNIT_NS:
            raise ValueError(f"unsupported time unit {unit!r}")
        self._time_unit = unit
        scale = 1.0 / TIME_UNIT_NS[unit]
        self.ax.set_xlabel(f"time [{unit}]")
        self.ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{value * scale:.6g}")
        )
        self.draw_idle()

    def set_output_names(self, names: Sequence[str]) -> None:
        names = [str(name) for name in names]
        if len(names) != len(self._pulse):
            raise ValueError("waveform output-name count must match pulse count")
        self._output_names = names
        for line, name in zip(self._line, names):
            line.set_label(name)
        self.ax.legend(loc="upper right")
        self.draw_idle()

    @staticmethod
    def _nearest_grid_value(value: float, step: float) -> float:
        scaled = value / step
        if scaled >= 0.0:
            return float(np.floor(scaled + 0.5) * step)
        return float(np.ceil(scaled - 0.5) * step)

    def set_grid(
        self,
        *,
        time_step_ns: float,
        voltage_step_mv: float,
        snap_enabled: bool,
        visible: bool,
    ) -> None:
        if not np.isfinite(time_step_ns) or time_step_ns <= 0.0:
            raise ValueError("time grid spacing must be a positive finite value")
        if not np.isfinite(voltage_step_mv) or voltage_step_mv <= 0.0:
            raise ValueError("voltage grid spacing must be a positive finite value")
        self._grid_time_ns = float(time_step_ns)
        self._grid_voltage_mv = float(voltage_step_mv)
        self._grid_snap_enabled = bool(snap_enabled)
        self._grid_visible = bool(visible)
        self.ax.xaxis.set_major_locator(MultipleLocator(self._grid_time_ns))
        self.ax.yaxis.set_major_locator(MultipleLocator(self._grid_voltage_mv))
        self.ax.grid(self._grid_visible)
        self.draw_idle()

    @property
    def grid_settings(self) -> Tuple[float, float, bool, bool]:
        return (
            self._grid_time_ns,
            self._grid_voltage_mv,
            self._grid_snap_enabled,
            self._grid_visible,
        )

    @property
    def voltage_view(self) -> str:
        return self._voltage_view

    def set_physical_waveforms(self, time_ns, waveforms_mv) -> None:
        time_values = np.asarray(time_ns, dtype=float)
        waveform_values = np.asarray(waveforms_mv, dtype=float)
        expected_shape = (len(self._pulse), time_values.size)
        if time_values.ndim != 1 or waveform_values.shape != expected_shape:
            raise ValueError(
                "physical waveforms must have shape "
                f"{expected_shape}, received {waveform_values.shape}"
            )
        self._physical_time_ns = time_values.copy()
        self._physical_values_mv = waveform_values.copy()
        for index, line in enumerate(self._physical_line):
            line.set_data(time_values, waveform_values[index])
        self.draw_idle()

    def set_voltage_view(self, mode: str) -> None:
        if mode not in {"both", "virtual", "physical"}:
            raise ValueError("voltage view must be 'both', 'virtual', or 'physical'")
        self._voltage_view = mode
        self._drag_flat = None
        self._drag_point = None
        for line in self._line:
            line.set_visible(mode in {"both", "virtual"})
        for line in self._physical_line:
            line.set_visible(mode in {"both", "physical"})
        for artist in self._sweep_artists:
            artist.set_visible(mode in {"both", "physical"})
        self.fit_view()

    def set_sweep_envelope(
        self,
        port_index: int,
        time_ns,
        endpoint_a_mv,
        endpoint_b_mv,
    ) -> None:
        self.set_sweep_envelopes(
            (("legacy", port_index, time_ns, endpoint_a_mv, endpoint_b_mv),)
        )

    def set_sweep_envelopes(self, envelopes) -> None:
        self.clear_sweep_envelope()
        time_bounds = []
        lower_bounds = []
        upper_bounds = []
        for _, port_index, time_ns, endpoint_a_mv, endpoint_b_mv in envelopes:
            time_values = np.asarray(time_ns, dtype=float)
            endpoint_a = np.asarray(endpoint_a_mv, dtype=float)
            endpoint_b = np.asarray(endpoint_b_mv, dtype=float)
            lower = np.minimum(endpoint_a, endpoint_b)
            upper = np.maximum(endpoint_a, endpoint_b)
            color = self._orig_colors[port_index]
            lower_line, = self.ax.plot(
                time_values, lower, ":", color=color, linewidth=1.0
            )
            upper_line, = self.ax.plot(
                time_values, upper, ":", color=color, linewidth=1.0
            )
            fill = self.ax.fill_between(
                time_values, lower, upper, color=color, alpha=0.09
            )
            self._sweep_artists.extend([lower_line, upper_line, fill])
            for artist in (lower_line, upper_line, fill):
                artist.set_visible(self._voltage_view in {"both", "physical"})
            if self._sweep_port_index is None:
                self._sweep_port_index = port_index
            time_bounds.append(time_values)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        if time_bounds:
            self._sweep_time_ns = np.concatenate(time_bounds)
            self._sweep_lower_mv = np.concatenate(lower_bounds)
            self._sweep_upper_mv = np.concatenate(upper_bounds)
        self.draw_idle()

    def clear_sweep_envelope(self) -> None:
        for artist in getattr(self, "_sweep_artists", ()):
            artist.remove()
        self._sweep_artists = []
        self._sweep_port_index = None
        self._sweep_time_ns = np.asarray([], dtype=float)
        self._sweep_lower_mv = np.asarray([], dtype=float)
        self._sweep_upper_mv = np.asarray([], dtype=float)
        self.draw_idle()

    def fit_view(self) -> None:
        """Fit the view to the pulse data."""
        x_values = []
        y_values = []
        if self._voltage_view in {"both", "virtual"}:
            x_values.extend(pulse.t for pulse in self._pulse)
            y_values.extend(pulse.v for pulse in self._pulse)
        if (
            self._voltage_view in {"both", "physical"}
            and self._physical_time_ns.size
            and self._physical_values_mv.size
        ):
            x_values.append(self._physical_time_ns)
            y_values.extend(self._physical_values_mv)
        if not x_values or not y_values:
            return
        x_min = min(float(np.min(values)) for values in x_values)
        x_max = max(float(np.max(values)) for values in x_values)
        y_min = min(float(np.min(values)) for values in y_values)
        y_max = max(float(np.max(values)) for values in y_values)
        margin_x = max(1.0, 0.03 * max(1.0, x_max - x_min))
        margin_y = max(0.5, 0.10 * max(1.0, y_max - y_min))
        if self._sweep_time_ns.size:
            x_min = min(x_min, float(np.min(self._sweep_time_ns)) - margin_x)
            x_max = max(x_max, float(np.max(self._sweep_time_ns)) + margin_x)
            y_min = min(y_min, float(np.min(self._sweep_lower_mv)) - margin_y)
            y_max = max(y_max, float(np.max(self._sweep_upper_mv)) + margin_y)
        self.ax.set_xlim(
            x_min - margin_x,
            x_max + margin_x
        )
        self.ax.set_ylim(
            y_min - margin_y,
            y_max + margin_y
        )
        self.draw_idle()

    def refresh(self) -> None:
        """Refresh plot"""
        for i, pulse in enumerate(self._pulse):
            self._line[i].set_data(pulse.t, pulse.v)
        self.draw_idle()

    def add_pulse(self, pulse: PulseSequence) -> None:
        """Add a new pulse to the plot."""
        self._pulse.append(pulse)
        line,                   = self.ax.plot(pulse.t, pulse.v, "-o", picker=5)
        self._line.append(line)
        self._orig_colors.append(line.get_color())
        physical_line, = self.ax.plot(
            pulse.t,
            pulse.v,
            "--",
            color=line.get_color(),
            linewidth=1.4,
            zorder=0.5,
        )
        self._physical_line.append(physical_line)
        line.set_visible(self._voltage_view in {"both", "virtual"})
        physical_line.set_visible(self._voltage_view in {"both", "physical"})
        self._selected_port_idx = len(self._pulse) - 1
        self.refresh()
        self.fit_view()
        self._update_highlight()

    def get_selected_port_idx(self) -> int:
        """Get the index of the currently selected port."""
        return self._selected_port_idx

    def set_selected_port_idx(self, idx: int) -> None:
        """Set the index of the currently selected port."""
        self._selected_port_idx = idx
        self._update_highlight()

    def remove_pulse(self, idx: int) -> None:
        """Remove the pulse/line at position `idx`."""
        if idx >= len(self._pulse):
            raise ValueError("Index out of bounds for pulse removal.")
        ln = self._line.pop(idx)
        ln.remove()
        physical_line = self._physical_line.pop(idx)
        physical_line.remove()
        self._pulse.pop(idx)
        self._orig_colors.pop(idx)
        self._physical_time_ns = np.asarray([], dtype=float)
        self._physical_values_mv = np.empty((0, 0), dtype=float)
        if self._selected_port_idx >= len(self._pulse):
            self._selected_port_idx = max(0, len(self._pulse) - 1)
        self._update_highlight()
        self.draw_idle()

    def _locate_flat_segment(self, event) -> Optional[Tuple[int, int]]:
        """Return range (i0,i1) if click is on *flat* part, else None."""
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return None
        line_x, line_y = self._line[self._selected_port_idx].get_data()
        # Check Flat Click Events
        for i0 in range(len(self._pulse[self._selected_port_idx].t)):
            if i0 % 2 != 0:
                continue
            if i0 == len(self._pulse[self._selected_port_idx].t) - 1:
                break
            i1 = i0 + 1
            # mid-point of flat
            xm = 0.5 * (line_x[i0] + line_x[i1])
            ym = line_y[i0]
            dx = abs(x - xm)
            dy = abs(y - ym)
            if (
                dx < 0.25 * (line_x[i1] - line_x[i0])
                and dy < 0.05 * (self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
            ):
                return (i0, i1)
        return None

    def _locate_point_segment(self, event) -> Optional[Tuple[int, int]]:
        """Return range (i0,i1) if click is on *flat* part, else None."""
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return None
        line_x, line_y = self._line[self._selected_port_idx].get_data()
        # Check Point Click Events
        for i0 in range(len(self._pulse[self._selected_port_idx].t)):
            dx = abs(x - line_x[i0])
            dy = abs(y - line_y[i0])
            if (
                dx < 0.05 * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
                and dy < 0.05 * (self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
            ):
                return (i0, i0+1)
        return None

    def _on_press(self, event) -> None:
        if (event.button != 1 or not event.inaxes):
            return
        if self._voltage_view == "physical":
            self._pan_origin = (event.xdata, event.ydata)
            return
        seg     = self._locate_flat_segment(event)
        point   = self._locate_point_segment(event)
        if seg:
            self._drag_flat = seg
        elif point:
            self._drag_point = point
        else:
            self._pan_origin = (event.xdata, event.ydata)

    def _on_move(self, event) -> None:
        # tooltip
        self._annot.set_visible(False)

        # drag flat
        if self._drag_flat and event.ydata is not None:
            new_v       = np.clip(
                event.ydata,
                self._pulse[self._selected_port_idx].v_bounds[0],
                self._pulse[self._selected_port_idx].v_bounds[1]
            )
            if self._grid_snap_enabled:
                new_v = self._nearest_grid_value(new_v, self._grid_voltage_mv)
                new_v = np.clip(
                    new_v,
                    np.ceil(self._pulse[self._selected_port_idx].v_bounds[0] / self._grid_voltage_mv)
                    * self._grid_voltage_mv,
                    np.floor(self._pulse[self._selected_port_idx].v_bounds[1] / self._grid_voltage_mv)
                    * self._grid_voltage_mv,
                )
            i0, i1      = self._drag_flat
            self._pulse[self._selected_port_idx].update_flat(self._drag_flat, new_v)
            self.flat_moved.emit(i0, i1, new_v)
            self.refresh()

            self._update_annot(
                self._pulse[self._selected_port_idx].t[i0],
                new_v,
                f"{_time_from_ns(self._pulse[self._selected_port_idx].t[i0], self._time_unit):0.3g} "
                f"{self._time_unit}\n{new_v:0.3g} mV"
            )
            self._annot.set_visible(True)

        # drag point
        if self._drag_point and event.xdata is not None:
            xmin, xmax  = self.ax.get_xlim()
            new_t       = np.clip(event.xdata, xmin, xmax)
            i0, i1      = self._drag_point
            if self._grid_snap_enabled and i0 > 0:
                new_t = self._nearest_grid_value(new_t, self._grid_time_ns)
                minimum = np.nextafter(
                    float(self._pulse[self._selected_port_idx].t[i0 - 1]),
                    np.inf,
                )
                if new_t < minimum:
                    new_t = np.ceil(minimum / self._grid_time_ns) * self._grid_time_ns
            self._pulse[self._selected_port_idx].update_point(self._drag_point, new_t)
            actual_t = float(self._pulse[self._selected_port_idx].t[i0])
            self.point_moved.emit(i0, i1, actual_t)
            self.refresh()

            self._update_annot(
                actual_t,
                self._pulse[self._selected_port_idx].v[i0],
                f"{_time_from_ns(actual_t, self._time_unit):0.3g} {self._time_unit}\n"
                f"{self._pulse[self._selected_port_idx].v[i0]:0.3g} mV"
            )
            self._annot.set_visible(True)
        # pan
        if self._pan_origin and event.xdata is not None and event.ydata is not None:
            x0, y0      = self._pan_origin
            dx          = x0 - event.xdata
            dy          = y0 - event.ydata
            xmin, xmax  = self.ax.get_xlim()
            ymin, ymax  = self.ax.get_ylim()
            self.ax.set_xlim(xmin + dx, xmax + dx)
            self.ax.set_ylim(ymin + dy, ymax + dy)
            self.draw_idle()

    def _on_release(self, _event):
        self._drag_flat  = None
        self._drag_point = None
        self._pan_origin = None

    def _on_scroll(self, event):
        if not event.inaxes:
            return

        factor = -0.1 if event.step > 0 else +0.1
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

        mods = (
            set(event.modifiers) if hasattr(event, "modifiers")
            else {event.key} if event.key else set()
        )

        # Ctrl + wheel : horizontal zoom
        if "ctrl" in mods:
            new_w = (xmax - xmin) * factor
            self.ax.set_xlim(xmin - 0.5*new_w, xmax + 0.5*new_w)

        # Shift + wheel : vertical zoom
        elif "shift" in mods:
            new_h = (ymax - ymin) * factor
            self.ax.set_ylim(ymin - 0.5*new_h, ymax + 0.5*new_h)

        # plain wheel : isotropic zoom
        else:
            new_w = (xmax - xmin) * factor
            new_h = (ymax - ymin) * factor
            self.ax.set_xlim(xmin - 0.5*new_w, xmax + 0.5*new_w)
            self.ax.set_ylim(ymin - 0.5*new_h, ymax + 0.5*new_h)

        self.draw_idle()

    def _update_highlight(self):
        for i, line in enumerate(self._line):
            if i == self._selected_port_idx:
                line.set_zorder(10)
                line.set_alpha(self._default_alpha)
                line.set_linewidth(self._highlight_lw)
                line.set_color(self._orig_colors[i])
            else:
                line.set_zorder(1)
                line.set_alpha(self._dim_alpha)
                line.set_linewidth(self._default_lw)
                line.set_color(self._lighten(self._orig_colors[i], 0.1))
            physical = self._physical_line[i]
            physical.set_color(line.get_color())
            physical.set_alpha(line.get_alpha())
            physical.set_linewidth(1.8 if i == self._selected_port_idx else 1.4)
            physical.set_zorder(9 if i == self._selected_port_idx else 0.5)
        self.draw_idle()

    @staticmethod
    def _lighten(c, amount: float = 0.5):
        r, g, b, a = colors.to_rgba(c)
        white      = 1.0
        return (
            r + (white - r)*amount,
            g + (white - g)*amount,
            b + (white - b)*amount,
            a
        )

    def _restore_full_intensity(self):
        for i, line in enumerate(self._line):
            line.set_alpha(self._default_alpha)
            line.set_linewidth(self._default_lw)
            line.set_color(self._orig_colors[i])
            line.set_zorder(1)
            physical = self._physical_line[i]
            physical.set_alpha(self._default_alpha)
            physical.set_linewidth(1.4)
            physical.set_color(self._orig_colors[i])
            physical.set_zorder(0.5)
        self.draw_idle()

    def _update_annot(self, x, y, text: str):
        self._annot.xy = (x, y)
        self._annot.set_text(text)

        xmid = sum(self.ax.get_xlim()) / 2
        ymid = sum(self.ax.get_ylim()) / 2

        dx, dy =  +12, +12
        ha,  va = "left", "bottom"

        if x > xmid:
            dx, ha = -dx, "right"
        if y > ymid:
            dy, va = -dy, "top"

        self._annot.set_position((dx, dy))
        self._annot.set_ha(ha)
        self._annot.set_va(va)
        self._annot.set_visible(True)
        self.ax.figure.canvas.draw_idle()


if _USE_PYQTGRAPH:
    try:
        from .dc_waveform_widgets import (
            RfPulsePreviewWidget,
            RfPulseTimelineWidget,
            TracePlotWidget,
            WaveformPlotWidget,
        )
    except ImportError:
        from dc_waveform_widgets import (
            RfPulsePreviewWidget,
            RfPulseTimelineWidget,
            TracePlotWidget,
            WaveformPlotWidget,
        )
    MatplotWidget = WaveformPlotWidget
else:
    TracePlotWidget = _MatplotlibTracePlotWidget
    MatplotWidget = _MatplotlibWaveformPlotWidget
    RfPulsePreviewWidget = None
    RfPulseTimelineWidget = None


def _trace_duration_spin(value_ns: float, unit: str) -> QtWidgets.QDoubleSpinBox:
    editor = QtWidgets.QDoubleSpinBox()
    editor.setDecimals(9)
    editor.setRange(1.0e-9, 1.0e12)
    editor.setSuffix(f" {unit}")
    value = _time_from_ns(float(value_ns), unit)
    editor.setValue(value)
    editor.setSingleStep(max(1.0e-6, abs(value) * 0.1))
    return editor


class TraceHoldEditDialog(QtWidgets.QDialog):
    """Edit one X/Y trace point and its shared hold duration."""

    def __init__(
        self,
        *,
        segment_index: int,
        x_output: int,
        y_output: int,
        x_mv: float,
        y_mv: float,
        x_bounds: Tuple[float, float],
        y_bounds: Tuple[float, float],
        hold_ns: float,
        segment_name: str,
        time_unit: str,
        parent=None,
    ):
        super().__init__(parent)
        self._time_unit = time_unit
        self._same_output = int(x_output) == int(y_output)
        self.setWindowTitle(f"Edit Hold {segment_name}")
        form = QtWidgets.QFormLayout(self)

        self.segment_name = QtWidgets.QLineEdit(str(segment_name))
        form.addRow("Segment name:", self.segment_name)

        self.x_voltage = QtWidgets.QDoubleSpinBox()
        self.x_voltage.setDecimals(9)
        self.x_voltage.setRange(float(x_bounds[0]), float(x_bounds[1]))
        self.x_voltage.setSuffix(" mV")
        self.x_voltage.setValue(float(x_mv))
        form.addRow(f"X virtual voltage (AWG {int(x_output)}):", self.x_voltage)

        self.y_voltage = QtWidgets.QDoubleSpinBox()
        self.y_voltage.setDecimals(9)
        self.y_voltage.setRange(float(y_bounds[0]), float(y_bounds[1]))
        self.y_voltage.setSuffix(" mV")
        self.y_voltage.setValue(float(y_mv))
        self.y_voltage.setEnabled(not self._same_output)
        form.addRow(f"Y virtual voltage (AWG {int(y_output)}):", self.y_voltage)

        self.hold_duration = _trace_duration_spin(hold_ns, time_unit)
        form.addRow("Hold duration:", self.hold_duration)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save
            | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.setMinimumWidth(390)

    def values(self) -> Tuple[float, float, float, str]:
        y_mv = (
            self.x_voltage.value()
            if self._same_output
            else self.y_voltage.value()
        )
        return (
            float(self.x_voltage.value()),
            float(y_mv),
            _time_to_ns(
                float(self.hold_duration.value()),
                self._time_unit,
            ),
            self.segment_name.text(),
        )


class TraceRampEditDialog(QtWidgets.QDialog):
    """Edit the shared duration of one trace transition."""

    def __init__(
        self,
        *,
        segment_index: int,
        ramp_ns: float,
        time_unit: str,
        parent=None,
    ):
        super().__init__(parent)
        self._time_unit = time_unit
        self.setWindowTitle(
            f"Edit Ramp P{int(segment_index) - 1} to P{int(segment_index)}"
        )
        form = QtWidgets.QFormLayout(self)
        self.ramp_duration = _trace_duration_spin(ramp_ns, time_unit)
        form.addRow("Ramp duration:", self.ramp_duration)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save
            | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.setMinimumWidth(340)

    def value_ns(self) -> float:
        return _time_to_ns(
            float(self.ramp_duration.value()),
            self._time_unit,
        )


class ControlPanel(QtWidgets.QWidget): # pylint: disable=too-few-public-methods
    """Input boxes plus a live table of all segments."""

    add_requested       = QtCore.pyqtSignal(float, float, float)
    update_plot         = QtCore.pyqtSignal()
    port_is_selected    = QtCore.pyqtSignal(int)
    sweep_requested     = QtCore.pyqtSignal(int, int)
    sweep_remove_requested = QtCore.pyqtSignal(int, int)
    ramp_sweep_requested = QtCore.pyqtSignal(int, int)
    ramp_sweep_remove_requested = QtCore.pyqtSignal(int, int)
    hold_sweep_requested = QtCore.pyqtSignal(int, int)
    hold_sweep_remove_requested = QtCore.pyqtSignal(int, int)
    segment_timing_changed = QtCore.pyqtSignal(int, int)
    segment_name_changed = QtCore.pyqtSignal(int, int)
    segment_structure_changed = QtCore.pyqtSignal(int, str, int)
    hardware_output_edit_requested = QtCore.pyqtSignal(int)
    output_name_changed = QtCore.pyqtSignal(int, str)
    port_idx: int       = 0

    def __init__(
            self,
            pulse: PulseSequence,
            time_unit: str = DEFAULT_TIME_UNIT,
            parent=None
        ):
        super().__init__(parent)
        self._pulse         = pulse
        self.idx            = ControlPanel.port_idx
        ControlPanel.port_idx += 1
        self._sweep_row: Optional[int] = None
        self._sweep_rows = set()
        self._sweep_color: Optional[QtGui.QColor] = None
        self._ramp_sweep_rows = set()
        self._hold_sweep_rows = set()
        self._time_unit = time_unit

        v_splitter          = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        layout              = QtWidgets.QVBoxLayout(self)

        form                = QtWidgets.QFormLayout()
        self.output_name = QtWidgets.QLineEdit(f"awg_{self.idx}")
        self.output_name.setMaxLength(128)
        self.output_name.setToolTip(
            "Shared display name for this AWG output. The original awg_N "
            "identifier is retained in saved data."
        )
        self.output_name.editingFinished.connect(
            lambda: self.output_name_changed.emit(
                self.idx,
                self.output_name.text(),
            )
        )
        self.edit_ramp = QtWidgets.QLineEdit(
            f"{_time_from_ns(DEFAULT_GUI_RAMP_NS, self._time_unit):.6g}"
        )
        self.edit_flat = QtWidgets.QLineEdit(
            f"{_time_from_ns(DEFAULT_GUI_FLAT_NS, self._time_unit):.6g}"
        )
        self.edit_v = QtWidgets.QLineEdit(f"{DEFAULT_INITIAL_VOLTAGE_MV:.6g}")

        for w in (self.edit_ramp, self.edit_flat, self.edit_v):
            w.setValidator(QtGui.QDoubleValidator(decimals=9))

        self._ramp_label = QtWidgets.QLabel()
        self._flat_label = QtWidgets.QLabel()
        form.addRow("Output name:", self.output_name)
        form.addRow(self._ramp_label, self.edit_ramp)
        form.addRow(self._flat_label, self.edit_flat)
        form.addRow("Virtual V [mV]:", self.edit_v)

        btn_add             = QtWidgets.QPushButton("Add segment")
        btn_add.clicked.connect(self._on_add)
        form.addRow(btn_add)

        btn_select_port     = QtWidgets.QPushButton("Edit this AWG output")
        btn_select_port.clicked.connect(self._select_port)
        form.addRow(btn_select_port)

        form_widget         = QtWidgets.QWidget()
        form_widget.setLayout(form)

        self.table          = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["#", "Name", "Ramp [ns]", "Flat [ns]", "Virtual V [mV]"]
        )
        self.table.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Expanding
        )
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_menu)

        header              = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
        )
        self.table.itemChanged.connect(self._on_item_changed)
        self._refresh_time_labels()

        v_splitter.addWidget(form_widget)
        v_splitter.addWidget(self.table)
        layout.addWidget(v_splitter)
        self.refresh_table()

        layout.setStretch(0, 1)

    def set_output_name(self, name: str) -> None:
        with QtCore.QSignalBlocker(self.output_name):
            self.output_name.setText(str(name))

    def _on_table_menu(self, pos: QtCore.QPoint) -> None:
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        self.table.selectRow(row)
        self.port_is_selected.emit(self.idx)
        menu = QtWidgets.QMenu(self)
        act_ins_above = menu.addAction("Insert segment above")
        act_ins_below = menu.addAction("Insert segment below")
        act_ins_above.setEnabled(row > 0)
        sweep_menu = menu.addMenu("Voltage sweep")
        act_sweep = sweep_menu.addAction("Configure sweep...")
        act_remove_sweep = sweep_menu.addAction("Remove sweep")
        act_remove_sweep.setEnabled(row in self._sweep_rows)
        ramp_sweep_menu = menu.addMenu("RAMP rate sweep")
        ramp_sweep_menu.setEnabled(row > 0)
        act_ramp_sweep = ramp_sweep_menu.addAction("Configure sweep...")
        act_remove_ramp_sweep = ramp_sweep_menu.addAction("Remove sweep")
        act_remove_ramp_sweep.setEnabled(row in self._ramp_sweep_rows)
        hold_sweep_menu = menu.addMenu("Hold time sweep")
        act_hold_sweep = hold_sweep_menu.addAction("Configure sweep...")
        act_remove_hold_sweep = hold_sweep_menu.addAction("Remove sweep")
        act_remove_hold_sweep.setEnabled(row in self._hold_sweep_rows)
        menu.addSeparator()
        act_del       = menu.addAction("Delete segment")
        act_del.setEnabled(row > 0)
        act_del.setToolTip(
            "The initial SET defines the waveform baseline and cannot be "
            "deleted."
        )

        chosen = menu.exec_(self.table.viewport().mapToGlobal(pos))
        if chosen == act_sweep:
            self.sweep_requested.emit(self.idx, row)
            return
        if chosen == act_remove_sweep:
            self.sweep_remove_requested.emit(self.idx, row)
            return
        if chosen == act_ramp_sweep:
            self.ramp_sweep_requested.emit(self.idx, row)
            return
        if chosen == act_remove_ramp_sweep:
            self.ramp_sweep_remove_requested.emit(self.idx, row)
            return
        if chosen == act_hold_sweep:
            self.hold_sweep_requested.emit(self.idx, row)
            return
        if chosen == act_remove_hold_sweep:
            self.hold_sweep_remove_requested.emit(self.idx, row)
            return
        action = {
            act_ins_above: "insert_above",
            act_ins_below: "insert_below",
            act_del: "delete",
        }.get(chosen)
        if action is None:
            return
        if not self._edit_segment_structure(action, row):
            QtWidgets.QMessageBox.warning(
                self,
                "Edit failed",
                "The requested insertion or deletion is not valid for this segment.",
            )

    def _edit_segment_structure(self, action: str, row: int) -> bool:
        """Apply one table structure edit and report its logical SET index."""
        row = int(row)
        if action == "insert_above":
            operation = "insert"
            segment_index = row
            ok = self._pulse.insert_flat_ramp(row * 2)
        elif action == "insert_below":
            operation = "insert"
            segment_index = row + 1
            ok = self._pulse.insert_flat_ramp(row * 2 + 2)
        elif action == "delete":
            operation = "delete"
            segment_index = row
            ok = self._pulse.delete_flat_ramp(row * 2)
        else:
            raise ValueError(f"unknown segment structure action {action!r}")
        if not ok:
            return False

        # Move the local marker state before the first repaint. The main-window
        # model remaps the sweep specifications below, but repainting first
        # would briefly mark the newly inserted row as the sweep target.
        self._remap_structure_markers(operation, segment_index)
        self.refresh_table()
        self.segment_structure_changed.emit(
            self.idx,
            operation,
            segment_index,
        )
        # A deleted row's selection otherwise moves onto the row below it.
        # Qt's selection brush then hides the sweep-target background, making
        # a correctly remapped sweep look as though it disappeared.
        self.table.clearSelection()
        self.table.setCurrentCell(-1, -1)
        return True

    def _remap_structure_markers(
        self,
        operation: str,
        segment_index: int,
    ) -> None:
        """Move local sweep markers with a shared segment structure edit."""
        self._sweep_rows = _remap_segment_rows(
            self._sweep_rows,
            operation,
            segment_index,
        )
        self._hold_sweep_rows = _remap_segment_rows(
            self._hold_sweep_rows,
            operation,
            segment_index,
        )
        self._ramp_sweep_rows = _remap_segment_rows(
            self._ramp_sweep_rows,
            operation,
            segment_index,
        )
        self._sweep_row = min(self._sweep_rows) if self._sweep_rows else None

    def _sweep_indicator_icon(
        self,
        *,
        voltage_sweep: bool,
        ramp_sweep: bool,
        hold_sweep: bool,
    ) -> QtGui.QIcon:
        """Return a compact marker that remains visible on selected rows."""
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setPen(QtCore.Qt.NoPen)
        voltage_color = QtGui.QColor(
            self._sweep_color or QtGui.QColor("#f2c14e")
        )
        voltage_color.setAlpha(255)
        ramp_color = QtGui.QColor("#3978c5")
        hold_color = QtGui.QColor("#2b9a66")
        active_colors = [
            color
            for enabled, color in (
                (voltage_sweep, voltage_color),
                (ramp_sweep, ramp_color),
                (hold_sweep, hold_color),
            )
            if enabled
        ]
        if len(active_colors) == 1:
            painter.setBrush(active_colors[0])
            painter.drawEllipse(QtCore.QRectF(3.0, 3.0, 10.0, 10.0))
        else:
            width = 13.0 / max(1, len(active_colors))
            for index, color in enumerate(active_colors):
                painter.setBrush(color)
                painter.drawRoundedRect(
                    QtCore.QRectF(1.0 + index * width, 2.0, width - 1.0, 12.0),
                    1.5,
                    1.5,
                )
        painter.end()
        return QtGui.QIcon(pixmap)

    def set_sweep_row(
        self,
        row: Optional[int],
        color: Optional[QtGui.QColor] = None,
    ) -> None:
        self.set_sweep_rows(() if row is None else (row,), color)

    def set_sweep_rows(
        self,
        rows,
        color: Optional[QtGui.QColor] = None,
    ) -> None:
        self.set_sweep_state(
            rows,
            self._ramp_sweep_rows,
            self._hold_sweep_rows,
            color,
        )

    def set_ramp_sweep_rows(self, rows) -> None:
        self.set_sweep_state(
            self._sweep_rows,
            rows,
            self._hold_sweep_rows,
            self._sweep_color,
        )

    def set_hold_sweep_rows(self, rows) -> None:
        self.set_sweep_state(
            self._sweep_rows,
            self._ramp_sweep_rows,
            rows,
            self._sweep_color,
        )

    def set_sweep_state(
        self,
        voltage_rows,
        ramp_rows,
        hold_rows,
        color: Optional[QtGui.QColor] = None,
    ) -> None:
        """Apply all sweep marker sets with one atomic table refresh."""
        self._sweep_rows = {int(row) for row in voltage_rows}
        self._sweep_row = min(self._sweep_rows) if self._sweep_rows else None
        self._sweep_color = QtGui.QColor(color) if color is not None else None
        self._ramp_sweep_rows = {int(row) for row in ramp_rows}
        self._hold_sweep_rows = {int(row) for row in hold_rows}
        self.refresh_table()

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        row, col = item.row(), item.column()
        if col == 1:
            try:
                self._pulse.rename_segment(row, item.text())
            except (IndexError, TypeError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid segment name",
                    str(exc),
                )
                self.refresh_table()
                return
            self.refresh_table()
            self.segment_name_changed.emit(self.idx, row)
            return
        try:
            val = float(item.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Bad edit", "Enter a number")
            return

        # Map table row
        flat_idx = row * 2              # because every table row is (i,i+1) with i even
        if col == 2:
            ok = self._pulse.edit_ramp(flat_idx, _time_to_ns(val, self._time_unit))
        elif col == 3:
            ok = self._pulse.edit_flat(flat_idx, _time_to_ns(val, self._time_unit))
        elif col == 4:
            ok = self._pulse.edit_voltage(flat_idx, val)
        else:
            return
        if not ok:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid edit",
                "Durations must be positive and the first SET has no incoming ramp.",
            )
            self.refresh_table()
            return
        self.refresh_table()
        if col in (2, 3):
            self.segment_timing_changed.emit(self.idx, row)
        else:
            self.update_plot.emit()

    def _on_add(self):
        self.port_is_selected.emit(self.idx)
        try:
            ramp = float(self.edit_ramp.text())
            flat = float(self.edit_flat.text())
            v    = float(self.edit_v.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Bad input",
                "Please enter valid numbers."
            )
            return
        self.add_requested.emit(
            _time_to_ns(ramp, self._time_unit),
            _time_to_ns(flat, self._time_unit),
            v,
        )

    def _refresh_time_labels(self) -> None:
        self._ramp_label.setText(f"Ramp [{self._time_unit}]:")
        self._flat_label.setText(f"Flat [{self._time_unit}]:")
        self.table.setHorizontalHeaderLabels(
            [
                "#",
                "Name",
                f"Ramp [{self._time_unit}]",
                f"Flat [{self._time_unit}]",
                "Virtual V [mV]",
            ]
        )

    def set_time_unit(self, unit: str) -> None:
        if unit not in TIME_UNIT_NS:
            raise ValueError(f"unsupported time unit {unit!r}")
        if unit == self._time_unit:
            return
        old_unit = self._time_unit
        for editor in (self.edit_ramp, self.edit_flat):
            try:
                value_ns = _time_to_ns(float(editor.text()), old_unit)
            except ValueError:
                continue
            editor.setText(f"{_time_from_ns(value_ns, unit):.6g}")
        self._time_unit = unit
        self._refresh_time_labels()
        self.refresh_table()

    def refresh_table(self):
        with QtCore.QSignalBlocker(self.table):
            segs = self._pulse.flat_segments()
            self.table.setRowCount(len(segs))
            for row, (i0, i1) in enumerate(segs):
                ramp  = self._pulse.t[i0] - self._pulse.t[i0 - 1] if i0 > 0 else 0
                flat  = self._pulse.t[i1] - self._pulse.t[i0]
                v     = self._pulse.v[i0]
                for col, val in enumerate(
                    [
                        row + 1,
                        self._pulse.segment_name(row),
                        _time_from_ns(ramp, self._time_unit),
                        _time_from_ns(flat, self._time_unit),
                        v,
                    ]
                ):
                    text = str(val) if col == 1 else f"{val:.6g}"
                    item = QtWidgets.QTableWidgetItem(text)
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    if row in self._sweep_rows:
                        highlight = QtGui.QColor(
                            self._sweep_color or QtGui.QColor("#f2c14e")
                        )
                        highlight.setAlpha(42)
                        item.setBackground(QtGui.QBrush(highlight))
                        item.setToolTip("Voltage sweep target")
                    if row in self._ramp_sweep_rows:
                        highlight = QtGui.QColor("#3978c5")
                        highlight.setAlpha(52)
                        item.setBackground(QtGui.QBrush(highlight))
                        tooltip = "RAMP duration/rate sweep target"
                        if row in self._sweep_rows:
                            tooltip += "; voltage sweep target"
                        item.setToolTip(tooltip)
                    if row in self._hold_sweep_rows:
                        highlight = QtGui.QColor("#2b9a66")
                        highlight.setAlpha(52)
                        item.setBackground(QtGui.QBrush(highlight))
                        tooltip = "SET hold-duration sweep target"
                        if row in self._sweep_rows:
                            tooltip += "; voltage sweep target"
                        if row in self._ramp_sweep_rows:
                            tooltip += "; incoming RAMP-rate sweep target"
                        item.setToolTip(tooltip)
                    if col == 0 and (
                        row in self._sweep_rows
                        or row in self._ramp_sweep_rows
                        or row in self._hold_sweep_rows
                    ):
                        item.setIcon(
                            self._sweep_indicator_icon(
                                voltage_sweep=row in self._sweep_rows,
                                ramp_sweep=row in self._ramp_sweep_rows,
                                hold_sweep=row in self._hold_sweep_rows,
                            )
                        )
                    if col == 0 or (col == 2 and row == 0):
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                    self.table.setItem(row, col, item)

    def _select_port(self):
        """Select this output and open its active hardware front panel."""
        self.port_is_selected.emit(self.idx)
        self.hardware_output_edit_requested.emit(self.idx)


class MultiControlPanel(QtWidgets.QWidget): # pylint: disable=too-few-public-methods
    """Multi-waveform editor with HWH-backed AWG output mapping."""

    front_panel_requested = QtCore.pyqtSignal(object)
    awg_channel_changed = QtCore.pyqtSignal(int, int)

    def __init__(
            self,
            pulse: PulseSequence,
            initial_color: str,
            add_port: Callable,
            time_unit: str = DEFAULT_TIME_UNIT,
            awg_channels: Sequence[int] = (DEFAULT_QSTL_AWG_CHANNELS[0],),
            parent=None
        ):
        super().__init__(parent)
        self._time_unit = time_unit
        self._selected_index = 0
        self._awg_channels = tuple(int(channel) for channel in awg_channels)
        self._output_display_names = tuple(
            f"awg_{index}" for index in range(len(self._awg_channels))
        )
        self._front_panel_configuration = None
        self._qcs_front_panel_configuration = None
        self._qcs_dc_channel_names: Tuple[str, ...] = ()
        self._hardware_backend = EXECUTION_BACKEND_QCS
        self._ctrl_pannels: List[ControlPanel] = [
            ControlPanel(pulse, time_unit=time_unit)
        ]
        self._color_map: List[str]              = [initial_color]
        self.splitter                           = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)

        self.front_panel_preview = HardwareFrontPanelPreview(self)
        self.front_panel_preview.set_scope("output")
        self.front_panel_preview.set_qcs_selection("dc", 0)
        self.front_panel_preview.setToolTip(
            "Select the DAC SMA used by the active AWG output"
        )
        self.front_panel_preview.activated.connect(
            lambda: self.front_panel_requested.emit(self)
        )
        self.mapping_summary = QtWidgets.QLabel(self)
        self.mapping_summary.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self.front_panel_group = QtWidgets.QGroupBox(
            "QCS DC Output Channel",
            self,
        )
        front_panel_layout = QtWidgets.QVBoxLayout(self.front_panel_group)
        front_panel_layout.setContentsMargins(6, 6, 6, 6)
        front_panel_layout.setSpacing(2)
        front_panel_layout.addWidget(self.front_panel_preview)
        front_panel_layout.addWidget(self.mapping_summary)

        # Port Add Button
        btn_add_port                            = QtWidgets.QPushButton("Add Port")
        btn_add_port.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Expanding
        )
        btn_widget                              = QtWidgets.QWidget()
        self._add_port_widget                   = btn_widget
        btn_layout                              = QtWidgets.QVBoxLayout(btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_widget.setFixedWidth(btn_add_port.sizeHint().width())
        btn_layout.addWidget(btn_add_port)
        btn_add_port.clicked.connect(add_port)

        self.splitter.addWidget(self._ctrl_pannels[0])
        self.splitter.addWidget(btn_widget)

        self.panel_scroll = QtWidgets.QScrollArea(self)
        self.panel_scroll.setWidgetResizable(True)
        self.panel_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.panel_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded
        )
        self.panel_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.panel_scroll.setWidget(self.splitter)
        self.panel_scroll.setMinimumHeight(330)

        # Control Panels list
        self.panel_table                        = QtWidgets.QTableWidget(0, 5)
        self.panel_table.setHorizontalHeaderLabels(
            ["AWG output", "Color", "QCS channel", "set_x", "set_y"]
        )
        self.panel_table.verticalHeader().setVisible(False)
        self.panel_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.panel_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        panel_table_header = self.panel_table.horizontalHeader()
        panel_table_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        panel_table_header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        panel_table_header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        panel_table_header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        panel_table_header.setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)

        self.btn_reset = QtWidgets.QPushButton("Reset highlight")

        control_panel_v_splitter                = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        control_panel_v_splitter.addWidget(self.panel_scroll)
        control_panel_v_splitter.addWidget(self.panel_table)
        control_panel_v_splitter.addWidget(self.btn_reset)

        layout                                  = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.front_panel_group)
        layout.addWidget(control_panel_v_splitter)
        self.set_awg_channels(self._awg_channels)
        self.update_port_strip_geometry()
        self.refresh_table()

    def refresh_table(self):
        """Refresh all control panels' tables."""
        for ctrl in self._ctrl_pannels:
            ctrl.refresh_table()

    def set_time_unit(self, unit: str) -> None:
        self._time_unit = unit
        for control in self._ctrl_pannels:
            control.set_time_unit(unit)

    def set_awg_channels(self, channels: Sequence[int]) -> None:
        channels = tuple(int(channel) for channel in channels)
        if len(channels) != len(self._ctrl_pannels):
            raise ValueError("AWG channel count must match the waveform output count")
        if any(channel < 0 for channel in channels):
            raise ValueError("AWG channels must be nonnegative")
        if len(set(channels)) != len(channels):
            raise ValueError("AWG channels must be unique")
        self._awg_channels = channels
        self._selected_index = min(self._selected_index, len(channels) - 1)
        self._refresh_mapping_summary()

    def set_output_display_names(self, names: Sequence[str]) -> None:
        names = tuple(str(name) for name in names)
        if len(names) != len(self._ctrl_pannels):
            raise ValueError("AWG output display-name count must match outputs")
        self._output_display_names = names
        for control, name in zip(self._ctrl_pannels, names):
            control.set_output_name(name)
        self._refresh_mapping_summary()

    def set_selected_port(self, index: int) -> None:
        if not 0 <= int(index) < len(self._ctrl_pannels):
            raise IndexError("selected AWG output is out of range")
        self._selected_index = int(index)
        self.front_panel_preview.set_qcs_selection(
            "dc",
            self._selected_index,
        )
        self._refresh_mapping_summary()
        QtCore.QTimer.singleShot(
            0,
            lambda: self.panel_scroll.ensureWidgetVisible(
                self._ctrl_pannels[self._selected_index], 12, 0
            ),
        )

    def set_front_panel_configuration(self, configuration) -> None:
        self._front_panel_configuration = configuration
        self.front_panel_preview.set_configuration(configuration)
        self._refresh_mapping_summary()

    def set_hardware_backend(self, backend: str) -> None:
        self._hardware_backend = str(backend).strip().lower()
        self.front_panel_preview.set_backend(self._hardware_backend)
        self.front_panel_group.setTitle(
            "QCS DC Output Channel"
            if self._hardware_backend == EXECUTION_BACKEND_QCS
            else "AWG Output Channel"
        )
        self._refresh_mapping_summary()

    def set_qcs_front_panel_configuration(
        self,
        configuration: Optional[Mapping[str, object]],
        dc_channel_names: Sequence[str],
    ) -> None:
        self._qcs_front_panel_configuration = configuration
        self._qcs_dc_channel_names = tuple(
            str(name) for name in dc_channel_names
        )
        self.front_panel_preview.set_qcs_configuration(configuration)
        self.front_panel_preview.set_qcs_selection(
            "dc",
            self._selected_index,
        )
        self._refresh_mapping_summary()

    def front_panel_values(self) -> dict:
        return {
            "output_ch": self._awg_channels[self._selected_index],
            "output_nqz": 1,
        }

    def apply_front_panel_settings(self, values: Mapping[str, object]) -> None:
        channel = int(values["output_ch"])
        if self._front_panel_configuration is not None:
            port_index = QickFrontPanelControl._find_port_for_channel(
                self._front_panel_configuration.outputs,
                channel,
            )
            if port_index is None:
                raise ValueError(
                    f"generator {channel} is not mapped to a front-panel DAC"
                )
            port = self._front_panel_configuration.port("output", port_index)
            channel_position = port.qick_channels.index(channel)
            block_path = port.block_paths[channel_position].lower()
            if "axis_awg_tuning_v1" not in block_path:
                raise ValueError(
                    f"generator {channel} is not an axis_awg_tuning_v1 channel"
                )
        self.awg_channel_changed.emit(
            self._selected_index,
            channel,
        )

    def _refresh_mapping_summary(self) -> None:
        canonical_name = f"awg_{self._selected_index}"
        display_name = (
            self._output_display_names[self._selected_index]
            if self._selected_index < len(self._output_display_names)
            else canonical_name
        )
        output_label = (
            f"{display_name} ({canonical_name})"
            if display_name != canonical_name
            else canonical_name
        )
        if self._hardware_backend == EXECUTION_BACKEND_QCS:
            if self._selected_index < len(self._qcs_dc_channel_names):
                name = self._qcs_dc_channel_names[self._selected_index]
                details = f"{output_label} | QCS {name}"
            else:
                details = (
                    f"{output_label} | "
                    "QCS DC channel not mapped"
                )
            self.mapping_summary.setText(details)
            self.front_panel_preview.set_qcs_selection(
                "dc",
                self._selected_index,
            )
            return
        if not self._awg_channels:
            self.mapping_summary.setText("No AWG output mapping")
            return
        channel = self._awg_channels[self._selected_index]
        details = f"{output_label} | QICK generator {channel}"
        if self._front_panel_configuration is not None:
            port_index = QickFrontPanelControl._find_port_for_channel(
                self._front_panel_configuration.outputs,
                channel,
            )
            if port_index is not None:
                port = self._front_panel_configuration.port(
                    "output", port_index
                )
                details += f" | {port.label} | {port.board_label}"
        self.mapping_summary.setText(details)
        self.front_panel_preview.set_channels(output_ch=channel)

    def update_port_strip_geometry(self) -> None:
        """Preserve a usable card width and expose overflow via a scrollbar."""
        card_width = 310
        for control in self._ctrl_pannels:
            control.setMinimumWidth(card_width)
        add_width = max(72, self._add_port_widget.sizeHint().width())
        total_width = card_width * len(self._ctrl_pannels) + add_width
        total_width += self.splitter.handleWidth() * len(self._ctrl_pannels)
        self.splitter.setMinimumWidth(total_width)
        self.splitter.setSizes(
            [card_width] * len(self._ctrl_pannels) + [add_width]
        )
        self.splitter.updateGeometry()


class GridSettingsDialog(QtWidgets.QDialog):
    """Configure fixed waveform-grid spacing and drag snapping."""

    def __init__(
        self,
        *,
        time_step_ns: float,
        voltage_step_mv: float,
        snap_enabled: bool,
        visible: bool,
        time_unit: str = DEFAULT_TIME_UNIT,
        parent=None,
    ):
        super().__init__(parent)
        self._time_unit = time_unit
        self.setWindowTitle("Waveform grid settings")
        form = QtWidgets.QFormLayout(self)

        self.time_step_ns = QtWidgets.QDoubleSpinBox()
        self.time_step_ns.setRange(1.0e-6, 1.0e12)
        self.time_step_ns.setDecimals(6)
        self.time_step_ns.setValue(_time_from_ns(time_step_ns, time_unit))
        self.time_step_ns.setSuffix(f" {time_unit}")

        self.voltage_step_mv = QtWidgets.QDoubleSpinBox()
        self.voltage_step_mv.setRange(1.0e-6, 1.0e9)
        self.voltage_step_mv.setDecimals(6)
        self.voltage_step_mv.setValue(voltage_step_mv)
        self.voltage_step_mv.setSuffix(" mV")

        self.snap_enabled = QtWidgets.QCheckBox("Snap dragged points to the grid")
        self.snap_enabled.setChecked(snap_enabled)
        self.visible = QtWidgets.QCheckBox("Show fixed grid lines")
        self.visible.setChecked(visible)
        origin_note = QtWidgets.QLabel(
            f"Time and voltage grids are anchored at 0 {time_unit} and 0 mV."
        )
        origin_note.setWordWrap(True)

        form.addRow("Time spacing:", self.time_step_ns)
        form.addRow("Voltage spacing:", self.voltage_step_mv)
        form.addRow(self.snap_enabled)
        form.addRow(self.visible)
        form.addRow(origin_note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> Tuple[float, float, bool, bool]:
        return (
            _time_to_ns(self.time_step_ns.value(), self._time_unit),
            self.voltage_step_mv.value(),
            self.snap_enabled.isChecked(),
            self.visible.isChecked(),
        )


class CrossCapacitanceDialog(QtWidgets.QDialog):
    """Edit the virtual-gate to physical-AWG voltage transform."""

    def __init__(self, output_count: int, matrix, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Virtual-to-physical cross-capacitance matrix")
        self.resize(max(620, 115 * output_count), max(330, 62 * output_count))
        values = np.asarray(matrix, dtype=float)
        expected_shape = (output_count, output_count)
        if values.shape != expected_shape:
            raise ValueError(
                f"cross-capacitance matrix shape must be {expected_shape}"
            )

        layout = QtWidgets.QVBoxLayout(self)
        equation = QtWidgets.QLabel(
            "Vphysical = C x Vvirtual (rows: physical AWGs, columns: virtual gates)",
            self,
        )
        equation.setWordWrap(True)
        layout.addWidget(equation)
        self.table = QtWidgets.QTableWidget(output_count, output_count, self)
        self.table.setHorizontalHeaderLabels(
            [f"virtual awg_{index}" for index in range(output_count)]
        )
        self.table.setVerticalHeaderLabels(
            [f"physical awg_{index}" for index in range(output_count)]
        )
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self._editors = []
        for row in range(output_count):
            editor_row = []
            for column in range(output_count):
                editor = QtWidgets.QDoubleSpinBox(self.table)
                editor.setRange(-10.0, 10.0)
                editor.setDecimals(6)
                editor.setSingleStep(0.01)
                editor.setValue(float(values[row, column]))
                if row == column:
                    editor.setValue(1.0)
                    editor.setReadOnly(True)
                    editor.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
                self.table.setCellWidget(row, column, editor)
                editor_row.append(editor)
            self._editors.append(editor_row)
        layout.addWidget(self.table)

        button_row = QtWidgets.QHBoxLayout()
        reset_button = QtWidgets.QPushButton("Reset identity")
        reset_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton)
        )
        reset_button.clicked.connect(self._reset_identity)
        button_row.addWidget(reset_button)
        button_row.addStretch(1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        button_row.addWidget(buttons)
        layout.addLayout(button_row)

    def _reset_identity(self) -> None:
        for row, editors in enumerate(self._editors):
            for column, editor in enumerate(editors):
                editor.setValue(1.0 if row == column else 0.0)

    def matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [editor.value() for editor in editor_row]
                for editor_row in self._editors
            ],
            dtype=float,
        )


class SweepSettingsDialog(QtWidgets.QDialog):
    """Configure the voltage sweep attached to one SET/output pair."""

    def __init__(
        self,
        *,
        output_name: str,
        segment_name: str,
        current_amplitude: float,
        full_scale_mv: float,
        initial: Optional[QickSweepSpec] = None,
        cartesian_base_count: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Voltage sweep settings")
        self._output_name = output_name
        self._segment_name = segment_name
        self._full_scale_mv = float(full_scale_mv)
        self._cartesian_base_count = int(cartesian_base_count)
        form = QtWidgets.QFormLayout(self)

        form.addRow("Output:", QtWidgets.QLabel(output_name))
        form.addRow("SET segment:", QtWidgets.QLabel(segment_name))
        form.addRow(
            "Current level:",
            QtWidgets.QLabel(
                f"{current_amplitude * self._full_scale_mv:.6g} mV"
            ),
        )

        default_start = max(-1.0, current_amplitude - 0.2) * self._full_scale_mv
        default_stop = min(1.0, current_amplitude + 0.2) * self._full_scale_mv
        if initial is not None:
            default_start = initial.start * self._full_scale_mv
            default_stop = initial.stop * self._full_scale_mv

        self.start = QtWidgets.QDoubleSpinBox()
        self.stop = QtWidgets.QDoubleSpinBox()
        for widget, value in ((self.start, default_start), (self.stop, default_stop)):
            widget.setRange(-self._full_scale_mv, self._full_scale_mv)
            widget.setDecimals(6)
            widget.setSingleStep(max(0.001, self._full_scale_mv / 100.0))
            widget.setSuffix(" mV")
            widget.setValue(value)
        self.count = QtWidgets.QSpinBox()
        self.count.setRange(1, 1_000_000)
        self.count.setValue(initial.count if initial is not None else 9)
        self.endpoint_summary = QtWidgets.QLabel()
        self.cartesian_summary = QtWidgets.QLabel()

        form.addRow("Start voltage:", self.start)
        form.addRow("Stop voltage:", self.stop)
        form.addRow("Sweep point count:", self.count)
        form.addRow("Endpoint levels:", self.endpoint_summary)
        form.addRow("Cartesian total:", self.cartesian_summary)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.start.valueChanged.connect(self._refresh_summary)
        self.stop.valueChanged.connect(self._refresh_summary)
        self.count.valueChanged.connect(self._refresh_summary)
        self._refresh_summary()

    def _refresh_summary(self, *_args) -> None:
        self.endpoint_summary.setText(
            f"{self.start.value():.6g} mV to {self.stop.value():.6g} mV"
        )
        self.cartesian_summary.setText(
            f"{self._cartesian_base_count} x {self.count.value()} = "
            f"{self._cartesian_base_count * self.count.value()} points"
        )

    def value(self) -> QickSweepSpec:
        return QickSweepSpec(
            segment_name=self._segment_name,
            output_name=self._output_name,
            start=self.start.value() / self._full_scale_mv,
            stop=self.stop.value() / self._full_scale_mv,
            count=self.count.value(),
        )


class RampRateSweepSettingsDialog(QtWidgets.QDialog):
    """Configure a RAMP rate by sweeping its duration in microseconds."""

    def __init__(
        self,
        *,
        segment_name: str,
        current_duration_us: float,
        voltage_delta_mv: float,
        initial: Optional[QickRampRateSweepSpec] = None,
        cartesian_base_count: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("RAMP rate sweep settings")
        self._segment_name = str(segment_name)
        self._voltage_delta_mv = float(voltage_delta_mv)
        self._cartesian_base_count = int(cartesian_base_count)
        form = QtWidgets.QFormLayout(self)

        form.addRow("RAMP segment:", QtWidgets.QLabel(self._segment_name))
        form.addRow(
            "Current duration:",
            QtWidgets.QLabel(f"{float(current_duration_us):.9g} us"),
        )
        form.addRow(
            "Nominal voltage change:",
            QtWidgets.QLabel(f"{self._voltage_delta_mv:.9g} mV"),
        )

        default_start = max(float(current_duration_us) * 0.5, 1.0e-6)
        default_stop = max(float(current_duration_us) * 1.5, 2.0e-6)
        if initial is not None:
            default_start = float(initial.start)
            default_stop = float(initial.stop)

        self.start = QtWidgets.QDoubleSpinBox()
        self.stop = QtWidgets.QDoubleSpinBox()
        for widget, value in ((self.start, default_start), (self.stop, default_stop)):
            widget.setRange(1.0e-6, 1.0e9)
            widget.setDecimals(9)
            widget.setSingleStep(max(1.0e-6, float(current_duration_us) / 20.0))
            widget.setSuffix(" us")
            widget.setValue(value)
        self.count = QtWidgets.QSpinBox()
        self.count.setRange(2, 1_000_000)
        self.count.setValue(initial.count if initial is not None else 9)
        self.rate_summary = QtWidgets.QLabel()
        self.rate_summary.setWordWrap(True)
        self.cartesian_summary = QtWidgets.QLabel()

        form.addRow("Start duration:", self.start)
        form.addRow("Stop duration:", self.stop)
        form.addRow("Sweep point count:", self.count)
        form.addRow("Nominal RAMP rates:", self.rate_summary)
        form.addRow("Cartesian total:", self.cartesian_summary)
        self.rate_summary.setToolTip(
            "The following SET remains the RAMP target. Voltage sweeps on "
            "either endpoint automatically update the signed hardware step."
        )

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.start.valueChanged.connect(self._refresh_summary)
        self.stop.valueChanged.connect(self._refresh_summary)
        self.count.valueChanged.connect(self._refresh_summary)
        self._refresh_summary()

    def _refresh_summary(self, *_args) -> None:
        start_rate = self._voltage_delta_mv / self.start.value()
        stop_rate = self._voltage_delta_mv / self.stop.value()
        self.rate_summary.setText(
            f"{start_rate:.9g} to {stop_rate:.9g} mV/us "
            "(nominal endpoint levels)"
        )
        self.cartesian_summary.setText(
            f"{self._cartesian_base_count} x {self.count.value()} = "
            f"{self._cartesian_base_count * self.count.value()} points"
        )

    def value(self) -> QickRampRateSweepSpec:
        return QickRampRateSweepSpec(
            segment_name=self._segment_name,
            start=self.start.value(),
            stop=self.stop.value(),
            count=self.count.value(),
        )


class HoldDurationSweepSettingsDialog(QtWidgets.QDialog):
    """Configure the hold-time sweep attached to one shared SET segment."""

    def __init__(
        self,
        *,
        segment_name: str,
        current_duration_us: float,
        initial: Optional[QickHoldDurationSweepSpec] = None,
        cartesian_base_count: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("SET hold time sweep settings")
        self._segment_name = str(segment_name)
        self._cartesian_base_count = int(cartesian_base_count)
        form = QtWidgets.QFormLayout(self)

        form.addRow("SET segment:", QtWidgets.QLabel(self._segment_name))
        form.addRow(
            "Current hold:",
            QtWidgets.QLabel(f"{float(current_duration_us):.9g} us"),
        )

        default_start = max(float(current_duration_us) * 0.5, 1.0e-6)
        default_stop = max(float(current_duration_us) * 1.5, 2.0e-6)
        if initial is not None:
            default_start = float(initial.start)
            default_stop = float(initial.stop)

        self.start = QtWidgets.QDoubleSpinBox()
        self.stop = QtWidgets.QDoubleSpinBox()
        for widget, value in (
            (self.start, default_start),
            (self.stop, default_stop),
        ):
            widget.setRange(1.0e-6, 1.0e9)
            widget.setDecimals(9)
            widget.setSingleStep(
                max(1.0e-6, float(current_duration_us) / 20.0)
            )
            widget.setSuffix(" us")
            widget.setValue(value)
        self.count = QtWidgets.QSpinBox()
        self.count.setRange(2, 1_000_000)
        self.count.setValue(initial.count if initial is not None else 9)
        self.cartesian_summary = QtWidgets.QLabel()

        form.addRow("Start hold:", self.start)
        form.addRow("Stop hold:", self.stop)
        form.addRow("Sweep point count:", self.count)
        form.addRow("Cartesian total:", self.cartesian_summary)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.count.valueChanged.connect(self._refresh_summary)
        self._refresh_summary()

    def _refresh_summary(self, *_args) -> None:
        self.cartesian_summary.setText(
            f"{self._cartesian_base_count} x {self.count.value()} = "
            f"{self._cartesian_base_count * self.count.value()} points"
        )

    def value(self) -> QickHoldDurationSweepSpec:
        return QickHoldDurationSweepSpec(
            segment_name=self._segment_name,
            start=self.start.value(),
            stop=self.stop.value(),
            count=self.count.value(),
        )


class RfPulseEditorPanel(QtWidgets.QWidget):
    """Main-window editor and waveform preview for one QICK RF pulse."""

    spec_applied = QtCore.pyqtSignal(object)
    spec_removed = QtCore.pyqtSignal()

    def __init__(self, pulse: PulseSequence, parent=None):
        super().__init__(parent)
        self._pulse = pulse
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        controls = QtWidgets.QWidget(splitter)
        form = QtWidgets.QFormLayout(controls)
        self.gen_ch = QtWidgets.QSpinBox()
        self.gen_ch.setRange(0, 255)
        self.segment = QtWidgets.QComboBox()
        self.delay_us = QtWidgets.QDoubleSpinBox()
        self.delay_us.setRange(0.0, 1.0e9)
        self.delay_us.setDecimals(6)
        self.delay_us.setSuffix(" us")
        self.duration_us = QtWidgets.QDoubleSpinBox()
        self.duration_us.setRange(0.001, 1.0e9)
        self.duration_us.setDecimals(6)
        self.duration_us.setValue(0.05)
        self.duration_us.setSuffix(" us")
        self.frequency_mhz = QtWidgets.QDoubleSpinBox()
        self.frequency_mhz.setRange(-10000.0, 10000.0)
        self.frequency_mhz.setDecimals(6)
        self.frequency_mhz.setValue(50.0)
        self.frequency_mhz.setSuffix(" MHz")
        self.gain = QtWidgets.QSpinBox()
        self.gain.setRange(-32768, 32767)
        self.gain.setValue(20000)
        self.att1_db = QtWidgets.QDoubleSpinBox()
        self.att2_db = QtWidgets.QDoubleSpinBox()
        for attenuator in (self.att1_db, self.att2_db):
            attenuator.setRange(0.0, 31.75)
            attenuator.setDecimals(2)
            attenuator.setSingleStep(0.25)
            attenuator.setSuffix(" dB")
        self.filter_type = QtWidgets.QComboBox()
        self.filter_type.addItems(["bypass", "lowpass", "highpass", "bandpass"])
        self.filter_cutoff = QtWidgets.QDoubleSpinBox()
        self.filter_cutoff.setRange(0.0, 100.0)
        self.filter_cutoff.setDecimals(6)
        self.filter_cutoff.setValue(2.5)
        self.filter_cutoff.setSuffix(" GHz")
        self.filter_bandwidth = QtWidgets.QDoubleSpinBox()
        self.filter_bandwidth.setRange(0.001, 100.0)
        self.filter_bandwidth.setDecimals(6)
        self.filter_bandwidth.setValue(1.0)
        self.filter_bandwidth.setSuffix(" GHz")
        self.phase_degrees = QtWidgets.QDoubleSpinBox()
        self.phase_degrees.setRange(-360.0, 360.0)
        self.phase_degrees.setDecimals(6)
        self.phase_degrees.setSuffix(" deg")
        self.nqz = QtWidgets.QSpinBox()
        self.nqz.setRange(1, 2)
        self.nqz.setValue(1)
        self.require_within = QtWidgets.QCheckBox("Require pulse inside selected SET")
        self.require_within.setChecked(True)
        self.absolute_time = QtWidgets.QLabel()

        form.addRow("RF generator index:", self.gen_ch)
        form.addRow("Anchor SET segment:", self.segment)
        form.addRow("Start delay:", self.delay_us)
        form.addRow("Duration:", self.duration_us)
        form.addRow("Frequency:", self.frequency_mhz)
        form.addRow("Gain:", self.gain)
        form.addRow("ATT1:", self.att1_db)
        form.addRow("ATT2:", self.att2_db)
        form.addRow("Output filter:", self.filter_type)
        form.addRow("Filter cutoff/center:", self.filter_cutoff)
        form.addRow("Filter bandwidth:", self.filter_bandwidth)
        form.addRow("Phase:", self.phase_degrees)
        form.addRow("Nyquist zone:", self.nqz)
        form.addRow(self.require_within)
        form.addRow("Absolute timing:", self.absolute_time)

        button_row = QtWidgets.QHBoxLayout()
        apply_button = QtWidgets.QPushButton("Apply RF pulse")
        apply_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton))
        clear_button = QtWidgets.QPushButton("Remove RF pulse")
        clear_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton))
        button_row.addWidget(apply_button)
        button_row.addWidget(clear_button)
        form.addRow(button_row)
        apply_button.clicked.connect(self._apply)
        clear_button.clicked.connect(self._remove)

        if RfPulsePreviewWidget is None:
            self.preview = QtWidgets.QLabel("Install pyqtgraph to preview RF pulses.")
            self.preview.setAlignment(QtCore.Qt.AlignCenter)
        else:
            self.preview = RfPulsePreviewWidget(splitter)
        splitter.addWidget(controls)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.refresh_segments(pulse)
        for widget in (
            self.gen_ch,
            self.segment,
            self.delay_us,
            self.duration_us,
            self.frequency_mhz,
            self.gain,
            self.att1_db,
            self.att2_db,
            self.filter_type,
            self.filter_cutoff,
            self.filter_bandwidth,
            self.phase_degrees,
            self.nqz,
            self.require_within,
        ):
            signal = getattr(widget, "valueChanged", None)
            if signal is None:
                signal = getattr(widget, "currentIndexChanged", None)
            if signal is None:
                signal = getattr(widget, "toggled", None)
            signal.connect(self._preview_current)
        self._preview_current()

    def refresh_segments(self, pulse: Optional[PulseSequence] = None) -> None:
        if pulse is not None:
            self._pulse = pulse
        previous = self.segment.currentData()
        self.segment.blockSignals(True)
        self.segment.clear()
        durations = []
        for index, (start, end) in enumerate(self._pulse.flat_segments()):
            internal_name = f"set_{index}"
            display_name = self._pulse.segment_name(index)
            start_us = float(self._pulse.t[start]) / 1000.0
            end_us = float(self._pulse.t[end]) / 1000.0
            durations.append(end_us - start_us)
            self.segment.addItem(
                f"{display_name}  [{start_us:.6g}, {end_us:.6g}] us",
                internal_name,
            )
        if previous is not None:
            match = self.segment.findData(previous)
            if match >= 0:
                self.segment.setCurrentIndex(match)
        elif durations:
            self.segment.setCurrentIndex(int(np.argmax(durations)))
        self.segment.blockSignals(False)
        self._preview_current()

    def current_spec(self) -> QickRfPulseSpec:
        return QickRfPulseSpec(
            gen_ch=self.gen_ch.value(),
            segment_name=str(self.segment.currentData()),
            delay_us=self.delay_us.value(),
            duration_us=self.duration_us.value(),
            frequency_mhz=self.frequency_mhz.value(),
            gain=self.gain.value(),
            att1_db=self.att1_db.value(),
            att2_db=self.att2_db.value(),
            phase_degrees=self.phase_degrees.value(),
            nqz=self.nqz.value(),
            require_within_segment=self.require_within.isChecked(),
            filter_type=self.filter_type.currentText(),
            filter_cutoff=self.filter_cutoff.value(),
            filter_bandwidth=self.filter_bandwidth.value(),
        )

    def _absolute_times(self, spec: QickRfPulseSpec) -> Tuple[float, float, float]:
        return rf_pulse_absolute_times_us(self._pulse, spec)

    def _preview_current(self, *_args) -> None:
        try:
            spec = self.current_spec()
            start_us, end_us, set_end_us = self._absolute_times(spec)
        except (IndexError, TypeError, ValueError):
            self.absolute_time.setText("Invalid")
            if hasattr(self.preview, "clear_preview"):
                self.preview.clear_preview()
            return
        self.absolute_time.setText(
            f"{start_us:.6g} to {end_us:.6g} us (SET ends {set_end_us:.6g} us)"
        )
        if hasattr(self.preview, "set_pulse"):
            self.preview.set_pulse(
                start_us=start_us,
                duration_us=spec.duration_us,
                frequency_mhz=spec.frequency_mhz,
                gain=spec.gain,
                phase_degrees=spec.phase_degrees,
                att1_db=spec.att1_db,
                att2_db=spec.att2_db,
            )

    def _apply(self) -> None:
        try:
            spec = self.current_spec()
            self._absolute_times(spec)
        except (IndexError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid RF pulse", str(exc))
            return
        self.spec_applied.emit(spec)

    def _remove(self) -> None:
        if hasattr(self.preview, "clear_preview"):
            self.preview.clear_preview()
        self.absolute_time.setText("Disabled")
        self.spec_removed.emit()

    def load_spec(self, spec: Optional[QickRfPulseSpec]) -> None:
        if spec is None:
            self._remove()
            return
        self.gen_ch.setValue(spec.gen_ch)
        segment_index = self.segment.findData(spec.segment_name)
        if segment_index >= 0:
            self.segment.setCurrentIndex(segment_index)
        self.delay_us.setValue(spec.delay_us)
        self.duration_us.setValue(spec.duration_us)
        self.frequency_mhz.setValue(spec.frequency_mhz)
        self.gain.setValue(spec.gain)
        self.att1_db.setValue(spec.att1_db)
        self.att2_db.setValue(spec.att2_db)
        self.filter_type.setCurrentText(spec.filter_type)
        self.filter_cutoff.setValue(spec.filter_cutoff)
        self.filter_bandwidth.setValue(spec.filter_bandwidth)
        self.phase_degrees.setValue(spec.phase_degrees)
        self.nqz.setValue(spec.nqz)
        self.require_within.setChecked(spec.require_within_segment)
        self._preview_current()


class RfPulsePortPanel(QtWidgets.QGroupBox):
    """Compact, always-available editor for one RF output generator."""

    changed = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal(object)
    front_panel_requested = QtCore.pyqtSignal(object)
    lo_frequency_change_requested = QtCore.pyqtSignal(object, float)

    def __init__(
        self,
        pulse: PulseSequence,
        index: int,
        *,
        default_gen_ch: Optional[int] = None,
        time_unit: str = DEFAULT_TIME_UNIT,
        parent=None,
    ):
        super().__init__(parent)
        self._pulse = pulse
        self._index = index
        self._time_unit = time_unit
        self._front_panel_configuration = None
        self._qcs_front_panel_configuration = None
        self._hardware_backend = EXECUTION_BACKEND_QCS
        self.setCheckable(True)
        self.setChecked(False)
        form = QtWidgets.QFormLayout(self)
        self._form_layout = form

        self.gen_ch = QtWidgets.QSpinBox()
        self.gen_ch.setRange(0, 255)
        self.gen_ch.setValue(index if default_gen_ch is None else default_gen_ch)
        self.output_board_type = QtWidgets.QComboBox()
        self.output_board_type.addItems(QICK_OUTPUT_BOARD_TYPES)
        self.qcs_module_model = QtWidgets.QComboBox()
        self.qcs_module_model.addItem("M5300A RF AWG", "M5300A")
        self.qcs_module_model.addItem("M5301A Precision AWG", "M5301A")
        self.qcs_module_model.setPlaceholderText("Not mapped")
        self.qcs_module_model.setEnabled(False)
        self.qcs_module_model.setToolTip(
            "The QCS module is determined by the front-panel mapping. Click "
            "the front-panel preview to select a different module or SMA output."
        )
        self.qcs_lo_frequency_ghz = QtWidgets.QDoubleSpinBox()
        self.qcs_lo_frequency_ghz.setRange(0.0, 18.0)
        self.qcs_lo_frequency_ghz.setDecimals(9)
        self.qcs_lo_frequency_ghz.setSingleStep(0.01)
        self.qcs_lo_frequency_ghz.setSuffix(" GHz")
        self.qcs_lo_frequency_ghz.setToolTip(
            "Physical M5300 local-oscillator frequency. This is separate "
            "from the RF waveform frequency below."
        )
        self.apply_qcs_lo_frequency = QtWidgets.QPushButton("Apply LO")
        self.apply_qcs_lo_frequency.setToolTip(
            "Save this LO frequency to the shared QCS ChannelMapper"
        )
        self.qcs_lo_frequency_status = QtWidgets.QLabel("LO not configured")
        self.qcs_lo_frequency_status.setStyleSheet(
            "QLabel { color: #6a737d; }"
        )
        self.qcs_lo_frequency_status.setWordWrap(True)
        self.qcs_lo_frequency_row = QtWidgets.QWidget()
        qcs_lo_layout = QtWidgets.QVBoxLayout(self.qcs_lo_frequency_row)
        qcs_lo_layout.setContentsMargins(0, 0, 0, 0)
        qcs_lo_layout.setSpacing(2)
        qcs_lo_editor_layout = QtWidgets.QHBoxLayout()
        qcs_lo_editor_layout.setContentsMargins(0, 0, 0, 0)
        qcs_lo_editor_layout.addWidget(self.qcs_lo_frequency_ghz, 1)
        qcs_lo_editor_layout.addWidget(self.apply_qcs_lo_frequency)
        qcs_lo_layout.addLayout(qcs_lo_editor_layout)
        qcs_lo_layout.addWidget(self.qcs_lo_frequency_status)
        self.segment = QtWidgets.QComboBox()
        self.delay = QtWidgets.QDoubleSpinBox()
        self.duration = QtWidgets.QDoubleSpinBox()
        for editor in (self.delay, self.duration):
            editor.setRange(0.0, 1.0e12)
            editor.setDecimals(9)
        self.duration.setMinimum(1.0e-9)
        self.duration.setValue(_time_from_ns(1000.0, time_unit))
        self.duration_sweep_enabled = QtWidgets.QCheckBox(
            "Sweep RF pulse duration"
        )
        self.duration_sweep_start = QtWidgets.QDoubleSpinBox()
        self.duration_sweep_stop = QtWidgets.QDoubleSpinBox()
        for editor in (
            self.duration_sweep_start,
            self.duration_sweep_stop,
        ):
            editor.setRange(1.0e-9, 1.0e12)
            editor.setDecimals(9)
            editor.setValue(_time_from_ns(1000.0, time_unit))
        self.duration_sweep_count = QtWidgets.QSpinBox()
        self.duration_sweep_count.setRange(1, 1_000_000)
        self.duration_sweep_count.setValue(1)
        self.segment_length_mode = QtWidgets.QComboBox()
        self.segment_length_mode.addItem(
            "Keep AWG segment length fixed",
            "fixed",
        )
        self.segment_length_mode.addItem(
            "Original AWG segment + RF duration",
            "extend_by_rf_duration",
        )
        self.frequency_mhz = QtWidgets.QDoubleSpinBox()
        self.frequency_mhz.setRange(-10000.0, 10000.0)
        self.frequency_mhz.setDecimals(6)
        self.frequency_mhz.setValue(50.0)
        self.frequency_mhz.setSuffix(" MHz")
        self.frequency_sweep_enabled = QtWidgets.QCheckBox(
            "Sweep RF frequency"
        )
        self.frequency_sweep_start_mhz = QtWidgets.QDoubleSpinBox()
        self.frequency_sweep_stop_mhz = QtWidgets.QDoubleSpinBox()
        for editor in (
            self.frequency_sweep_start_mhz,
            self.frequency_sweep_stop_mhz,
        ):
            editor.setRange(-10000.0, 10000.0)
            editor.setDecimals(6)
            editor.setValue(50.0)
            editor.setSuffix(" MHz")
        self.frequency_sweep_count = QtWidgets.QSpinBox()
        self.frequency_sweep_count.setRange(1, 1_000_000)
        self.frequency_sweep_count.setValue(1)
        self.gain = QtWidgets.QSpinBox()
        self.gain.setRange(-32768, 32767)
        self.gain.setValue(20000)
        self.qcs_amplitude = QtWidgets.QDoubleSpinBox()
        self.qcs_amplitude.setRange(-1.0, 1.0)
        self.qcs_amplitude.setDecimals(9)
        self.qcs_amplitude.setSingleStep(0.01)
        self.qcs_amplitude.setValue(self._gain_to_qcs_amplitude(self.gain.value()))
        self.qcs_amplitude.setToolTip(
            "QCS RF waveform amplitude relative to the configured "
            "signal-generator output range"
        )
        self.qcs_power_calibration_group = QtWidgets.QGroupBox(
            "M5300A 50 Ohm output-power calibration"
        )
        self.qcs_power_calibration_group.setCheckable(True)
        self.qcs_power_calibration_group.setChecked(False)
        qcs_power_calibration_form = QtWidgets.QFormLayout(
            self.qcs_power_calibration_group
        )
        self.qcs_power_calibration_database_path = QtWidgets.QLineEdit(
            DEFAULT_POWER_CALIBRATION_DB_PATH
        )
        self.qcs_power_calibration_database_path.setPlaceholderText(
            "Select the M5300A/M5200A calibration database"
        )
        self.browse_qcs_power_calibration = QtWidgets.QToolButton()
        self.browse_qcs_power_calibration.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        self.browse_qcs_power_calibration.setToolTip(
            "Choose the dedicated QCS M5300A/M5200A RF-power "
            "calibration database"
        )
        qcs_power_database_row = QtWidgets.QHBoxLayout()
        qcs_power_database_row.addWidget(
            self.qcs_power_calibration_database_path,
            1,
        )
        qcs_power_database_row.addWidget(
            self.browse_qcs_power_calibration
        )
        self.qcs_power_calibration_run_id = QtWidgets.QSpinBox()
        self.qcs_power_calibration_run_id.setRange(0, 2_147_483_647)
        self.qcs_power_calibration_run_id.setSpecialValueText(
            "Latest compatible"
        )
        self.qcs_power_calibration_run_id.setToolTip(
            "Calibration record ID 0 selects the newest calibration matching "
            "the active M5300 SMA connector, LO, and RF frequency. This is "
            "separate from the QCoDeS Run ID shown in Plottr Inspectr."
        )
        self.qcs_target_output_power_dbm = QtWidgets.QDoubleSpinBox()
        self.qcs_target_output_power_dbm.setRange(-200.0, 100.0)
        self.qcs_target_output_power_dbm.setDecimals(6)
        self.qcs_target_output_power_dbm.setSuffix(" dBm")
        self.qcs_target_output_power_dbm.setValue(-20.0)
        self.qcs_power_calibration_status = QtWidgets.QLabel(
            "Enable calibration to choose connector power in dBm. The "
            "matching relative amplitude is resolved from the active "
            "ChannelMapper when Run QCS starts."
        )
        self.qcs_power_calibration_status.setWordWrap(True)
        self.qcs_power_calibration_status.setStyleSheet(
            "QLabel { color: #4f5b66; }"
        )
        qcs_power_calibration_form.addRow(
            "Calibration DB:",
            qcs_power_database_row,
        )
        qcs_power_calibration_form.addRow(
            "Calibration record ID:",
            self.qcs_power_calibration_run_id,
        )
        qcs_power_calibration_form.addRow(
            "Target M5300 connector power:",
            self.qcs_target_output_power_dbm,
        )
        qcs_power_calibration_form.addRow(
            "Status:",
            self.qcs_power_calibration_status,
        )
        self._power_calibration_resolved_signature = None
        self.power_calibration_group = QtWidgets.QGroupBox(
            "Calibrated output power"
        )
        self.power_calibration_group.setCheckable(True)
        self.power_calibration_group.setChecked(False)
        power_calibration_form = QtWidgets.QFormLayout(
            self.power_calibration_group
        )
        self.power_calibration_database_path = QtWidgets.QLineEdit(
            DEFAULT_POWER_CALIBRATION_DB_PATH
        )
        self.power_calibration_database_path.setPlaceholderText(
            "Select gain_pwr_calb.db"
        )
        browse_power_calibration = QtWidgets.QToolButton()
        browse_power_calibration.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        browse_power_calibration.setToolTip(
            "Choose an RF output power calibration database"
        )
        browse_power_calibration.clicked.connect(
            self._browse_power_calibration_database
        )
        power_calibration_database_row = QtWidgets.QHBoxLayout()
        power_calibration_database_row.addWidget(
            self.power_calibration_database_path,
            1,
        )
        power_calibration_database_row.addWidget(browse_power_calibration)
        self.power_calibration_run_id = QtWidgets.QSpinBox()
        self.power_calibration_run_id.setRange(0, 2_147_483_647)
        self.power_calibration_run_id.setSpecialValueText(
            "Latest compatible"
        )
        self.power_calibration_run = QtWidgets.QComboBox()
        self.power_calibration_run.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.power_calibration_run.setMinimumContentsLength(38)
        self.power_calibration_run.addItem(
            "Auto: best exact PCB / filter / Nyquist match",
            0,
        )
        self.refresh_power_calibration_runs = QtWidgets.QPushButton(
            "Refresh Runs"
        )
        self.refresh_power_calibration_runs.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
        )
        self.refresh_power_calibration_runs.setToolTip(
            "Inspect calibration runs and rank them against the current "
            "PCB, frequency, filter, and Nyquist settings"
        )
        power_calibration_run_row = QtWidgets.QHBoxLayout()
        power_calibration_run_row.addWidget(
            self.power_calibration_run,
            1,
        )
        power_calibration_run_row.addWidget(
            self.refresh_power_calibration_runs
        )
        self._power_calibration_candidates = ()
        self.target_output_power_dbm = QtWidgets.QDoubleSpinBox()
        self.target_output_power_dbm.setRange(-200.0, 100.0)
        self.target_output_power_dbm.setDecimals(6)
        self.target_output_power_dbm.setSuffix(" dBm")
        self.target_output_power_dbm.setValue(-20.0)
        self.power_sweep_enabled = QtWidgets.QCheckBox(
            "Sweep calibrated output power"
        )
        self.power_sweep_start_dbm = QtWidgets.QDoubleSpinBox()
        self.power_sweep_stop_dbm = QtWidgets.QDoubleSpinBox()
        for editor in (
            self.power_sweep_start_dbm,
            self.power_sweep_stop_dbm,
        ):
            editor.setRange(-200.0, 100.0)
            editor.setDecimals(6)
            editor.setValue(-20.0)
            editor.setSuffix(" dBm")
        self.power_sweep_count = QtWidgets.QSpinBox()
        self.power_sweep_count.setRange(1, 1_000_000)
        self.power_sweep_count.setValue(1)
        self.apply_power_calibration = QtWidgets.QPushButton(
            "Apply calibrated gain"
        )
        self.apply_power_calibration.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        )
        self.power_calibration_status = QtWidgets.QLabel(
            "Select a calibration database and target connector power."
        )
        self.power_calibration_status.setWordWrap(True)
        power_calibration_form.addRow(
            "Calibration DB:",
            power_calibration_database_row,
        )
        power_calibration_form.addRow(
            "Calibration run:",
            power_calibration_run_row,
        )
        power_calibration_form.addRow(
            "Target output power:",
            self.target_output_power_dbm,
        )
        power_calibration_form.addRow(self.power_sweep_enabled)
        power_calibration_form.addRow(
            "Power sweep start:",
            self.power_sweep_start_dbm,
        )
        power_calibration_form.addRow(
            "Power sweep stop:",
            self.power_sweep_stop_dbm,
        )
        power_calibration_form.addRow(
            "Power sweep points:",
            self.power_sweep_count,
        )
        power_calibration_form.addRow(self.apply_power_calibration)
        power_calibration_form.addRow("Status:", self.power_calibration_status)
        self.att1_db = QtWidgets.QDoubleSpinBox()
        self.att2_db = QtWidgets.QDoubleSpinBox()
        for attenuator in (self.att1_db, self.att2_db):
            attenuator.setRange(0.0, 31.75)
            attenuator.setDecimals(2)
            attenuator.setSingleStep(0.25)
            attenuator.setSuffix(" dB")
        self.filter_type = QtWidgets.QComboBox()
        self.filter_type.addItems(["bypass", "lowpass", "highpass", "bandpass"])
        self.filter_cutoff = QtWidgets.QDoubleSpinBox()
        self.filter_cutoff.setRange(0.0, 100.0)
        self.filter_cutoff.setDecimals(6)
        self.filter_cutoff.setValue(2.5)
        self.filter_cutoff.setSuffix(" GHz")
        self.filter_bandwidth = QtWidgets.QDoubleSpinBox()
        self.filter_bandwidth.setRange(0.001, 100.0)
        self.filter_bandwidth.setDecimals(6)
        self.filter_bandwidth.setValue(1.0)
        self.filter_bandwidth.setSuffix(" GHz")
        self.phase_degrees = QtWidgets.QDoubleSpinBox()
        self.phase_degrees.setRange(-360.0, 360.0)
        self.phase_degrees.setDecimals(6)
        self.phase_degrees.setSuffix(" deg")
        self.nqz = QtWidgets.QSpinBox()
        self.nqz.setRange(1, 2)
        self.nqz.setValue(1)
        self.require_within = QtWidgets.QCheckBox("Keep pulse inside anchor SET")
        self.require_within.setChecked(True)

        self.front_panel_preview = HardwareFrontPanelPreview(self)
        self.front_panel_preview.set_qcs_selection(
            "rf",
            self.gen_ch.value(),
        )
        self.front_panel_preview.activated.connect(
            lambda: self.front_panel_requested.emit(self)
        )
        front_panel_row = QtWidgets.QVBoxLayout()
        front_panel_row.setContentsMargins(0, 0, 0, 0)
        front_panel_row.setSpacing(2)
        front_panel_row.addWidget(QtWidgets.QLabel("Front panel:"))
        front_panel_row.addWidget(self.front_panel_preview)
        form.addRow(front_panel_row)
        # Selecting a physical RF connector is configuration, not execution.
        # Keep the picker usable before this optional RF output is enabled;
        # an unchecked QGroupBox otherwise disables all of its children.
        self.front_panel_preview.setEnabled(True)
        self.toggled.connect(self._keep_front_panel_preview_enabled)
        QtCore.QTimer.singleShot(
            0,
            self._keep_front_panel_preview_enabled,
        )
        self.gen_ch_label = QtWidgets.QLabel("QCS RF logical channel:")
        form.addRow(self.gen_ch_label, self.gen_ch)
        self.output_board_label = QtWidgets.QLabel("Output board:")
        self.qcs_module_label = QtWidgets.QLabel("Module:")
        form.addRow(self.output_board_label, self.output_board_type)
        form.addRow(self.qcs_module_label, self.qcs_module_model)
        form.addRow("M5300 LO frequency:", self.qcs_lo_frequency_row)
        self.segment_label = QtWidgets.QLabel("Anchor SET:")
        form.addRow(self.segment_label, self.segment)
        self._delay_label = QtWidgets.QLabel()
        self._duration_label = QtWidgets.QLabel()
        form.addRow(self._delay_label, self.delay)
        form.addRow(self._duration_label, self.duration)
        form.addRow(self.duration_sweep_enabled)
        self._duration_sweep_start_label = QtWidgets.QLabel()
        self._duration_sweep_stop_label = QtWidgets.QLabel()
        form.addRow(
            self._duration_sweep_start_label,
            self.duration_sweep_start,
        )
        form.addRow(
            self._duration_sweep_stop_label,
            self.duration_sweep_stop,
        )
        self.duration_sweep_count_label = QtWidgets.QLabel(
            "Duration sweep points:"
        )
        form.addRow(self.duration_sweep_count_label, self.duration_sweep_count)
        self.segment_length_mode_label = QtWidgets.QLabel(
            "AWG segment timing:"
        )
        form.addRow(self.segment_length_mode_label, self.segment_length_mode)
        self.frequency_label = QtWidgets.QLabel("Frequency:")
        form.addRow(self.frequency_label, self.frequency_mhz)
        form.addRow(self.frequency_sweep_enabled)
        self.frequency_sweep_start_label = QtWidgets.QLabel(
            "Frequency sweep start:"
        )
        self.frequency_sweep_stop_label = QtWidgets.QLabel(
            "Frequency sweep stop:"
        )
        self.frequency_sweep_count_label = QtWidgets.QLabel(
            "Frequency sweep points:"
        )
        form.addRow(
            self.frequency_sweep_start_label,
            self.frequency_sweep_start_mhz,
        )
        form.addRow(
            self.frequency_sweep_stop_label,
            self.frequency_sweep_stop_mhz,
        )
        form.addRow(
            self.frequency_sweep_count_label,
            self.frequency_sweep_count,
        )
        self.gain_label = QtWidgets.QLabel("Gain:")
        self.qcs_amplitude_label = QtWidgets.QLabel("Relative amplitude:")
        form.addRow(self.gain_label, self.gain)
        form.addRow(self.qcs_amplitude_label, self.qcs_amplitude)
        form.addRow(self.qcs_power_calibration_group)
        form.addRow(self.power_calibration_group)
        self.shared_path_note = QtWidgets.QLabel(
            "ATT and filter settings are edited from the HWH-backed Front Panel."
        )
        self.shared_path_note.setWordWrap(True)
        self.shared_path_note.setStyleSheet("QLabel { color: #4f5b66; }")
        form.addRow(self.shared_path_note)
        self.qcs_path_note = QtWidgets.QLabel(
            "Module and SMA connector follow the QCS front-panel mapping. "
            "Use Relative amplitude directly, or enable M5300A calibration "
            "to select connector power in dBm."
        )
        self.qcs_path_note.setWordWrap(True)
        self.qcs_path_note.setStyleSheet("QLabel { color: #4f5b66; }")
        form.addRow(self.qcs_path_note)
        self.phase_label = QtWidgets.QLabel("Phase:")
        form.addRow(self.phase_label, self.phase_degrees)
        self.nqz_label = QtWidgets.QLabel("Nyquist zone:")
        form.addRow(self.nqz_label, self.nqz)
        form.addRow(self.require_within)
        self.remove_button = QtWidgets.QPushButton("Remove RF Port")
        self.remove_button.clicked.connect(
            lambda: self.remove_requested.emit(self)
        )
        form.addRow(self.remove_button)

        self.refresh_segments(pulse)
        self.set_time_unit(time_unit, force=True)
        for widget in (
            self.gen_ch,
            self.output_board_type,
            self.segment,
            self.delay,
            self.duration,
            self.duration_sweep_enabled,
            self.duration_sweep_start,
            self.duration_sweep_stop,
            self.duration_sweep_count,
            self.segment_length_mode,
            self.frequency_mhz,
            self.frequency_sweep_enabled,
            self.frequency_sweep_start_mhz,
            self.frequency_sweep_stop_mhz,
            self.frequency_sweep_count,
            self.gain,
            self.att1_db,
            self.att2_db,
            self.filter_type,
            self.filter_cutoff,
            self.filter_bandwidth,
            self.phase_degrees,
            self.nqz,
            self.require_within,
            self.power_sweep_enabled,
            self.power_sweep_start_dbm,
            self.power_sweep_stop_dbm,
            self.power_sweep_count,
        ):
            signal = getattr(widget, "valueChanged", None)
            if signal is None:
                signal = getattr(widget, "currentIndexChanged", None)
            if signal is None:
                signal = getattr(widget, "toggled", None)
            signal.connect(self.changed.emit)
        self.qcs_amplitude.valueChanged.connect(
            self._apply_qcs_amplitude_to_gain
        )
        self.browse_qcs_power_calibration.clicked.connect(
            self._browse_qcs_power_calibration_database
        )
        self.qcs_power_calibration_group.toggled.connect(
            self._update_qcs_power_calibration_controls
        )
        self.qcs_power_calibration_group.toggled.connect(self.changed.emit)
        self.qcs_power_calibration_database_path.textChanged.connect(
            self.changed.emit
        )
        self.qcs_power_calibration_run_id.valueChanged.connect(
            self.changed.emit
        )
        self.qcs_target_output_power_dbm.valueChanged.connect(
            self.changed.emit
        )
        self.gain.valueChanged.connect(self._sync_qcs_amplitude_from_gain)
        self.toggled.connect(self.changed.emit)
        self.output_board_type.currentTextChanged.connect(
            self._update_board_controls
        )
        self.gen_ch.valueChanged.connect(self._sync_front_panel_selection)
        self.apply_qcs_lo_frequency.clicked.connect(
            self._request_qcs_lo_frequency_change
        )
        self.duration_sweep_enabled.toggled.connect(
            self._update_duration_sweep_controls
        )
        self.frequency_sweep_enabled.toggled.connect(
            self._update_frequency_sweep_controls
        )
        self.power_sweep_enabled.toggled.connect(
            self._update_power_sweep_controls
        )
        self.apply_power_calibration.clicked.connect(
            self._apply_calibrated_output_power
        )
        self.refresh_power_calibration_runs.clicked.connect(
            self._refresh_power_calibration_runs
        )
        self.power_calibration_run.currentIndexChanged.connect(
            self._power_calibration_run_changed
        )
        self.power_calibration_group.toggled.connect(
            self._update_power_calibration_controls
        )
        for calibration_input in (
            self.power_calibration_database_path,
            self.power_calibration_run_id,
            self.target_output_power_dbm,
            self.frequency_mhz,
            self.frequency_sweep_enabled,
            self.frequency_sweep_start_mhz,
            self.frequency_sweep_stop_mhz,
            self.frequency_sweep_count,
            self.power_sweep_enabled,
            self.power_sweep_start_dbm,
            self.power_sweep_stop_dbm,
            self.power_sweep_count,
            self.gain,
            self.att1_db,
            self.att2_db,
            self.filter_type,
            self.filter_cutoff,
            self.filter_bandwidth,
            self.nqz,
        ):
            signal = getattr(calibration_input, "textChanged", None)
            if signal is None:
                signal = getattr(calibration_input, "valueChanged", None)
            if signal is None:
                signal = getattr(
                    calibration_input,
                    "currentTextChanged",
                    None,
                )
            if signal is None:
                signal = getattr(calibration_input, "toggled", None)
            signal.connect(self._mark_power_calibration_stale)
        self._update_board_controls()
        self._update_duration_sweep_controls()
        self._update_frequency_sweep_controls()
        self._update_power_calibration_controls()
        self._update_power_sweep_controls()
        self._update_qcs_power_calibration_controls()
        self.set_hardware_backend(self._hardware_backend)
        self.set_index(index)

    def _request_qcs_lo_frequency_change(self) -> None:
        """Request one persisted physical M5300 LO update from MainWindow."""

        if (
            self._hardware_backend != EXECUTION_BACKEND_QCS
            or self.qcs_module_model.currentData() != "M5300A"
        ):
            return
        self.lo_frequency_change_requested.emit(
            self,
            float(self.qcs_lo_frequency_ghz.value()) * 1.0e9,
        )

    @staticmethod
    def _gain_to_qcs_amplitude(gain: int) -> float:
        gain = int(gain)
        return float(gain) / (32767.0 if gain >= 0 else 32768.0)

    @staticmethod
    def _qcs_amplitude_to_gain(amplitude: float) -> int:
        amplitude = float(np.clip(amplitude, -1.0, 1.0))
        scale = 32767.0 if amplitude >= 0.0 else 32768.0
        return int(np.clip(round(amplitude * scale), -32768, 32767))

    def _sync_qcs_amplitude_from_gain(self, *_args) -> None:
        with QtCore.QSignalBlocker(self.qcs_amplitude):
            self.qcs_amplitude.setValue(
                self._gain_to_qcs_amplitude(self.gain.value())
            )

    def _apply_qcs_amplitude_to_gain(self, *_args) -> None:
        gain = self._qcs_amplitude_to_gain(self.qcs_amplitude.value())
        with QtCore.QSignalBlocker(self.gain):
            self.gain.setValue(gain)
        with QtCore.QSignalBlocker(self.qcs_amplitude):
            self.qcs_amplitude.setValue(
                self._gain_to_qcs_amplitude(gain)
            )
        self._mark_power_calibration_stale()
        self.changed.emit()

    def _set_form_row_visible(
        self,
        field: QtWidgets.QWidget,
        visible: bool,
    ) -> None:
        """Show or hide both halves of one QFormLayout row."""

        label = self._form_layout.labelForField(field)
        if label is not None:
            label.setVisible(bool(visible))
        field.setVisible(bool(visible))

    def _update_time_labels(self) -> None:
        is_qcs = self._hardware_backend == EXECUTION_BACKEND_QCS
        self._delay_label.setText(
            (
                "Pre-delay"
                if is_qcs
                else "Delay"
            )
            + f" [{self._time_unit}]:"
        )
        self._duration_label.setText(
            (
                "Waveform duration"
                if is_qcs
                else "Duration"
            )
            + f" [{self._time_unit}]:"
        )
        sweep_prefix = (
            "Waveform duration sweep"
            if is_qcs
            else "Sweep"
        )
        self._duration_sweep_start_label.setText(
            f"{sweep_prefix} start [{self._time_unit}]:"
        )
        self._duration_sweep_stop_label.setText(
            f"{sweep_prefix} stop [{self._time_unit}]:"
        )

    def _update_backend_presentation(self) -> None:
        is_qcs = self._hardware_backend == EXECUTION_BACKEND_QCS
        self.gen_ch_label.setText(
            "RF virtual channel index:" if is_qcs else "Generator index:"
        )
        self.gen_ch.setEnabled(not is_qcs)
        self._set_form_row_visible(self.output_board_type, not is_qcs)
        self._set_form_row_visible(self.qcs_module_model, is_qcs)
        self._set_form_row_visible(self.qcs_lo_frequency_row, False)
        self._set_form_row_visible(self.gain, not is_qcs)
        self._set_form_row_visible(self.qcs_amplitude, is_qcs)
        self._set_form_row_visible(
            self.qcs_power_calibration_group,
            is_qcs,
        )
        self._set_form_row_visible(self.power_calibration_group, not is_qcs)
        self._set_form_row_visible(self.shared_path_note, not is_qcs)
        self._set_form_row_visible(self.qcs_path_note, is_qcs)
        self._set_form_row_visible(self.nqz, not is_qcs)
        self.segment_label.setText(
            "Waveform segment:" if is_qcs else "Anchor SET:"
        )
        self.duration_sweep_enabled.setText(
            "Sweep waveform duration"
            if is_qcs
            else "Sweep RF pulse duration"
        )
        self.duration_sweep_count_label.setText(
            "Waveform duration sweep points:"
            if is_qcs
            else "Duration sweep points:"
        )
        self.segment_length_mode_label.setText(
            "Segment duration behavior:"
            if is_qcs
            else "AWG segment timing:"
        )
        current_mode = self.segment_length_mode.currentData()
        with QtCore.QSignalBlocker(self.segment_length_mode):
            self.segment_length_mode.setItemText(
                0,
                (
                    "Keep segment duration fixed"
                    if is_qcs
                    else "Keep AWG segment length fixed"
                ),
            )
            self.segment_length_mode.setItemText(
                1,
                (
                    "Extend segment by RF waveform duration"
                    if is_qcs
                    else "Original AWG segment + RF duration"
                ),
            )
            mode_index = self.segment_length_mode.findData(current_mode)
            if mode_index >= 0:
                self.segment_length_mode.setCurrentIndex(mode_index)
        self.frequency_label.setText(
            "RF frequency:" if is_qcs else "Frequency:"
        )
        self.frequency_sweep_start_label.setText(
            "RF frequency sweep start:"
            if is_qcs
            else "Frequency sweep start:"
        )
        self.frequency_sweep_stop_label.setText(
            "RF frequency sweep stop:"
            if is_qcs
            else "Frequency sweep stop:"
        )
        self.frequency_sweep_count_label.setText(
            "RF frequency sweep points:"
            if is_qcs
            else "Frequency sweep points:"
        )
        self.phase_label.setText(
            "Instantaneous phase:" if is_qcs else "Phase:"
        )
        self.require_within.setText(
            "Keep waveform inside selected segment"
            if is_qcs
            else "Keep pulse inside anchor SET"
        )
        self.remove_button.setText(
            "Remove RF Output" if is_qcs else "Remove RF Port"
        )
        self._sync_qcs_amplitude_from_gain()
        self._sync_qcs_module_model()
        self._update_qcs_power_calibration_controls()
        self._update_time_labels()

    def _update_duration_sweep_controls(self, *_args) -> None:
        enabled = self.duration_sweep_enabled.isChecked()
        for widget in (
            self.duration_sweep_start,
            self.duration_sweep_stop,
            self.duration_sweep_count,
            self.segment_length_mode,
        ):
            widget.setEnabled(enabled)

    def _update_frequency_sweep_controls(self, *_args) -> None:
        enabled = self.frequency_sweep_enabled.isChecked()
        for widget in (
            self.frequency_sweep_start_mhz,
            self.frequency_sweep_stop_mhz,
            self.frequency_sweep_count,
        ):
            widget.setEnabled(enabled)

    def _update_power_sweep_controls(self, *_args) -> None:
        enabled = (
            self.power_calibration_group.isChecked()
            and self.power_sweep_enabled.isChecked()
            and self.output_board_type.currentText() == "RF_Out"
        )
        for widget in (
            self.power_sweep_start_dbm,
            self.power_sweep_stop_dbm,
            self.power_sweep_count,
        ):
            widget.setEnabled(enabled)

    def _set_power_calibration_status(
        self,
        message: str,
        *,
        state: str = "neutral",
    ) -> None:
        colors = {
            "neutral": "#4f5b66",
            "stale": "#9a6700",
            "success": "#1a7f37",
            "error": "#cf222e",
        }
        self.power_calibration_status.setText(str(message))
        self.power_calibration_status.setStyleSheet(
            f"QLabel {{ color: {colors[state]}; }}"
        )

    def _update_power_calibration_controls(self, *_args) -> None:
        rf_output = self.output_board_type.currentText() == "RF_Out"
        self.power_calibration_group.setEnabled(rf_output)
        self.apply_power_calibration.setEnabled(
            rf_output
            and self.power_calibration_group.isChecked()
            and bool(self.power_calibration_database_path.text().strip())
        )
        calibration_enabled = (
            rf_output and self.power_calibration_group.isChecked()
        )
        self.power_sweep_enabled.setEnabled(calibration_enabled)
        self.power_calibration_run.setEnabled(calibration_enabled)
        self.refresh_power_calibration_runs.setEnabled(
            calibration_enabled
            and bool(self.power_calibration_database_path.text().strip())
        )
        if not rf_output:
            self._set_power_calibration_status(
                "Calibrated power selection requires an RF_Out board.",
                state="error",
            )
        elif not self.power_calibration_group.isChecked():
            self._set_power_calibration_status(
                "Enable this group to select output power from calibration.",
            )
        self._update_power_sweep_controls()

    def _mark_power_calibration_stale(self, *_args) -> None:
        self._power_calibration_resolved_signature = None
        self._update_power_calibration_controls()
        if (
            self.output_board_type.currentText() == "RF_Out"
            and self.power_calibration_group.isChecked()
        ):
            self._set_power_calibration_status(
                "Calibration inputs changed. Apply the calibrated gain again.",
                state="stale",
            )

    def _update_qcs_power_calibration_controls(self, *_args) -> None:
        is_qcs = self._hardware_backend == EXECUTION_BACKEND_QCS
        is_m5300 = is_qcs and self.qcs_module_model.currentData() == "M5300A"
        self.qcs_power_calibration_group.setEnabled(is_m5300)
        calibrated = bool(
            is_m5300 and self.qcs_power_calibration_group.isChecked()
        )
        self.qcs_amplitude.setEnabled(is_qcs and not calibrated)
        if not is_m5300:
            self.qcs_power_calibration_status.setText(
                "Calibrated connector power requires this RF output to be "
                "mapped to an M5300A SMA."
            )
            self.qcs_power_calibration_status.setStyleSheet(
                "QLabel { color: #cf222e; }"
            )
        elif calibrated:
            self.qcs_power_calibration_status.setText(
                "The relative amplitude will be resolved at Run QCS from "
                "the exact M5300 connector, LO, RF frequency, and requested "
                "dBm. The calibration M5200 and mapper are provenance only."
            )
            self.qcs_power_calibration_status.setStyleSheet(
                "QLabel { color: #1a7f37; }"
            )
        else:
            self.qcs_power_calibration_status.setText(
                "Enable calibration to choose M5300 connector power in dBm; "
                "otherwise Relative amplitude is used directly."
            )
            self.qcs_power_calibration_status.setStyleSheet(
                "QLabel { color: #4f5b66; }"
            )

    def _browse_qcs_power_calibration_database(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose QCS M5300A output-power calibration database",
            self.qcs_power_calibration_database_path.text().strip()
            or DEFAULT_POWER_CALIBRATION_DB_PATH,
            "SQLite calibration database (*.db)",
        )
        if path:
            self.qcs_power_calibration_database_path.setText(path)

    def qcs_power_calibration_config(
        self,
    ) -> Optional[QcsRfPowerCalibrationConfig]:
        """Return the fixed-power request for this mapped QCS RF output."""

        if not self.qcs_power_calibration_group.isChecked():
            return None
        if self.qcs_module_model.currentData() != "M5300A":
            raise ValueError(
                f"RF Output {self._index + 1} calibrated QCS power requires "
                "an M5300A mapping"
            )
        if self.frequency_sweep_enabled.isChecked():
            raise ValueError(
                f"RF Output {self._index + 1} calibrated QCS power currently "
                "requires one fixed RF frequency; disable Sweep RF frequency"
            )
        database_path = self.qcs_power_calibration_database_path.text().strip()
        if not database_path:
            raise ValueError(
                f"RF Output {self._index + 1} QCS power-calibration database "
                "is missing"
            )
        if not Path(database_path).expanduser().is_file():
            raise ValueError(
                f"RF Output {self._index + 1} QCS power-calibration database "
                f"was not found: {database_path}"
            )
        return QcsRfPowerCalibrationConfig(
            database_path=database_path,
            run_id=self.qcs_power_calibration_run_id.value(),
            target_power_dbm=self.qcs_target_output_power_dbm.value(),
        )

    def _browse_power_calibration_database(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose RF output power calibration database",
            self.power_calibration_database_path.text().strip()
            or DEFAULT_POWER_CALIBRATION_DB_PATH,
            "QCoDeS SQLite database (*.db)",
        )
        if path:
            self.power_calibration_database_path.setText(path)
            self._refresh_power_calibration_runs()

    def _power_calibration_run_changed(self, *_args) -> None:
        run_id = int(self.power_calibration_run.currentData() or 0)
        with QtCore.QSignalBlocker(self.power_calibration_run_id):
            self.power_calibration_run_id.setValue(run_id)
        self._mark_power_calibration_stale()

    def _set_power_calibration_run_id(self, run_id: int) -> None:
        run_id = int(run_id)
        with QtCore.QSignalBlocker(self.power_calibration_run_id):
            self.power_calibration_run_id.setValue(run_id)
        index = self.power_calibration_run.findData(run_id)
        with QtCore.QSignalBlocker(self.power_calibration_run):
            if index < 0 and run_id > 0:
                self.power_calibration_run.addItem(
                    f"Run {run_id} | refresh to inspect metadata",
                    run_id,
                )
                index = self.power_calibration_run.findData(run_id)
            self.power_calibration_run.setCurrentIndex(max(0, index))

    def _calibration_frequency_points(self) -> np.ndarray:
        if not self.frequency_sweep_enabled.isChecked():
            return np.asarray([self.frequency_mhz.value()], dtype=float)
        return np.linspace(
            self.frequency_sweep_start_mhz.value(),
            self.frequency_sweep_stop_mhz.value(),
            self.frequency_sweep_count.value(),
            dtype=float,
        )

    def _calibration_power_points(self) -> np.ndarray:
        if not self.power_sweep_enabled.isChecked():
            return np.asarray(
                [self.target_output_power_dbm.value()],
                dtype=float,
            )
        return np.linspace(
            self.power_sweep_start_dbm.value(),
            self.power_sweep_stop_dbm.value(),
            self.power_sweep_count.value(),
            dtype=float,
        )

    def _refresh_power_calibration_runs(
        self,
        *,
        update_status: bool = True,
        catalog=None,
    ):
        database_path = self.power_calibration_database_path.text().strip()
        if not database_path:
            if update_status:
                self._set_power_calibration_status(
                    "Select a calibration database first.",
                    state="error",
                )
            return ()
        previous_run_id = int(self.power_calibration_run_id.value())
        try:
            if catalog is None:
                catalog = CalibrationDatabase(database_path)
            list_candidates = getattr(
                catalog,
                "output_calibration_candidates",
                None,
            )
            if not callable(list_candidates):
                return ()
            candidates = tuple(
                list_candidates(
                    self.output_board_type.currentText(),
                    self._calibration_frequency_points(),
                    nqz=self.nqz.value(),
                    output_filter_type=self.filter_type.currentText(),
                    output_filter_cutoff_ghz=self.filter_cutoff.value(),
                    output_filter_bandwidth_ghz=(
                        self.filter_bandwidth.value()
                    ),
                )
            )
        except (
            LookupError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            sqlite3.Error,
        ) as exc:
            self._power_calibration_candidates = ()
            if update_status:
                self._set_power_calibration_status(str(exc), state="error")
            return ()

        self._power_calibration_candidates = candidates
        exact = next(
            (candidate for candidate in candidates if candidate.exact_match),
            None,
        )
        auto_label = (
            f"Auto: Run {exact.summary.run_id} (best exact match)"
            if exact is not None
            else "Auto: no exact match - select a run below"
        )
        with QtCore.QSignalBlocker(self.power_calibration_run):
            self.power_calibration_run.clear()
            self.power_calibration_run.addItem(auto_label, 0)
            for candidate in candidates:
                self.power_calibration_run.addItem(
                    candidate.display_label,
                    candidate.summary.run_id,
                )
                index = self.power_calibration_run.count() - 1
                self.power_calibration_run.setItemData(
                    index,
                    candidate.detail_text,
                    QtCore.Qt.ToolTipRole,
                )
            selected_index = self.power_calibration_run.findData(
                previous_run_id
            )
            if previous_run_id > 0 and selected_index < 0:
                self.power_calibration_run.addItem(
                    f"Run {previous_run_id} | not found in current DB",
                    previous_run_id,
                )
                selected_index = self.power_calibration_run.count() - 1
            self.power_calibration_run.setCurrentIndex(
                max(0, selected_index)
            )
        with QtCore.QSignalBlocker(self.power_calibration_run_id):
            self.power_calibration_run_id.setValue(
                int(self.power_calibration_run.currentData() or 0)
            )
        if update_status:
            if exact is not None:
                self._set_power_calibration_status(
                    f"Best match is Run {exact.summary.run_id}. "
                    "The run list shows PCB, filter, Nyquist zone, and "
                    "frequency coverage.",
                    state="success",
                )
            elif candidates:
                self._set_power_calibration_status(
                    "No exact PCB/filter/Nyquist/frequency match. "
                    "Review the candidate details and select a Run "
                    "explicitly to override.",
                    state="stale",
                )
            else:
                self._set_power_calibration_status(
                    "No output calibration runs were found in this database.",
                    state="error",
                )
        return candidates

    def _apply_calibrated_output_power(self) -> Optional[int]:
        try:
            if self.output_board_type.currentText() != "RF_Out":
                raise ValueError(
                    "calibrated output power is available only for RF_Out"
                )
            if not self.power_calibration_group.isChecked():
                raise ValueError("enable Calibrated output power first")
            database_path = self.power_calibration_database_path.text().strip()
            if not database_path:
                raise ValueError("select a calibration database")
            frequency_points = self._calibration_frequency_points()
            power_points = self._calibration_power_points()
            frequency_mhz = float(frequency_points[0])
            catalog = CalibrationDatabase(database_path)
            candidates = self._refresh_power_calibration_runs(
                update_status=False,
                catalog=catalog,
            )
            selected_run_id = int(self.power_calibration_run_id.value())
            selected_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.summary.run_id == selected_run_id
                ),
                None,
            )
            manual_override = (
                selected_run_id > 0
                and selected_candidate is not None
                and not selected_candidate.exact_match
            )
            if (
                manual_override
                and (
                    self.frequency_sweep_enabled.isChecked()
                    or self.power_sweep_enabled.isChecked()
                )
            ):
                raise LookupError(
                    "RF frequency/power sweeps require an exact PCB, filter, "
                    "Nyquist-zone, and frequency-coverage calibration match"
                )
            if selected_run_id == 0 and candidates:
                selected_candidate = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.exact_match
                    ),
                    None,
                )
                if selected_candidate is None:
                    raise LookupError(
                        "no exact PCB/filter/Nyquist/frequency calibration "
                        "match; select one of the listed runs explicitly "
                        "to use a manual override"
                    )
                selected_run_id = selected_candidate.summary.run_id
            if manual_override:
                calibration = catalog.output_calibration(
                    selected_candidate.summary.board_type,
                    frequency_points,
                    run_id=selected_run_id,
                )
            else:
                calibration = catalog.output_calibration(
                    "RF_Out",
                    frequency_points,
                    run_id=selected_run_id or None,
                    nqz=self.nqz.value(),
                    output_filter_type=self.filter_type.currentText(),
                    output_filter_cutoff_ghz=self.filter_cutoff.value(),
                    output_filter_bandwidth_ghz=(
                        self.filter_bandwidth.value()
                    ),
                )
            target_power_dbm = float(power_points[0])
            if hasattr(calibration, "build_gain_schedule"):
                schedule = calibration.build_gain_schedule(
                    frequency_points,
                    target_power_dbm,
                    output_att1_db=self.att1_db.value(),
                    output_att2_db=self.att2_db.value(),
                    max_entries=int(frequency_points.size),
                )
                gain = int(schedule.gain_codes[0])
            elif frequency_points.size == 1:
                response_dbm = float(
                    calibration.frequency_response_dbm(
                        frequency_points
                    )[0]
                )
                gain = int(
                    calibration.nominal_gain_for_power(
                        target_power_dbm,
                        reference_response_dbm=response_dbm,
                        output_att1_db=self.att1_db.value(),
                        output_att2_db=self.att2_db.value(),
                    )
                )
            else:
                raise RuntimeError(
                    "the selected calibration cannot build a frequency "
                    "dependent gain schedule"
                )
            predicted_power_dbm = float(
                calibration.output_power_dbm(
                    [frequency_mhz],
                    [gain],
                    output_att1_db=self.att1_db.value(),
                    output_att2_db=self.att2_db.value(),
                )[0]
            )
        except (
            LookupError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            sqlite3.Error,
        ) as exc:
            self._power_calibration_resolved_signature = None
            self._set_power_calibration_status(str(exc), state="error")
            QtWidgets.QMessageBox.warning(
                self,
                "No compatible output calibration",
                str(exc),
            )
            return None

        with QtCore.QSignalBlocker(self.gain):
            self.gain.setValue(int(gain))
        self._power_calibration_resolved_signature = (
            str(Path(database_path).expanduser()),
            int(calibration.summary.run_id),
            float(frequency_mhz),
            int(self.nqz.value()),
            str(self.filter_type.currentText()),
            float(self.filter_cutoff.value()),
            float(self.filter_bandwidth.value()),
            float(self.att1_db.value()),
            float(self.att2_db.value()),
            float(target_power_dbm),
            float(frequency_points[-1]),
            int(frequency_points.size),
            float(power_points[-1]),
            int(power_points.size),
            int(gain),
        )
        self._set_power_calibration_status(
            f"Run {calibration.summary.run_id}: gain {gain} gives "
            f"{predicted_power_dbm:.6g} dBm at {frequency_mhz:.6g} MHz. "
            f"Current ATT1/ATT2 {self.att1_db.value():.2f}/"
            f"{self.att2_db.value():.2f} dB; Nyquist zone and "
            f"{self.filter_type.currentText()} filter "
            + (
                "manually overridden ("
                + selected_candidate.detail_text
                + ")."
                if manual_override
                else "matched."
            ),
            state="stale" if manual_override else "success",
        )
        self.changed.emit()
        return int(gain)

    def validate_power_calibration(self) -> None:
        """Reject an enabled calibrated-power request that is not current."""
        if not self.isChecked() or not self.power_calibration_group.isChecked():
            return
        if self.output_board_type.currentText() != "RF_Out":
            raise ValueError(
                f"RF Output {self._index + 1} calibrated power requires "
                "an RF_Out board"
            )
        if (
            self.frequency_sweep_enabled.isChecked()
            or self.power_sweep_enabled.isChecked()
        ):
            database_path = self.power_calibration_database_path.text().strip()
            if not database_path:
                raise ValueError(
                    f"RF Output {self._index + 1} calibration database is missing"
                )
            CalibrationDatabase(database_path).output_calibration(
                "RF_Out",
                self._calibration_frequency_points(),
                run_id=(
                    None
                    if self.power_calibration_run_id.value() == 0
                    else int(self.power_calibration_run_id.value())
                ),
                nqz=self.nqz.value(),
                output_filter_type=self.filter_type.currentText(),
                output_filter_cutoff_ghz=self.filter_cutoff.value(),
                output_filter_bandwidth_ghz=self.filter_bandwidth.value(),
            )
            return
        if self._power_calibration_resolved_signature is None:
            raise ValueError(
                f"RF Output {self._index + 1} calibrated power is not applied "
                "for the current frequency, ATT, Nyquist zone, and filter "
                "settings; click Apply calibrated gain"
            )

    def _update_board_controls(self, *_args) -> None:
        has_attenuators = self.output_board_type.currentText() == "RF_Out"
        self.att1_db.setEnabled(has_attenuators)
        self.att2_db.setEnabled(has_attenuators)
        self.filter_type.setEnabled(has_attenuators)
        self.filter_cutoff.setEnabled(has_attenuators)
        self.filter_bandwidth.setEnabled(has_attenuators)
        if has_attenuators:
            tooltip = "RF_Out onboard attenuator"
        else:
            tooltip = "DC_Out has no onboard ATT1/ATT2; these values are ignored"
        self.att1_db.setToolTip(tooltip)
        self.att2_db.setToolTip(tooltip)
        self.filter_type.setToolTip(tooltip)
        self.filter_cutoff.setToolTip(tooltip)
        self.filter_bandwidth.setToolTip(tooltip)
        self._mark_power_calibration_stale()

    def set_front_panel_configuration(self, configuration) -> None:
        self._front_panel_configuration = configuration
        self.front_panel_preview.set_configuration(configuration)
        self._sync_front_panel_selection()

    def set_hardware_backend(self, backend: str) -> None:
        backend = str(backend).strip().lower()
        if backend not in EXECUTION_BACKENDS:
            raise ValueError(f"unsupported execution backend {backend!r}")
        self._hardware_backend = backend
        self.front_panel_preview.set_backend(self._hardware_backend)
        self._keep_front_panel_preview_enabled()
        self._update_backend_presentation()

    def _keep_front_panel_preview_enabled(self, *_args) -> None:
        """Allow physical RF configuration before the waveform is enabled."""

        self.front_panel_preview.setEnabled(True)
        self.front_panel_preview.currentWidget().setEnabled(True)
        if hasattr(self, "qcs_lo_frequency_row"):
            is_m5300 = (
                self._hardware_backend == EXECUTION_BACKEND_QCS
                and self.qcs_module_model.currentData() == "M5300A"
            )
            self.qcs_lo_frequency_row.setEnabled(is_m5300)
            self.qcs_lo_frequency_ghz.setEnabled(is_m5300)
            self.apply_qcs_lo_frequency.setEnabled(is_m5300)

    def set_qcs_front_panel_configuration(
        self,
        configuration: Optional[Mapping[str, object]],
    ) -> None:
        self._qcs_front_panel_configuration = configuration
        self.front_panel_preview.set_qcs_configuration(configuration)
        self.front_panel_preview.set_qcs_selection(
            "rf",
            self.gen_ch.value(),
        )
        self._sync_qcs_module_model()

    def _sync_qcs_module_model(self) -> None:
        """Reflect the module bound by the QCS front-panel configuration."""

        selected_model = None
        selected_lo_frequency_hz = None
        binding_tooltip = (
            "No QCS RF module is mapped to this virtual channel. Click the "
            "front-panel preview to configure its module and SMA output."
        )
        if self._qcs_front_panel_configuration is not None:
            try:
                configuration = normalize_qcs_hardware_configuration(
                    self._qcs_front_panel_configuration
                )
            except (TypeError, ValueError):
                configuration = None
            if configuration is not None:
                mapping = next(
                    (
                        candidate
                        for candidate in configuration["channel_mappings"]
                        if (
                            candidate["role"] == "rf"
                            and int(candidate["logical_index"])
                            == self.gen_ch.value()
                        )
                    ),
                    None,
                )
                if mapping is not None:
                    module = next(
                        (
                            candidate
                            for candidate in configuration["modules"]
                            if int(candidate["slot"]) == int(mapping["slot"])
                        ),
                        None,
                    )
                    if module is not None:
                        selected_model = str(module["model"])
                        selected_lo_frequency_hz = mapping.get(
                            "lo_frequency_hz"
                        )
                        binding_tooltip = (
                            f"{selected_model}, chassis "
                            f"{configuration['chassis']}, slot "
                            f"{mapping['slot']}, SMA CH {mapping['channel']}; "
                            f"QCS virtual channel "
                            f"{mapping['virtual_name']!r}. Click the front-panel "
                            "preview to change the physical binding."
                        )
        with QtCore.QSignalBlocker(self.qcs_module_model):
            model_index = self.qcs_module_model.findData(selected_model)
            self.qcs_module_model.setCurrentIndex(model_index)
        self.qcs_module_model.setToolTip(binding_tooltip)
        is_m5300 = (
            self._hardware_backend == EXECUTION_BACKEND_QCS
            and selected_model == "M5300A"
        )
        self._set_form_row_visible(self.qcs_lo_frequency_row, is_m5300)
        if is_m5300:
            if selected_lo_frequency_hz is None:
                self.qcs_lo_frequency_status.setText(
                    "LO is not configured; choose a frequency before running."
                )
                self.qcs_lo_frequency_status.setStyleSheet(
                    "QLabel { color: #cf222e; }"
                )
            else:
                with QtCore.QSignalBlocker(self.qcs_lo_frequency_ghz):
                    self.qcs_lo_frequency_ghz.setValue(
                        float(selected_lo_frequency_hz) / 1.0e9
                    )
                self.qcs_lo_frequency_status.setText(
                    "Current ChannelMapper LO: "
                    f"{float(selected_lo_frequency_hz) / 1.0e9:.9g} GHz"
                )
                self.qcs_lo_frequency_status.setStyleSheet(
                    "QLabel { color: #1a7f37; }"
                )
        self.apply_qcs_lo_frequency.setEnabled(is_m5300)
        # An unchecked QGroupBox disables its child editors.  Re-evaluate the
        # explicit exceptions after a mapping change so an M5301 -> M5300
        # reassignment does not leave the newly visible LO editor disabled.
        self._keep_front_panel_preview_enabled()
        self._update_qcs_power_calibration_controls()

    def _sync_front_panel_selection(self, *_args) -> None:
        self.front_panel_preview.set_qcs_selection(
            "rf",
            self.gen_ch.value(),
        )
        self._sync_qcs_module_model()
        self.front_panel_preview.set_channels(output_ch=self.gen_ch.value())
        if self._front_panel_configuration is None:
            return
        for port in self._front_panel_configuration.outputs:
            if self.gen_ch.value() in port.qick_channels:
                if port.board_type in QICK_OUTPUT_BOARD_TYPES:
                    self.output_board_type.setCurrentText(port.board_type)
                return

    def apply_front_panel_settings(self, values: Mapping[str, object]) -> None:
        self.gen_ch.setValue(int(values["output_ch"]))
        self.output_board_type.setCurrentText(str(values["output_board_type"]))
        self.att1_db.setValue(float(values["output_att1_db"]))
        self.att2_db.setValue(float(values["output_att2_db"]))
        self.filter_type.setCurrentText(str(values["output_filter_type"]))
        self.filter_cutoff.setValue(float(values["output_filter_cutoff_ghz"]))
        self.filter_bandwidth.setValue(
            float(values["output_filter_bandwidth_ghz"])
        )
        self.nqz.setValue(int(values.get("output_nqz", self.nqz.value())))
        self._update_board_controls()
        self._sync_front_panel_selection()
        self.changed.emit()

    def set_index(self, index: int) -> None:
        self._index = index
        self.setTitle(f"RF Output {index + 1}")

    def set_time_unit(self, unit: str, *, force: bool = False) -> None:
        if unit not in TIME_UNIT_NS:
            raise ValueError(f"unsupported time unit {unit!r}")
        if not force and unit == self._time_unit:
            return
        old_unit = self._time_unit
        delay_ns = _time_to_ns(self.delay.value(), old_unit)
        duration_ns = _time_to_ns(self.duration.value(), old_unit)
        sweep_start_ns = _time_to_ns(
            self.duration_sweep_start.value(),
            old_unit,
        )
        sweep_stop_ns = _time_to_ns(
            self.duration_sweep_stop.value(),
            old_unit,
        )
        self._time_unit = unit
        with QtCore.QSignalBlocker(self.delay), \
                QtCore.QSignalBlocker(self.duration), \
                QtCore.QSignalBlocker(self.duration_sweep_start), \
                QtCore.QSignalBlocker(self.duration_sweep_stop):
            self.delay.setValue(_time_from_ns(delay_ns, unit))
            self.duration.setValue(_time_from_ns(duration_ns, unit))
            self.duration_sweep_start.setValue(
                _time_from_ns(sweep_start_ns, unit)
            )
            self.duration_sweep_stop.setValue(
                _time_from_ns(sweep_stop_ns, unit)
            )
            self.delay.setSuffix(f" {unit}")
            self.duration.setSuffix(f" {unit}")
            self.duration_sweep_start.setSuffix(f" {unit}")
            self.duration_sweep_stop.setSuffix(f" {unit}")
        self._update_time_labels()

    def refresh_segments(self, pulse: Optional[PulseSequence] = None) -> None:
        if pulse is not None:
            self._pulse = pulse
        previous = self.segment.currentData()
        with QtCore.QSignalBlocker(self.segment):
            self.segment.clear()
            for index, _segment in enumerate(self._pulse.flat_segments()):
                self.segment.addItem(
                    self._pulse.segment_name(index),
                    f"set_{index}",
                )
            match = self.segment.findData(previous)
            if match >= 0:
                self.segment.setCurrentIndex(match)

    def configured_spec(
        self,
        *,
        for_qcs: bool = False,
    ) -> QickRfPulseSpec:
        """Return the editor values even when this output is disabled."""
        return QickRfPulseSpec(
            gen_ch=self.gen_ch.value(),
            segment_name=str(self.segment.currentData()),
            delay_us=_time_to_ns(self.delay.value(), self._time_unit) / 1000.0,
            duration_us=_time_to_ns(self.duration.value(), self._time_unit) / 1000.0,
            frequency_mhz=self.frequency_mhz.value(),
            gain=self.gain.value(),
            att1_db=self.att1_db.value(),
            att2_db=self.att2_db.value(),
            phase_degrees=self.phase_degrees.value(),
            nqz=self.nqz.value(),
            require_within_segment=self.require_within.isChecked(),
            filter_type=self.filter_type.currentText(),
            filter_cutoff=self.filter_cutoff.value(),
            filter_bandwidth=self.filter_bandwidth.value(),
            output_board_type=self.output_board_type.currentText(),
            duration_sweep_enabled=self.duration_sweep_enabled.isChecked(),
            duration_sweep_start_us=(
                _time_to_ns(
                    self.duration_sweep_start.value(),
                    self._time_unit,
                )
                / 1000.0
            ),
            duration_sweep_stop_us=(
                _time_to_ns(
                    self.duration_sweep_stop.value(),
                    self._time_unit,
                )
                / 1000.0
            ),
            duration_sweep_count=self.duration_sweep_count.value(),
            segment_length_mode=str(
                self.segment_length_mode.currentData()
            ),
            frequency_sweep_enabled=(
                self.frequency_sweep_enabled.isChecked()
            ),
            frequency_sweep_start_mhz=(
                self.frequency_sweep_start_mhz.value()
            ),
            frequency_sweep_stop_mhz=(
                self.frequency_sweep_stop_mhz.value()
            ),
            frequency_sweep_count=self.frequency_sweep_count.value(),
            power_sweep_enabled=(
                not for_qcs
                and self.power_calibration_group.isChecked()
                and self.power_sweep_enabled.isChecked()
            ),
            power_sweep_start_dbm=self.power_sweep_start_dbm.value(),
            power_sweep_stop_dbm=self.power_sweep_stop_dbm.value(),
            power_sweep_count=self.power_sweep_count.value(),
            power_calibration_enabled=(
                not for_qcs and self.power_calibration_group.isChecked()
            ),
            power_calibration_database_path=(
                self.power_calibration_database_path.text().strip()
            ),
            power_calibration_run_id=self.power_calibration_run_id.value(),
            target_output_power_dbm=self.target_output_power_dbm.value(),
        )

    def spec(self, *, for_qcs: bool = False) -> Optional[QickRfPulseSpec]:
        if not self.isChecked():
            return None
        return self.configured_spec(for_qcs=for_qcs)

    def settings_dict(self) -> dict:
        spec = self.configured_spec()
        return {
            "enabled": self.isChecked(),
            "gen_ch": spec.gen_ch,
            "segment_name": spec.segment_name,
            "delay_us": spec.delay_us,
            "duration_us": spec.duration_us,
            "frequency_mhz": spec.frequency_mhz,
            "gain": spec.gain,
            "output_board_type": spec.output_board_type,
            "att1_db": spec.att1_db,
            "att2_db": spec.att2_db,
            "filter_type": spec.filter_type,
            "filter_cutoff": spec.filter_cutoff,
            "filter_bandwidth": spec.filter_bandwidth,
            "phase_degrees": spec.phase_degrees,
            "nqz": spec.nqz,
            "require_within_segment": spec.require_within_segment,
            "duration_sweep_enabled": spec.duration_sweep_enabled,
            "duration_sweep_start_us": spec.duration_sweep_start_us,
            "duration_sweep_stop_us": spec.duration_sweep_stop_us,
            "duration_sweep_count": spec.duration_sweep_count,
            "segment_length_mode": spec.segment_length_mode,
            "frequency_sweep_enabled": spec.frequency_sweep_enabled,
            "frequency_sweep_start_mhz": spec.frequency_sweep_start_mhz,
            "frequency_sweep_stop_mhz": spec.frequency_sweep_stop_mhz,
            "frequency_sweep_count": spec.frequency_sweep_count,
            "power_sweep_enabled": spec.power_sweep_enabled,
            "power_sweep_start_dbm": spec.power_sweep_start_dbm,
            "power_sweep_stop_dbm": spec.power_sweep_stop_dbm,
            "power_sweep_count": spec.power_sweep_count,
            "power_calibration_enabled": spec.power_calibration_enabled,
            "power_calibration_database_path": (
                spec.power_calibration_database_path
            ),
            "power_calibration_run_id": spec.power_calibration_run_id,
            "target_output_power_dbm": spec.target_output_power_dbm,
            "qcs_power_calibration_enabled": (
                self.qcs_power_calibration_group.isChecked()
            ),
            "qcs_power_calibration_database_path": (
                self.qcs_power_calibration_database_path.text().strip()
            ),
            "qcs_power_calibration_run_id": (
                self.qcs_power_calibration_run_id.value()
            ),
            "qcs_target_output_power_dbm": (
                self.qcs_target_output_power_dbm.value()
            ),
        }

    def load_settings(self, data: dict) -> None:
        if not isinstance(data, dict):
            raise TypeError("each RF output setting must be a JSON object")
        enabled = data.get("enabled", True)
        require_within = data.get("require_within_segment", True)
        duration_sweep_enabled = data.get(
            "duration_sweep_enabled",
            False,
        )
        if not isinstance(enabled, bool):
            raise TypeError("RF output enabled must be boolean")
        if not isinstance(require_within, bool):
            raise TypeError("RF require_within_segment must be boolean")
        if not isinstance(duration_sweep_enabled, bool):
            raise TypeError("RF duration_sweep_enabled must be boolean")
        frequency_sweep_enabled = data.get(
            "frequency_sweep_enabled",
            False,
        )
        power_sweep_enabled = data.get("power_sweep_enabled", False)
        if not isinstance(frequency_sweep_enabled, bool):
            raise TypeError("RF frequency_sweep_enabled must be boolean")
        if not isinstance(power_sweep_enabled, bool):
            raise TypeError("RF power_sweep_enabled must be boolean")
        power_calibration_enabled = data.get(
            "power_calibration_enabled",
            False,
        )
        if not isinstance(power_calibration_enabled, bool):
            raise TypeError("RF power_calibration_enabled must be boolean")
        qcs_power_calibration_enabled = data.get(
            "qcs_power_calibration_enabled",
            False,
        )
        if not isinstance(qcs_power_calibration_enabled, bool):
            raise TypeError(
                "RF qcs_power_calibration_enabled must be boolean"
            )
        spec = QickRfPulseSpec(
            gen_ch=int(data["gen_ch"]),
            segment_name=_resolve_stored_set_segment_name(
                data["segment_name"],
                tuple(
                    self.segment.itemData(index)
                    for index in range(self.segment.count())
                ),
                label="RF output",
            ),
            delay_us=float(data["delay_us"]),
            duration_us=float(data["duration_us"]),
            frequency_mhz=float(data["frequency_mhz"]),
            gain=int(data["gain"]),
            att1_db=float(data["att1_db"]),
            att2_db=float(data["att2_db"]),
            phase_degrees=float(data.get("phase_degrees", 0.0)),
            nqz=int(data.get("nqz", 1)),
            require_within_segment=require_within,
            filter_type=str(data.get("filter_type", "bypass")),
            filter_cutoff=float(data.get("filter_cutoff", 2.5)),
            filter_bandwidth=float(data.get("filter_bandwidth", 1.0)),
            output_board_type=str(data.get("output_board_type", "RF_Out")),
            duration_sweep_enabled=duration_sweep_enabled,
            duration_sweep_start_us=float(
                data.get(
                    "duration_sweep_start_us",
                    data.get("duration_us", 1.0),
                )
            ),
            duration_sweep_stop_us=float(
                data.get(
                    "duration_sweep_stop_us",
                    data.get("duration_us", 1.0),
                )
            ),
            duration_sweep_count=int(
                data.get("duration_sweep_count", 1)
            ),
            segment_length_mode=str(
                data.get("segment_length_mode", "fixed")
            ),
            frequency_sweep_enabled=frequency_sweep_enabled,
            frequency_sweep_start_mhz=float(
                data.get(
                    "frequency_sweep_start_mhz",
                    data.get("frequency_mhz", 50.0),
                )
            ),
            frequency_sweep_stop_mhz=float(
                data.get(
                    "frequency_sweep_stop_mhz",
                    data.get("frequency_mhz", 50.0),
                )
            ),
            frequency_sweep_count=int(
                data.get("frequency_sweep_count", 1)
            ),
            power_sweep_enabled=(
                power_calibration_enabled and power_sweep_enabled
            ),
            power_sweep_start_dbm=float(
                data.get(
                    "power_sweep_start_dbm",
                    data.get("target_output_power_dbm", -20.0),
                )
            ),
            power_sweep_stop_dbm=float(
                data.get(
                    "power_sweep_stop_dbm",
                    data.get("target_output_power_dbm", -20.0),
                )
            ),
            power_sweep_count=int(data.get("power_sweep_count", 1)),
            power_calibration_enabled=power_calibration_enabled,
            power_calibration_database_path=str(
                data.get(
                    "power_calibration_database_path",
                    DEFAULT_POWER_CALIBRATION_DB_PATH,
                )
            ),
            power_calibration_run_id=int(
                data.get("power_calibration_run_id", 0)
            ),
            target_output_power_dbm=float(
                data.get("target_output_power_dbm", -20.0)
            ),
        )
        segment = self.segment.findData(spec.segment_name)
        if segment < 0:
            raise ValueError(f"unknown RF output anchor {spec.segment_name!r}")
        self.gen_ch.setValue(spec.gen_ch)
        self.segment.setCurrentIndex(segment)
        self.delay.setValue(_time_from_ns(spec.delay_us * 1000.0, self._time_unit))
        self.duration.setValue(
            _time_from_ns(spec.duration_us * 1000.0, self._time_unit)
        )
        self.duration_sweep_enabled.setChecked(
            spec.duration_sweep_enabled
        )
        self.duration_sweep_start.setValue(
            _time_from_ns(
                spec.duration_sweep_start_us * 1000.0,
                self._time_unit,
            )
        )
        self.duration_sweep_stop.setValue(
            _time_from_ns(
                spec.duration_sweep_stop_us * 1000.0,
                self._time_unit,
            )
        )
        self.duration_sweep_count.setValue(spec.duration_sweep_count)
        mode_index = self.segment_length_mode.findData(
            spec.segment_length_mode
        )
        if mode_index >= 0:
            self.segment_length_mode.setCurrentIndex(mode_index)
        self.frequency_mhz.setValue(spec.frequency_mhz)
        self.frequency_sweep_enabled.setChecked(
            spec.frequency_sweep_enabled
        )
        self.frequency_sweep_start_mhz.setValue(
            spec.frequency_sweep_start_mhz
        )
        self.frequency_sweep_stop_mhz.setValue(
            spec.frequency_sweep_stop_mhz
        )
        self.frequency_sweep_count.setValue(spec.frequency_sweep_count)
        self.gain.setValue(spec.gain)
        self.output_board_type.setCurrentText(spec.output_board_type)
        self.att1_db.setValue(spec.att1_db)
        self.att2_db.setValue(spec.att2_db)
        self.filter_type.setCurrentText(spec.filter_type)
        self.filter_cutoff.setValue(spec.filter_cutoff)
        self.filter_bandwidth.setValue(spec.filter_bandwidth)
        self.phase_degrees.setValue(spec.phase_degrees)
        self.nqz.setValue(spec.nqz)
        self.require_within.setChecked(spec.require_within_segment)
        self.power_calibration_database_path.setText(
            str(
                data.get(
                    "power_calibration_database_path",
                    DEFAULT_POWER_CALIBRATION_DB_PATH,
                )
            )
        )
        self._set_power_calibration_run_id(
            int(data.get("power_calibration_run_id", 0))
        )
        self.target_output_power_dbm.setValue(
            spec.target_output_power_dbm
        )
        self.power_sweep_enabled.setChecked(spec.power_sweep_enabled)
        self.power_sweep_start_dbm.setValue(spec.power_sweep_start_dbm)
        self.power_sweep_stop_dbm.setValue(spec.power_sweep_stop_dbm)
        self.power_sweep_count.setValue(spec.power_sweep_count)
        self.power_calibration_group.setChecked(
            power_calibration_enabled
        )
        self.qcs_power_calibration_database_path.setText(
            str(
                data.get(
                    "qcs_power_calibration_database_path",
                    DEFAULT_POWER_CALIBRATION_DB_PATH,
                )
            )
        )
        self.qcs_power_calibration_run_id.setValue(
            int(data.get("qcs_power_calibration_run_id", 0))
        )
        self.qcs_target_output_power_dbm.setValue(
            float(data.get("qcs_target_output_power_dbm", -20.0))
        )
        self.qcs_power_calibration_group.setChecked(
            qcs_power_calibration_enabled
        )
        self._update_board_controls()
        self._update_duration_sweep_controls()
        self._update_frequency_sweep_controls()
        self._update_power_calibration_controls()
        self._update_power_sweep_controls()
        self._update_qcs_power_calibration_controls()
        self.setChecked(enabled)


class RfPortsPanel(QtWidgets.QWidget):
    """Add/remove editor for up to eight RF waveform outputs."""

    specs_changed = QtCore.pyqtSignal(object)
    front_panel_requested = QtCore.pyqtSignal(object)
    lo_frequency_change_requested = QtCore.pyqtSignal(int, float)
    MAX_PORTS = 8

    def __init__(self, pulse: PulseSequence, *, time_unit: str, parent=None):
        super().__init__(parent)
        self._pulse = pulse
        self._time_unit = time_unit
        self._panels: List[RfPulsePortPanel] = []
        self._front_panel_configuration = None
        self._qcs_front_panel_configuration = None
        self._hardware_backend = EXECUTION_BACKEND_QCS
        layout = QtWidgets.QVBoxLayout(self)
        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._content = QtWidgets.QWidget(self._scroll)
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.addStretch(1)
        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll)
        self.add_button = QtWidgets.QPushButton("Add RF Port")
        self.add_button.clicked.connect(lambda: self.add_port())
        layout.addWidget(self.add_button)
        self.add_port()

    def add_port(
        self,
        spec: Optional[QickRfPulseSpec] = None,
        *,
        settings: Optional[dict] = None,
    ) -> None:
        if len(self._panels) >= self.MAX_PORTS:
            return
        if spec is not None and settings is not None:
            raise ValueError("use either RF spec or RF settings, not both")
        panel = RfPulsePortPanel(
            self._pulse,
            len(self._panels),
            default_gen_ch=DEFAULT_QSTL_RF_CHANNELS[len(self._panels)],
            time_unit=self._time_unit,
            parent=self._content,
        )
        panel.changed.connect(self._emit_specs)
        panel.remove_requested.connect(self.remove_port)
        panel.front_panel_requested.connect(self.front_panel_requested.emit)
        panel.lo_frequency_change_requested.connect(
            self._forward_lo_frequency_change_request
        )
        if self._front_panel_configuration is not None:
            panel.set_front_panel_configuration(self._front_panel_configuration)
        panel.set_hardware_backend(self._hardware_backend)
        panel.set_qcs_front_panel_configuration(
            self._qcs_front_panel_configuration
        )
        self._content_layout.insertWidget(self._content_layout.count() - 1, panel)
        self._panels.append(panel)
        if spec is not None:
            self._load_spec(panel, spec)
        elif settings is not None:
            panel.load_settings(settings)
        self.add_button.setEnabled(len(self._panels) < self.MAX_PORTS)
        self._emit_specs()

    def remove_port(self, panel: RfPulsePortPanel) -> None:
        if panel not in self._panels:
            return
        self._panels.remove(panel)
        panel.deleteLater()
        for index, current in enumerate(self._panels):
            current.set_index(index)
        self.add_button.setEnabled(True)
        self._emit_specs()

    def _load_spec(self, panel: RfPulsePortPanel, spec: QickRfPulseSpec) -> None:
        panel.setChecked(True)
        panel.gen_ch.setValue(spec.gen_ch)
        segment = panel.segment.findData(spec.segment_name)
        if segment >= 0:
            panel.segment.setCurrentIndex(segment)
        panel.delay.setValue(
            _time_from_ns(spec.delay_us * 1000.0, panel._time_unit)
        )
        panel.duration.setValue(
            _time_from_ns(spec.duration_us * 1000.0, panel._time_unit)
        )
        panel.frequency_mhz.setValue(spec.frequency_mhz)
        panel.gain.setValue(spec.gain)
        panel.output_board_type.setCurrentText(spec.output_board_type)
        panel.att1_db.setValue(spec.att1_db)
        panel.att2_db.setValue(spec.att2_db)
        panel.phase_degrees.setValue(spec.phase_degrees)
        panel.nqz.setValue(spec.nqz)
        panel.require_within.setChecked(spec.require_within_segment)

    def apply_path_settings(self, values: Mapping[str, object]) -> int:
        """Apply a committed RF path to the matching Experiment output editor."""
        output_ch = int(values["output_ch"])
        target = next(
            (panel for panel in self._panels if panel.gen_ch.value() == output_ch),
            self._panels[0] if self._panels else None,
        )
        if target is None:
            self.add_port()
            target = self._panels[0]
        target.gen_ch.setValue(output_ch)
        target.output_board_type.setCurrentText(str(values["output_board_type"]))
        target.att1_db.setValue(float(values["output_att1_db"]))
        target.att2_db.setValue(float(values["output_att2_db"]))
        target.filter_type.setCurrentText(str(values["output_filter_type"]))
        target.filter_cutoff.setValue(float(values["output_filter_cutoff_ghz"]))
        target.filter_bandwidth.setValue(
            float(values["output_filter_bandwidth_ghz"])
        )
        target.nqz.setValue(int(values.get("output_nqz", target.nqz.value())))
        target._update_board_controls()
        self._emit_specs()
        return self._panels.index(target)

    def set_front_panel_configuration(self, configuration) -> None:
        self._front_panel_configuration = configuration
        for panel in self._panels:
            panel.set_front_panel_configuration(configuration)

    def set_hardware_backend(self, backend: str) -> None:
        self._hardware_backend = str(backend).strip().lower()
        self.add_button.setText(
            "Add RF Output"
            if self._hardware_backend == EXECUTION_BACKEND_QCS
            else "Add RF Port"
        )
        for panel in self._panels:
            panel.set_hardware_backend(self._hardware_backend)

    def validate_qcs_rf_channel_names(
        self,
        rf_channel_names: Mapping[int, str],
    ) -> Tuple[int, ...]:
        """Validate that active RF editors can be bound without data loss."""
        raw_indices = tuple(int(index) for index in rf_channel_names)
        if len(set(raw_indices)) != len(raw_indices):
            raise ValueError(
                "QCS RF logical channel numbers must be unique"
            )
        logical_indices = tuple(sorted(raw_indices))
        if len(logical_indices) > self.MAX_PORTS:
            raise ValueError(
                f"QCS defines {len(logical_indices)} RF channels, but the "
                f"GUI supports at most {self.MAX_PORTS}"
            )
        enabled_count = sum(panel.isChecked() for panel in self._panels)
        if enabled_count > len(logical_indices):
            raise ValueError(
                f"{enabled_count} RF outputs are enabled, but the QCS "
                f"hardware configuration defines only "
                f"{len(logical_indices)} RF role(s). Add RF mappings or "
                "disable the extra RF outputs before applying."
            )
        return logical_indices

    def set_qcs_front_panel_preview_configuration(
        self,
        configuration: Optional[Mapping[str, object]],
    ) -> None:
        """Refresh QCS diagrams without changing RF logical identities."""
        self._qcs_front_panel_configuration = configuration
        for panel in self._panels:
            panel.set_qcs_front_panel_configuration(configuration)

    def set_qcs_front_panel_configuration(
        self,
        configuration: Optional[Mapping[str, object]],
        rf_channel_names: Mapping[int, str],
    ) -> None:
        logical_indices = self.validate_qcs_rf_channel_names(
            rf_channel_names
        )
        self._qcs_front_panel_configuration = configuration
        assignments = {}
        with QtCore.QSignalBlocker(self):
            while len(self._panels) < len(logical_indices):
                previous_count = len(self._panels)
                self.add_port()
                if len(self._panels) == previous_count:
                    raise ValueError(
                        "QCS RF roles exceed the available RF editor panels"
                    )

            remaining = list(logical_indices)
            enabled_panels = [
                panel for panel in self._panels if panel.isChecked()
            ]
            disabled_panels = [
                panel for panel in self._panels if not panel.isChecked()
            ]

            for panel in enabled_panels:
                current = panel.gen_ch.value()
                if current in remaining:
                    assignments[panel] = current
                    remaining.remove(current)
            for panel in enabled_panels:
                if panel not in assignments:
                    assignments[panel] = remaining.pop(0)

            for panel in disabled_panels:
                current = panel.gen_ch.value()
                if current in remaining:
                    assignments[panel] = current
                    remaining.remove(current)
            for panel in disabled_panels:
                if not remaining:
                    break
                if panel not in assignments:
                    assignments[panel] = remaining.pop(0)

            for panel, logical_index in assignments.items():
                with QtCore.QSignalBlocker(panel.gen_ch):
                    panel.gen_ch.setValue(logical_index)
            self.set_qcs_front_panel_preview_configuration(configuration)
        self._emit_specs()

    def _forward_lo_frequency_change_request(
        self,
        panel: RfPulsePortPanel,
        frequency_hz: float,
    ) -> None:
        if panel not in self._panels:
            return
        self.lo_frequency_change_requested.emit(
            int(panel.gen_ch.value()),
            float(frequency_hz),
        )

    def specs(
        self,
        *,
        for_qcs: bool = False,
    ) -> Tuple[QickRfPulseSpec, ...]:
        specs = tuple(
            spec
            for panel in self._panels
            if (spec := panel.spec(for_qcs=for_qcs)) is not None
        )
        channels = tuple(spec.gen_ch for spec in specs)
        if len(set(channels)) != len(channels):
            raise ValueError("enabled RF output ports must use unique generator indices")
        return specs

    def configured_specs(self) -> Tuple[QickRfPulseSpec, ...]:
        """Return every editor value, including currently disabled outputs."""
        return tuple(panel.configured_spec() for panel in self._panels)

    def settings(self) -> Tuple[dict, ...]:
        return tuple(panel.settings_dict() for panel in self._panels)

    def load_settings(self, entries: Sequence[dict]) -> None:
        entries = tuple(entries)
        if len(entries) > self.MAX_PORTS:
            raise ValueError(f"at most {self.MAX_PORTS} RF output ports are supported")
        with QtCore.QSignalBlocker(self):
            for panel in self._panels:
                self._content_layout.removeWidget(panel)
                panel.setParent(None)
                panel.deleteLater()
            self._panels.clear()
            for entry in entries:
                self.add_port(settings=entry)
            self.add_button.setEnabled(len(self._panels) < self.MAX_PORTS)
        self._emit_specs()

    def _emit_specs(self, *_args) -> None:
        try:
            specs = self.specs()
        except (TypeError, ValueError):
            return
        self.specs_changed.emit(specs)

    def refresh_segments(self, pulse: Optional[PulseSequence] = None) -> None:
        if pulse is not None:
            self._pulse = pulse
        for panel in self._panels:
            panel.refresh_segments(self._pulse)
        self._emit_specs()

    def remap_segment_references(
        self,
        pulse: PulseSequence,
        operation: str,
        segment_index: int,
    ) -> dict:
        """Keep RF pulse and all RF sweep anchors on their logical SETs."""
        self._pulse = pulse
        sweep_key_remap = {}
        fallback_index = min(
            max(0, int(segment_index)),
            max(0, pulse.set_count - 1),
        )
        for panel in self._panels:
            old_name = str(panel.segment.currentData())
            new_name = _remap_set_segment_name(
                old_name,
                operation,
                segment_index,
            )
            was_enabled = panel.isChecked()
            gen_ch = int(panel.gen_ch.value())
            old_sweep_keys = []
            if panel.duration_sweep_enabled.isChecked():
                old_sweep_keys.append((f"rf_gen_{gen_ch}", old_name))
            if panel.frequency_sweep_enabled.isChecked():
                old_sweep_keys.append((
                    f"rf_gen_{gen_ch}_frequency",
                    old_name,
                ))
            if (
                panel.power_calibration_group.isChecked()
                and panel.power_sweep_enabled.isChecked()
            ):
                old_sweep_keys.append((
                    f"rf_gen_{gen_ch}_power",
                    old_name,
                ))
            with QtCore.QSignalBlocker(panel):
                panel.refresh_segments(pulse)
                if new_name is None:
                    panel.setChecked(False)
                    fallback_name = f"set_{fallback_index}"
                    match = panel.segment.findData(fallback_name)
                else:
                    match = panel.segment.findData(new_name)
                if match < 0:
                    raise RuntimeError(
                        f"cannot remap RF anchor {old_name!r} after "
                        f"{operation} at SET {segment_index}"
                    )
                panel.segment.setCurrentIndex(match)
            if was_enabled:
                for old_sweep_key in old_sweep_keys:
                    sweep_key_remap[old_sweep_key] = (
                        None
                        if new_name is None
                        else (old_sweep_key[0], new_name)
                    )
        return sweep_key_remap

    def set_time_unit(self, unit: str) -> None:
        self._time_unit = unit
        for panel in self._panels:
            panel.set_time_unit(unit)


class RfReadoutPanel(QtWidgets.QGroupBox):
    """Backend-aware digitizer acquisition configuration."""

    spec_changed = QtCore.pyqtSignal(object)
    front_panel_requested = QtCore.pyqtSignal(object)
    qcs_acquisition_mode_changed = QtCore.pyqtSignal(bool)

    def __init__(self, pulse: PulseSequence, *, time_unit: str, parent=None):
        super().__init__("RF Readout 1", parent)
        self._pulse = pulse
        self._time_unit = time_unit
        self._front_panel_configuration = None
        self._qcs_front_panel_configuration = None
        self._hardware_backend = EXECUTION_BACKEND_QCS
        self._qcs_hardware_demodulation = True
        self._qcs_acquisition_mode_editing_enabled = True
        self._qcs_acquisition_timing_rate_hz = (
            DEFAULT_QCS_SAMPLE_RATE_HZ
        )
        self._fir_sample_rate_hz: Optional[float] = None
        self._fir_trigger_delay_us = 0.0
        self._fir_uses_fpga_trigger_delay: Optional[bool] = None
        self.setCheckable(True)
        self.setChecked(False)
        form = QtWidgets.QFormLayout(self)
        self._form_layout = form
        self.ro_ch = QtWidgets.QSpinBox()
        self.ro_ch.setRange(0, 255)
        self.input_board_type = QtWidgets.QComboBox()
        self.input_board_type.addItems(QICK_INPUT_BOARD_TYPES)
        self.segment = QtWidgets.QComboBox()
        self.delay = QtWidgets.QDoubleSpinBox()
        self.delay.setRange(0.0, 1.0e12)
        self.delay.setDecimals(9)
        self.samples = QtWidgets.QSpinBox()
        self.samples.setRange(1, 10_000_000)
        self.samples.setValue(64)
        self.qcs_acquisition_duration = QtWidgets.QDoubleSpinBox()
        self.qcs_acquisition_duration.setDecimals(15)
        self.qcs_acquisition_duration.setKeyboardTracking(False)
        default_qcs_duration_ns = (
            64 * 1.0e9 / self._qcs_acquisition_timing_rate_hz
        )
        self.qcs_acquisition_duration.setRange(
            _time_from_ns(
                1.0e9 / self._qcs_acquisition_timing_rate_hz,
                self._time_unit,
            ),
            _time_from_ns(
                self.samples.maximum()
                * 1.0e9
                / self._qcs_acquisition_timing_rate_hz,
                self._time_unit,
            ),
        )
        self.qcs_acquisition_duration.setValue(
            _time_from_ns(default_qcs_duration_ns, self._time_unit)
        )
        self.qcs_acquisition_duration.setSuffix(f" {self._time_unit}")
        self.qcs_acquisition_duration.setToolTip(
            "Requested QCS acquisition time. The GUI rounds upward to a "
            "whole M5200 sample, or to a 16-sample block for Single I/Q, so "
            "the programmed acquisition is never shorter than requested."
        )
        self.qcs_single_iq_radio = QtWidgets.QRadioButton(
            "Single I/Q value"
        )
        self.qcs_trace_radio = QtWidgets.QRadioButton("Trace")
        self.qcs_single_iq_radio.setToolTip(
            "Use a QCS IntegrationFilter and return one integrated complex "
            "I/Q value per shot."
        )
        self.qcs_trace_radio.setToolTip(
            "Return the requested raw M5200 trace samples for every shot. "
            "The trace can cross later segments; after the programmed AWG "
            "waveform ends, QCS pads its lane at baseline. "
            "Large traces and multi-point sweeps can require substantially "
            "more transfer and storage time."
        )
        self.qcs_acquisition_mode_group = QtWidgets.QButtonGroup(self)
        self.qcs_acquisition_mode_group.setExclusive(True)
        self.qcs_acquisition_mode_group.addButton(
            self.qcs_single_iq_radio
        )
        self.qcs_acquisition_mode_group.addButton(self.qcs_trace_radio)
        self.qcs_single_iq_radio.setChecked(True)
        qcs_acquisition_mode_layout = QtWidgets.QHBoxLayout()
        qcs_acquisition_mode_layout.setContentsMargins(0, 0, 0, 0)
        qcs_acquisition_mode_layout.addWidget(self.qcs_single_iq_radio)
        qcs_acquisition_mode_layout.addWidget(self.qcs_trace_radio)
        qcs_acquisition_mode_layout.addStretch(1)
        self.qcs_acquisition_mode_widget = QtWidgets.QWidget(self)
        self.qcs_acquisition_mode_widget.setLayout(
            qcs_acquisition_mode_layout
        )
        self.frequency_mhz = QtWidgets.QDoubleSpinBox()
        self.frequency_mhz.setRange(-10000.0, 10000.0)
        self.frequency_mhz.setDecimals(6)
        self.frequency_mhz.setValue(50.0)
        self.frequency_mhz.setSuffix(" MHz")
        self.attenuation_db = QtWidgets.QDoubleSpinBox()
        self.attenuation_db.setRange(0.0, 31.75)
        self.attenuation_db.setDecimals(2)
        self.attenuation_db.setSingleStep(0.25)
        self.attenuation_db.setValue(20.0)
        self.attenuation_db.setSuffix(" dB")
        self.dc_gain_db = QtWidgets.QDoubleSpinBox()
        self.dc_gain_db.setRange(-6.0, 26.0)
        self.dc_gain_db.setDecimals(2)
        self.dc_gain_db.setSingleStep(1.0)
        self.dc_gain_db.setValue(0.0)
        self.dc_gain_db.setSuffix(" dB")
        self.dc_measure_mode = QtWidgets.QCheckBox("Convert FIR I/Q to current")
        self.dc_measure_mode.setToolTip(
            "DC_In only: use identity ADC-to-voltage conversion, then divide "
            "I and Q by the measurement gain"
        )
        self.measurement_unit = QtWidgets.QComboBox()
        self.measurement_unit.addItem("ADC units", "adc")
        self.measurement_unit.addItem("Voltage", "voltage")
        self.measurement_unit.addItem("Current", "current")
        self.measurement_unit.setToolTip(
            "Select the stored and plotted FIR I/Q representation. Voltage "
            "and current are available for DC_In; current divides calibrated "
            "voltage by the measurement gain."
        )
        self.dc_measure_gain_v_per_a = QtWidgets.QDoubleSpinBox()
        self.dc_measure_gain_v_per_a.setRange(1.0e-9, 1.0e15)
        self.dc_measure_gain_v_per_a.setDecimals(6)
        self.dc_measure_gain_v_per_a.setValue(
            DEFAULT_DC_MEASURE_GAIN_V_PER_A
        )
        self.dc_measure_gain_v_per_a.setSuffix(" V/A")
        self.dc_measure_gain_v_per_a.setToolTip(
            "Current conversion: I_current = I_voltage / gain and "
            "Q_current = Q_voltage / gain"
        )
        self.input_condition_stack = QtWidgets.QStackedWidget()
        self.input_condition_stack.addWidget(self.attenuation_db)
        self.input_condition_stack.addWidget(self.dc_gain_db)
        self.input_condition_stack.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        self.input_condition_stack.setFixedHeight(
            max(
                self.attenuation_db.sizeHint().height(),
                self.dc_gain_db.sizeHint().height(),
            )
        )
        self.filter_type = QtWidgets.QComboBox()
        self.filter_type.addItems(["bypass", "lowpass", "highpass", "bandpass"])
        self.filter_cutoff = QtWidgets.QDoubleSpinBox()
        self.filter_cutoff.setRange(0.0, 100.0)
        self.filter_cutoff.setDecimals(6)
        self.filter_cutoff.setValue(2.5)
        self.filter_cutoff.setSuffix(" GHz")
        self.filter_bandwidth = QtWidgets.QDoubleSpinBox()
        self.filter_bandwidth.setRange(0.001, 100.0)
        self.filter_bandwidth.setDecimals(6)
        self.filter_bandwidth.setValue(1.0)
        self.filter_bandwidth.setSuffix(" GHz")
        self.nqz = QtWidgets.QSpinBox()
        self.nqz.setRange(1, 2)
        self.nqz.setValue(1)
        self.margin_samples = QtWidgets.QSpinBox()
        self.margin_samples.setRange(0, 10_000_000)
        self.margin_samples.setValue(1024)
        self.override_fpga_trigger_delay = QtWidgets.QCheckBox(
            "Override HWH default"
        )
        self.fpga_trigger_delay_us = QtWidgets.QDoubleSpinBox()
        self.fpga_trigger_delay_us.setRange(0.0, 10_000_000.0)
        self.fpga_trigger_delay_us.setDecimals(6)
        self.fpga_trigger_delay_us.setSuffix(" us")
        self.fpga_trigger_delay_us.setToolTip(
            "FPGA delay from trigger arrival to FIR-DDR storage. The value is "
            "converted to the register unit reported by the loaded HWH."
        )
        fpga_delay_row = QtWidgets.QHBoxLayout()
        fpga_delay_row.setContentsMargins(0, 0, 0, 0)
        fpga_delay_row.addWidget(self.override_fpga_trigger_delay)
        fpga_delay_row.addWidget(self.fpga_trigger_delay_us, 1)
        self.fpga_delay_widget = QtWidgets.QWidget()
        self.fpga_delay_widget.setLayout(fpga_delay_row)
        self.force_overwrite = QtWidgets.QCheckBox("Allow overwrite of reserved DDR range")
        self.post_run_read_delay = QtWidgets.QDoubleSpinBox()
        self.post_run_read_delay.setRange(0.0, 60.0)
        self.post_run_read_delay.setDecimals(6)
        self.post_run_read_delay.setValue(0.1)
        self.post_run_read_delay.setSuffix(" s")
        self.post_run_read_delay.setToolTip(
            "Wait after the tProcessor program completes before reading DDR"
        )
        self.dc_voltage_calibration_enabled = QtWidgets.QCheckBox(
            "Apply DC input voltage calibration"
        )
        self.dc_voltage_calibration_path = QtWidgets.QLineEdit()
        self.dc_voltage_calibration_path.setPlaceholderText(
            "QCoDeS calibration DB containing a DC Voltage run"
        )
        self.dc_voltage_calibration_browse = QtWidgets.QToolButton()
        self.dc_voltage_calibration_browse.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        self.dc_voltage_calibration_browse.setToolTip(
            "Choose DC voltage calibration database"
        )
        self.dc_voltage_calibration_run_id = QtWidgets.QSpinBox()
        self.dc_voltage_calibration_run_id.setRange(0, (1 << 31) - 1)
        self.dc_voltage_calibration_run_id.setSpecialValueText(
            "Latest matching channel/gain"
        )
        calibration_path_row = QtWidgets.QHBoxLayout()
        calibration_path_row.setContentsMargins(0, 0, 0, 0)
        calibration_path_row.addWidget(self.dc_voltage_calibration_path, 1)
        calibration_path_row.addWidget(self.dc_voltage_calibration_browse)
        self.calibration_path_widget = QtWidgets.QWidget()
        self.calibration_path_widget.setLayout(calibration_path_row)

        self.front_panel_preview = HardwareFrontPanelPreview(self)
        self.front_panel_preview.set_qcs_selection("acquisition", 0)
        self.front_panel_preview.activated.connect(
            lambda: self.front_panel_requested.emit(self)
        )
        front_panel_row = QtWidgets.QVBoxLayout()
        front_panel_row.setContentsMargins(0, 0, 0, 0)
        front_panel_row.setSpacing(2)
        front_panel_row.addWidget(QtWidgets.QLabel("Front panel:"))
        front_panel_row.addWidget(self.front_panel_preview)
        form.addRow(front_panel_row)
        # The physical input must remain selectable even while acquisition is
        # unchecked. QGroupBox normally disables every child in that state.
        self.front_panel_preview.setEnabled(True)
        self.toggled.connect(self._keep_front_panel_preview_enabled)
        QtCore.QTimer.singleShot(
            0,
            self._keep_front_panel_preview_enabled,
        )
        self.ro_ch_label = QtWidgets.QLabel("Readout index:")
        form.addRow(self.ro_ch_label, self.ro_ch)
        form.addRow("Input board:", self.input_board_type)
        form.addRow(
            "Acquisition result:",
            self.qcs_acquisition_mode_widget,
        )
        self.segment_label = QtWidgets.QLabel("Anchor SET:")
        form.addRow(self.segment_label, self.segment)
        self._delay_label = QtWidgets.QLabel()
        form.addRow(self._delay_label, self.delay)
        self.samples_label = QtWidgets.QLabel("Stored FIR samples:")
        form.addRow(self.samples_label, self.samples)
        self.qcs_acquisition_duration_label = QtWidgets.QLabel()
        form.addRow(
            self.qcs_acquisition_duration_label,
            self.qcs_acquisition_duration,
        )
        self.frequency_label = QtWidgets.QLabel(
            "Readout/DDC frequency:"
        )
        form.addRow(self.frequency_label, self.frequency_mhz)
        self.input_condition_label = QtWidgets.QLabel("Input attenuation:")
        form.addRow(self.input_condition_label, self.input_condition_stack)
        form.addRow("Stored/display unit:", self.measurement_unit)
        form.addRow("DC measurement gain:", self.dc_measure_gain_v_per_a)
        form.addRow(self.dc_voltage_calibration_enabled)
        form.addRow("Calibration DB:", self.calibration_path_widget)
        form.addRow("Calibration Run ID:", self.dc_voltage_calibration_run_id)
        form.addRow("Input filter:", self.filter_type)
        form.addRow("Filter cutoff/center:", self.filter_cutoff)
        form.addRow("Filter bandwidth:", self.filter_bandwidth)
        form.addRow("FIR input margin:", self.margin_samples)
        form.addRow(
            "FPGA trigger-to-store delay:",
            self.fpga_delay_widget,
        )
        form.addRow("DDR read delay after run:", self.post_run_read_delay)
        form.addRow(self.force_overwrite)
        self.fir_profile_note = QtWidgets.QLabel(
            "The loaded HWH selects the post-FIR DDR rate (1 MSPS or 50 kSPS)."
        )
        self.fir_profile_note.setWordWrap(True)
        self.fir_profile_note.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        form.addRow("HWH FIR DDR:", self.fir_profile_note)
        self.qcs_acquisition_note = QtWidgets.QLabel()
        self.qcs_acquisition_note.setWordWrap(True)
        self.qcs_acquisition_note.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        form.addRow("Acquisition mode:", self.qcs_acquisition_note)
        self._qick_only_rows = (
            self.input_board_type,
            self.input_condition_stack,
            self.measurement_unit,
            self.dc_measure_gain_v_per_a,
            self.dc_voltage_calibration_enabled,
            self.calibration_path_widget,
            self.dc_voltage_calibration_run_id,
            self.filter_type,
            self.filter_cutoff,
            self.filter_bandwidth,
            self.margin_samples,
            self.fpga_delay_widget,
            self.post_run_read_delay,
            self.force_overwrite,
            self.fir_profile_note,
        )

        self.refresh_segments(pulse)
        self.set_time_unit(time_unit, force=True)
        for widget in (
            self.ro_ch,
            self.input_board_type,
            self.segment,
            self.delay,
            self.samples,
            self.frequency_mhz,
            self.attenuation_db,
            self.dc_gain_db,
            self.measurement_unit,
            self.dc_measure_mode,
            self.dc_measure_gain_v_per_a,
            self.dc_voltage_calibration_enabled,
            self.dc_voltage_calibration_path,
            self.dc_voltage_calibration_run_id,
            self.filter_type,
            self.filter_cutoff,
            self.filter_bandwidth,
            self.nqz,
            self.margin_samples,
            self.override_fpga_trigger_delay,
            self.fpga_trigger_delay_us,
            self.post_run_read_delay,
            self.force_overwrite,
        ):
            signal = getattr(widget, "valueChanged", None)
            if signal is None:
                signal = getattr(widget, "currentIndexChanged", None)
            if signal is None:
                signal = getattr(widget, "toggled", None)
            if signal is None:
                signal = getattr(widget, "textChanged", None)
            signal.connect(self._emit_spec)
        self.samples.valueChanged.connect(self._update_fir_profile_note)
        self.samples.valueChanged.connect(
            self._update_qcs_acquisition_note
        )
        self.qcs_acquisition_duration.valueChanged.connect(
            self._qcs_acquisition_duration_changed
        )
        self.qcs_single_iq_radio.toggled.connect(
            self._qcs_single_iq_selected
        )
        self.qcs_trace_radio.toggled.connect(
            self._qcs_trace_selected
        )
        self.override_fpga_trigger_delay.toggled.connect(
            self._update_fpga_trigger_delay_controls
        )
        self.fpga_trigger_delay_us.valueChanged.connect(
            self._update_fir_profile_note
        )
        self.toggled.connect(self._emit_spec)
        self.input_board_type.currentTextChanged.connect(
            self._update_board_controls
        )
        self.measurement_unit.currentIndexChanged.connect(
            self._measurement_representation_changed
        )
        self.dc_measure_mode.toggled.connect(
            self._legacy_dc_measure_mode_changed
        )
        self.dc_voltage_calibration_enabled.toggled.connect(
            self._calibration_toggled
        )
        self.dc_voltage_calibration_browse.clicked.connect(
            self._browse_dc_voltage_calibration
        )
        self.ro_ch.valueChanged.connect(self._sync_front_panel_selection)
        self._update_board_controls()
        self._update_fpga_trigger_delay_controls()
        self.set_hardware_backend(self._hardware_backend)

    def _update_board_controls(self, *_args) -> None:
        rf_input = self.input_board_type.currentText() == "RF_In"
        if rf_input and self.measurement_unit.currentData() != "adc":
            with QtCore.QSignalBlocker(self.measurement_unit):
                self.measurement_unit.setCurrentIndex(
                    self.measurement_unit.findData("adc")
                )
        if rf_input and self.dc_voltage_calibration_enabled.isChecked():
            with QtCore.QSignalBlocker(self.dc_voltage_calibration_enabled):
                self.dc_voltage_calibration_enabled.setChecked(False)
        representation = str(self.measurement_unit.currentData())
        with QtCore.QSignalBlocker(self.dc_measure_mode):
            self.dc_measure_mode.setChecked(representation == "current")
        self.input_condition_stack.setCurrentIndex(0 if rf_input else 1)
        self.input_condition_label.setText(
            "Input attenuation:" if rf_input else "DC input gain:"
        )
        self.filter_type.setEnabled(rf_input)
        self.filter_cutoff.setEnabled(rf_input)
        self.filter_bandwidth.setEnabled(rf_input)
        self.measurement_unit.setEnabled(not rf_input)
        self.dc_measure_mode.setEnabled(not rf_input)
        self.dc_measure_gain_v_per_a.setEnabled(
            not rf_input and representation == "current"
        )
        calibration_enabled = (
            not rf_input and self.dc_voltage_calibration_enabled.isChecked()
        )
        self.dc_voltage_calibration_enabled.setEnabled(not rf_input)
        self.dc_voltage_calibration_path.setEnabled(calibration_enabled)
        self.dc_voltage_calibration_browse.setEnabled(calibration_enabled)
        self.dc_voltage_calibration_run_id.setEnabled(calibration_enabled)
        tooltip = (
            "RF_In onboard filter"
            if rf_input
            else "DC_In has no onboard RF filter; these values are ignored"
        )
        self.filter_type.setToolTip(tooltip)
        self.filter_cutoff.setToolTip(tooltip)
        self.filter_bandwidth.setToolTip(tooltip)

    def _measurement_representation_changed(self, *_args) -> None:
        self._update_board_controls()
        self._emit_spec()

    def _legacy_dc_measure_mode_changed(self, checked: bool) -> None:
        representation = (
            "current"
            if checked
            else (
                "voltage"
                if self.dc_voltage_calibration_enabled.isChecked()
                else "adc"
            )
        )
        with QtCore.QSignalBlocker(self.measurement_unit):
            self.measurement_unit.setCurrentIndex(
                self.measurement_unit.findData(representation)
            )
        self._update_board_controls()
        self._emit_spec()

    def _calibration_toggled(self, checked: bool) -> None:
        if (
            checked
            and self.input_board_type.currentText() == "DC_In"
            and self.measurement_unit.currentData() == "adc"
        ):
            with QtCore.QSignalBlocker(self.measurement_unit):
                self.measurement_unit.setCurrentIndex(
                    self.measurement_unit.findData("voltage")
                )
        self._update_board_controls()
        self._emit_spec()

    def _browse_dc_voltage_calibration(self) -> None:
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose DC voltage calibration database",
            self.dc_voltage_calibration_path.text().strip(),
            "QCoDeS SQLite database (*.db)",
        )
        if path:
            self.dc_voltage_calibration_path.setText(path)

    def set_front_panel_configuration(self, configuration) -> None:
        self._front_panel_configuration = configuration
        self.front_panel_preview.set_configuration(configuration)
        self._fir_sample_rate_hz = getattr(
            configuration,
            "fir_sample_rate_hz",
            None,
        )
        self._fir_trigger_delay_us = float(
            getattr(configuration, "fir_trigger_delay_us", 0.0)
        )
        self._fir_uses_fpga_trigger_delay = (
            str(
                getattr(
                    configuration,
                    "fir_trigger_delay_units",
                    "none",
                )
            )
            != "none"
        )
        if not self.override_fpga_trigger_delay.isChecked():
            with QtCore.QSignalBlocker(self.fpga_trigger_delay_us):
                self.fpga_trigger_delay_us.setValue(
                    self._fir_trigger_delay_us
                )
        self._update_fpga_trigger_delay_controls()
        self._update_fir_profile_note()
        self._sync_front_panel_selection()

    def _set_form_row_visible(
        self,
        field: QtWidgets.QWidget,
        visible: bool,
    ) -> None:
        """Show or hide both halves of one QFormLayout row."""

        label = self._form_layout.labelForField(field)
        if label is not None:
            label.setVisible(bool(visible))
        field.setVisible(bool(visible))

    def _update_backend_presentation(self) -> None:
        is_qick = self._hardware_backend == EXECUTION_BACKEND_QICK
        self.setTitle(
            "RF Readout 1" if is_qick else "QCS Acquisition 1"
        )
        self.ro_ch_label.setVisible(is_qick)
        self.ro_ch.setVisible(is_qick)
        for field in self._qick_only_rows:
            self._set_form_row_visible(field, is_qick)
        self._set_form_row_visible(
            self.qcs_acquisition_note,
            not is_qick,
        )
        self._set_form_row_visible(
            self.qcs_acquisition_mode_widget,
            not is_qick,
        )
        self._set_form_row_visible(self.samples, is_qick)
        self._set_form_row_visible(
            self.qcs_acquisition_duration,
            not is_qick,
        )
        self._set_form_row_visible(
            self.frequency_mhz,
            is_qick or self._qcs_hardware_demodulation,
        )
        self.segment_label.setText(
            "Anchor SET:" if is_qick else "Acquisition segment:"
        )
        self.samples_label.setText("Stored FIR samples:")
        self.samples.setSingleStep(1)
        self.qcs_acquisition_duration_label.setText(
            (
                "Total I/Q averaging time"
                if self._qcs_hardware_demodulation
                else "Raw trace duration"
            )
            + f" [{self._time_unit}]:"
        )
        self._update_qcs_acquisition_duration_constraints()
        self.frequency_label.setText(
            "Readout/DDC frequency:"
            if is_qick
            else "Integration-filter RF frequency:"
        )
        self._delay_label.setText(
            (
                "Trigger delay"
                if is_qick
                else "Acquisition pre-delay"
            )
            + f" [{self._time_unit}]:"
        )
        self.samples.setToolTip(
            "Number of post-FIR samples stored in QICK DDR"
        )
        self.frequency_mhz.setToolTip(
            "QICK digital downconversion frequency"
            if is_qick
            else (
                "RF frequency of the QCS IntegrationFilter used for hardware "
                "I/Q demodulation"
            )
        )
        self._update_qcs_acquisition_note()

    def set_hardware_backend(self, backend: str) -> None:
        backend = str(backend).strip().lower()
        if backend not in EXECUTION_BACKENDS:
            raise ValueError(f"unsupported execution backend {backend!r}")
        self._hardware_backend = backend
        self.front_panel_preview.set_backend(self._hardware_backend)
        self._keep_front_panel_preview_enabled()
        self._update_backend_presentation()

    def _keep_front_panel_preview_enabled(self, *_args) -> None:
        """Keep physical input and QCS result mode independently editable."""

        self.front_panel_preview.setEnabled(True)
        self.front_panel_preview.currentWidget().setEnabled(True)
        self.qcs_acquisition_mode_widget.setEnabled(
            self._qcs_acquisition_mode_editing_enabled
        )
        self.qcs_single_iq_radio.setEnabled(
            self._qcs_acquisition_mode_editing_enabled
        )
        self.qcs_trace_radio.setEnabled(
            self._qcs_acquisition_mode_editing_enabled
        )
        self.qcs_acquisition_duration.setEnabled(
            self._qcs_acquisition_mode_editing_enabled
        )

    def set_qcs_acquisition_mode_editing_enabled(
        self,
        enabled: bool,
    ) -> None:
        """Prevent result-mode changes while a hardware run is active."""

        self._qcs_acquisition_mode_editing_enabled = bool(enabled)
        self._keep_front_panel_preview_enabled()

    def set_qcs_hardware_demodulation(self, enabled: bool) -> None:
        enabled = bool(enabled)
        target = (
            self.qcs_single_iq_radio
            if enabled
            else self.qcs_trace_radio
        )
        if not target.isChecked():
            with (
                QtCore.QSignalBlocker(self.qcs_single_iq_radio),
                QtCore.QSignalBlocker(self.qcs_trace_radio),
            ):
                target.setChecked(True)
        self._qcs_hardware_demodulation = enabled
        if self._hardware_backend == EXECUTION_BACKEND_QCS:
            self._update_backend_presentation()

    def _qcs_single_iq_selected(self, checked: bool) -> None:
        if not checked:
            return
        self._qcs_hardware_demodulation = True
        if self._hardware_backend == EXECUTION_BACKEND_QCS:
            self._update_backend_presentation()
        self.qcs_acquisition_mode_changed.emit(True)
        self._emit_spec()

    def _qcs_trace_selected(self, checked: bool) -> None:
        if not checked:
            return
        self._qcs_hardware_demodulation = False
        if self._hardware_backend == EXECUTION_BACKEND_QCS:
            self._update_backend_presentation()
        self.qcs_acquisition_mode_changed.emit(False)
        self._emit_spec()

    def _qcs_acquisition_duration_seconds(self) -> float:
        requested_s = (
            _time_to_ns(
                self.qcs_acquisition_duration.value(),
                self._time_unit,
            )
            * 1.0e-9
        )
        requested_samples = (
            requested_s * self._qcs_acquisition_timing_rate_hz
        )
        nearest_sample = round(requested_samples)
        tolerance = max(1.0e-9, 8.0 * ulp(requested_samples))
        if abs(requested_samples - nearest_sample) <= tolerance:
            return nearest_sample / self._qcs_acquisition_timing_rate_hz
        return requested_s

    def _qcs_acquisition_sample_count(self) -> int:
        """Round requested time upward to a valid M5200 sample count."""

        requested_samples = (
            self._qcs_acquisition_duration_seconds()
            * self._qcs_acquisition_timing_rate_hz
        )
        nearest_sample = round(requested_samples)
        tolerance = max(1.0e-12, 8.0 * ulp(requested_samples))
        if abs(requested_samples - nearest_sample) <= tolerance:
            whole_samples = int(nearest_sample)
        else:
            whole_samples = int(ceil(requested_samples))
        whole_samples = max(1, whole_samples)
        if self._qcs_hardware_demodulation:
            block = QCS_M5200_INTEGRATION_BLOCK_SAMPLES
            whole_samples = ((whole_samples + block - 1) // block) * block
        maximum_samples = self._qcs_max_acquisition_samples()
        if whole_samples > maximum_samples:
            mode_name = (
                "total I/Q averaging"
                if self._qcs_hardware_demodulation
                else "raw trace"
            )
            raise ValueError(
                f"QCS {mode_name} time exceeds the GUI limit of "
                f"{maximum_samples:,} M5200 samples"
            )
        return whole_samples

    def _qcs_max_acquisition_samples(self) -> int:
        """Return the result-mode-specific QCS acquisition ceiling."""

        if self._qcs_hardware_demodulation:
            duration_samples = int(
                floor(
                    QCS_MAX_TOTAL_IQ_AVERAGING_DURATION_S
                    * self._qcs_acquisition_timing_rate_hz
                )
            )
            return min(
                QCS_MAX_TOTAL_IQ_AVERAGING_SAMPLES,
                duration_samples,
            )
        return QCS_NOISE_MAX_RAW_TRACE_SAMPLES

    def _update_qcs_acquisition_duration_constraints(self) -> None:
        quantum = (
            QCS_M5200_INTEGRATION_BLOCK_SAMPLES
            if self._qcs_hardware_demodulation
            else 1
        )
        maximum_samples = (
            self._qcs_max_acquisition_samples() // quantum * quantum
        )
        seconds_per_sample = 1.0 / self._qcs_acquisition_timing_rate_hz
        with QtCore.QSignalBlocker(self.qcs_acquisition_duration):
            self.qcs_acquisition_duration.setRange(
                _time_from_ns(
                    seconds_per_sample * 1.0e9,
                    self._time_unit,
                ),
                _time_from_ns(
                    maximum_samples * seconds_per_sample * 1.0e9,
                    self._time_unit,
                ),
            )
            self.qcs_acquisition_duration.setSingleStep(
                _time_from_ns(
                    quantum * seconds_per_sample * 1.0e9,
                    self._time_unit,
                )
            )
        if self._qcs_hardware_demodulation:
            self.qcs_acquisition_duration.setToolTip(
                "Total effective M5200 I/Q averaging time. Up to 100 us is "
                "executed in one bounded QCS pass. Longer requests are "
                "accumulated from repeated bounded passes and reduced to one "
                "sample-weighted complex I/Q result per point/repetition. "
                "The selected AWG segment must cover the pre-delay plus one "
                "pass (at most 100 us), not the complete accumulated time. "
                "This is not one continuous raw trace."
            )
        else:
            maximum_duration_s = (
                maximum_samples / self._qcs_acquisition_timing_rate_hz
            )
            self.qcs_acquisition_duration.setToolTip(
                "Raw M5200 trace duration. Every sample is returned and "
                "stored, so this mode keeps its separate "
                f"{maximum_samples:,}-sample "
                f"({maximum_duration_s * 1.0e3:.6g} ms) payload limit."
            )

    def _qcs_acquisition_duration_changed(self, *_args) -> None:
        self._update_qcs_acquisition_note()
        self._emit_spec()

    def set_qcs_acquisition_timing_rate(self, sample_rate_hz: float) -> None:
        sample_rate_hz = float(sample_rate_hz)
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
            raise ValueError(
                "QCS acquisition timing rate must be positive and finite"
            )
        self._qcs_acquisition_timing_rate_hz = sample_rate_hz
        self._update_qcs_acquisition_duration_constraints()
        self._update_qcs_acquisition_note()

    def set_qcs_front_panel_configuration(
        self,
        configuration: Optional[Mapping[str, object]],
    ) -> None:
        self._qcs_front_panel_configuration = configuration
        self.front_panel_preview.set_qcs_configuration(configuration)
        self.front_panel_preview.set_qcs_selection("acquisition", 0)

    def _update_fpga_trigger_delay_controls(self, *_args) -> None:
        supported = self._fir_uses_fpga_trigger_delay is not False
        self.override_fpga_trigger_delay.setEnabled(supported)
        self.fpga_trigger_delay_us.setEnabled(
            supported and self.override_fpga_trigger_delay.isChecked()
        )
        self._update_fir_profile_note()

    def _update_fir_profile_note(self, *_args) -> None:
        if self._fir_sample_rate_hz is None:
            self.fir_profile_note.setText(
                "FIR DDR rate is unavailable in the loaded HWH metadata"
            )
            return
        sample_period_us = 1_000_000.0 / float(self._fir_sample_rate_hz)
        trace_us = self.samples.value() * sample_period_us
        if self._fir_uses_fpga_trigger_delay:
            if self.override_fpga_trigger_delay.isChecked():
                delay = (
                    f"; FPGA delay override "
                    f"{self.fpga_trigger_delay_us.value():g} us"
                )
            else:
                delay = f"; HWH FPGA delay {self._fir_trigger_delay_us:g} us"
        else:
            delay = "; no FPGA trigger-delay register"
        self.fir_profile_note.setText(
            f"{format_sample_rate_hz(self._fir_sample_rate_hz)}, "
            f"{sample_period_us:g} us/sample; "
            f"{self.samples.value():,} samples = {trace_us:g} us"
            f"{delay}"
        )

    def _update_qcs_acquisition_note(self, *_args) -> None:
        sample_count = self._qcs_acquisition_sample_count()
        requested_duration = self.qcs_acquisition_duration.value()
        effective_duration_s = (
            sample_count / self._qcs_acquisition_timing_rate_hz
        )
        effective_duration = _time_from_ns(
            effective_duration_s * 1.0e9,
            self._time_unit,
        )
        timing_text = (
            f"Requested {requested_duration:.12g} {self._time_unit} -> "
            f"{sample_count:,} M5200 samples at "
            f"{format_sample_rate_hz(self._qcs_acquisition_timing_rate_hz)}; "
            f"programmed time {effective_duration:.12g} {self._time_unit}. "
        )
        if self._qcs_hardware_demodulation:
            pass_count = max(
                1,
                int(
                    ceil(
                        sample_count
                        / QCS_SPARAMETER_MAX_INTEGRATION_SAMPLES
                    )
                ),
            )
            if pass_count == 1:
                pass_text = "one bounded QCS execution pass"
            else:
                pass_text = (
                    f"{pass_count:,} bounded QCS execution passes; the GUI "
                    "can stop between passes and retains the completed "
                    "running average"
                )
            self.qcs_acquisition_note.setText(
                timing_text
                + "Hardware demodulation returns one accumulated complex "
                "I/Q value per point/repetition. The requested total uses "
                f"{pass_text}; each pass is at most 100 us and is internally "
                "split into hardware-safe IntegrationFilter windows. The "
                "selected AWG segment must cover one pass plus pre-delay."
            )
            return
        self.qcs_acquisition_note.setText(
            timing_text
            + "Raw acquisition returns every trace sample. The selected segment "
            "anchors the trace start; the trace may continue across later "
            "segments or beyond the end of the programmed waveform. Beyond "
            "the waveform, QCS pads AWG lanes at baseline; add an explicit "
            "final segment when voltage must remain applied."
        )

    def _sync_front_panel_selection(self, *_args) -> None:
        self.front_panel_preview.set_qcs_selection("acquisition", 0)
        self.front_panel_preview.set_channels(input_ch=self.ro_ch.value())
        if self._front_panel_configuration is None:
            return
        for port in self._front_panel_configuration.inputs:
            if self.ro_ch.value() in port.qick_channels:
                if port.board_type in QICK_INPUT_BOARD_TYPES:
                    self.input_board_type.setCurrentText(port.board_type)
                return

    def apply_front_panel_settings(self, values: Mapping[str, object]) -> None:
        self.ro_ch.setValue(int(values["readout_ch"]))
        self.input_board_type.setCurrentText(str(values["input_board_type"]))
        self.attenuation_db.setValue(float(values["readout_attenuation_db"]))
        self.dc_gain_db.setValue(float(values["readout_dc_gain_db"]))
        self.filter_type.setCurrentText(str(values["readout_filter_type"]))
        self.filter_cutoff.setValue(float(values["readout_filter_cutoff_ghz"]))
        self.filter_bandwidth.setValue(
            float(values["readout_filter_bandwidth_ghz"])
        )
        self.nqz.setValue(int(values.get("readout_nqz", self.nqz.value())))
        self._update_board_controls()
        self._sync_front_panel_selection()
        self._emit_spec()

    def set_time_unit(self, unit: str, *, force: bool = False) -> None:
        if unit not in TIME_UNIT_NS:
            raise ValueError(f"unsupported time unit {unit!r}")
        if not force and unit == self._time_unit:
            return
        delay_ns = _time_to_ns(self.delay.value(), self._time_unit)
        qcs_acquisition_duration_ns = _time_to_ns(
            self.qcs_acquisition_duration.value(),
            self._time_unit,
        )
        self._time_unit = unit
        with QtCore.QSignalBlocker(self.delay):
            self.delay.setValue(_time_from_ns(delay_ns, unit))
            self.delay.setSuffix(f" {unit}")
        self._update_qcs_acquisition_duration_constraints()
        with QtCore.QSignalBlocker(self.qcs_acquisition_duration):
            self.qcs_acquisition_duration.setValue(
                _time_from_ns(qcs_acquisition_duration_ns, unit)
            )
            self.qcs_acquisition_duration.setSuffix(f" {unit}")
        self._delay_label.setText(
            (
                "Trigger delay"
                if self._hardware_backend == EXECUTION_BACKEND_QICK
                else "Acquisition pre-delay"
            )
            + f" [{unit}]:"
        )
        self.qcs_acquisition_duration_label.setText(
            (
                "Total I/Q averaging time"
                if self._qcs_hardware_demodulation
                else "Raw trace duration"
            )
            + f" [{unit}]:"
        )
        self._update_qcs_acquisition_note()

    def refresh_segments(self, pulse: Optional[PulseSequence] = None) -> None:
        if pulse is not None:
            self._pulse = pulse
        previous = self.segment.currentData()
        with QtCore.QSignalBlocker(self.segment):
            self.segment.clear()
            for index, _segment in enumerate(self._pulse.flat_segments()):
                self.segment.addItem(
                    self._pulse.segment_name(index),
                    f"set_{index}",
                )
            match = self.segment.findData(previous)
            if match >= 0:
                self.segment.setCurrentIndex(match)

    def remap_segment_reference(
        self,
        pulse: PulseSequence,
        operation: str,
        segment_index: int,
    ) -> Tuple[str, Optional[str]]:
        """Keep acquisition attached to its logical SET after an edit.

        If its target SET is deleted, acquisition is disabled and a nearby
        remaining SET is selected only as an explicit future-edit fallback.
        This prevents Qt from silently acquiring the first SET while a stale
        readout specification still points at the deleted segment.
        """

        self._pulse = pulse
        old_name = str(self.segment.currentData())
        new_name = _remap_set_segment_name(
            old_name,
            operation,
            segment_index,
        )
        fallback_index = min(
            max(0, int(segment_index)),
            max(0, pulse.set_count - 1),
        )
        with QtCore.QSignalBlocker(self):
            self.refresh_segments(pulse)
            if new_name is None:
                self.setChecked(False)
                target_name = f"set_{fallback_index}"
            else:
                target_name = new_name
            match = self.segment.findData(target_name)
            if match < 0:
                raise RuntimeError(
                    f"cannot remap acquisition anchor {old_name!r} after "
                    f"{operation} at SET {segment_index}"
                )
            self.segment.setCurrentIndex(match)
        self._emit_spec()
        return old_name, new_name

    def configured_spec(self) -> QickDdrReadoutSpec:
        """Return the editor values even when capture is disabled."""
        return QickDdrReadoutSpec(
            ro_ch=self.ro_ch.value(),
            segment_name=str(self.segment.currentData()),
            delay_us=_time_to_ns(self.delay.value(), self._time_unit) / 1000.0,
            samples_per_trigger=(
                self._qcs_acquisition_sample_count()
                if self._hardware_backend == EXECUTION_BACKEND_QCS
                else self.samples.value()
            ),
            readout_frequency_mhz=self.frequency_mhz.value(),
            margin_input_samples=self.margin_samples.value(),
            fpga_trigger_delay_us=(
                self.fpga_trigger_delay_us.value()
                if (
                    self.override_fpga_trigger_delay.isChecked()
                    and self._fir_uses_fpga_trigger_delay is not False
                )
                else None
            ),
            force_overwrite=self.force_overwrite.isChecked(),
            post_run_read_delay_seconds=self.post_run_read_delay.value(),
            attenuation_db=self.attenuation_db.value(),
            filter_type=self.filter_type.currentText(),
            filter_cutoff=self.filter_cutoff.value(),
            filter_bandwidth=self.filter_bandwidth.value(),
            input_board_type=self.input_board_type.currentText(),
            dc_gain_db=self.dc_gain_db.value(),
            dc_measure_mode=self.dc_measure_mode.isChecked(),
            dc_measure_gain_v_per_a=self.dc_measure_gain_v_per_a.value(),
            measurement_representation=str(
                self.measurement_unit.currentData()
            ),
            dc_voltage_calibration_enabled=(
                self.dc_voltage_calibration_enabled.isChecked()
            ),
            dc_voltage_calibration_database_path=(
                self.dc_voltage_calibration_path.text().strip()
            ),
            dc_voltage_calibration_run_id=(
                self.dc_voltage_calibration_run_id.value()
            ),
            nqz=self.nqz.value(),
        )

    def spec(self) -> Optional[QickDdrReadoutSpec]:
        if not self.isChecked():
            return None
        return self.configured_spec()

    def set_dc_measurement(self, enabled: bool, gain_v_per_a: float) -> None:
        """Update a readout's DC current conversion from linked controls."""
        if enabled and self.input_board_type.currentText() != "DC_In":
            raise ValueError("DC measure mode requires the DC_In input board")
        with QtCore.QSignalBlocker(self.dc_measure_mode):
            self.dc_measure_mode.setChecked(bool(enabled))
        representation = "current" if enabled else "adc"
        with QtCore.QSignalBlocker(self.measurement_unit):
            self.measurement_unit.setCurrentIndex(
                self.measurement_unit.findData(representation)
            )
        with QtCore.QSignalBlocker(self.dc_measure_gain_v_per_a):
            self.dc_measure_gain_v_per_a.setValue(float(gain_v_per_a))
        self._update_board_controls()
        self._emit_spec()

    def set_dc_voltage_calibration(
        self,
        database_path: str,
        run_id: int,
    ) -> None:
        """Select a completed DC calibration for subsequent measurements."""
        if self.input_board_type.currentText() != "DC_In":
            self.input_board_type.setCurrentText("DC_In")
        self.set_dc_voltage_calibration_selection(
            True,
            database_path,
            run_id,
        )

    def set_dc_voltage_calibration_selection(
        self,
        enabled: bool,
        database_path: str,
        run_id: int,
    ) -> None:
        """Update a readout's DC calibration from linked controls."""
        if enabled and self.input_board_type.currentText() != "DC_In":
            raise ValueError("DC voltage calibration requires the DC_In input board")
        with QtCore.QSignalBlocker(self.dc_voltage_calibration_path):
            self.dc_voltage_calibration_path.setText(str(database_path))
        with QtCore.QSignalBlocker(self.dc_voltage_calibration_run_id):
            self.dc_voltage_calibration_run_id.setValue(int(run_id))
        with QtCore.QSignalBlocker(self.dc_voltage_calibration_enabled):
            self.dc_voltage_calibration_enabled.setChecked(bool(enabled))
        if enabled and self.measurement_unit.currentData() == "adc":
            with QtCore.QSignalBlocker(self.measurement_unit):
                self.measurement_unit.setCurrentIndex(
                    self.measurement_unit.findData("voltage")
                )
        self._update_board_controls()
        self._emit_spec()

    def settings_dict(self) -> dict:
        spec = self.configured_spec()
        return {
            "enabled": self.isChecked(),
            "ro_ch": spec.ro_ch,
            "segment_name": spec.segment_name,
            "delay_us": spec.delay_us,
            # QICK and QCS keep independent dormant controls. QCS derives its
            # sample count from the requested time when a program is built.
            "samples_per_trigger": self.samples.value(),
            "qcs_acquisition_duration_s": (
                self._qcs_acquisition_duration_seconds()
            ),
            "readout_frequency_mhz": spec.readout_frequency_mhz,
            "margin_input_samples": spec.margin_input_samples,
            "fpga_trigger_delay_us": spec.fpga_trigger_delay_us,
            "force_overwrite": spec.force_overwrite,
            "post_run_read_delay_seconds": spec.post_run_read_delay_seconds,
            "input_board_type": spec.input_board_type,
            "attenuation_db": spec.attenuation_db,
            "dc_gain_db": spec.dc_gain_db,
            "dc_measure_mode": spec.dc_measure_mode,
            "dc_measure_gain_v_per_a": spec.dc_measure_gain_v_per_a,
            "measurement_representation": (
                spec.effective_measurement_representation
            ),
            "dc_voltage_calibration_enabled": (
                spec.dc_voltage_calibration_enabled
            ),
            "dc_voltage_calibration_database_path": (
                spec.dc_voltage_calibration_database_path
            ),
            "dc_voltage_calibration_run_id": (
                spec.dc_voltage_calibration_run_id
            ),
            "filter_type": spec.filter_type,
            "filter_cutoff": spec.filter_cutoff,
            "filter_bandwidth": spec.filter_bandwidth,
            "nqz": spec.nqz,
        }

    def load_settings(self, data: dict) -> None:
        if not isinstance(data, dict):
            raise TypeError("RF readout setting must be a JSON object")
        enabled = data.get("enabled", False)
        force_overwrite = data.get("force_overwrite", False)
        if not isinstance(enabled, bool):
            raise TypeError("RF readout enabled must be boolean")
        if not isinstance(force_overwrite, bool):
            raise TypeError("RF readout force_overwrite must be boolean")
        dc_measure_mode = data.get("dc_measure_mode", False)
        if not isinstance(dc_measure_mode, bool):
            raise TypeError("RF readout dc_measure_mode must be boolean")
        raw_representation = data.get("measurement_representation")
        if raw_representation is None or raw_representation == "auto":
            raw_representation = (
                "current"
                if dc_measure_mode
                else (
                    "voltage"
                    if bool(
                        data.get(
                            "dc_voltage_calibration_enabled",
                            False,
                        )
                    )
                    else "adc"
                )
            )
        spec = QickDdrReadoutSpec(
            ro_ch=int(data["ro_ch"]),
            segment_name=_resolve_stored_set_segment_name(
                data["segment_name"],
                tuple(
                    self.segment.itemData(index)
                    for index in range(self.segment.count())
                ),
                label="RF readout",
            ),
            delay_us=float(data["delay_us"]),
            samples_per_trigger=int(data["samples_per_trigger"]),
            readout_frequency_mhz=float(data.get("readout_frequency_mhz", 0.0)),
            margin_input_samples=int(data.get("margin_input_samples", 1024)),
            fpga_trigger_delay_us=(
                None
                if data.get("fpga_trigger_delay_us") is None
                else float(data["fpga_trigger_delay_us"])
            ),
            force_overwrite=force_overwrite,
            post_run_read_delay_seconds=float(
                data.get("post_run_read_delay_seconds", 0.1)
            ),
            attenuation_db=float(data.get("attenuation_db", 20.0)),
            filter_type=str(data.get("filter_type", "bypass")),
            filter_cutoff=float(data.get("filter_cutoff", 2.5)),
            filter_bandwidth=float(data.get("filter_bandwidth", 1.0)),
            input_board_type=str(data.get("input_board_type", "RF_In")),
            dc_gain_db=float(data.get("dc_gain_db", 0.0)),
            dc_measure_mode=dc_measure_mode,
            dc_measure_gain_v_per_a=float(
                data.get(
                    "dc_measure_gain_v_per_a",
                    DEFAULT_DC_MEASURE_GAIN_V_PER_A,
                )
            ),
            measurement_representation=str(raw_representation),
            dc_voltage_calibration_enabled=bool(
                data.get("dc_voltage_calibration_enabled", False)
            ),
            dc_voltage_calibration_database_path=str(
                data.get("dc_voltage_calibration_database_path", "")
            ),
            dc_voltage_calibration_run_id=int(
                data.get("dc_voltage_calibration_run_id", 0)
            ),
            nqz=int(data.get("nqz", 1)),
        )
        raw_qcs_duration_s = data.get("qcs_acquisition_duration_s")
        qcs_duration_s = (
            spec.samples_per_trigger / self._qcs_acquisition_timing_rate_hz
            if raw_qcs_duration_s is None
            else float(raw_qcs_duration_s)
        )
        if not np.isfinite(qcs_duration_s) or qcs_duration_s <= 0.0:
            raise ValueError("QCS acquisition duration must be positive and finite")
        segment = self.segment.findData(spec.segment_name)
        if segment < 0:
            raise ValueError(f"unknown RF readout anchor {spec.segment_name!r}")
        filter_index = self.filter_type.findText(spec.filter_type)
        if filter_index < 0:
            raise ValueError(f"unknown RF input filter {spec.filter_type!r}")
        with QtCore.QSignalBlocker(self):
            self.ro_ch.setValue(spec.ro_ch)
            self.segment.setCurrentIndex(segment)
            self.delay.setValue(
                _time_from_ns(spec.delay_us * 1000.0, self._time_unit)
            )
            self.samples.setValue(spec.samples_per_trigger)
            with QtCore.QSignalBlocker(self.qcs_acquisition_duration):
                self.qcs_acquisition_duration.setValue(
                    _time_from_ns(qcs_duration_s * 1.0e9, self._time_unit)
                )
            self.frequency_mhz.setValue(spec.readout_frequency_mhz)
            self.input_board_type.setCurrentText(spec.input_board_type)
            self.margin_samples.setValue(spec.margin_input_samples)
            self.override_fpga_trigger_delay.setChecked(
                spec.fpga_trigger_delay_us is not None
            )
            if spec.fpga_trigger_delay_us is not None:
                self.fpga_trigger_delay_us.setValue(
                    spec.fpga_trigger_delay_us
                )
            elif self._fir_sample_rate_hz is not None:
                self.fpga_trigger_delay_us.setValue(
                    self._fir_trigger_delay_us
                )
            self.force_overwrite.setChecked(spec.force_overwrite)
            self.post_run_read_delay.setValue(
                spec.post_run_read_delay_seconds
            )
            self.attenuation_db.setValue(spec.attenuation_db)
            self.dc_gain_db.setValue(spec.dc_gain_db)
            self.dc_measure_mode.setChecked(spec.dc_measure_mode)
            self.measurement_unit.setCurrentIndex(
                self.measurement_unit.findData(
                    spec.effective_measurement_representation
                )
            )
            self.dc_measure_gain_v_per_a.setValue(
                spec.dc_measure_gain_v_per_a
            )
            self.dc_voltage_calibration_enabled.setChecked(
                spec.dc_voltage_calibration_enabled
            )
            self.dc_voltage_calibration_path.setText(
                spec.dc_voltage_calibration_database_path
            )
            self.dc_voltage_calibration_run_id.setValue(
                spec.dc_voltage_calibration_run_id
            )
            self.filter_type.setCurrentIndex(filter_index)
            self.filter_cutoff.setValue(spec.filter_cutoff)
            self.filter_bandwidth.setValue(spec.filter_bandwidth)
            self.nqz.setValue(spec.nqz)
            self.setChecked(enabled)
        self._update_board_controls()
        self._update_fpga_trigger_delay_controls()
        self._emit_spec()

    def apply_path_settings(self, values: Mapping[str, object]) -> None:
        """Apply a committed RF path to this Experiment readout editor."""
        self.ro_ch.setValue(int(values["readout_ch"]))
        self.input_board_type.setCurrentText(str(values["input_board_type"]))
        self.attenuation_db.setValue(float(values["readout_attenuation_db"]))
        self.dc_gain_db.setValue(float(values["readout_dc_gain_db"]))
        self.filter_type.setCurrentText(str(values["readout_filter_type"]))
        self.filter_cutoff.setValue(float(values["readout_filter_cutoff_ghz"]))
        self.filter_bandwidth.setValue(
            float(values["readout_filter_bandwidth_ghz"])
        )
        self.nqz.setValue(int(values.get("readout_nqz", self.nqz.value())))
        self._update_board_controls()
        self._emit_spec()

    def _emit_spec(self, *_args) -> None:
        try:
            spec = self.spec()
        except (TypeError, ValueError):
            return
        self.spec_changed.emit(spec)


class QickSetupDialog(QtWidgets.QDialog):
    """Edit the QICK connection and timing values shared by every tab."""

    def __init__(
        self,
        *,
        connection: QickConnectionConfig,
        fabric_mhz: float,
        tproc_mhz: float,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("QICK Connection and Clocks")
        self.setMinimumWidth(460)
        form = QtWidgets.QFormLayout(self)

        self.qick_host = QtWidgets.QLineEdit(connection.host, self)
        self.ns_port = QtWidgets.QSpinBox(self)
        self.ns_port.setRange(1, 65535)
        self.ns_port.setValue(connection.ns_port)
        self.proxy_name = QtWidgets.QLineEdit(connection.proxy_name, self)
        self.fabric_mhz = QtWidgets.QDoubleSpinBox(self)
        self.fabric_mhz.setRange(1.0, 5000.0)
        self.fabric_mhz.setDecimals(6)
        self.fabric_mhz.setSuffix(" MHz")
        self.fabric_mhz.setValue(float(fabric_mhz))
        self.tproc_mhz = QtWidgets.QDoubleSpinBox(self)
        self.tproc_mhz.setRange(1.0, 5000.0)
        self.tproc_mhz.setDecimals(6)
        self.tproc_mhz.setSuffix(" MHz")
        self.tproc_mhz.setValue(float(tproc_mhz))

        form.addRow("QICK IP/host:", self.qick_host)
        form.addRow("Pyro nameserver port:", self.ns_port)
        form.addRow("Pyro proxy name:", self.proxy_name)
        form.addRow("AWG fabric clock:", self.fabric_mhz)
        form.addRow("tProcessor clock:", self.tproc_mhz)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> Tuple[QickConnectionConfig, float, float]:
        connection = QickConnectionConfig(
            host=self.qick_host.text().strip(),
            ns_port=self.ns_port.value(),
            proxy_name=self.proxy_name.text().strip(),
        )
        return connection, self.fabric_mhz.value(), self.tproc_mhz.value()


class ExperimentPanel(QtWidgets.QWidget):
    """QCoDeS database, AWG scale, and direct-run controls."""

    run_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    show_program_requested = QtCore.pyqtSignal()
    awg_metadata_requested = QtCore.pyqtSignal()
    sweep_axes_changed = QtCore.pyqtSignal()
    sweep_update_requested = QtCore.pyqtSignal(object, float, float, int)
    sweep_remove_requested = QtCore.pyqtSignal(object)
    bias_t_changed = QtCore.pyqtSignal(bool, str, float, str, float, float)
    ddr_capacity_refresh_requested = QtCore.pyqtSignal()
    qcs_hardware_configuration_requested = QtCore.pyqtSignal()
    qcs_output_count_changed = QtCore.pyqtSignal(int)
    qcs_output_sync_warning = QtCore.pyqtSignal(str)
    run_state_changed = QtCore.pyqtSignal(bool)
    execution_backend_changed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        fabric_mhz: float,
        tproc_mhz: float,
        full_scale_mv: float,
        awg_channels: Sequence[int],
        repetitions: int,
        iq_repetition_policy: str = IQ_REPETITION_POLICY_PRESERVE,
        bias_t_enabled: bool = False,
        bias_t_compensation_type: str = "dc",
        bias_t_compensation_mv: Optional[float] = None,
        bias_t_mode: str = "fixed_voltage",
        bias_t_duration_us: float = DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
        bias_t_filter_tau_us: float = DEFAULT_BIAS_T_FILTER_TAU_US,
        compile_validation_mode: str = DEFAULT_GUI_COMPILE_VALIDATION_MODE,
        parent=None,
    ):
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget(scroll)
        form = QtWidgets.QFormLayout(content)
        self._form_layout = form
        scroll.setWidget(content)
        outer.addWidget(scroll)

        self.qick_host = QtWidgets.QLineEdit(DEFAULT_QICK_HOST)
        self.ns_port = QtWidgets.QSpinBox()
        self.ns_port.setRange(1, 65535)
        self.ns_port.setValue(DEFAULT_QICK_NS_PORT)
        self.proxy_name = QtWidgets.QLineEdit(DEFAULT_QICK_PROXY_NAME)

        self.backend_selector = QtWidgets.QComboBox(self)
        self.backend_selector.addItem("Keysight QCS", EXECUTION_BACKEND_QCS)
        self.backend_selector.addItem("QICK", EXECUTION_BACKEND_QICK)
        self.backend_selector.setCurrentIndex(
            self.backend_selector.findData(DEFAULT_EXECUTION_BACKEND)
        )
        self.backend_selector.hide()
        self.execution_system_label = QtWidgets.QLabel("Keysight QCS")
        self.execution_system_label.setToolTip(
            "Hardware execution follows the applied M5000 configuration. "
            "Legacy QICK settings remain loadable for compatibility."
        )
        self._last_execution_backend = None
        self.qcs_connection_group = QtWidgets.QGroupBox(
            "Keysight QCS connection and channel mapping"
        )
        qcs_form = QtWidgets.QFormLayout(self.qcs_connection_group)
        self.qcs_mapper_path = QtWidgets.QLineEdit(DEFAULT_QCS_MAPPER_PATH)
        self.qcs_mapper_path.setPlaceholderText(
            "Path to the native Keysight QCS .qcs ChannelMapper file"
        )
        self.browse_qcs_mapper = QtWidgets.QToolButton()
        self.browse_qcs_mapper.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        self.browse_qcs_mapper.setToolTip(
            "Choose a Keysight QCS channel-mapper file"
        )
        self.browse_qcs_mapper.clicked.connect(self._browse_qcs_mapper)
        self.browse_qcs_mapper.hide()
        qcs_mapper_row = QtWidgets.QHBoxLayout()
        qcs_mapper_row.addWidget(self.qcs_mapper_path, 1)
        qcs_mapper_row.addWidget(self.browse_qcs_mapper)
        self.qcs_dc_channel_names = QtWidgets.QLineEdit(
            ", ".join(
                f"dc_ch_{index + 1}" for index in range(len(awg_channels))
            )
        )
        self.qcs_dc_channel_names.setPlaceholderText(
            "dc_ch_1, dc_ch_2 (one virtual name per AWG output)"
        )
        self.qcs_dc_full_scale_v = QtWidgets.QDoubleSpinBox()
        self.qcs_dc_full_scale_v.setRange(0.001, 1.0e6)
        self.qcs_dc_full_scale_v.setDecimals(6)
        self.qcs_dc_full_scale_v.setSuffix(" V")
        self.qcs_dc_full_scale_v.setValue(DEFAULT_QCS_FULL_SCALE_V)
        self.qcs_dc_full_scale_v.setToolTip(
            "Effective M5301A connector voltage represented by QCS DC "
            "amplitude +1.0. Enter the present AWG full scale here; a "
            "successful M5301A / 1 Mohm-scope calibration updates this same "
            "value using the fitted origin gain."
        )
        self.qcs_rf_channel_names = QtWidgets.QLineEdit()
        self.qcs_rf_channel_names.setPlaceholderText(
            "0=rf_drive, 2=rf_probe"
        )
        self.qcs_acquisition_channel_name = QtWidgets.QLineEdit()
        self.qcs_acquisition_channel_name.setPlaceholderText(
            "QCS digitizer virtual channel (required to run)"
        )
        for derived_field in (
            self.qcs_mapper_path,
            self.qcs_dc_channel_names,
            self.qcs_rf_channel_names,
            self.qcs_acquisition_channel_name,
        ):
            derived_field.setReadOnly(True)
            derived_field.setToolTip(
                "Derived from Configure M5000 Hardware; edit the front-panel "
                "configuration to change this value."
            )
        self.qcs_hw_demod = QtWidgets.QCheckBox(
            "Use hardware I/Q demodulation",
            self.qcs_connection_group,
        )
        self.qcs_hw_demod.setChecked(True)
        self.qcs_hw_demod.setToolTip(
            "Pass an IntegrationFilter to the QCS acquisition so the "
            "hardware returns integrated I/Q data. Disable this to request "
            "raw trace data."
        )
        # Compatibility state for settings and execution code. The user-facing
        # selector lives in the QCS Acquisition tab as two explicit radios.
        self.qcs_hw_demod.hide()
        # Kept as a fixed, hidden compatibility value for settings and code
        # paths that predate the hardware-derived M5200 rate. Users do not
        # configure or need to inspect it in the Experiment panel.
        self.qcs_sample_rate_hz = QtWidgets.QDoubleSpinBox(
            self.qcs_connection_group
        )
        self.qcs_sample_rate_hz.setRange(
            DEFAULT_QCS_SAMPLE_RATE_HZ,
            DEFAULT_QCS_SAMPLE_RATE_HZ,
        )
        self.qcs_sample_rate_hz.setDecimals(0)
        self.qcs_sample_rate_hz.setGroupSeparatorShown(True)
        self.qcs_sample_rate_hz.setSuffix(" Hz")
        self.qcs_sample_rate_hz.setSpecialValueText(
            format_sample_rate_hz(DEFAULT_QCS_SAMPLE_RATE_HZ)
        )
        self.qcs_sample_rate_hz.setValue(DEFAULT_QCS_SAMPLE_RATE_HZ)
        self.qcs_sample_rate_hz.setReadOnly(True)
        self.qcs_sample_rate_hz.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.qcs_sample_rate_hz.setToolTip(
            "Keysight QCS 2.5.5 defines the M5200 digitizer sample rate as "
            "4.8 GSa/s. The mapped hardware rate is verified again when the "
            "program is compiled."
        )
        self.qcs_sample_rate_hz.hide()
        self.qcs_init_time_us = QtWidgets.QDoubleSpinBox()
        self.qcs_init_time_us.setRange(0.0, 1.0e9)
        self.qcs_init_time_us.setDecimals(9)
        self.qcs_init_time_us.setSingleStep(0.01)
        self.qcs_init_time_us.setKeyboardTracking(False)
        self.qcs_init_time_us.setSuffix(" us")
        self.qcs_init_time_us.setValue(DEFAULT_QCS_INIT_TIME_US)
        self.qcs_init_time_us.setToolTip(
            "Delay from the end of one QCS program iteration to the start "
            "of the next. This is not connection or compilation time. The "
            "70 ns default follows QTTVideoMode/QCSVideoProcessor.py; larger "
            "values directly slow every hardware-sweep point. Values are "
            "rounded upward to a 10 ns HCL/fabric-safe quantum."
        )
        self.qcs_init_time_us.editingFinished.connect(
            self._commit_qcs_inter_iteration_delay
        )
        self._qcs_blocking = True
        self._qcs_front_panel_draft_pending = False
        self._qcs_hardware_configuration = None
        self._qcs_hardware_configuration_state = (
            QCS_HARDWARE_STATE_EXTERNAL
        )
        self._qcs_hardware_authoritative_mapper_path = None
        self._qcs_hardware_mapper_sha256 = None
        self.qcs_hardware_configuration_button = QtWidgets.QPushButton(
            "Configure M5000 front panel..."
        )
        self.qcs_hardware_configuration_button.setToolTip(
            "Build, load, validate, and save the physical Keysight QCS "
            "ChannelMapper"
        )
        self.qcs_hardware_configuration_button.clicked.connect(
            lambda: self.qcs_hardware_configuration_requested.emit()
        )
        qcs_form.addRow("Channel mapper file:", qcs_mapper_row)
        qcs_form.addRow(self.qcs_hardware_configuration_button)
        qcs_form.addRow("DC virtual channel names:", self.qcs_dc_channel_names)
        qcs_form.addRow("DC full scale (+/-):", self.qcs_dc_full_scale_v)
        qcs_form.addRow("RF generator map:", self.qcs_rf_channel_names)
        qcs_form.addRow(
            "Acquisition virtual channel:",
            self.qcs_acquisition_channel_name,
        )
        qcs_form.addRow(
            "Inter-point / repetition delay:",
            self.qcs_init_time_us,
        )

        self.database_path = QtWidgets.QLineEdit(DEFAULT_QCODES_DB_PATH)
        browse_database = QtWidgets.QToolButton()
        browse_database.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)
        )
        browse_database.setToolTip("Choose QCoDeS SQLite database")
        browse_database.clicked.connect(self._browse_database)
        database_row = QtWidgets.QHBoxLayout()
        database_row.addWidget(self.database_path, 1)
        database_row.addWidget(browse_database)

        self.experiment_name = QtWidgets.QLineEdit("QCS pulse experiment")
        self.sample_name = QtWidgets.QLineEdit("PulseGenerator")
        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setPlaceholderText("Optional experiment notes")
        self.notes.setMaximumHeight(90)

        self.fabric_mhz = QtWidgets.QDoubleSpinBox()
        self.fabric_mhz.setRange(1.0, 5000.0)
        self.fabric_mhz.setDecimals(6)
        self.fabric_mhz.setSuffix(" MHz")
        self.tproc_mhz = QtWidgets.QDoubleSpinBox()
        self.tproc_mhz.setRange(1.0, 5000.0)
        self.tproc_mhz.setDecimals(6)
        self.tproc_mhz.setSuffix(" MHz")
        self.full_scale_mv = QtWidgets.QDoubleSpinBox()
        self.full_scale_mv.setRange(1.0, 1.0e6)
        self.full_scale_mv.setDecimals(6)
        self.full_scale_mv.setSuffix(" mV")
        # Voltage-sweep coordinates are normalized by the execution
        # backend's active AWG scale.  Keep that display scale separate from
        # the dormant QICK full-scale widget so QCS does not inherit QICK's
        # historical +/-800 mV limit.
        self._sweep_voltage_scale_mv = float(full_scale_mv)
        self.awg_channels = QtWidgets.QLineEdit()
        self.repetitions = QtWidgets.QSpinBox()
        self.repetitions.setRange(1, 1_000_000)
        self.iq_repetition_policy = QtWidgets.QComboBox()
        self.iq_repetition_policy.addItem(
            "Preserve every repetition",
            IQ_REPETITION_POLICY_PRESERVE,
        )
        self.iq_repetition_policy.addItem(
            "Coherently average; save one I/Q per point",
            IQ_REPETITION_POLICY_COHERENT_AVERAGE,
        )
        self.iq_repetition_policy.setToolTip(
            "For a single-I/Q acquisition, preserve one complex value from "
            "every repetition or coherently average I and Q and save one "
            "complex value per sweep point. Hardware still executes every "
            "configured repetition. Raw traces always preserve repetitions."
        )
        self.iq_repetition_policy_note = QtWidgets.QLabel()
        self.iq_repetition_policy_note.setWordWrap(True)
        self.iq_repetition_policy_note.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        iq_repetition_policy_layout = QtWidgets.QVBoxLayout()
        iq_repetition_policy_layout.setContentsMargins(0, 0, 0, 0)
        iq_repetition_policy_layout.setSpacing(2)
        iq_repetition_policy_layout.addWidget(self.iq_repetition_policy)
        iq_repetition_policy_layout.addWidget(self.iq_repetition_policy_note)
        self.iq_repetition_policy_widget = QtWidgets.QWidget()
        self.iq_repetition_policy_widget.setLayout(
            iq_repetition_policy_layout
        )
        self._ddr_readout_spec: Optional[QickDdrReadoutSpec] = None
        self._ddr_capacity_words_32b: Optional[int] = None
        self._ddr_samples_per_axi_word = 8

        self.ddr_usage_group = QtWidgets.QGroupBox("PL DDR capture memory")
        ddr_usage_layout = QtWidgets.QVBoxLayout(self.ddr_usage_group)
        self.ddr_usage_progress = QtWidgets.QProgressBar()
        self.ddr_usage_progress.setRange(0, 10_000)
        self.ddr_usage_progress.setValue(0)
        self.ddr_usage_progress.setTextVisible(True)
        self.ddr_usage_progress.setFormat("Capacity unavailable")
        self.ddr_usage_progress.setFixedHeight(
            max(22, self.ddr_usage_progress.sizeHint().height())
        )
        self.ddr_usage_summary = QtWidgets.QLabel()
        self.ddr_usage_summary.setWordWrap(True)
        self.ddr_usage_summary.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self.ddr_usage_detail = QtWidgets.QLabel()
        self.ddr_usage_detail.setWordWrap(True)
        self.ddr_usage_detail.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self.ddr_capacity_refresh = QtWidgets.QPushButton(
            "Identify QICK / Refresh Capacity"
        )
        self.ddr_capacity_refresh.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
        )
        self.ddr_capacity_refresh.setToolTip(
            "Connect to QICK, read the HWH-backed DDR buffer size, and "
            "recalculate the percentage."
        )
        self.ddr_capacity_refresh.clicked.connect(
            self.ddr_capacity_refresh_requested.emit
        )
        ddr_usage_layout.addWidget(self.ddr_usage_progress)
        ddr_usage_layout.addWidget(self.ddr_usage_summary)
        ddr_usage_layout.addWidget(self.ddr_usage_detail)
        ddr_usage_layout.addWidget(
            self.ddr_capacity_refresh,
            0,
            QtCore.Qt.AlignRight,
        )

        self.qcs_waveform_usage_group = QtWidgets.QGroupBox(
            "QCS/HCL M5301 rendered-waveform budget"
        )
        qcs_waveform_layout = QtWidgets.QVBoxLayout(
            self.qcs_waveform_usage_group
        )
        self.qcs_waveform_usage_progress = QtWidgets.QProgressBar()
        self.qcs_waveform_usage_progress.setRange(
            0,
            QCS_M5301_MAX_RENDERED_SAMPLES,
        )
        self.qcs_waveform_usage_progress.setValue(0)
        self.qcs_waveform_usage_progress.setTextVisible(True)
        self.qcs_waveform_usage_progress.setFormat("Calculating capacity...")
        self.qcs_waveform_usage_progress.setFixedHeight(
            max(22, self.qcs_waveform_usage_progress.sizeHint().height())
        )
        self.qcs_waveform_usage_summary = QtWidgets.QLabel()
        self.qcs_waveform_usage_summary.setWordWrap(True)
        self.qcs_waveform_usage_summary.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self.qcs_waveform_usage_detail = QtWidgets.QLabel()
        self.qcs_waveform_usage_detail.setWordWrap(True)
        self.qcs_waveform_usage_detail.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        qcs_waveform_layout.addWidget(self.qcs_waveform_usage_progress)
        qcs_waveform_layout.addWidget(self.qcs_waveform_usage_summary)
        qcs_waveform_layout.addWidget(self.qcs_waveform_usage_detail)
        self.qcs_waveform_usage_group.setToolTip(
            "The tested QCS 2.5.5 HCL backend accepts 98,304 rendered M5301 "
            "samples (40.960 us) per physical DC output and program. This is "
            "the QCS/HCL program budget, not the module's onboard-memory "
            "specification. Ramps and a minimum seed for independently "
            "established nonzero levels consume capacity. The rest of each "
            "fixed plateau uses QCS Hold; zero-voltage delays consume no "
            "waveform samples."
        )
        self.awg_metadata_mode = QtWidgets.QComboBox()
        self.awg_metadata_mode.addItem(
            "Parametric recipe only (recommended)",
            AWG_METADATA_MODE_PARAMETRIC,
        )
        self.awg_metadata_mode.addItem(
            "Expanded vertices for every sweep point",
            AWG_METADATA_MODE_EXPANDED,
        )
        self.awg_metadata_mode.setCurrentIndex(
            self.awg_metadata_mode.findData(DEFAULT_AWG_METADATA_MODE)
        )
        self.awg_metadata_mode.setToolTip(
            "Parametric mode stores the base waveform and sweep rules without "
            "allocating per-point vertex arrays."
        )
        self.awg_metadata_button = QtWidgets.QPushButton(
            "Preview / Export AWG Metadata..."
        )
        self.awg_metadata_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)
        )
        self.awg_metadata_button.clicked.connect(
            self.awg_metadata_requested.emit
        )
        metadata_hint = QtWidgets.QLabel(
            "The default recipe is compact. Expanded per-point vertices are "
            "generated only when explicitly selected or exported."
        )
        metadata_hint.setWordWrap(True)
        self.compile_validation_mode = QtWidgets.QComboBox()
        self.compile_validation_mode.addItem(
            "Boundary and corners (fast, recommended)",
            COMPILE_VALIDATION_BOUNDARY,
        )
        self.compile_validation_mode.addItem(
            "Every Cartesian point (slow, exhaustive)",
            COMPILE_VALIDATION_FULL,
        )
        compile_mode = normalize_compile_validation_mode(
            compile_validation_mode
        )
        self.compile_validation_mode.setCurrentIndex(
            self.compile_validation_mode.findData(compile_mode)
        )
        self.compile_validation_mode.setToolTip(
            "Boundary mode validates the sweep origin, endpoints, Cartesian "
            "corners, and duration-table rows without expanding every sweep "
            "combination. Full mode validates every Cartesian point and can "
            "take a long time for large sweeps."
        )
        self._map_sweep_specs: Tuple[QickSweepAxisSpec, ...] = ()
        self.sweep_parameter_group = QtWidgets.QGroupBox("Sweep parameters")
        sweep_parameter_layout = QtWidgets.QVBoxLayout(
            self.sweep_parameter_group
        )
        self.sweep_parameter_table = QtWidgets.QTableWidget(0, 5)
        self.sweep_parameter_table.setHorizontalHeaderLabels(
            ["Type", "Target", "Start", "Stop", "Points"]
        )
        self.sweep_parameter_table.verticalHeader().setVisible(False)
        self.sweep_parameter_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self.sweep_parameter_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self.sweep_parameter_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )
        self.sweep_parameter_table.setMaximumHeight(180)
        sweep_parameter_header = self.sweep_parameter_table.horizontalHeader()
        sweep_parameter_header.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        sweep_parameter_header.setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch
        )
        for column in (2, 3, 4):
            sweep_parameter_header.setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeToContents
            )
        self.qcs_sweep_execution_indicator = QtWidgets.QFrame(
            self.sweep_parameter_group
        )
        self.qcs_sweep_execution_indicator.setObjectName(
            "qcsSweepExecutionIndicator"
        )
        self.qcs_sweep_execution_indicator.setFrameShape(
            QtWidgets.QFrame.StyledPanel
        )
        sweep_execution_layout = QtWidgets.QVBoxLayout(
            self.qcs_sweep_execution_indicator
        )
        sweep_execution_layout.setContentsMargins(8, 6, 8, 6)
        sweep_execution_layout.setSpacing(2)
        self.qcs_sweep_execution_mode_label = QtWidgets.QLabel(
            "QCS sweep execution: No sweep"
        )
        mode_font = self.qcs_sweep_execution_mode_label.font()
        mode_font.setBold(True)
        self.qcs_sweep_execution_mode_label.setFont(mode_font)
        self.qcs_sweep_execution_reason_label = QtWidgets.QLabel(
            "No voltage or RF sweep is configured."
        )
        self.qcs_sweep_execution_reason_label.setWordWrap(True)
        self.qcs_sweep_execution_reason_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        sweep_execution_layout.addWidget(self.qcs_sweep_execution_mode_label)
        sweep_execution_layout.addWidget(self.qcs_sweep_execution_reason_label)
        sweep_parameter_layout.addWidget(self.qcs_sweep_execution_indicator)
        self.set_qcs_sweep_execution_status(
            QcsSweepExecutionPreview(
                mode="none",
                reasons=("No voltage or RF sweep is configured.",),
            )
        )
        sweep_parameter_layout.addWidget(self.sweep_parameter_table)

        self.sweep_parameter_empty = QtWidgets.QLabel(
            "No sweep configured. Configure an AWG segment, RF duration, "
            "RF frequency, or calibrated RF power sweep first."
        )
        self.sweep_parameter_empty.setWordWrap(True)
        sweep_parameter_layout.addWidget(self.sweep_parameter_empty)

        self.sweep_parameter_editor = QtWidgets.QGroupBox("Selected sweep")
        sweep_parameter_form = QtWidgets.QFormLayout(
            self.sweep_parameter_editor
        )
        self.sweep_parameter_target = QtWidgets.QLabel("-")
        self.sweep_parameter_target.setWordWrap(True)
        self.sweep_parameter_start = QtWidgets.QDoubleSpinBox()
        self.sweep_parameter_stop = QtWidgets.QDoubleSpinBox()
        for editor in (
            self.sweep_parameter_start,
            self.sweep_parameter_stop,
        ):
            editor.setDecimals(9)
            editor.setRange(-1.0e12, 1.0e12)
        self.sweep_parameter_count = QtWidgets.QSpinBox()
        self.sweep_parameter_count.setRange(1, 1_000_000)
        sweep_parameter_form.addRow("Target:", self.sweep_parameter_target)
        sweep_parameter_form.addRow("Start:", self.sweep_parameter_start)
        sweep_parameter_form.addRow("Stop:", self.sweep_parameter_stop)
        sweep_parameter_form.addRow("Points:", self.sweep_parameter_count)
        sweep_parameter_buttons = QtWidgets.QHBoxLayout()
        self.sweep_parameter_apply = QtWidgets.QPushButton("Apply changes")
        self.sweep_parameter_apply.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        )
        self.sweep_parameter_remove = QtWidgets.QPushButton(
            "Remove selected sweep"
        )
        self.sweep_parameter_remove.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton)
        )
        sweep_parameter_buttons.addWidget(self.sweep_parameter_apply)
        sweep_parameter_buttons.addWidget(self.sweep_parameter_remove)
        sweep_parameter_form.addRow(sweep_parameter_buttons)
        sweep_parameter_layout.addWidget(self.sweep_parameter_editor)
        self.sweep_parameter_table.itemSelectionChanged.connect(
            self._load_selected_sweep_parameter
        )
        self.sweep_parameter_apply.clicked.connect(
            self._apply_selected_sweep_parameter
        )
        self.sweep_parameter_remove.clicked.connect(
            self._remove_selected_sweep_parameter
        )
        self.sweep_parameter_editor.setEnabled(False)

        self.sweep_map_group = QtWidgets.QGroupBox("AWG 2D sweep map")
        sweep_map_form = QtWidgets.QFormLayout(self.sweep_map_group)
        self.sweep_map_x = QtWidgets.QComboBox()
        self.sweep_map_y = QtWidgets.QComboBox()
        self.sweep_map_status = QtWidgets.QLabel(
            "Add at least two AWG sweep variables to enable the map."
        )
        self.sweep_map_status.setWordWrap(True)
        sweep_map_form.addRow("X sweep variable:", self.sweep_map_x)
        sweep_map_form.addRow("Y sweep variable:", self.sweep_map_y)
        sweep_map_form.addRow(self.sweep_map_status)
        self.sweep_map_x.currentIndexChanged.connect(
            self._on_sweep_axis_changed
        )
        self.sweep_map_y.currentIndexChanged.connect(
            self._on_sweep_axis_changed
        )
        self.bias_t_group = QtWidgets.QGroupBox("Bias-T compensation")
        self.bias_t_group.setCheckable(True)
        self.bias_t_group.setChecked(bool(bias_t_enabled))
        bias_t_form = QtWidgets.QFormLayout(self.bias_t_group)
        self.bias_t_type = QtWidgets.QComboBox()
        self.bias_t_type.addItem("DC compensation", "dc")
        self.bias_t_type.addItem("Filter compensation", "filter")
        type_index = self.bias_t_type.findData(str(bias_t_compensation_type))
        if type_index < 0:
            raise ValueError(
                "Bias-T compensation type must be one of "
                f"{BIAS_T_COMPENSATION_TYPES}"
            )
        self.bias_t_type.setCurrentIndex(type_index)
        self.bias_t_mode = QtWidgets.QComboBox()
        self.bias_t_mode.addItem("Fixed voltage (adjust time)", "fixed_voltage")
        self.bias_t_mode.addItem("Fixed time (adjust voltage)", "fixed_time")
        mode_index = self.bias_t_mode.findData(str(bias_t_mode))
        if mode_index < 0:
            raise ValueError(
                f"Bias-T compensation mode must be one of {BIAS_T_COMPENSATION_MODES}"
            )
        self.bias_t_mode.setCurrentIndex(mode_index)
        self.bias_t_compensation_mv = QtWidgets.QDoubleSpinBox()
        self.bias_t_compensation_mv.setRange(0.001, float(full_scale_mv))
        self.bias_t_compensation_mv.setDecimals(6)
        self.bias_t_compensation_mv.setSuffix(" mV")
        self.bias_t_compensation_mv.setValue(
            float(
                full_scale_mv * DEFAULT_BIAS_T_COMPENSATION_FRACTION
                if bias_t_compensation_mv is None
                else bias_t_compensation_mv
            )
        )
        self.bias_t_duration_us = QtWidgets.QDoubleSpinBox()
        self.bias_t_duration_us.setRange(1.0e-6, 1.0e9)
        self.bias_t_duration_us.setDecimals(6)
        self.bias_t_duration_us.setSuffix(" us")
        self.bias_t_duration_us.setValue(float(bias_t_duration_us))
        self.bias_t_filter_tau_us = QtWidgets.QDoubleSpinBox()
        self.bias_t_filter_tau_us.setRange(1.0e-6, 1.0e12)
        self.bias_t_filter_tau_us.setDecimals(6)
        self.bias_t_filter_tau_us.setSuffix(" us")
        self.bias_t_filter_tau_us.setValue(float(bias_t_filter_tau_us))
        bias_t_form.addRow("Compensation type:", self.bias_t_type)
        bias_t_form.addRow("DC control mode:", self.bias_t_mode)
        bias_t_form.addRow("DC voltage:", self.bias_t_compensation_mv)
        bias_t_form.addRow("DC time:", self.bias_t_duration_us)
        bias_t_form.addRow("Filter time constant (tau):", self.bias_t_filter_tau_us)
        self.set_qick_values(
            fabric_mhz=fabric_mhz,
            tproc_mhz=tproc_mhz,
            full_scale_mv=full_scale_mv,
            awg_channels=awg_channels,
            repetitions=repetitions,
        )
        self.set_iq_repetition_policy(iq_repetition_policy)

        form.addRow("Execution system:", self.execution_system_label)
        form.addRow(self.qcs_connection_group)
        form.addRow("QCoDeS DB file:", database_row)
        form.addRow("Experiment name:", self.experiment_name)
        form.addRow("Sample name:", self.sample_name)
        self.full_scale_mv_label = QtWidgets.QLabel("AWG full scale (+/-):")
        form.addRow(self.full_scale_mv_label, self.full_scale_mv)
        form.addRow("Repetitions per sweep point:", self.repetitions)
        form.addRow(
            "Single-I/Q repetition save:",
            self.iq_repetition_policy_widget,
        )
        form.addRow(self.ddr_usage_group)
        form.addRow(self.qcs_waveform_usage_group)
        self.compile_validation_label = QtWidgets.QLabel(
            "Compile validation:"
        )
        form.addRow(
            self.compile_validation_label,
            self.compile_validation_mode,
        )
        form.addRow("AWG waveform metadata:", self.awg_metadata_mode)
        form.addRow(metadata_hint)
        form.addRow(self.awg_metadata_button)
        form.addRow(self.sweep_parameter_group)
        self.sweep_map_group.setVisible(False)
        form.addRow(self.bias_t_group)
        form.addRow("Notes:", self.notes)

        self.full_scale_mv.valueChanged.connect(self._update_bias_t_range)
        self.full_scale_mv.valueChanged.connect(
            lambda _value: self.set_sweep_specs(self._map_sweep_specs)
        )
        self.repetitions.valueChanged.connect(self._refresh_ddr_usage)
        self.repetitions.valueChanged.connect(
            self._refresh_iq_repetition_policy_controls
        )
        self.bias_t_group.toggled.connect(self._emit_bias_t_changed)
        self.bias_t_compensation_mv.valueChanged.connect(self._emit_bias_t_changed)
        self.bias_t_duration_us.valueChanged.connect(self._emit_bias_t_changed)
        self.bias_t_filter_tau_us.valueChanged.connect(self._emit_bias_t_changed)
        self.bias_t_type.currentIndexChanged.connect(self._on_bias_t_mode_changed)
        self.bias_t_mode.currentIndexChanged.connect(self._on_bias_t_mode_changed)
        self._update_bias_t_mode_controls()
        self._refresh_ddr_usage()

        self.run_button = QtWidgets.QPushButton("Run QICK Experiment")
        self.run_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        )
        self.run_button.clicked.connect(self.run_requested.emit)
        self.stop_button = QtWidgets.QPushButton("Stop QCS Experiment")
        self.stop_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop)
        )
        self.stop_button.setToolTip(
            "Abort the active QCS program, stop before the next sweep point, "
            "and reset mapped DC outputs to zero"
        )
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.show_program_button = QtWidgets.QPushButton("Show QICK Program")
        self.show_program_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView)
        )
        self.show_program_button.setToolTip(
            "Compile the current settings and show the tProcessor assembly"
        )
        self.show_program_button.clicked.connect(self.show_program_requested.emit)
        action_row = QtWidgets.QHBoxLayout()
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.stop_button)
        action_row.addWidget(self.show_program_button)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.progress.hide()
        self.run_status = QtWidgets.QLabel("Ready")
        self.run_status.setWordWrap(True)
        self._running = False
        self._stop_available = False
        self.run_timing_group = QtWidgets.QGroupBox("Run timing and events")
        run_timing_layout = QtWidgets.QVBoxLayout(self.run_timing_group)
        self.run_elapsed_label = QtWidgets.QLabel(
            "Elapsed: 00:00:00.000 | Not running"
        )
        self.run_elapsed_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self.run_event_log = QtWidgets.QPlainTextEdit()
        self.run_event_log.setReadOnly(True)
        self.run_event_log.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.run_event_log.setPlaceholderText(
            "Program build, acquisition, result retrieval, and database "
            "events will appear here."
        )
        self.run_event_log.setMaximumBlockCount(1000)
        self.run_event_log.setMinimumHeight(150)
        self.run_event_log.setMaximumHeight(220)
        run_log_actions = QtWidgets.QHBoxLayout()
        run_log_actions.addStretch(1)
        self.copy_run_log_button = QtWidgets.QToolButton()
        self.copy_run_log_button.setText("Copy log")
        self.copy_run_log_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonTextBesideIcon
        )
        self.copy_run_log_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)
        )
        self.copy_run_log_button.clicked.connect(self._copy_run_event_log)
        self.clear_run_log_button = QtWidgets.QToolButton()
        self.clear_run_log_button.setText("Clear")
        self.clear_run_log_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonTextBesideIcon
        )
        self.clear_run_log_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton)
        )
        self.clear_run_log_button.clicked.connect(self.run_event_log.clear)
        run_log_actions.addWidget(self.copy_run_log_button)
        run_log_actions.addWidget(self.clear_run_log_button)
        run_timing_layout.addWidget(self.run_elapsed_label)
        run_timing_layout.addWidget(self.run_event_log)
        run_timing_layout.addLayout(run_log_actions)
        self._run_elapsed_timer = QtCore.QElapsedTimer()
        self._run_timeline_active = False
        self._run_stage_starts = {}
        self._run_last_event_ms = 0
        self._run_final_elapsed_ms = 0
        self._run_started_at = ""
        self._run_elapsed_update_timer = QtCore.QTimer(self)
        self._run_elapsed_update_timer.setInterval(250)
        self._run_elapsed_update_timer.timeout.connect(
            self._update_run_elapsed_label
        )
        form.addRow(action_row)
        form.addRow(self.progress)
        form.addRow("Status:", self.run_status)
        form.addRow(self.run_timing_group)
        self.backend_selector.currentIndexChanged.connect(
            self._update_execution_backend_controls
        )
        self.qcs_hw_demod.toggled.connect(
            self._refresh_iq_repetition_policy_controls
        )
        self._update_execution_backend_controls()

    def set_ddr_readout_spec(
        self,
        spec: Optional[QickDdrReadoutSpec],
    ) -> None:
        """Update the capture estimate from the Experiment RF-readout editor."""
        self._ddr_readout_spec = spec
        self._refresh_ddr_usage()
        self._refresh_iq_repetition_policy_controls()

    def iq_repetition_policy_value(self, *, effective: bool = False) -> str:
        """Return the configured or currently applicable save policy."""

        policy = normalize_iq_repetition_policy(
            self.iq_repetition_policy.currentData()
        )
        if effective and not self._single_iq_repetition_policy_available():
            return IQ_REPETITION_POLICY_PRESERVE
        return policy

    def set_iq_repetition_policy(self, policy: str) -> None:
        """Restore a validated repetition-save policy from application settings."""

        policy = normalize_iq_repetition_policy(policy)
        index = self.iq_repetition_policy.findData(policy)
        if index < 0:
            raise ValueError(f"I/Q repetition policy {policy!r} is unavailable")
        self.iq_repetition_policy.setCurrentIndex(index)
        self._refresh_iq_repetition_policy_controls()

    def _single_iq_repetition_policy_available(self) -> bool:
        spec = self._ddr_readout_spec
        if spec is None:
            return False
        if self.execution_backend() == EXECUTION_BACKEND_QCS:
            return self.qcs_hw_demod.isChecked()
        return int(spec.samples_per_trigger) == 1

    def _refresh_iq_repetition_policy_controls(self, *_args) -> None:
        """Enable averaging only when one I/Q value is acquired per shot."""

        available = self._single_iq_repetition_policy_available()
        self.iq_repetition_policy.setEnabled(
            available and not getattr(self, "_running", False)
        )
        if self._ddr_readout_spec is None:
            note = (
                "Enable acquisition to select how repeated single-I/Q "
                "values are saved."
            )
        elif not available:
            note = (
                "Trace acquisition always preserves every repetition; this "
                "selection is retained and becomes active in Single I/Q mode."
            )
        elif self.repetitions.value() == 1:
            note = (
                "Only one repetition is configured, so both choices save the "
                "same value."
            )
        else:
            note = (
                "Preserve keeps every shot. Coherent average computes mean I "
                "and mean Q and saves one complex value per sweep point."
            )
        self.iq_repetition_policy_note.setText(note)

    def set_ddr_memory_configuration(self, configuration) -> None:
        """Use the live HWH/runtime DDR capacity reported by QICK."""
        self._ddr_capacity_words_32b = getattr(
            configuration,
            "ddr_capacity_words_32b",
            None,
        )
        self._ddr_samples_per_axi_word = int(
            getattr(configuration, "ddr_samples_per_axi_word", 8)
        )
        self._refresh_ddr_usage()

    def _set_ddr_usage_style(self, state: str) -> None:
        colors = {
            "inactive": ("#eceff1", "#9aa4ad"),
            "normal": ("#e7eef0", "#267b8a"),
            "warning": ("#f6efe1", "#c58a1c"),
            "error": ("#f5e5e5", "#b33a3a"),
        }
        background, chunk = colors[state]
        self.ddr_usage_progress.setStyleSheet(
            "QProgressBar {"
            " border: 1px solid #8b949c;"
            " background: %s;"
            " color: #202428;"
            " text-align: center;"
            " padding: 1px;"
            "}"
            "QProgressBar::chunk { background: %s; }"
            % (background, chunk)
        )

    def _refresh_ddr_usage(self, *_args) -> None:
        sweep_points = prod(spec.count for spec in self._map_sweep_specs)
        repetitions = self.repetitions.value()
        sweep_axes = len(self._map_sweep_specs)
        spec = self._ddr_readout_spec

        if spec is None:
            self.ddr_usage_progress.setValue(0)
            self.ddr_usage_progress.setFormat("RF readout disabled")
            self.ddr_usage_summary.setText(
                f"{sweep_axes} sweep axis/axes -> {sweep_points:,} Cartesian "
                f"point(s) x {repetitions:,} repetition(s)."
            )
            self.ddr_usage_detail.setText(
                "Enable RF Readout to calculate the PL DDR capture allocation."
            )
            self._set_ddr_usage_style("inactive")
            return

        usage = calculate_ddr_capture_memory_usage(
            sweep_points=sweep_points,
            repetitions=repetitions,
            samples_per_trigger=int(spec.samples_per_trigger),
            start_address_bytes=int(spec.address),
            capacity_words_32b=self._ddr_capacity_words_32b,
            samples_per_axi_word=self._ddr_samples_per_axi_word,
            force_overwrite=bool(spec.force_overwrite),
        )
        self.ddr_usage_summary.setText(
            f"{sweep_axes} sweep axis/axes -> {usage.sweep_points:,} "
            f"Cartesian point(s) x {usage.repetitions:,} repetition(s) = "
            f"{usage.trigger_count:,} DDR trigger(s)."
        )
        detail = (
            f"{usage.samples_per_trigger:,} I/Q sample(s)/trigger -> "
            f"{usage.physical_words_per_trigger:,} padded 32-bit "
            f"word(s)/trigger. Valid data {format_binary_bytes(usage.valid_data_bytes)}; "
            f"AXI padding {format_binary_bytes(usage.padding_bytes)}; "
            f"reserved {format_binary_bytes(usage.reserved_bytes)}."
        )

        percent = usage.address_usage_percent
        if percent is None:
            self.ddr_usage_progress.setValue(0)
            self.ddr_usage_progress.setFormat(
                f"{format_binary_bytes(usage.reserved_bytes)} reserved | "
                "capacity unknown"
            )
            self.ddr_usage_detail.setText(
                detail
                + " Identify QICK to read ddr4_buf.maxlen and calculate the "
                "PL DDR percentage."
            )
            self._set_ddr_usage_style("inactive")
            return

        self.ddr_usage_progress.setValue(
            min(10_000, max(0, int(round(percent * 100.0))))
        )
        self.ddr_usage_progress.setFormat(
            f"{percent:.3f}% | "
            f"{format_binary_bytes(usage.end_address_bytes)} / "
            f"{format_binary_bytes(usage.capacity_bytes)}"
        )
        address_detail = (
            f" Address range: 0x{usage.start_address_bytes:08X} to "
            f"0x{usage.end_address_bytes:08X}; HWH/runtime capacity "
            f"{format_binary_bytes(usage.capacity_bytes)}."
        )
        if usage.exceeds_capacity:
            if usage.force_overwrite:
                warning = (
                    " The request exceeds PL DDR capacity; overwrite is "
                    "enabled, so earlier trigger data will be overwritten."
                )
            else:
                warning = (
                    " The request exceeds PL DDR capacity and the QICK "
                    "driver will reject the run."
                )
            self._set_ddr_usage_style("error")
        elif percent >= 80.0:
            warning = " High PL DDR utilization."
            self._set_ddr_usage_style("warning")
        else:
            warning = ""
            self._set_ddr_usage_style("normal")
        self.ddr_usage_detail.setText(detail + address_detail + warning)

    def _set_qcs_waveform_usage_style(self, state: str) -> None:
        colors = {
            "inactive": ("#eceff1", "#9aa4ad"),
            "normal": ("#e7eef0", "#267b8a"),
            "warning": ("#f6efe1", "#c58a1c"),
            "error": ("#f5e5e5", "#b33a3a"),
        }
        background, chunk = colors[state]
        self.qcs_waveform_usage_progress.setStyleSheet(
            "QProgressBar {"
            " border: 1px solid #8b949c;"
            " background: %s;"
            " color: #202428;"
            " text-align: center;"
            " padding: 1px;"
            "}"
            "QProgressBar::chunk { background: %s; }"
            % (background, chunk)
        )

    def set_qcs_waveform_capacity(
        self,
        report: QcsM5301CapacityReport,
        physical_channel_names: Sequence[str] = (),
    ) -> None:
        """Display the worst per-output QCS/HCL rendered-waveform use."""
        channel_names = {
            channel.output_name: str(physical_channel_names[index])
            for index, channel in enumerate(report.channels)
            if index < len(physical_channel_names)
            and str(physical_channel_names[index]).strip()
        }
        worst = report.worst_channel
        used = int(worst.rendered_samples)
        maximum = int(report.maximum_samples)
        percent = 100.0 * report.usage_fraction
        self.qcs_waveform_usage_progress.setRange(0, maximum)
        self.qcs_waveform_usage_progress.setValue(
            min(maximum, max(0, used))
        )
        self.qcs_waveform_usage_progress.setFormat(
            f"{used:,} / {maximum:,} samples ({percent:.2f}%)"
        )
        physical_name = channel_names.get(worst.output_name)
        output_label = worst.output_name
        if physical_name is not None:
            output_label += f" -> {physical_name}"
        point_label = (
            ""
            if report.sweep_point_count == 1
            else f" at sweep point {worst.point_index + 1:,}"
        )
        self.qcs_waveform_usage_summary.setText(
            f"Worst physical output: {output_label}{point_label}; "
            f"{worst.rendered_duration_us:.6f} / 40.960000 us."
        )

        channel_details = []
        for channel in report.channels:
            label = channel.output_name
            if channel.output_name in channel_names:
                label += f" -> {channel_names[channel.output_name]}"
            detail = (
                f"{label}: {channel.rendered_samples:,} samples "
                f"({channel.rendered_duration_us:.6f} us)"
            )
            channel_details.append(detail)
        if report.exhaustive:
            inspection = (
                f"All {report.sweep_point_count:,} software-sweep point(s) "
                "checked."
            )
        else:
            inspection = (
                f"Live preview checked {report.inspected_point_count:,} "
                f"boundary/midpoint point(s) of {report.sweep_point_count:,}. "
                "Every point is checked before a QCS run starts."
            )
        explanation = (
            " Each output has its own budget; outputs are not added together. "
            "Ramps and the minimum seed for each independently established "
            "nonzero level count. The rest of each fixed plateau uses QCS "
            "Hold and does not consume waveform samples; zero-voltage delays "
            "do not either. "
            "Fixed output levels remain explicit DCWaveform amplitudes; "
            "mapped physical-channel offsets are not used."
        )
        if report.exceeds_capacity:
            explanation += (
                " This setup exceeds the measured QCS/HCL limit; the QCS run "
                "is blocked until the waveform is shortened."
            )
        self.qcs_waveform_usage_detail.setText(
            "; ".join(channel_details) + ". " + inspection + explanation
        )
        if report.exceeds_capacity:
            self._set_qcs_waveform_usage_style("error")
        elif percent >= 80.0:
            self._set_qcs_waveform_usage_style("warning")
        else:
            self._set_qcs_waveform_usage_style("normal")

    def set_qcs_waveform_capacity_error(self, message: str) -> None:
        """Show an invalid QCS waveform setup without waiting for Run."""
        maximum = QCS_M5301_MAX_RENDERED_SAMPLES
        self.qcs_waveform_usage_progress.setRange(0, maximum)
        self.qcs_waveform_usage_progress.setValue(maximum)
        self.qcs_waveform_usage_progress.setFormat("Invalid waveform setup")
        self.qcs_waveform_usage_summary.setText(
            "QCS cannot compile the current M5301 waveform."
        )
        self.qcs_waveform_usage_detail.setText(str(message))
        self._set_qcs_waveform_usage_style("error")

    def set_qcs_mixed_waveform_capacity_pending(self) -> None:
        """Explain that hybrid capacity is authoritative only per slice."""

        maximum = QCS_M5301_MAX_RENDERED_SAMPLES
        self.qcs_waveform_usage_progress.setRange(0, maximum)
        self.qcs_waveform_usage_progress.setValue(0)
        self.qcs_waveform_usage_progress.setFormat(
            "Per-slice capacity preflight at Run"
        )
        self.qcs_waveform_usage_summary.setText(
            "Hybrid sweep: M5301 capacity depends on the software-loop "
            "coordinate."
        )
        self.qcs_waveform_usage_detail.setText(
            "Before the first hardware program is submitted, every planned "
            "software-loop coordinate is checked using its exact fixed "
            "timing and M5301 offset/Hold representation."
        )
        self._set_qcs_waveform_usage_style("normal")

    def set_qcs_sweep_execution_status(
        self,
        preview: QcsSweepExecutionPreview,
    ) -> None:
        """Show the predicted or compiled QCS repetition mode."""

        mode = str(preview.mode).strip().lower()
        if mode not in {"none", "hardware", "hybrid", "software", "invalid"}:
            raise ValueError(f"unsupported QCS sweep execution mode {mode!r}")
        reasons = tuple(
            str(reason).strip() for reason in preview.reasons if str(reason).strip()
        )
        labels = {
            "none": "No sweep",
            "hardware": (
                "Hardware sweep"
                if preview.exact
                else "Hardware sweep (planned)"
            ),
            "hybrid": "Hardware sweep inside software loop",
            "software": "Software sweep",
            "invalid": "Invalid sweep",
        }
        colors = {
            "none": ("#455a64", "#eceff1"),
            "hardware": ("#1b5e20", "#e8f5e9"),
            "hybrid": ("#0d4775", "#e3f2fd"),
            "software": ("#8a4b00", "#fff3e0"),
            "invalid": ("#b71c1c", "#ffebee"),
        }
        foreground, background = colors[mode]
        self.qcs_sweep_execution_mode_label.setText(
            f"QCS sweep execution: {labels[mode]}"
        )
        reason_text = " ".join(reasons)
        self.qcs_sweep_execution_reason_label.setText(reason_text)
        tooltip = reason_text or self.qcs_sweep_execution_mode_label.text()
        self.qcs_sweep_execution_indicator.setToolTip(tooltip)
        self.qcs_sweep_execution_indicator.setStyleSheet(
            "QFrame#qcsSweepExecutionIndicator {"
            f"background-color: {background}; color: {foreground}; "
            f"border: 1px solid {foreground}; border-radius: 3px;"
            "}"
        )

    def _update_bias_t_range(self, full_scale_mv: float) -> None:
        self.bias_t_compensation_mv.setMaximum(max(0.001, float(full_scale_mv)))

    def _emit_bias_t_changed(self, *_args) -> None:
        self.bias_t_changed.emit(
            self.bias_t_group.isChecked(),
            str(self.bias_t_type.currentData()),
            self.bias_t_compensation_mv.value(),
            str(self.bias_t_mode.currentData()),
            self.bias_t_duration_us.value(),
            self.bias_t_filter_tau_us.value(),
        )

    def _update_bias_t_mode_controls(self) -> None:
        filter_mode = self.bias_t_type.currentData() == "filter"
        fixed_time = self.bias_t_mode.currentData() == "fixed_time"
        self.bias_t_mode.setEnabled(not filter_mode)
        self.bias_t_compensation_mv.setEnabled(not filter_mode and not fixed_time)
        self.bias_t_duration_us.setEnabled(not filter_mode and fixed_time)
        self.bias_t_filter_tau_us.setEnabled(filter_mode)

    def _on_bias_t_mode_changed(self, *_args) -> None:
        self._update_bias_t_mode_controls()
        self._emit_bias_t_changed()

    def _browse_database(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Choose QCoDeS database",
            self.database_path.text().strip() or DEFAULT_QCODES_DB_PATH,
            "QCoDeS SQLite database (*.db)",
        )
        if path:
            if Path(path).suffix.lower() != ".db":
                path = str(Path(path).with_suffix(".db"))
            self.database_path.setText(path)

    def _browse_qcs_mapper(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose Keysight QCS channel mapper",
            self.qcs_mapper_path.text().strip(),
            "Keysight QCS mapper (*.qcs);;All files (*)",
        )
        if path:
            self.qcs_mapper_path.setText(path)

    def execution_backend(self) -> str:
        backend = str(self.backend_selector.currentData())
        if backend not in EXECUTION_BACKENDS:
            raise ValueError(f"unsupported execution backend {backend!r}")
        return backend

    def set_execution_backend(self, backend: str) -> None:
        backend = str(backend).strip().lower()
        if backend not in EXECUTION_BACKENDS:
            raise ValueError(f"unsupported execution backend {backend!r}")
        index = self.backend_selector.findData(backend)
        if index < 0:
            raise ValueError(f"execution backend {backend!r} is unavailable")
        self.backend_selector.setCurrentIndex(index)
        self._update_execution_backend_controls()

    def _update_execution_backend_controls(self, *_args) -> None:
        backend = self.execution_backend()
        is_qick = backend == EXECUTION_BACKEND_QICK
        self.execution_system_label.setText(
            "QICK (legacy settings)"
            if is_qick
            else "Keysight QCS / M5000"
        )
        self.qcs_connection_group.setVisible(not is_qick)
        self.qcs_connection_group.setEnabled(not is_qick and not self._running)
        self.ddr_usage_group.setVisible(is_qick)
        self.ddr_usage_group.setEnabled(is_qick and not self._running)
        self.qcs_waveform_usage_group.setVisible(not is_qick)
        self.qcs_waveform_usage_group.setEnabled(
            not is_qick and not self._running
        )
        self.qcs_sweep_execution_indicator.setVisible(not is_qick)
        self.full_scale_mv_label.setVisible(is_qick)
        self.full_scale_mv.setVisible(is_qick)
        self.compile_validation_label.setVisible(is_qick)
        self.compile_validation_mode.setVisible(is_qick)
        self.compile_validation_mode.setEnabled(is_qick and not self._running)
        self.show_program_button.setVisible(True)
        self.show_program_button.setEnabled(not self._running)
        self.stop_button.setVisible(not is_qick)
        self.stop_button.setEnabled(
            not is_qick and self._running and self._stop_available
        )
        self.show_program_button.setToolTip(
            "Compile the current settings and show the tProcessor assembly"
            if is_qick
            else (
                "Show an offline QCS Python source preview without loading a "
                "mapper, compiling a Program, or contacting hardware"
            )
        )
        self.show_program_button.setText(
            "Show QICK Program" if is_qick else "Show QCS Program"
        )
        self.run_button.setText(
            "Run QICK Experiment" if is_qick else "Run QCS Experiment"
        )
        if hasattr(self, "sweep_map_status"):
            self._update_sweep_map_status()
        if hasattr(self, "iq_repetition_policy"):
            self._refresh_iq_repetition_policy_controls()
        if backend != self._last_execution_backend:
            self._last_execution_backend = backend
            self.execution_backend_changed.emit(backend)

    @staticmethod
    def _comma_separated_names(text: str, description: str) -> Tuple[str, ...]:
        names = tuple(item.strip() for item in str(text).split(",") if item.strip())
        if not names:
            raise ValueError(f"{description} must contain at least one name")
        if len(set(names)) != len(names):
            raise ValueError(f"{description} must contain unique names")
        return names

    def _parse_qcs_dc_channel_names(
        self,
        output_count: Optional[int] = None,
    ) -> Tuple[str, ...]:
        names = self._comma_separated_names(
            self.qcs_dc_channel_names.text(),
            "QCS DC virtual channel names",
        )
        if output_count is not None and len(names) != int(output_count):
            raise ValueError(
                f"exactly {int(output_count)} QCS DC virtual channel names are "
                "required, one for each AWG output"
            )
        return names

    def _parse_qcs_rf_channel_names(self) -> dict:
        text = self.qcs_rf_channel_names.text().strip()
        if not text:
            return {}
        mapping = {}
        used_names = set()
        for entry in text.split(","):
            fields = entry.split("=", 1)
            if len(fields) != 2:
                raise ValueError(
                    "QCS RF channel map entries must use gen_ch=name"
                )
            channel_text, name_text = (field.strip() for field in fields)
            try:
                channel = int(channel_text)
            except ValueError as exc:
                raise ValueError(
                    "QCS RF generator identifiers must be integers"
                ) from exc
            if channel < 0:
                raise ValueError(
                    "QCS RF generator identifiers must be nonnegative"
                )
            if not name_text:
                raise ValueError("QCS RF virtual channel names must not be empty")
            if channel in mapping:
                raise ValueError(
                    f"QCS RF generator {channel} is mapped more than once"
                )
            if name_text in used_names:
                raise ValueError(
                    "QCS RF virtual channel names must be unique"
                )
            mapping[channel] = name_text
            used_names.add(name_text)
        return mapping

    @staticmethod
    def _normalized_qcs_mapper_path(path: str) -> str:
        return str(Path(path).expanduser().resolve()).casefold()

    @staticmethod
    def _qcs_state_after_mapper_change(state: str) -> str:
        if state in {
            QCS_HARDWARE_STATE_IMPORTED,
            QCS_HARDWARE_STATE_IMPORTED_DIRTY,
        }:
            return QCS_HARDWARE_STATE_IMPORTED_DIRTY
        return QCS_HARDWARE_STATE_DRAFT

    def _commit_qcs_inter_iteration_delay(self) -> float:
        """Round the displayed delay to the exact value HCL will receive."""

        effective_us = (
            quantize_qcs_inter_iteration_delay(
                self.qcs_init_time_us.value() * 1.0e-6
            )
            * 1.0e6
        )
        self.qcs_init_time_us.setValue(effective_us)
        return effective_us

    def qcs_settings_dict(self) -> dict:
        self._commit_qcs_inter_iteration_delay()
        strict = self.execution_backend() == EXECUTION_BACKEND_QCS
        acquisition_name = self.qcs_acquisition_channel_name.text().strip()
        raw_dc_names_text = self.qcs_dc_channel_names.text()
        raw_rf_names_text = self.qcs_rf_channel_names.text()
        dc_names_text_override = None
        rf_names_text_override = None
        acquisition_name_text_override = None
        try:
            dc_names = (
                self._parse_qcs_dc_channel_names()
                if raw_dc_names_text.strip() or strict
                else ()
            )
        except ValueError:
            if strict:
                raise
            dc_names = ()
            dc_names_text_override = raw_dc_names_text
        try:
            rf_names = self._parse_qcs_rf_channel_names()
        except ValueError:
            if strict:
                raise
            rf_names = {}
            rf_names_text_override = raw_rf_names_text

        mapper_path = self.qcs_mapper_path.text().strip()
        hardware_configuration = self._qcs_hardware_configuration
        state = self._qcs_hardware_configuration_state
        mapper_file_sha256 = self._qcs_hardware_mapper_sha256
        if state not in QCS_HARDWARE_STATES:
            state = QCS_HARDWARE_STATE_EXTERNAL
        if hardware_configuration is None:
            state = QCS_HARDWARE_STATE_EXTERNAL
            mapper_file_sha256 = None
            self._qcs_hardware_authoritative_mapper_path = None
        else:
            hardware_configuration = normalize_qcs_hardware_configuration(
                hardware_configuration
            )
            if state == QCS_HARDWARE_STATE_EXTERNAL:
                state = QCS_HARDWARE_STATE_DRAFT
            if (
                dc_names_text_override is not None
                or rf_names_text_override is not None
            ):
                dc_count = sum(
                    mapping["role"] == "dc"
                    for mapping in hardware_configuration[
                        "channel_mappings"
                    ]
                )
                bound_dc, bound_rf, _bound_acquisition = qcs_role_bindings(
                    hardware_configuration,
                    required_dc_count=dc_count,
                )
                if dc_names_text_override is not None:
                    dc_names = tuple(bound_dc)
                if rf_names_text_override is not None:
                    rf_names = {
                        int(channel): name
                        for channel, name in bound_rf.items()
                    }
            original_fingerprint = qcs_hardware_mapper_fingerprint(
                hardware_configuration
            )
            try:
                synchronized_configuration = (
                    synchronize_qcs_hardware_role_names(
                        hardware_configuration,
                        dc_names,
                        rf_names,
                        acquisition_name or None,
                    )
                )
            except (TypeError, ValueError) as exc:
                if strict:
                    raise ValueError(
                        "QCS DC, RF, or acquisition roles differ from the "
                        "known hardware configuration. Use Configure M5000 "
                        "front panel to change role assignments."
                    ) from exc
                dc_count = sum(
                    mapping["role"] == "dc"
                    for mapping in hardware_configuration[
                        "channel_mappings"
                    ]
                )
                bound_dc, bound_rf, bound_acquisition = qcs_role_bindings(
                    hardware_configuration,
                    required_dc_count=dc_count,
                )
                dc_names = tuple(bound_dc)
                rf_names = {
                    int(channel): name
                    for channel, name in bound_rf.items()
                }
                acquisition_name = bound_acquisition or ""
                dc_names_text_override = raw_dc_names_text
                rf_names_text_override = raw_rf_names_text
                acquisition_name_text_override = (
                    self.qcs_acquisition_channel_name.text()
                )
                synchronized_configuration = hardware_configuration

            hardware_configuration = synchronized_configuration
            mapper_changed = (
                qcs_hardware_mapper_fingerprint(hardware_configuration)
                != original_fingerprint
            )
            if mapper_changed:
                state = self._qcs_state_after_mapper_change(state)
            if (
                state
                in {
                    QCS_HARDWARE_STATE_SAVED,
                    QCS_HARDWARE_STATE_IMPORTED,
                }
                and (
                    self._qcs_hardware_authoritative_mapper_path is None
                    or self._normalized_qcs_mapper_path(mapper_path)
                    != self._qcs_hardware_authoritative_mapper_path
                )
            ):
                state = self._qcs_state_after_mapper_change(state)
        if state not in {
            QCS_HARDWARE_STATE_SAVED,
            QCS_HARDWARE_STATE_IMPORTED,
        }:
            mapper_file_sha256 = None

        self._qcs_hardware_configuration = hardware_configuration
        self._qcs_hardware_configuration_state = state
        self._qcs_hardware_mapper_sha256 = mapper_file_sha256
        result = {
            "mapper_path": mapper_path,
            "dc_channel_names": list(dc_names),
            "dc_full_scale_v": self.qcs_dc_full_scale_v.value(),
            "rf_channel_names": {
                str(channel): name
                for channel, name in rf_names.items()
            },
            "acquisition_channel_name": acquisition_name or None,
            "hw_demod": self.qcs_hw_demod.isChecked(),
            "sample_rate_hz": self.qcs_sample_rate_hz.value(),
            "init_time_s": self.qcs_init_time_us.value() * 1.0e-6,
            "blocking": bool(self._qcs_blocking),
            "hardware_configuration": hardware_configuration,
            "hardware_configuration_state": state,
            "hardware_mapper_sha256": mapper_file_sha256,
        }
        if dc_names_text_override is not None:
            result["dc_channel_names_text"] = dc_names_text_override
        if rf_names_text_override is not None:
            result["rf_channel_names_text"] = rf_names_text_override
        if acquisition_name_text_override is not None:
            result["acquisition_channel_name_text"] = (
                acquisition_name_text_override
            )
        return result

    def set_qcs_front_panel_draft_pending(self, pending: bool) -> None:
        """Block hardware use while the graphical mapper is incomplete."""

        self._qcs_front_panel_draft_pending = bool(pending)

    def qcs_connection_values(self, output_count: int) -> QcsConnectionConfig:
        if self._qcs_front_panel_draft_pending:
            raise ValueError(
                "The QCS front-panel assignment is incomplete or its native "
                "mapper is still being validated. Finish selecting the "
                "highlighted SMA channels and wait for the background mapper "
                "update before running hardware."
            )
        settings = self.qcs_settings_dict()
        state = settings["hardware_configuration_state"]
        if state == QCS_HARDWARE_STATE_DRAFT:
            raise ValueError(
                "The QCS front-panel configuration has unsaved physical "
                "changes. Open Configure M5000 front panel and save a native "
                ".qcs mapper before running."
            )
        if state == QCS_HARDWARE_STATE_IMPORTED_DIRTY:
            raise ValueError(
                "The imported QCS topology has unsaved edits. Reload the "
                "original mapper to discard them, or use Restore diagram "
                "layout and save a new mapper before running."
            )
        if state in {
            QCS_HARDWARE_STATE_SAVED,
            QCS_HARDWARE_STATE_IMPORTED,
        }:
            expected_digest = settings["hardware_mapper_sha256"]
            if expected_digest is None:
                raise ValueError(
                    "The QCS mapper identity is unknown. Reload or save the "
                    "front-panel mapper before running."
                )
            actual_digest = qcs_mapper_file_sha256(
                settings["mapper_path"]
            )
            if actual_digest != expected_digest:
                raise ValueError(
                    "The QCS ChannelMapper file changed after the hardware "
                    "configuration was applied. Reload or save it before "
                    "running."
                )
            if state == QCS_HARDWARE_STATE_IMPORTED:
                validate_imported_qcs_role_configuration(
                    settings["hardware_configuration"],
                    settings["mapper_path"],
                )
        dc_names = self._parse_qcs_dc_channel_names(output_count)
        return QcsConnectionConfig(
            mapper_path=settings["mapper_path"],
            dc_channel_names=dc_names,
            mapper_sha256=settings["hardware_mapper_sha256"],
            dc_full_scale_v=settings["dc_full_scale_v"],
            rf_channel_names={
                int(channel): name
                for channel, name in settings["rf_channel_names"].items()
            },
            acquisition_channel_name=settings["acquisition_channel_name"],
            hw_demod=settings["hw_demod"],
            init_time_s=settings["init_time_s"],
            blocking=settings["blocking"],
        )

    def _qcs_dc_topology_source(
        self,
        *,
        target_output_count: Optional[int] = None,
    ) -> tuple[Optional[dict], list[str]]:
        configuration = self._qcs_hardware_configuration
        if configuration is None:
            return None, [
                item.strip()
                for item in self.qcs_dc_channel_names.text().split(",")
                if item.strip()
            ]

        current_dc_count = sum(
            mapping["role"] == "dc"
            for mapping in configuration["channel_mappings"]
        )
        current_dc, current_rf, current_acquisition = qcs_role_bindings(
            configuration,
            required_dc_count=current_dc_count,
        )
        try:
            visible_dc = list(self._parse_qcs_dc_channel_names())
        except ValueError as exc:
            self.qcs_output_sync_warning.emit(
                "Invalid dormant QCS DC names were reset to the known "
                f"hardware bindings before changing outputs: {exc}"
            )
            return configuration, current_dc

        if len(visible_dc) == current_dc_count:
            try:
                synchronized = synchronize_qcs_hardware_role_names(
                    configuration,
                    visible_dc,
                    {
                        int(channel): name
                        for channel, name in current_rf.items()
                    },
                    current_acquisition,
                )
            except (TypeError, ValueError) as exc:
                self.qcs_output_sync_warning.emit(
                    "QCS DC names could not be synchronized with the known "
                    f"hardware bindings and were reset before changing "
                    f"outputs: {exc}"
                )
                return configuration, current_dc
            return synchronized, visible_dc
        if (
            target_output_count is not None
            and target_output_count > current_dc_count
            and len(visible_dc) == target_output_count
        ):
            existing_names = visible_dc[:current_dc_count]
            if set(existing_names) == set(current_dc):
                try:
                    configuration = synchronize_qcs_hardware_role_names(
                        configuration,
                        existing_names,
                        {
                            int(channel): name
                            for channel, name in current_rf.items()
                        },
                        current_acquisition,
                    )
                except (TypeError, ValueError) as exc:
                    self.qcs_output_sync_warning.emit(
                        "QCS DC names could not be synchronized with the "
                        "known hardware bindings and were reset before "
                        f"adding an output: {exc}"
                    )
                    return configuration, current_dc
            return configuration, visible_dc

        self.qcs_output_sync_warning.emit(
            "QCS DC names did not match the current waveform count and were "
            "reset to the known hardware bindings before changing outputs."
        )
        return configuration, current_dc

    def set_qcs_output_count(self, output_count: int) -> None:
        output_count = int(output_count)
        if output_count < 1:
            raise ValueError("QCS requires at least one DC output channel")
        original_configuration = self._qcs_hardware_configuration
        base_configuration, names = self._qcs_dc_topology_source(
            target_output_count=output_count
        )
        names = names[:output_count]
        used = set(names)
        spare_names = iter(())
        if base_configuration is not None:
            modules_by_slot = {
                module["slot"]: module
                for module in base_configuration["modules"]
            }
            spare_names = iter(
                mapping["virtual_name"]
                for mapping in base_configuration["channel_mappings"]
                if (
                    mapping["role"] == "unassigned"
                    and QCS_MODULE_MODELS[
                        modules_by_slot[mapping["slot"]]["model"]
                    ]["instrument"]
                    == "M5301AWG"
                    and mapping["virtual_name"] not in used
                )
            )
        next_index = 1
        while len(names) < output_count:
            candidate = next(spare_names, None)
            if candidate is None:
                candidate = f"dc_ch_{next_index}"
                next_index += 1
            if candidate in used:
                continue
            names.append(candidate)
            used.add(candidate)
        resized_configuration = None
        if base_configuration is not None:
            resized_configuration = resize_qcs_dc_mappings(
                base_configuration,
                names,
            )
        self.qcs_dc_channel_names.setText(", ".join(names))
        if resized_configuration is not None:
            mapper_changed = (
                qcs_hardware_mapper_fingerprint(resized_configuration)
                != qcs_hardware_mapper_fingerprint(
                    original_configuration
                )
            )
            self._qcs_hardware_configuration = resized_configuration
            if mapper_changed:
                self._qcs_hardware_configuration_state = (
                    self._qcs_state_after_mapper_change(
                        self._qcs_hardware_configuration_state
                    )
                )
                self._qcs_hardware_mapper_sha256 = None
        self.qcs_output_count_changed.emit(output_count)

    def remove_qcs_output(
        self,
        output_index: int,
        *,
        output_count: Optional[int] = None,
    ) -> None:
        requested_output_count = (
            None if output_count is None else int(output_count)
        )
        if requested_output_count is not None and requested_output_count < 1:
            raise ValueError("QCS requires at least one DC output channel")
        original_configuration = self._qcs_hardware_configuration
        base_configuration, names = self._qcs_dc_topology_source()
        output_index = int(output_index)
        removed = False
        if 0 <= output_index < len(names):
            names.pop(output_index)
            removed = True
        resized_configuration = None
        if (
            removed
            and base_configuration is not None
            and names
        ):
            resized_configuration = resize_qcs_dc_mappings(
                base_configuration,
                names,
                removed_index=output_index,
            )
        self.qcs_dc_channel_names.setText(", ".join(names))
        if resized_configuration is not None:
            mapper_changed = (
                qcs_hardware_mapper_fingerprint(resized_configuration)
                != qcs_hardware_mapper_fingerprint(
                    original_configuration
                )
            )
            self._qcs_hardware_configuration = resized_configuration
            if mapper_changed:
                self._qcs_hardware_configuration_state = (
                    self._qcs_state_after_mapper_change(
                        self._qcs_hardware_configuration_state
                    )
                )
                self._qcs_hardware_mapper_sha256 = None
        resulting_output_count = (
            len(names)
            if requested_output_count is None
            else requested_output_count
        )
        if resulting_output_count < 1:
            raise ValueError("QCS requires at least one DC output channel")
        self.qcs_output_count_changed.emit(resulting_output_count)

    def set_qcs_settings(self, settings: Mapping[str, object], output_count: int) -> None:
        settings = dict(settings)
        dc_names = settings.get("dc_channel_names")
        if dc_names is None:
            dc_names = tuple(
                f"dc_ch_{index + 1}" for index in range(int(output_count))
            )
        rf_names = dict(settings.get("rf_channel_names", {}))
        acquisition_name = settings.get("acquisition_channel_name")
        mapper_path = str(
            settings.get("mapper_path", DEFAULT_QCS_MAPPER_PATH)
        )
        dc_names = tuple(str(name).strip() for name in dc_names)
        normalized_rf_names = {
            int(channel): str(name).strip()
            for channel, name in rf_names.items()
        }
        hardware_configuration = settings.get("hardware_configuration")
        mapper_file_sha256 = normalize_qcs_mapper_sha256(
            settings.get("hardware_mapper_sha256")
        )
        state = str(
            settings.get(
                "hardware_configuration_state",
                (
                    QCS_HARDWARE_STATE_EXTERNAL
                    if hardware_configuration is None
                    else QCS_HARDWARE_STATE_DRAFT
                ),
            )
        )
        if state not in QCS_HARDWARE_STATES:
            raise ValueError(
                f"unsupported QCS hardware configuration state {state!r}"
            )
        if hardware_configuration is None:
            state = QCS_HARDWARE_STATE_EXTERNAL
            normalized_hardware_configuration = None
            mapper_file_sha256 = None
        else:
            if state == QCS_HARDWARE_STATE_EXTERNAL:
                state = QCS_HARDWARE_STATE_DRAFT
            normalized_hardware_configuration = (
                normalize_qcs_hardware_configuration(
                    hardware_configuration,
                    required_dc_count=int(output_count),
                )
            )
            bound_dc, bound_rf, bound_acquisition = qcs_role_bindings(
                normalized_hardware_configuration,
                required_dc_count=int(output_count),
            )
            expected_rf = {
                str(channel): name
                for channel, name in normalized_rf_names.items()
            }
            if (
                bound_dc != list(dc_names)
                or bound_rf != expected_rf
                or bound_acquisition
                != (
                    None
                    if acquisition_name is None
                    else str(acquisition_name).strip() or None
                )
            ):
                raise ValueError(
                    "QCS hardware role bindings must exactly match the "
                    "experiment DC, RF, and acquisition channel names"
                )
            if (
                state == QCS_HARDWARE_STATE_SAVED
                and mapper_file_sha256 is None
            ):
                state = QCS_HARDWARE_STATE_DRAFT
            elif (
                state == QCS_HARDWARE_STATE_IMPORTED
                and mapper_file_sha256 is None
            ):
                state = QCS_HARDWARE_STATE_IMPORTED_DIRTY

        dc_names_text = settings.get("dc_channel_names_text")
        rf_names_text = settings.get("rf_channel_names_text")
        acquisition_name_text = settings.get(
            "acquisition_channel_name_text"
        )
        if dc_names_text is not None and not isinstance(dc_names_text, str):
            raise TypeError("QCS dc_channel_names_text must be a string")
        if rf_names_text is not None and not isinstance(rf_names_text, str):
            raise TypeError("QCS rf_channel_names_text must be a string")
        if (
            acquisition_name_text is not None
            and not isinstance(acquisition_name_text, str)
        ):
            raise TypeError(
                "QCS acquisition_channel_name_text must be a string"
            )
        self.qcs_mapper_path.setText(
            mapper_path
        )
        self.qcs_dc_channel_names.setText(
            (
                dc_names_text
                if dc_names_text is not None
                else ", ".join(dc_names)
            )
        )
        self.qcs_dc_full_scale_v.setValue(
            float(
                settings.get(
                    "dc_full_scale_v", DEFAULT_QCS_FULL_SCALE_V
                )
            )
        )
        self.qcs_rf_channel_names.setText(
            (
                rf_names_text
                if rf_names_text is not None
                else ", ".join(
                    f"{channel}={name}"
                    for channel, name in sorted(
                        normalized_rf_names.items(),
                        key=lambda item: item[0],
                    )
                )
            )
        )
        self.qcs_acquisition_channel_name.setText(
            (
                acquisition_name_text
                if acquisition_name_text is not None
                else (
                    ""
                    if acquisition_name is None
                    else str(acquisition_name)
                )
            )
        )
        self.qcs_hw_demod.setChecked(bool(settings.get("hw_demod", True)))
        # Versions through settings schema 36 stored a user-editable timing
        # rate which defaulted incorrectly to the QICK 1 MSPS FIR rate. QCS
        # M5200 timing is fixed by the hardware and mapper, so migrate it.
        self.qcs_sample_rate_hz.setValue(DEFAULT_QCS_SAMPLE_RATE_HZ)
        self.qcs_init_time_us.setValue(
            float(settings.get("init_time_s", DEFAULT_QCS_INIT_TIME_US * 1.0e-6))
            * 1.0e6
        )
        self._commit_qcs_inter_iteration_delay()
        self._qcs_blocking = bool(settings.get("blocking", True))
        self._qcs_hardware_configuration = normalized_hardware_configuration
        self._qcs_hardware_configuration_state = state
        self._qcs_hardware_mapper_sha256 = mapper_file_sha256
        self._qcs_front_panel_draft_pending = False
        self._qcs_hardware_authoritative_mapper_path = (
            self._normalized_qcs_mapper_path(mapper_path)
            if state
            in {
                QCS_HARDWARE_STATE_SAVED,
                QCS_HARDWARE_STATE_IMPORTED,
            }
            else None
        )

    @staticmethod
    def _sweep_axis_key(value) -> Optional[Tuple[str, str]]:
        if value is None:
            return None
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise TypeError("AWG sweep axis key must contain output and segment")
        return str(value[0]), str(value[1])

    def _sweep_axis_label(self, spec) -> str:
        axis_kind = getattr(spec, "axis_kind", "amplitude")
        if axis_kind == "rf_duration":
            return (
                f"RF gen {spec.gen_ch} / {spec.segment_name} duration | "
                f"{spec.start:.6g} to {spec.stop:.6g} us | "
                f"{spec.count} points"
            )
        if axis_kind == "rf_frequency":
            return (
                f"RF gen {spec.gen_ch} / {spec.segment_name} frequency | "
                f"{spec.start:.6g} to {spec.stop:.6g} MHz | "
                f"{spec.count} points"
            )
        if axis_kind == "rf_power":
            return (
                f"RF gen {spec.gen_ch} / {spec.segment_name} power | "
                f"{spec.start:.6g} to {spec.stop:.6g} dBm | "
                f"{spec.count} points"
            )
        if axis_kind == "ramp_duration":
            return (
                f"{spec.segment_name} RAMP duration (rate derived) | "
                f"{spec.start:.6g} to {spec.stop:.6g} us | "
                f"{spec.count} points"
            )
        if axis_kind == "hold_duration":
            return (
                f"{spec.segment_name} SET hold duration | "
                f"{spec.start:.6g} to {spec.stop:.6g} us | "
                f"{spec.count} points"
            )
        scale_mv = self._sweep_voltage_scale_mv
        return (
            f"{spec.output_name} / {spec.segment_name} | "
            f"{spec.start * scale_mv:.6g} to "
            f"{spec.stop * scale_mv:.6g} mV | {spec.count} points"
        )

    @staticmethod
    def _sweep_parameter_key(spec) -> Tuple[str, str, str]:
        return (
            str(getattr(spec, "axis_kind", "amplitude")),
            str(spec.output_name),
            str(spec.segment_name),
        )

    def _sweep_parameter_values(
        self,
        spec,
    ) -> Tuple[str, str, float, float, str]:
        axis_kind = str(getattr(spec, "axis_kind", "amplitude"))
        if axis_kind == "rf_duration":
            return (
                "RF duration",
                f"RF gen {spec.gen_ch} / {spec.segment_name}",
                float(spec.start),
                float(spec.stop),
                "us",
            )
        if axis_kind == "rf_frequency":
            return (
                "RF frequency",
                f"RF gen {spec.gen_ch} / {spec.segment_name}",
                float(spec.start),
                float(spec.stop),
                "MHz",
            )
        if axis_kind == "rf_power":
            return (
                "RF output power",
                f"RF gen {spec.gen_ch} / {spec.segment_name}",
                float(spec.start),
                float(spec.stop),
                "dBm",
            )
        if axis_kind == "ramp_duration":
            return (
                "RAMP duration",
                f"All AWG outputs / {spec.segment_name}",
                float(spec.start),
                float(spec.stop),
                "us",
            )
        if axis_kind == "hold_duration":
            return (
                "SET hold duration",
                f"All AWG outputs / {spec.segment_name}",
                float(spec.start),
                float(spec.stop),
                "us",
            )
        scale_mv = self._sweep_voltage_scale_mv
        return (
            "Voltage",
            f"{spec.output_name} / {spec.segment_name}",
            float(spec.start) * scale_mv,
            float(spec.stop) * scale_mv,
            "mV",
        )

    def _selected_sweep_parameter_key(
        self,
    ) -> Optional[Tuple[str, str, str]]:
        row = self.sweep_parameter_table.currentRow()
        if row < 0:
            return None
        item = self.sweep_parameter_table.item(row, 0)
        if item is None:
            return None
        value = item.data(QtCore.Qt.UserRole)
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            return None
        return tuple(str(part) for part in value)

    def _selected_sweep_parameter(self):
        selected_key = self._selected_sweep_parameter_key()
        if selected_key is None:
            return None
        return next(
            (
                spec
                for spec in self._map_sweep_specs
                if self._sweep_parameter_key(spec) == selected_key
            ),
            None,
        )

    def _refresh_sweep_parameter_table(
        self,
        *,
        selected_key: Optional[Tuple[str, str, str]] = None,
    ) -> None:
        if selected_key is None:
            selected_key = self._selected_sweep_parameter_key()
        selected_row = -1
        with QtCore.QSignalBlocker(self.sweep_parameter_table):
            self.sweep_parameter_table.setRowCount(len(self._map_sweep_specs))
            for row, spec in enumerate(self._map_sweep_specs):
                sweep_type, target, start, stop, unit = (
                    self._sweep_parameter_values(spec)
                )
                key = self._sweep_parameter_key(spec)
                values = (
                    sweep_type,
                    target,
                    f"{start:.9g} {unit}",
                    f"{stop:.9g} {unit}",
                    str(spec.count),
                )
                for column, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    if column == 0:
                        item.setData(QtCore.Qt.UserRole, key)
                    if column >= 2:
                        item.setTextAlignment(
                            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
                        )
                    self.sweep_parameter_table.setItem(row, column, item)
                if key == selected_key:
                    selected_row = row
            if selected_row < 0 and self._map_sweep_specs:
                selected_row = 0
            if selected_row >= 0:
                self.sweep_parameter_table.selectRow(selected_row)
            else:
                self.sweep_parameter_table.clearSelection()
                self.sweep_parameter_table.setCurrentCell(-1, -1)
        has_sweeps = bool(self._map_sweep_specs)
        self.sweep_parameter_table.setVisible(has_sweeps)
        self.sweep_parameter_empty.setVisible(not has_sweeps)
        self._load_selected_sweep_parameter()

    def _load_selected_sweep_parameter(self) -> None:
        spec = self._selected_sweep_parameter()
        self.sweep_parameter_editor.setEnabled(spec is not None)
        if spec is None:
            self.sweep_parameter_target.setText("-")
            return
        _sweep_type, target, start, stop, unit = self._sweep_parameter_values(
            spec
        )
        self.sweep_parameter_target.setText(target)
        if unit == "mV":
            endpoint_limit = max(1.0, self._sweep_voltage_scale_mv)
            minimum, maximum = -endpoint_limit, endpoint_limit
            step = max(0.001, endpoint_limit / 1000.0)
        else:
            minimum, maximum = 1.0e-9, 1.0e12
            step = max(1.0e-6, min(abs(stop - start) / 100.0, 1.0))
        with QtCore.QSignalBlocker(self.sweep_parameter_start), \
                QtCore.QSignalBlocker(self.sweep_parameter_stop), \
                QtCore.QSignalBlocker(self.sweep_parameter_count):
            for editor in (
                self.sweep_parameter_start,
                self.sweep_parameter_stop,
            ):
                editor.setRange(minimum, maximum)
                editor.setSingleStep(step)
                editor.setSuffix(f" {unit}")
            self.sweep_parameter_start.setValue(start)
            self.sweep_parameter_stop.setValue(stop)
            self.sweep_parameter_count.setValue(int(spec.count))

    def _apply_selected_sweep_parameter(self) -> None:
        spec = self._selected_sweep_parameter()
        if spec is None:
            return
        self.sweep_update_requested.emit(
            spec,
            self.sweep_parameter_start.value(),
            self.sweep_parameter_stop.value(),
            self.sweep_parameter_count.value(),
        )

    def _remove_selected_sweep_parameter(self) -> None:
        spec = self._selected_sweep_parameter()
        if spec is not None:
            self.sweep_remove_requested.emit(spec)

    def selected_sweep_axis_keys(
        self,
        *,
        require_two: bool = False,
    ) -> Optional[Tuple[Tuple[str, str], Tuple[str, str]]]:
        if self.sweep_map_x.count() < 2 or self.sweep_map_y.count() < 2:
            if require_two:
                raise ValueError(
                    "configure at least two AWG sweep variables for a 2D map"
                )
            return None
        x_key = self._sweep_axis_key(self.sweep_map_x.currentData())
        y_key = self._sweep_axis_key(self.sweep_map_y.currentData())
        if x_key is None or y_key is None:
            if require_two:
                raise ValueError(
                    "configure at least two AWG sweep variables for a 2D map"
                )
            return None
        if x_key == y_key:
            raise ValueError("X and Y sweep variables must be different")
        return x_key, y_key

    def sweep_map_settings(self) -> dict:
        selected = self.selected_sweep_axis_keys()
        if selected is None:
            return {}
        return {
            "x_axis": {
                "output_name": selected[0][0],
                "segment_name": selected[0][1],
            },
            "y_axis": {
                "output_name": selected[1][0],
                "segment_name": selected[1][1],
            },
        }

    def set_sweep_specs(
        self,
        specs: Sequence,
        *,
        selected_keys: Optional[
            Tuple[Tuple[str, str], Tuple[str, str]]
        ] = None,
        selected_parameter_key: Optional[
            Tuple[str, str, str]
        ] = None,
    ) -> None:
        """Refresh selectable AWG sweep axes while preserving valid choices."""
        previous = self.selected_sweep_axis_keys()
        if selected_parameter_key is None:
            selected_parameter_key = self._selected_sweep_parameter_key()
        self._map_sweep_specs = tuple(specs)
        available = tuple(
            (str(spec.output_name), str(spec.segment_name))
            for spec in self._map_sweep_specs
        )
        desired = selected_keys if selected_keys is not None else previous
        if desired is not None:
            desired = (
                self._sweep_axis_key(desired[0]),
                self._sweep_axis_key(desired[1]),
            )
        if (
            desired is None
            or desired[0] not in available
            or desired[1] not in available
            or desired[0] == desired[1]
        ):
            desired = (available[0], available[1]) if len(available) >= 2 else None

        with QtCore.QSignalBlocker(self.sweep_map_x), QtCore.QSignalBlocker(
            self.sweep_map_y
        ):
            self.sweep_map_x.clear()
            self.sweep_map_y.clear()
            for spec, key in zip(self._map_sweep_specs, available):
                label = self._sweep_axis_label(spec)
                self.sweep_map_x.addItem(label, key)
                self.sweep_map_y.addItem(label, key)
            if desired is not None:
                self.sweep_map_x.setCurrentIndex(available.index(desired[0]))
                self.sweep_map_y.setCurrentIndex(available.index(desired[1]))

        enabled = len(available) >= 2
        self.sweep_map_x.setEnabled(enabled)
        self.sweep_map_y.setEnabled(enabled)
        self._refresh_sweep_parameter_table(
            selected_key=selected_parameter_key
        )
        self._update_sweep_map_status()
        current = self.selected_sweep_axis_keys()
        if current != previous:
            self.sweep_axes_changed.emit()
        self._refresh_ddr_usage()

    def set_sweep_voltage_scale(self, full_scale_mv: float) -> None:
        """Display normalized voltage sweeps using the active AWG scale."""
        full_scale_mv = float(full_scale_mv)
        if not np.isfinite(full_scale_mv) or full_scale_mv <= 0.0:
            raise ValueError("AWG voltage full scale must be positive")
        if np.isclose(
            full_scale_mv,
            self._sweep_voltage_scale_mv,
            rtol=0.0,
            atol=1.0e-12,
        ):
            return
        selected_key = self._selected_sweep_parameter_key()
        self._sweep_voltage_scale_mv = full_scale_mv
        self._refresh_sweep_parameter_table(selected_key=selected_key)
        self._update_sweep_map_status()
        self._refresh_ddr_usage()

    def _on_sweep_axis_changed(self, _index: int) -> None:
        if self.sweep_map_x.count() >= 2:
            x_key = self._sweep_axis_key(self.sweep_map_x.currentData())
            y_key = self._sweep_axis_key(self.sweep_map_y.currentData())
            if x_key == y_key:
                target = (
                    self.sweep_map_y
                    if self.sender() is self.sweep_map_x
                    else self.sweep_map_x
                )
                other_key = (
                    x_key if target is self.sweep_map_y else y_key
                )
                replacement = next(
                    index
                    for index in range(target.count())
                    if self._sweep_axis_key(target.itemData(index)) != other_key
                )
                with QtCore.QSignalBlocker(target):
                    target.setCurrentIndex(replacement)
        self._update_sweep_map_status()
        self.sweep_axes_changed.emit()

    def _update_sweep_map_status(self) -> None:
        selected = self.selected_sweep_axis_keys()
        if selected is None:
            self.sweep_map_status.setText(
                "Add at least two AWG sweep variables to enable the map."
            )
            return
        omitted = max(0, len(self._map_sweep_specs) - 2)
        point_count = prod(spec.count for spec in self._map_sweep_specs)
        suffix = (
            f" The other {omitted} sweep axis/axes will be coherently averaged."
            if omitted
            else ""
        )
        averaging_values = (
            "QCS acquisition values"
            if self.execution_backend() == EXECUTION_BACKEND_QCS
            else "FIR samples"
        )
        self.sweep_map_status.setText(
            f"{point_count:,} Cartesian points; I/Q are averaged over "
            f"repetitions and {averaging_values}.{suffix}"
        )

    def _parse_awg_channels(self, output_count: int) -> Tuple[int, ...]:
        fields = [field.strip() for field in self.awg_channels.text().split(",")]
        if any(not field for field in fields):
            raise ValueError("AWG generator indices must be comma-separated integers")
        try:
            channels = tuple(int(field) for field in fields)
        except ValueError as exc:
            raise ValueError("AWG generator indices must be integers") from exc
        if len(channels) != output_count:
            raise ValueError(
                f"exactly {output_count} AWG generator indices are required"
            )
        if any(channel < 0 for channel in channels) or len(set(channels)) != len(channels):
            raise ValueError("AWG generator indices must be unique and nonnegative")
        return channels

    def values(
        self,
        output_count: int,
        *,
        require_run_config: bool = True,
        database_path: Optional[str] = None,
        validate_qcs_hardware: bool = True,
    ) -> dict:
        backend = self.execution_backend()
        if backend == EXECUTION_BACKEND_QCS:
            # QICK fields are deliberately dormant under QCS. Keep exported
            # settings loadable without allowing a blank QICK host/proxy or a
            # malformed hidden AWG-index list to veto QCS hardware.
            defaults = QickConnectionConfig()
            connection = QickConnectionConfig(
                host=self.qick_host.text().strip() or defaults.host,
                ns_port=self.ns_port.value(),
                proxy_name=(
                    self.proxy_name.text().strip() or defaults.proxy_name
                ),
            )
            run = self.run_config_values(
                require_run_config=require_run_config,
                database_path=database_path,
            )
            try:
                awg_channels = self._parse_awg_channels(output_count)
            except ValueError:
                awg_channels = tuple(range(output_count))
        else:
            connection, run = self.connection_values(
                require_run_config=require_run_config,
                database_path=database_path,
            )
            awg_channels = self._parse_awg_channels(output_count)
        return {
            "connection": connection,
            "run": run,
            "execution_backend": backend,
            "qcs_connection": (
                self.qcs_connection_values(output_count)
                if (
                    backend == EXECUTION_BACKEND_QCS
                    and validate_qcs_hardware
                )
                else None
            ),
            "fabric_mhz": self.fabric_mhz.value(),
            "tproc_mhz": self.tproc_mhz.value(),
            "full_scale_mv": self.full_scale_mv.value(),
            "awg_channels": awg_channels,
            "repetitions_per_sweep": self.repetitions.value(),
            "iq_repetition_policy": self.iq_repetition_policy_value(),
            "awg_metadata_mode": normalize_awg_metadata_mode(
                self.awg_metadata_mode.currentData()
            ),
            "compile_validation_mode": normalize_compile_validation_mode(
                self.compile_validation_mode.currentData()
            ),
            "bias_t_compensation_enabled": self.bias_t_group.isChecked(),
            "bias_t_compensation_type": str(self.bias_t_type.currentData()),
            "bias_t_compensation_voltage_mv": self.bias_t_compensation_mv.value(),
            "bias_t_compensation_mode": str(self.bias_t_mode.currentData()),
            "bias_t_compensation_duration_us": self.bias_t_duration_us.value(),
            "bias_t_filter_tau_us": self.bias_t_filter_tau_us.value(),
        }

    def connection_values(
        self,
        *,
        require_run_config: bool = True,
        database_path: Optional[str] = None,
    ) -> Tuple[QickConnectionConfig, Optional[QcodesRunConfig]]:
        """Return shared QICK/QCoDeS settings without validating AWG fields."""
        connection = QickConnectionConfig(
            host=self.qick_host.text().strip(),
            ns_port=self.ns_port.value(),
            proxy_name=self.proxy_name.text().strip(),
        )
        return connection, self.run_config_values(
            require_run_config=require_run_config,
            database_path=database_path,
        )

    def run_config_values(
        self,
        *,
        require_run_config: bool = True,
        database_path: Optional[str] = None,
    ) -> Optional[QcodesRunConfig]:
        """Return QCoDeS settings without constructing a QICK connection."""
        return (
            QcodesRunConfig(
                database_path=(
                    self.database_path.text().strip()
                    if database_path is None
                    else str(database_path).strip()
                ),
                experiment_name=self.experiment_name.text().strip(),
                sample_name=self.sample_name.text().strip(),
                notes=self.notes.toPlainText(),
            )
            if require_run_config
            else None
        )

    def settings_dict(self, output_count: int) -> dict:
        values = self.values(
            output_count,
            validate_qcs_hardware=False,
        )
        connection = values["connection"]
        run = values["run"]
        return {
            "qick_host": connection.host,
            "ns_port": connection.ns_port,
            "proxy_name": connection.proxy_name,
            "database_path": run.database_path,
            "experiment_name": run.experiment_name,
            "sample_name": run.sample_name,
            "notes": run.notes,
            "execution_backend": values["execution_backend"],
            "qcs": self.qcs_settings_dict(),
        }

    def set_qick_values(
        self,
        *,
        fabric_mhz: float,
        tproc_mhz: float,
        full_scale_mv: float,
        awg_channels: Sequence[int],
        repetitions: int,
    ) -> None:
        self.fabric_mhz.setValue(float(fabric_mhz))
        self.tproc_mhz.setValue(float(tproc_mhz))
        self.full_scale_mv.setValue(float(full_scale_mv))
        self.awg_channels.setText(", ".join(str(value) for value in awg_channels))
        self.repetitions.setValue(int(repetitions))

    def set_awg_metadata_mode(self, mode: str) -> None:
        mode = normalize_awg_metadata_mode(mode)
        index = self.awg_metadata_mode.findData(mode)
        if index < 0:
            raise ValueError(f"unsupported AWG metadata mode {mode!r}")
        self.awg_metadata_mode.setCurrentIndex(index)

    def set_compile_validation_mode(self, mode: str) -> None:
        mode = normalize_compile_validation_mode(mode)
        index = self.compile_validation_mode.findData(mode)
        if index < 0:
            raise ValueError(f"unsupported compile validation mode {mode!r}")
        self.compile_validation_mode.setCurrentIndex(index)

    def set_connection_values(self, connection: QickConnectionConfig) -> None:
        """Mirror the shared Setup connection into legacy execution fields."""
        self.qick_host.setText(connection.host)
        self.ns_port.setValue(connection.ns_port)
        self.proxy_name.setText(connection.proxy_name)

    def set_bias_t_values(
        self,
        *,
        enabled: bool,
        compensation_type: str = "dc",
        compensation_mv: float,
        mode: str = "fixed_voltage",
        duration_us: float = DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
        filter_tau_us: float = DEFAULT_BIAS_T_FILTER_TAU_US,
    ) -> None:
        type_index = self.bias_t_type.findData(str(compensation_type))
        if type_index < 0:
            raise ValueError(
                "Bias-T compensation type must be one of "
                f"{BIAS_T_COMPENSATION_TYPES}"
            )
        mode_index = self.bias_t_mode.findData(str(mode))
        if mode_index < 0:
            raise ValueError(
                f"Bias-T compensation mode must be one of {BIAS_T_COMPENSATION_MODES}"
            )
        with QtCore.QSignalBlocker(self.bias_t_group), QtCore.QSignalBlocker(
            self.bias_t_type
        ), QtCore.QSignalBlocker(self.bias_t_compensation_mv), QtCore.QSignalBlocker(
            self.bias_t_mode
        ), QtCore.QSignalBlocker(self.bias_t_duration_us), QtCore.QSignalBlocker(
            self.bias_t_filter_tau_us
        ):
            self.bias_t_group.setChecked(bool(enabled))
            self.bias_t_type.setCurrentIndex(type_index)
            self.bias_t_compensation_mv.setValue(float(compensation_mv))
            self.bias_t_mode.setCurrentIndex(mode_index)
            self.bias_t_duration_us.setValue(float(duration_us))
            self.bias_t_filter_tau_us.setValue(float(filter_tau_us))
        self._update_bias_t_mode_controls()
        self._emit_bias_t_changed()

    def load_settings(
        self,
        connection: QickConnectionConfig,
        run: QcodesRunConfig,
        *,
        fabric_mhz: float,
        tproc_mhz: float,
        full_scale_mv: float,
        awg_channels: Sequence[int],
        repetitions: int,
        iq_repetition_policy: str = IQ_REPETITION_POLICY_PRESERVE,
        bias_t_enabled: bool = False,
        bias_t_compensation_type: str = "dc",
        bias_t_compensation_mv: float = DEFAULT_BIAS_T_COMPENSATION_MV,
        bias_t_mode: str = "fixed_voltage",
        bias_t_duration_us: float = DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
        bias_t_filter_tau_us: float = DEFAULT_BIAS_T_FILTER_TAU_US,
        awg_metadata_mode: str = DEFAULT_AWG_METADATA_MODE,
        compile_validation_mode: str = DEFAULT_GUI_COMPILE_VALIDATION_MODE,
        execution_backend: str = DEFAULT_EXECUTION_BACKEND,
        qcs_settings: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.qick_host.setText(connection.host)
        self.ns_port.setValue(connection.ns_port)
        self.proxy_name.setText(connection.proxy_name)
        self.database_path.setText(run.database_path)
        self.experiment_name.setText(run.experiment_name)
        self.sample_name.setText(run.sample_name)
        self.notes.setPlainText(run.notes)
        self.set_qick_values(
            fabric_mhz=fabric_mhz,
            tproc_mhz=tproc_mhz,
            full_scale_mv=full_scale_mv,
            awg_channels=awg_channels,
            repetitions=repetitions,
        )
        self.set_iq_repetition_policy(iq_repetition_policy)
        self.set_awg_metadata_mode(awg_metadata_mode)
        self.set_compile_validation_mode(compile_validation_mode)
        self.set_qcs_settings(qcs_settings or {}, len(awg_channels))
        self.set_execution_backend(execution_backend)
        self.set_bias_t_values(
            enabled=bias_t_enabled,
            compensation_type=bias_t_compensation_type,
            compensation_mv=bias_t_compensation_mv,
            mode=bias_t_mode,
            duration_us=bias_t_duration_us,
            filter_tau_us=bias_t_filter_tau_us,
        )

    def set_running(
        self,
        running: bool,
        message: str,
        *,
        show_progress: bool = True,
        allow_stop: bool = False,
    ) -> None:
        self._running = bool(running)
        self._stop_available = bool(running and allow_stop)
        self.run_button.setEnabled(not running)
        self.backend_selector.setEnabled(not running)
        self._update_execution_backend_controls()
        if running:
            self.progress.setValue(0)
        self.progress.setVisible(running and show_progress)
        self.run_status.setText(message)
        self.run_state_changed.emit(self._running)

    def set_stopping(self, message: str = "Stopping QCS experiment...") -> None:
        """Disable repeated stop requests while cancellation and reset finish."""
        self._stop_available = False
        self.stop_button.setEnabled(False)
        self.run_status.setText(str(message))

    def set_stop_available(self, available: bool) -> None:
        """Update Stop availability without changing the overall run state."""
        self._stop_available = bool(available and self._running)
        self.stop_button.setEnabled(
            self.execution_backend() == EXECUTION_BACKEND_QCS
            and self._stop_available
        )

    @staticmethod
    def _format_elapsed_ms(elapsed_ms: int) -> str:
        elapsed_ms = max(0, int(elapsed_ms))
        hours, remainder = divmod(elapsed_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, milliseconds = divmod(remainder, 1_000)
        return (
            f"{hours:02d}:{minutes:02d}:{seconds:02d}."
            f"{milliseconds:03d}"
        )

    @staticmethod
    def _run_stage_label(key: str) -> str:
        return {
            "experiment": "Experiment",
            "validation": "Run validation",
            "connection": "Hardware connection",
            "channel_mapping": "QCS channel mapping",
            "program_build": "QCS program build",
            "execution": "Hardware execution",
            "cancellation": "QCS cancellation",
            "awg_recipe": "AWG recipe",
            "awg_vertices": "Expanded AWG vertices",
            "rf_setup": "RF readout setup",
            "compile": "tProcessor compile",
            "ddr_arm": "FIR DDR arm",
            "acquisition": "Acquisition",
            "ddr_wait": "DDR read delay",
            "ddr_readback": "FIR DDR readback",
            "qcodes_save": "QCoDeS save",
        }.get(str(key), str(key).replace("_", " ").title())

    def _current_run_elapsed_ms(self) -> int:
        if not self._run_elapsed_timer.isValid():
            return 0
        if not self._run_timeline_active:
            return int(self._run_final_elapsed_ms)
        return int(self._run_elapsed_timer.elapsed())

    def _update_run_elapsed_label(self) -> None:
        elapsed = self._format_elapsed_ms(self._current_run_elapsed_ms())
        if self._run_timeline_active:
            suffix = f"Running | Started {self._run_started_at}"
        elif self._run_started_at:
            suffix = f"Finished | Started {self._run_started_at}"
        else:
            suffix = "Not running"
        self.run_elapsed_label.setText(f"Elapsed: {elapsed} | {suffix}")

    def _copy_run_event_log(self) -> None:
        QtWidgets.QApplication.clipboard().setText(
            self.run_event_log.toPlainText()
        )

    def start_run_timeline(self, message: str) -> None:
        self.run_event_log.clear()
        self._run_stage_starts.clear()
        self._run_last_event_ms = 0
        self._run_final_elapsed_ms = 0
        self._run_started_at = QtCore.QDateTime.currentDateTime().toString(
            "yyyy-MM-dd HH:mm:ss.zzz"
        )
        self._run_elapsed_timer.start()
        self._run_timeline_active = True
        self.clear_run_log_button.setEnabled(False)
        self._run_elapsed_update_timer.start()
        self.record_run_event("experiment", "started", message)
        self._update_run_elapsed_label()

    def record_run_event(self, key: str, state: str, message: str) -> None:
        if not self._run_timeline_active:
            return
        elapsed_ms = self._current_run_elapsed_ms()
        key = str(key)
        state = str(state).lower()
        label = self._run_stage_label(key)
        duration_text = ""
        if state == "started":
            self._run_stage_starts[key] = elapsed_ms
            state_text = "STARTED"
        elif state == "completed":
            started_ms = self._run_stage_starts.pop(
                key, self._run_last_event_ms
            )
            duration_text = (
                f" | stage {self._format_elapsed_ms(elapsed_ms - started_ms)}"
            )
            state_text = "COMPLETED"
        elif state in {"failed", "stopped"}:
            started_ms = self._run_stage_starts.pop(
                key, self._run_last_event_ms
            )
            duration_text = (
                f" | stage {self._format_elapsed_ms(elapsed_ms - started_ms)}"
            )
            state_text = "STOPPED" if state == "stopped" else "FAILED"
        else:
            state_text = "INFO"
        completed_at = QtCore.QDateTime.currentDateTime().toString(
            "HH:mm:ss.zzz"
        )
        self.run_event_log.appendPlainText(
            f"[{completed_at}] +{self._format_elapsed_ms(elapsed_ms)}"
            f"{duration_text} | {label} {state_text} | {message}"
        )
        self._run_last_event_ms = elapsed_ms
        scrollbar = self.run_event_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        self._update_run_elapsed_label()

    def finish_run_timeline(
        self,
        message: str,
        *,
        success: bool,
        stopped: bool = False,
    ) -> None:
        if not self._run_timeline_active:
            return
        if not success:
            pending_stages = [
                key
                for key in self._run_stage_starts
                if key != "experiment"
            ]
            for key in pending_stages:
                self.record_run_event(
                    key,
                    "stopped" if stopped else "failed",
                    (
                        "Stage interrupted by user stop"
                        if stopped
                        else "Stage interrupted by experiment failure"
                    ),
                )
        self.record_run_event(
            "experiment",
            "completed" if success else ("stopped" if stopped else "failed"),
            message,
        )
        self._run_final_elapsed_ms = int(self._run_elapsed_timer.elapsed())
        self._run_timeline_active = False
        self._run_elapsed_update_timer.stop()
        self.clear_run_log_button.setEnabled(True)
        self._update_run_elapsed_label()

    def update_progress(self, percent: int, message: str) -> None:
        percent = max(0, min(100, int(percent)))
        self.progress.setValue(percent)
        self.run_status.setText(f"{percent}% - {message}")

    def show_result(self, result) -> None:
        rf_settings = getattr(result, "rf_settings", {}) or {}
        program_summary = getattr(result, "program_summary", {}) or {}
        qcs_sweep_mode = str(
            program_summary.get("sweep_execution_mode", "")
        )
        if qcs_sweep_mode:
            if qcs_sweep_mode == "single_point":
                preview = QcsSweepExecutionPreview(
                    mode="none",
                    reasons=("The completed QCS Program contained one point.",),
                )
            elif qcs_sweep_mode == "hybrid_hardware_software":
                preview = QcsSweepExecutionPreview(
                    mode="hybrid",
                    reasons=(
                        "Confirmed by the completed QCS run: "
                        f"{int(program_summary.get('software_iteration_count', 0)):,} "
                        "Python iteration(s), each containing "
                        f"{int(program_summary.get('hardware_points_per_iteration', 0)):,} "
                        "native hardware-sweep point(s).",
                    ),
                )
            elif bool(program_summary.get("hardware_sweep", False)):
                preview = QcsSweepExecutionPreview(
                    mode="hardware",
                    reasons=(
                        "Confirmed by the completed QCS Program; DC voltage "
                        "was swept through waveform amplitudes.",
                    ),
                )
            else:
                reasons = tuple(
                    str(value)
                    for value in program_summary.get(
                        "software_sweep_reasons", ()
                    )
                )
                preview = QcsSweepExecutionPreview(
                    mode="software",
                    reasons=(
                        reasons
                        or ("Confirmed by the completed QCS Program.",)
                    ),
                )
            self.set_qcs_sweep_execution_status(preview)
        output_details = tuple(rf_settings.get("output_details", ()))
        rf_summary = ""
        if output_details:
            if rf_settings.get("backend") == EXECUTION_BACKEND_QCS:
                entries = []
                for item in output_details:
                    calibration = item.get("power_calibration")
                    if isinstance(calibration, Mapping):
                        entries.append(
                            f"RF {item['gen_ch']}: "
                            f"{float(calibration['target_power_dbm']):.6g} dBm "
                            f"from calibration Run "
                            f"{int(calibration['run_id'])}, relative amplitude "
                            f"{float(item['amplitude']):.9g} at "
                            f"{float(item['frequency_hz']) / 1.0e6:.9g} MHz"
                        )
                    else:
                        entries.append(
                            f"RF {item['gen_ch']}: relative amplitude "
                            f"{float(item['amplitude']):.9g} at "
                            f"{float(item['frequency_hz']) / 1.0e6:.9g} MHz"
                        )
                rf_summary = "\nQCS RF output: " + "; ".join(entries)
            else:
                entries = [
                    (
                        f"gen {item['gen_ch']}: ATT1/ATT2 "
                        f"{item['commanded_att1_db']:.2f}/"
                        f"{item['commanded_att2_db']:.2f} dB, "
                        f"{item['filter_type']} filter"
                    )
                    for item in output_details
                ]
                rf_summary = (
                    "\nRF output retained from Front Panel Update: "
                    + "; ".join(entries)
                )
        fir_summary = ""
        ddr_result = getattr(result, "ddr_result", None)
        sample_rate_hz = getattr(ddr_result, "sample_rate_hz", None)
        iq = getattr(ddr_result, "iq", None)
        if sample_rate_hz is not None and iq is not None:
            samples_per_trace = int(np.asarray(iq).shape[-2])
            trace_time_us = (
                samples_per_trace * 1_000_000.0 / float(sample_rate_hz)
            )
            if rf_settings.get("backend") == EXECUTION_BACKEND_QCS:
                readout_details = rf_settings.get("readout_details", {})
                if readout_details.get("hw_demod", False):
                    integration_us = (
                        float(readout_details.get("duration_s", 0.0))
                        * 1.0e6
                    )
                    fir_summary = (
                        f"\nQCS hardware-demodulated IQ: "
                        f"{samples_per_trace:,} value(s)/shot, "
                        f"{integration_us:g} us integration window"
                    )
                else:
                    fir_summary = (
                        f"\nQCS trace: "
                        f"{format_sample_rate_hz(sample_rate_hz)}, "
                        f"{samples_per_trace:,} samples/trace = "
                        f"{trace_time_us:g} us"
                    )
            else:
                fir_summary = (
                    f"\nFIR DDR: {format_sample_rate_hz(sample_rate_hz)}, "
                    f"{samples_per_trace:,} samples/trace = "
                    f"{trace_time_us:g} us"
                )
        self.set_running(
            False,
            f"Run {result.run_id}, {result.row_count} IQ samples\n"
            f"{result.database_path}{fir_summary}{rf_summary}",
        )


class QickExperimentWorker(QtCore.QObject):
    """Run blocking Pyro/QICK/QCoDeS work outside the GUI thread."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress_changed = QtCore.pyqtSignal(int, str)
    event_changed = QtCore.pyqtSignal(str, str, str)

    def __init__(self, kwargs: dict, parent=None):
        super().__init__(parent)
        self._kwargs = kwargs

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            kwargs = dict(self._kwargs)
            kwargs["progress_callback"] = self.progress_changed.emit
            kwargs["event_callback"] = self.event_changed.emit
            result = run_qick_qcodes_experiment(**kwargs)
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(result)


class QcsExperimentWorker(QtCore.QObject):
    """Run blocking Keysight QCS/QCoDeS work outside the GUI thread."""

    finished = QtCore.pyqtSignal(object)
    partial_result = QtCore.pyqtSignal(object)
    stopped = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)
    progress_changed = QtCore.pyqtSignal(int, str)
    event_changed = QtCore.pyqtSignal(str, str, str)
    stop_status_changed = QtCore.pyqtSignal(str)

    def __init__(self, kwargs: dict, parent=None):
        super().__init__(parent)
        self._kwargs = kwargs
        self._cancellation = QcsCancellationController(
            status_callback=self.stop_status_changed.emit
        )

    def request_stop(self) -> bool:
        """Request a thread-safe cooperative stop plus HCL program abort."""
        return self._cancellation.request_stop()

    def is_stop_requested(self) -> bool:
        return self._cancellation.is_stop_requested()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            kwargs = dict(self._kwargs)
            kwargs["progress_callback"] = self.progress_changed.emit
            kwargs["event_callback"] = self.event_changed.emit
            kwargs["partial_callback"] = self.partial_result.emit
            kwargs["cancellation"] = self._cancellation
            result = run_qcs_qcodes_experiment(**kwargs)
        except QcsExperimentCancelled as exc:
            self._cancellation.mark_finished()
            stored = getattr(exc, "stored_result", None)
            if stored is not None:
                message = (
                    "Stopped by user; completed QCS data was saved as "
                    f"QCoDeS Run {int(stored.run_id)} at "
                    f"{stored.database_path}"
                )
            else:
                message = (
                    "Stopped by user; QCS execution ended and DC safety "
                    "cleanup completed"
                )
            self.stopped.emit(message)
            return
        except Exception:
            self._cancellation.mark_finished()
            self.failed.emit(traceback.format_exc())
            return
        self._cancellation.mark_finished()
        self.finished.emit(result)


class QickProgramWorker(QtCore.QObject):
    """Compile a QICK program from live HWH metadata without running it."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, connection_config, program_kwargs: dict, parent=None):
        super().__init__(parent)
        self._connection_config = connection_config
        self._program_kwargs = dict(program_kwargs)

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            _soc, soccfg = connect_qick(self._connection_config)
            program = build_qick_program(soccfg, **self._program_kwargs)
            program.compile()
            summary = program.summary() if hasattr(program, "summary") else {}
            result = {
                "assembly": program.asm(),
                "instruction_count": len(program.prog_list),
                "machine_word_count": len(program.binprog),
                "summary": summary,
                "program": program,
            }
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(result)


class QickConfigurationWorker(QtCore.QObject):
    """Fetch the live HWH-derived config and reconstruct physical SMA routing."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, connection_config, parent=None):
        super().__init__(parent)
        self._connection_config = connection_config

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            _soc, soccfg = connect_qick(self._connection_config)
            configuration = identify_qick_front_panel(soccfg)
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(configuration)


class QcsHardwareIdentificationWorker(QtCore.QObject):
    """Fetch the QCS controller's read-only installed-module inventory."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, ip_address: str, parent=None):
        super().__init__(parent)
        self._ip_address = str(ip_address)

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            configuration = identify_qcs_hardware_configuration(
                self._ip_address
            )
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(configuration)


class QcsMapperSaveWorker(QtCore.QObject):
    """Build or validate one native QCS mapper outside the GUI thread."""

    finished = QtCore.pyqtSignal(int, object)
    failed = QtCore.pyqtSignal(int, str)

    def __init__(
        self,
        revision: int,
        configuration: Mapping[str, object],
        mapper_path: str,
        *,
        write_mapper: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._revision = int(revision)
        self._configuration = dict(configuration)
        self._mapper_path = str(mapper_path)
        self._write_mapper = bool(write_mapper)

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            if self._write_mapper:
                mapper_path = _qcs_front_panel_module.save_qcs_channel_mapper(
                    self._configuration,
                    self._mapper_path,
                )
                state = QCS_HARDWARE_STATE_SAVED
            else:
                (
                    _qcs_front_panel_module
                    .validate_imported_qcs_role_configuration(
                        self._configuration,
                        self._mapper_path,
                    )
                )
                mapper_path = Path(self._mapper_path).expanduser().resolve()
                state = QCS_HARDWARE_STATE_IMPORTED
            result = {
                "mapper_path": str(mapper_path),
                "hardware_configuration_state": state,
                "hardware_mapper_sha256": (
                    _qcs_front_panel_module.qcs_mapper_file_sha256(
                        mapper_path
                    )
                ),
            }
        except Exception:
            self.failed.emit(self._revision, traceback.format_exc())
            return
        self.finished.emit(self._revision, result)


class QickRfOutputConfigurationWorker(QtCore.QObject):
    """Apply one committed RF output path without running an experiment."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, connection_config, spec: QickRfPulseSpec, parent=None):
        super().__init__(parent)
        self._connection_config = connection_config
        self._spec = spec

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            soc, _soccfg = connect_qick(self._connection_config)
            details = configure_rf_output(soc, self._spec)
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(details)


class DetailedErrorMessageBox(QtWidgets.QMessageBox):
    """Error message with expandable traceback and one-click clipboard copy."""

    def __init__(self, title: str, summary: str, details: str, parent=None):
        super().__init__(parent)
        self._copy_text = f"{title}\n{summary}\n\n{details}".strip()
        self.setIcon(QtWidgets.QMessageBox.Critical)
        self.setWindowTitle(title)
        self.setText(summary)
        self.setDetailedText(details)
        self.setStandardButtons(QtWidgets.QMessageBox.Close)
        self.copy_button = self.addButton(
            "Copy Details", QtWidgets.QMessageBox.ActionRole
        )
        self.copy_button.setToolTip("Copy the complete error and traceback")
        self.copy_button.clicked.connect(self.copy_details)

    def copy_details(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self._copy_text)


class QickAssemblyDialog(QtWidgets.QDialog):
    """Read-only tProcessor assembly viewer with copy and save actions."""

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self._assembly = str(result["assembly"])
        self.setWindowTitle("QICK tProcessor Assembly")
        self.resize(1000, 760)

        layout = QtWidgets.QVBoxLayout(self)
        instruction_count = int(result.get("instruction_count", 0))
        machine_word_count = int(result.get("machine_word_count", 0))
        self.summary_label = QtWidgets.QLabel(
            f"{instruction_count:,} assembly instructions, "
            f"{machine_word_count:,} compiled machine words"
        )
        layout.addWidget(self.summary_label)

        self.assembly_text = QtWidgets.QPlainTextEdit(self)
        self.assembly_text.setReadOnly(True)
        self.assembly_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.assembly_text.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        )
        self.assembly_text.setPlainText(self._assembly)
        layout.addWidget(self.assembly_text, 1)

        self.status_label = QtWidgets.QLabel(
            "Compiled from the current GUI settings and connected QICK HWH."
        )
        layout.addWidget(self.status_label)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.copy_button = buttons.addButton(
            "Copy", QtWidgets.QDialogButtonBox.ActionRole
        )
        self.save_button = buttons.addButton(
            "Save As...", QtWidgets.QDialogButtonBox.ActionRole
        )
        self.copy_button.clicked.connect(self._copy_assembly)
        self.save_button.clicked.connect(self._save_assembly)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _copy_assembly(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self._assembly)
        self.status_label.setText("Assembly copied to the clipboard.")

    def _save_assembly(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save QICK tProcessor assembly",
            "qick_program.asm",
            "QICK assembly (*.asm);;Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        output_path = Path(path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".asm")
        output_path.write_text(self._assembly, encoding="utf-8")
        self.status_label.setText(f"Saved to {output_path}")


class QcsProgramCodeDialog(QtWidgets.QDialog):
    """Read-only offline QCS source viewer with copy and save actions."""

    def __init__(self, code: str, parent=None):
        super().__init__(parent)
        self._code = str(code)
        self.setWindowTitle("QCS Program Source Preview")
        self.resize(1000, 760)

        layout = QtWidgets.QVBoxLayout(self)
        self.summary_label = QtWidgets.QLabel(
            "Generated locally from the current AWG Tuning settings. No "
            "ChannelMapper was loaded, and no QCS compile, connection, or "
            "hardware submission was performed."
        )
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.code_text = QtWidgets.QPlainTextEdit(self)
        self.code_text.setReadOnly(True)
        self.code_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.code_text.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        )
        self.code_text.setPlainText(self._code)
        layout.addWidget(self.code_text, 1)

        self.status_label = QtWidgets.QLabel("Offline source preview ready.")
        layout.addWidget(self.status_label)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.copy_button = buttons.addButton(
            "Copy", QtWidgets.QDialogButtonBox.ActionRole
        )
        self.save_button = buttons.addButton(
            "Save As...", QtWidgets.QDialogButtonBox.ActionRole
        )
        self.copy_button.clicked.connect(self._copy_code)
        self.save_button.clicked.connect(self._save_code)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _copy_code(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self._code)
        self.status_label.setText("QCS source copied to the clipboard.")

    def _save_code(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save QCS Program source preview",
            "qcs_program_preview.py",
            "Python files (*.py);;Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        output_path = Path(path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".py")
        output_path.write_text(self._code, encoding="utf-8")
        self.status_label.setText(f"Saved to {output_path}")


class AwgMetadataDialog(QtWidgets.QDialog):
    """Preview and export parametric or expanded AWG waveform metadata."""

    def __init__(
        self,
        sequence,
        *,
        fabric_mhz: float,
        full_scale_mv: float,
        parent=None,
    ):
        super().__init__(parent)
        self._sequence = sequence
        self._fabric_mhz = float(fabric_mhz)
        self._full_scale_mv = float(full_scale_mv)
        self._recipe = dict(build_awg_waveform_recipe(
            sequence,
            fabric_mhz=self._fabric_mhz,
            full_scale_mv=self._full_scale_mv,
        ))
        self._point_record = None
        self.setWindowTitle("AWG Waveform Metadata")
        self.resize(1100, 760)

        layout = QtWidgets.QVBoxLayout(self)
        point_count = int(self._recipe["point_count"])
        sweep_shape = tuple(int(value) for value in self._recipe["sweep_shape"])
        output_count = len(self._recipe["output_names"])
        self.summary_label = QtWidgets.QLabel(
            f"Compact recipe: {output_count} output(s), "
            f"{point_count:,} sweep point(s), shape {sweep_shape or (1,)}. "
            "Point previews are expanded on demand."
        )
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        selector_layout = QtWidgets.QHBoxLayout()
        selector_layout.addWidget(QtWidgets.QLabel("Sweep point index:"))
        self.point_index = QtWidgets.QSpinBox(self)
        self.point_index.setRange(0, min(point_count - 1, 2_147_483_647))
        self.point_index.valueChanged.connect(self._refresh_point)
        selector_layout.addWidget(self.point_index)
        self.coordinate_label = QtWidgets.QLabel()
        self.coordinate_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        selector_layout.addWidget(self.coordinate_label, 1)
        layout.addLayout(selector_layout)

        tabs = QtWidgets.QTabWidget(self)
        self.vertex_table = QtWidgets.QTableWidget(self)
        self.vertex_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.vertex_table.setAlternatingRowColors(True)
        tabs.addTab(self.vertex_table, "Expanded Point Preview")

        self.recipe_text = QtWidgets.QPlainTextEdit(self)
        self.recipe_text.setReadOnly(True)
        self.recipe_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.recipe_text.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        )
        self.recipe_text.setPlainText(
            json.dumps(
                self._recipe,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        tabs.addTab(self.recipe_text, "Parametric Recipe")
        layout.addWidget(tabs, 1)

        self.status_label = QtWidgets.QLabel(
            "The QCoDeS default stores only this compact recipe."
        )
        self.status_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.save_recipe_button = buttons.addButton(
            "Save Recipe JSON...",
            QtWidgets.QDialogButtonBox.ActionRole,
        )
        self.save_point_button = buttons.addButton(
            "Save Selected Point JSON...",
            QtWidgets.QDialogButtonBox.ActionRole,
        )
        self.save_all_button = buttons.addButton(
            "Save All Points JSONL...",
            QtWidgets.QDialogButtonBox.ActionRole,
        )
        self.save_recipe_button.clicked.connect(self._save_recipe)
        self.save_point_button.clicked.connect(self._save_selected_point)
        self.save_all_button.clicked.connect(self._save_all_points)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh_point()

    @property
    def recipe(self) -> Mapping:
        return self._recipe

    @property
    def point_record(self) -> Mapping:
        return self._point_record

    def _refresh_point(self) -> None:
        record = dict(build_awg_vertex_record(
            self._sequence,
            self.point_index.value(),
            fabric_mhz=self._fabric_mhz,
            full_scale_mv=self._full_scale_mv,
        ))
        self._point_record = record
        coordinates = record["sweep_coordinate"]
        coordinate_parts = []
        for axis, value in zip(self._recipe["sweep_axes"], coordinates):
            target = " / ".join(
                part
                for part in (
                    str(axis.get("output_name", "")).strip(),
                    str(axis.get("segment_name", "")).strip(),
                )
                if part
            )
            unit = str(axis.get("coordinate_unit", "")).strip()
            coordinate_parts.append(
                f"{target or axis.get('axis_kind', 'sweep')} = "
                f"{float(value):.9g}{(' ' + unit) if unit else ''}"
            )
        self.coordinate_label.setText(
            " | ".join(coordinate_parts)
            if coordinate_parts
            else "Base waveform (no sweep axes)"
        )

        output_names = tuple(record["output_names"])
        headers = ["Time [cycles]", "Time [us]"]
        for name in output_names:
            headers.extend([
                f"{name} virtual [mV]",
                f"{name} physical [mV]",
            ])
        self.vertex_table.setColumnCount(len(headers))
        self.vertex_table.setHorizontalHeaderLabels(headers)
        times_cycles = record["time_cycles"]
        times_us = record["time_us"]
        self.vertex_table.setRowCount(len(times_cycles))
        for row, (time_cycles, time_us) in enumerate(
            zip(times_cycles, times_us)
        ):
            values = [f"{float(time_cycles):.9g}", f"{float(time_us):.9g}"]
            for name in output_names:
                values.extend([
                    f"{float(record['virtual_values_mv'][name][row]):.9g}",
                    f"{float(record['physical_values_mv'][name][row]):.9g}",
                ])
            for column, value in enumerate(values):
                self.vertex_table.setItem(
                    row,
                    column,
                    QtWidgets.QTableWidgetItem(value),
                )
        self.vertex_table.resizeColumnsToContents()

    @staticmethod
    def _write_json(path: str, payload: Mapping) -> Path:
        output_path = Path(path)
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")
        output_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return output_path

    def _save_recipe(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save compact AWG waveform recipe",
            "awg_waveform_recipe.json",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        output_path = self._write_json(path, self._recipe)
        self.status_label.setText(f"Compact recipe saved to {output_path}")

    def _save_selected_point(self) -> None:
        point_index = self.point_index.value()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save expanded AWG waveform point",
            f"awg_waveform_point_{point_index}.json",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        payload = {
            "schema": "qick-awg-waveform-selected-point-v1",
            "recipe": self._recipe,
            "waveform_point": self._point_record,
        }
        output_path = self._write_json(path, payload)
        self.status_label.setText(
            f"Expanded point {point_index:,} saved to {output_path}"
        )

    def _save_all_points(self) -> None:
        point_count = int(self._recipe["point_count"])
        if point_count > 100_000:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Export all expanded AWG vertices?",
                (
                    f"This will explicitly expand {point_count:,} sweep points. "
                    "The JSONL writer streams points to disk, but the file can "
                    "be very large and take a long time. Continue?"
                ),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save all expanded AWG waveform points",
            "awg_waveform_vertices.jsonl",
            "JSON Lines (*.jsonl);;All files (*)",
        )
        if not path:
            return

        output_path = Path(path)
        if output_path.suffix.lower() != ".jsonl":
            output_path = output_path.with_suffix(".jsonl")
        progress = QtWidgets.QProgressDialog(
            "Expanding AWG waveform points...",
            "Cancel",
            0,
            point_count,
            self,
        )
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def update_progress(completed: int, total: int) -> bool:
            progress.setMaximum(int(total))
            progress.setValue(int(completed))
            progress.setLabelText(
                f"Expanded {completed:,} of {total:,} AWG waveform points"
            )
            QtWidgets.QApplication.processEvents()
            return not progress.wasCanceled()

        try:
            output_path = write_awg_vertex_metadata_jsonl(
                self._sequence,
                output_path,
                fabric_mhz=self._fabric_mhz,
                full_scale_mv=self._full_scale_mv,
                progress_callback=update_progress,
            )
        except RuntimeError as exc:
            output_path.unlink(missing_ok=True)
            if progress.wasCanceled():
                self.status_label.setText("Expanded AWG metadata export canceled.")
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "AWG metadata export failed",
                    str(exc),
                )
            return
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            QtWidgets.QMessageBox.warning(
                self,
                "AWG metadata export failed",
                str(exc),
            )
            return
        finally:
            progress.close()
        self.status_label.setText(
            f"Expanded {point_count:,} point(s) to {output_path}"
        )


class QickExportDialog(QtWidgets.QDialog):
    """Collect QICK timing, channel-map, repetition, and sweep settings."""

    def __init__(
        self,
        pulse_count: int,
        set_names: Tuple[str, ...],
        parent=None,
        default_set_index: int = 0,
        initial_rf_spec: Optional[QickRfPulseSpec] = None,
        initial_rf_specs: Optional[Sequence[QickRfPulseSpec]] = None,
        initial_ddr_readout_spec: Optional[QickDdrReadoutSpec] = None,
        initial_sweep: Optional[QickSweepAxisSpec] = None,
        initial_sweeps: Optional[Sequence[QickSweepAxisSpec]] = None,
        initial_cross_capacitance=None,
        initial_fabric_mhz: float = DEFAULT_QICK_FABRIC_MHZ,
        initial_tproc_mhz: float = DEFAULT_QICK_TPROC_MHZ,
        initial_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
        initial_awg_channels: Optional[Sequence[int]] = None,
        initial_repetitions: int = 1,
        initial_bias_t_enabled: bool = False,
        initial_bias_t_compensation_type: str = "dc",
        initial_bias_t_compensation_mv: Optional[float] = None,
        initial_bias_t_mode: str = "fixed_voltage",
        initial_bias_t_duration_us: float = DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
        initial_bias_t_filter_tau_us: float = DEFAULT_BIAS_T_FILTER_TAU_US,
    ):
        super().__init__(parent)
        self.setWindowTitle("QICK export settings")
        self.resize(660, 780)
        self._pulse_count = pulse_count
        if initial_rf_spec is not None and initial_rf_specs is not None:
            raise ValueError("use either initial_rf_spec or initial_rf_specs, not both")
        self._rf_pulse_specs = tuple(
            initial_rf_specs
            if initial_rf_specs is not None
            else (() if initial_rf_spec is None else (initial_rf_spec,))
        )
        self._ddr_readout_spec = initial_ddr_readout_spec
        if initial_sweep is not None and initial_sweeps is not None:
            raise ValueError("use either initial_sweep or initial_sweeps, not both")
        supplied_sweeps = list(
            initial_sweeps
            if initial_sweeps is not None
            else (() if initial_sweep is None else (initial_sweep,))
        )
        self._dialog_ramp_sweeps = [
            spec
            for spec in supplied_sweeps
            if isinstance(spec, QickRampRateSweepSpec)
        ]
        self._dialog_hold_sweeps = [
            spec
            for spec in supplied_sweeps
            if isinstance(spec, QickHoldDurationSweepSpec)
        ]
        self._dialog_sweep_specs = [
            spec
            for spec in supplied_sweeps
            if isinstance(spec, QickSweepSpec)
        ]
        self._cross_capacitance = np.asarray(
            np.eye(pulse_count)
            if initial_cross_capacitance is None
            else initial_cross_capacitance,
            dtype=float,
        ).copy()
        if self._cross_capacitance.shape != (pulse_count, pulse_count):
            raise ValueError("cross-capacitance matrix size must match pulse count")

        outer_layout = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget(scroll)
        form = QtWidgets.QFormLayout(content)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        self.fabric_mhz = QtWidgets.QDoubleSpinBox()
        self.fabric_mhz.setRange(1.0, 5000.0)
        self.fabric_mhz.setDecimals(6)
        self.fabric_mhz.setValue(float(initial_fabric_mhz))
        self.fabric_mhz.setSuffix(" MHz")

        self.tproc_mhz = QtWidgets.QDoubleSpinBox()
        self.tproc_mhz.setRange(1.0, 5000.0)
        self.tproc_mhz.setDecimals(6)
        self.tproc_mhz.setValue(float(initial_tproc_mhz))
        self.tproc_mhz.setSuffix(" MHz")

        self.full_scale_mv = QtWidgets.QDoubleSpinBox()
        self.full_scale_mv.setRange(1.0, 1.0e6)
        self.full_scale_mv.setDecimals(6)
        self.full_scale_mv.setValue(float(initial_full_scale_mv))
        self.full_scale_mv.setSuffix(" mV")
        self._sweep_display_scale_mv = float(initial_full_scale_mv)

        if initial_awg_channels is None:
            initial_awg_channels = DEFAULT_QSTL_AWG_CHANNELS[:pulse_count]
        self.awg_channels = QtWidgets.QLineEdit(
            ", ".join(str(index) for index in initial_awg_channels)
        )
        self.repetitions = QtWidgets.QSpinBox()
        self.repetitions.setRange(1, 1_000_000)
        self.repetitions.setValue(int(initial_repetitions))
        self.bias_t_group = QtWidgets.QGroupBox("Bias-T compensation")
        self.bias_t_group.setCheckable(True)
        self.bias_t_group.setChecked(bool(initial_bias_t_enabled))
        bias_t_form = QtWidgets.QFormLayout(self.bias_t_group)
        self.bias_t_type = QtWidgets.QComboBox()
        self.bias_t_type.addItem("DC compensation", "dc")
        self.bias_t_type.addItem("Filter compensation", "filter")
        type_index = self.bias_t_type.findData(
            str(initial_bias_t_compensation_type)
        )
        if type_index < 0:
            raise ValueError(
                "Bias-T compensation type must be one of "
                f"{BIAS_T_COMPENSATION_TYPES}"
            )
        self.bias_t_type.setCurrentIndex(type_index)
        self.bias_t_mode = QtWidgets.QComboBox()
        self.bias_t_mode.addItem("Fixed voltage (adjust time)", "fixed_voltage")
        self.bias_t_mode.addItem("Fixed time (adjust voltage)", "fixed_time")
        mode_index = self.bias_t_mode.findData(str(initial_bias_t_mode))
        if mode_index < 0:
            raise ValueError(
                f"Bias-T compensation mode must be one of {BIAS_T_COMPENSATION_MODES}"
            )
        self.bias_t_mode.setCurrentIndex(mode_index)
        self.bias_t_compensation_mv = QtWidgets.QDoubleSpinBox()
        self.bias_t_compensation_mv.setRange(0.001, float(initial_full_scale_mv))
        self.bias_t_compensation_mv.setDecimals(6)
        self.bias_t_compensation_mv.setSuffix(" mV")
        self.bias_t_compensation_mv.setValue(float(
            initial_full_scale_mv * DEFAULT_BIAS_T_COMPENSATION_FRACTION
            if initial_bias_t_compensation_mv is None
            else initial_bias_t_compensation_mv
        ))
        self.bias_t_duration_us = QtWidgets.QDoubleSpinBox()
        self.bias_t_duration_us.setRange(1.0e-6, 1.0e9)
        self.bias_t_duration_us.setDecimals(6)
        self.bias_t_duration_us.setSuffix(" us")
        self.bias_t_duration_us.setValue(float(initial_bias_t_duration_us))
        self.bias_t_filter_tau_us = QtWidgets.QDoubleSpinBox()
        self.bias_t_filter_tau_us.setRange(1.0e-6, 1.0e12)
        self.bias_t_filter_tau_us.setDecimals(6)
        self.bias_t_filter_tau_us.setSuffix(" us")
        self.bias_t_filter_tau_us.setValue(float(initial_bias_t_filter_tau_us))
        bias_t_form.addRow("Compensation type:", self.bias_t_type)
        bias_t_form.addRow("DC control mode:", self.bias_t_mode)
        bias_t_form.addRow("DC voltage:", self.bias_t_compensation_mv)
        bias_t_form.addRow("DC time:", self.bias_t_duration_us)
        bias_t_form.addRow("Filter time constant (tau):", self.bias_t_filter_tau_us)
        self.full_scale_mv.valueChanged.connect(
            lambda value: self.bias_t_compensation_mv.setMaximum(
                max(0.001, float(value))
            )
        )
        self.bias_t_mode.currentIndexChanged.connect(
            self._update_bias_t_mode_controls
        )
        self.bias_t_type.currentIndexChanged.connect(
            self._update_bias_t_mode_controls
        )
        self._update_bias_t_mode_controls()

        form.addRow("AWG fabric clock:", self.fabric_mhz)
        form.addRow("tProcessor clock:", self.tproc_mhz)
        form.addRow("QICK full scale (+/-):", self.full_scale_mv)
        form.addRow("AWG generator indices:", self.awg_channels)
        form.addRow("Repetitions per sweep point:", self.repetitions)
        form.addRow(self.bias_t_group)

        sweep_group = QtWidgets.QGroupBox("Independent Cartesian voltage sweeps")
        sweep_group.setCheckable(True)
        sweep_group.setChecked(False)
        sweep_form = QtWidgets.QFormLayout(sweep_group)
        self.sweep_group = sweep_group
        self.sweep_output = QtWidgets.QComboBox()
        self.sweep_output.addItems([f"awg_{index}" for index in range(pulse_count)])
        self.sweep_segment = QtWidgets.QComboBox()
        self.sweep_segment.addItems(list(set_names))
        self.sweep_start = QtWidgets.QDoubleSpinBox()
        self.sweep_stop = QtWidgets.QDoubleSpinBox()
        for widget, value in (
            (self.sweep_start, -0.2 * self._sweep_display_scale_mv),
            (self.sweep_stop, 0.8 * self._sweep_display_scale_mv),
        ):
            widget.setRange(
                -self._sweep_display_scale_mv,
                self._sweep_display_scale_mv,
            )
            widget.setDecimals(6)
            widget.setSingleStep(max(0.001, self._sweep_display_scale_mv / 100.0))
            widget.setSuffix(" mV")
            widget.setValue(value)
        self.sweep_count = QtWidgets.QSpinBox()
        self.sweep_count.setRange(1, 1_000_000)
        self.sweep_count.setValue(9)
        sweep_form.addRow("Output:", self.sweep_output)
        sweep_form.addRow("SET segment:", self.sweep_segment)
        sweep_form.addRow("Start voltage:", self.sweep_start)
        sweep_form.addRow("Stop voltage:", self.sweep_stop)
        sweep_form.addRow("Point count:", self.sweep_count)
        self.sweep_table = QtWidgets.QTableWidget(0, 5)
        self.sweep_table.setHorizontalHeaderLabels(
            ["Output", "SET", "Start (mV)", "Stop (mV)", "Count"]
        )
        self.sweep_table.verticalHeader().setVisible(False)
        self.sweep_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.sweep_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.sweep_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.sweep_table.setMaximumHeight(140)
        sweep_header = self.sweep_table.horizontalHeader()
        sweep_header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        sweep_form.addRow("Active axes:", self.sweep_table)
        sweep_buttons = QtWidgets.QHBoxLayout()
        add_sweep = QtWidgets.QPushButton("Add/update axis")
        remove_sweep = QtWidgets.QPushButton("Remove selected")
        add_sweep.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton))
        remove_sweep.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton))
        sweep_buttons.addWidget(add_sweep)
        sweep_buttons.addWidget(remove_sweep)
        sweep_form.addRow(sweep_buttons)
        self.sweep_total = QtWidgets.QLabel()
        sweep_form.addRow("Cartesian total:", self.sweep_total)
        form.addRow(sweep_group)
        add_sweep.clicked.connect(self._upsert_current_sweep)
        remove_sweep.clicked.connect(self._remove_selected_sweep)
        self.sweep_table.itemSelectionChanged.connect(self._load_selected_sweep)
        self.sweep_count.valueChanged.connect(self._refresh_sweep_total)
        self.sweep_group.toggled.connect(self._refresh_sweep_total)
        self.full_scale_mv.valueChanged.connect(
            self._rescale_sweep_voltage_controls
        )
        if self._dialog_sweep_specs:
            self.sweep_group.setChecked(True)
        self._refresh_sweep_table(select_row=0 if self._dialog_sweep_specs else None)

        cross_row = QtWidgets.QHBoxLayout()
        self.cross_capacitance_summary = QtWidgets.QLabel()
        edit_cross_capacitance = QtWidgets.QPushButton("Edit matrix...")
        edit_cross_capacitance.clicked.connect(self._edit_cross_capacitance)
        cross_row.addWidget(self.cross_capacitance_summary)
        cross_row.addStretch(1)
        cross_row.addWidget(edit_cross_capacitance)
        form.addRow("Virtual-to-physical matrix:", cross_row)
        self._refresh_cross_capacitance_summary()

        note = QtWidgets.QLabel(
            "All exported ports must have identical SET/RAMP timing. "
            "Sweep endpoints are entered and displayed in mV; command generation "
            "normalizes them using the configured full scale. RF output and "
            "readout settings come from the main-window RF tabs."
        )
        note.setWordWrap(True)
        form.addRow(note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _current_sweep_spec(self) -> QickSweepSpec:
        full_scale_mv = self.full_scale_mv.value()
        return QickSweepSpec(
            segment_name=self.sweep_segment.currentText(),
            output_name=self.sweep_output.currentText(),
            start=self.sweep_start.value() / full_scale_mv,
            stop=self.sweep_stop.value() / full_scale_mv,
            count=self.sweep_count.value(),
        )

    def _update_bias_t_mode_controls(self, *_args) -> None:
        filter_mode = self.bias_t_type.currentData() == "filter"
        fixed_time = self.bias_t_mode.currentData() == "fixed_time"
        self.bias_t_mode.setEnabled(not filter_mode)
        self.bias_t_compensation_mv.setEnabled(not filter_mode and not fixed_time)
        self.bias_t_duration_us.setEnabled(not filter_mode and fixed_time)
        self.bias_t_filter_tau_us.setEnabled(filter_mode)

    def _rescale_sweep_voltage_controls(self, value: float) -> None:
        new_scale_mv = float(value)
        old_scale_mv = self._sweep_display_scale_mv
        if old_scale_mv <= 0.0 or new_scale_mv <= 0.0:
            return
        for widget in (self.sweep_start, self.sweep_stop):
            with QtCore.QSignalBlocker(widget):
                normalized_value = widget.value() / old_scale_mv
                widget.setRange(-new_scale_mv, new_scale_mv)
                widget.setSingleStep(max(0.001, new_scale_mv / 100.0))
                widget.setValue(normalized_value * new_scale_mv)
        self._sweep_display_scale_mv = new_scale_mv
        selected_row = self.sweep_table.currentRow()
        self._refresh_sweep_table()
        if 0 <= selected_row < self.sweep_table.rowCount():
            with QtCore.QSignalBlocker(self.sweep_table):
                self.sweep_table.selectRow(selected_row)

    def _refresh_sweep_table(self, select_row: Optional[int] = None) -> None:
        with QtCore.QSignalBlocker(self.sweep_table):
            self.sweep_table.setRowCount(len(self._dialog_sweep_specs))
            for row, spec in enumerate(self._dialog_sweep_specs):
                values = (
                    spec.output_name,
                    spec.segment_name,
                    f"{spec.start * self.full_scale_mv.value():.6g}",
                    f"{spec.stop * self.full_scale_mv.value():.6g}",
                    str(spec.count),
                )
                for column, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.sweep_table.setItem(row, column, item)
            if select_row is not None and self._dialog_sweep_specs:
                self.sweep_table.selectRow(
                    min(max(0, select_row), len(self._dialog_sweep_specs) - 1)
                )
        if select_row is not None and self._dialog_sweep_specs:
            self._load_selected_sweep()
        self._refresh_sweep_total()

    def _load_selected_sweep(self) -> None:
        row = self.sweep_table.currentRow()
        if row < 0 or row >= len(self._dialog_sweep_specs):
            return
        spec = self._dialog_sweep_specs[row]
        output_index = self.sweep_output.findText(spec.output_name)
        segment_index = self.sweep_segment.findText(spec.segment_name)
        if output_index >= 0:
            self.sweep_output.setCurrentIndex(output_index)
        if segment_index >= 0:
            self.sweep_segment.setCurrentIndex(segment_index)
        self.sweep_start.setValue(spec.start * self.full_scale_mv.value())
        self.sweep_stop.setValue(spec.stop * self.full_scale_mv.value())
        self.sweep_count.setValue(spec.count)

    def _upsert_current_sweep(self) -> None:
        spec = self._current_sweep_spec()
        target = (spec.segment_name, spec.output_name)
        for row, current in enumerate(self._dialog_sweep_specs):
            if (current.segment_name, current.output_name) == target:
                self._dialog_sweep_specs[row] = spec
                self._refresh_sweep_table(select_row=row)
                return
        self._dialog_sweep_specs.append(spec)
        self._refresh_sweep_table(select_row=len(self._dialog_sweep_specs) - 1)

    def _remove_selected_sweep(self) -> None:
        row = self.sweep_table.currentRow()
        if row < 0 or row >= len(self._dialog_sweep_specs):
            return
        self._dialog_sweep_specs.pop(row)
        self._refresh_sweep_table(
            select_row=(
                min(row, len(self._dialog_sweep_specs) - 1)
                if self._dialog_sweep_specs
                else None
            )
        )

    def _effective_sweeps(self) -> Tuple[QickSweepAxisSpec, ...]:
        duration_prefix = (
            tuple(self._dialog_ramp_sweeps)
            + tuple(self._dialog_hold_sweeps)
        )
        if not self.sweep_group.isChecked():
            return duration_prefix
        if not self._dialog_sweep_specs:
            return duration_prefix + (self._current_sweep_spec(),)
        specs = list(self._dialog_sweep_specs)
        row = self.sweep_table.currentRow()
        if 0 <= row < len(specs):
            specs[row] = self._current_sweep_spec()
        targets = [(spec.segment_name, spec.output_name) for spec in specs]
        if len(set(targets)) != len(targets):
            raise ValueError("each Cartesian voltage sweep must target a unique output/SET")
        return duration_prefix + tuple(specs)

    def _refresh_sweep_total(self, *_args) -> None:
        counts = [
            spec.count
            for spec in (
                *self._dialog_ramp_sweeps,
                *self._dialog_hold_sweeps,
            )
        ]
        if self.sweep_group.isChecked():
            voltage_counts = [spec.count for spec in self._dialog_sweep_specs]
            row = self.sweep_table.currentRow()
            if 0 <= row < len(voltage_counts):
                voltage_counts[row] = self.sweep_count.value()
            elif not voltage_counts:
                voltage_counts = [self.sweep_count.value()]
            counts.extend(voltage_counts)
        if not counts:
            self.sweep_total.setText("disabled")
            return
        self.sweep_total.setText(
            " x ".join(str(count) for count in counts)
            + f" = {prod(counts)} combinations"
        )

    def _edit_cross_capacitance(self) -> None:
        dialog = CrossCapacitanceDialog(
            self._pulse_count,
            self._cross_capacitance,
            self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        self._cross_capacitance = dialog.matrix()
        self._refresh_cross_capacitance_summary()

    def _refresh_cross_capacitance_summary(self) -> None:
        off_diagonal = self._cross_capacitance.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        nonzero = int(np.count_nonzero(np.abs(off_diagonal) > 1.0e-15))
        self.cross_capacitance_summary.setText(
            f"{self._pulse_count} x {self._pulse_count}, {nonzero} active terms"
        )

    def _parse_channels(self) -> Tuple[int, ...]:
        fields = [field.strip() for field in self.awg_channels.text().split(",")]
        if any(not field for field in fields):
            raise ValueError("AWG generator indices must be comma-separated integers")
        try:
            channels = tuple(int(field) for field in fields)
        except (TypeError, ValueError) as exc:
            raise ValueError("AWG generator indices must be integers") from exc
        if len(channels) != self._pulse_count:
            raise ValueError(f"exactly {self._pulse_count} AWG generator indices are required")
        if len(set(channels)) != len(channels) or any(channel < 0 for channel in channels):
            raise ValueError("AWG generator indices must be unique and nonnegative")
        return channels

    def values(self) -> dict:
        awg_channels = self._parse_channels()
        sweep_specs = self._effective_sweeps()
        sweep = sweep_specs[0] if len(sweep_specs) == 1 else None
        sweeps = sweep_specs if len(sweep_specs) > 1 else None
        rf_channels = tuple(spec.gen_ch for spec in self._rf_pulse_specs)
        if any(channel in awg_channels for channel in rf_channels):
            raise ValueError("RF generator indices must differ from all AWG tuning indices")
        if len(set(rf_channels)) != len(rf_channels):
            raise ValueError("RF generator indices must be unique")
        return {
            "fabric_mhz": self.fabric_mhz.value(),
            "tproc_mhz": self.tproc_mhz.value(),
            "full_scale_mv": self.full_scale_mv.value(),
            "awg_channels": awg_channels,
            "repetitions_per_sweep": self.repetitions.value(),
            "bias_t_compensation_enabled": self.bias_t_group.isChecked(),
            "bias_t_compensation_type": str(self.bias_t_type.currentData()),
            "bias_t_compensation_voltage_mv": self.bias_t_compensation_mv.value(),
            "bias_t_compensation_mode": str(self.bias_t_mode.currentData()),
            "bias_t_compensation_duration_us": self.bias_t_duration_us.value(),
            "bias_t_filter_tau_us": self.bias_t_filter_tau_us.value(),
            "sweep": sweep,
            "sweeps": sweeps,
            "cross_capacitance": tuple(
                tuple(float(value) for value in row)
                for row in self._cross_capacitance
            ),
            "rf_pulse_spec": (
                self._rf_pulse_specs[0] if len(self._rf_pulse_specs) == 1 else None
            ),
            "rf_pulse_specs": self._rf_pulse_specs,
            "ddr_readout_spec": self._ddr_readout_spec,
        }

    def accept(self) -> None:
        try:
            self.values()
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid QICK settings", str(exc))
            return
        super().accept()


class MainWindow(QtWidgets.QMainWindow): # pylint: disable=too-few-public-methods
    """Main window for the DCWaveform generator application."""

    sweep_state_changed = QtCore.pyqtSignal(object)

    @property
    def _sweep_spec(self) -> Optional[QickSweepSpec]:
        """Legacy single-sweep view used by older tests and callers."""
        return next(
            (
                spec
                for spec in self._sweep_specs
                if isinstance(spec, QickSweepSpec)
            ),
            None,
        )

    @_sweep_spec.setter
    def _sweep_spec(self, value: Optional[QickSweepSpec]) -> None:
        self._sweep_specs = [] if value is None else [value]

    def __init__(self):
        super().__init__()
        _install_value_input_wheel_guard()
        self.setWindowTitle("DC Waveform Generator - Keysight QCS")

        self._settings_path: Optional[Path] = None
        self._suspend_qcs_output_sync = False
        self._time_unit = DEFAULT_TIME_UNIT
        self._pulse: List[PulseSequence] = [
            PulseSequence(
                DEFAULT_INITIAL_VOLTAGE_MV,
                initial_duration_ns=DEFAULT_GUI_DURATION_NS,
            )
        ]
        self._awg_output_display_names: List[str] = ["awg_0"]
        self._awg_voltage_scale_mv = float(
            DEFAULT_QCS_FULL_SCALE_V * 1000.0
        )
        self._awg_voltage_backend = DEFAULT_EXECUTION_BACKEND
        self._loading_awg_settings = False
        self._pulse[0].v_bounds = (
            -self._awg_voltage_scale_mv,
            self._awg_voltage_scale_mv,
        )
        self._rf_pulse_spec: Optional[QickRfPulseSpec] = None
        self._rf_pulse_specs: List[QickRfPulseSpec] = []
        self._ddr_readout_spec: Optional[QickDdrReadoutSpec] = None
        self._rf_panel: Optional[RfPulseEditorPanel] = None
        self._dock_rf: Optional[QtWidgets.QDockWidget] = None
        self._sweep_specs: List[QickSweepAxisSpec] = []
        self._cross_capacitance = np.eye(1, dtype=float)
        self._qick_fabric_mhz = float(DEFAULT_QICK_FABRIC_MHZ)
        self._qick_tproc_mhz = float(DEFAULT_QICK_TPROC_MHZ)
        self._qick_full_scale_mv = float(DEFAULT_QICK_FULL_SCALE_MV)
        self._qick_awg_channels = (DEFAULT_QSTL_AWG_CHANNELS[0],)
        self._qick_repetitions_per_sweep = 1
        self._bias_t_compensation_enabled = False
        self._bias_t_compensation_type = "dc"
        self._bias_t_compensation_voltage_mv = float(
            DEFAULT_BIAS_T_COMPENSATION_MV
        )
        self._bias_t_compensation_mode = "fixed_voltage"
        self._bias_t_compensation_duration_us = float(
            DEFAULT_BIAS_T_COMPENSATION_DURATION_US
        )
        self._bias_t_filter_tau_us = float(DEFAULT_BIAS_T_FILTER_TAU_US)
        self._experiment_thread: Optional[QtCore.QThread] = None
        self._experiment_worker: Optional[QtCore.QObject] = None
        self._active_experiment_backend = DEFAULT_EXECUTION_BACKEND
        self._last_experiment_result = None
        self._active_qcs_partial_result = None
        self._last_stability_result = None
        self._trace_overlay_result = None
        self._trace_overlay_pinned = False
        self._trace_overlay_quantity = "magnitude"
        self._stability_overlay_thread: Optional[QtCore.QThread] = None
        self._stability_overlay_worker: Optional[QtCore.QObject] = None
        self._stability_plot_load_thread: Optional[QtCore.QThread] = None
        self._stability_plot_load_worker: Optional[QtCore.QObject] = None
        self._awg_sweep_load_thread: Optional[QtCore.QThread] = None
        self._awg_sweep_load_worker: Optional[QtCore.QObject] = None
        self._last_awg_sweep_map_result = None
        self._grid_time_ns = 1000.0
        self._grid_voltage_mv = 100.0
        self._grid_snap_enabled = False
        self._grid_visible = True
        self._grid_configured = False
        self._plot                          = MatplotWidget(self._pulse[0])
        self._selected_port_idx             = 0
        initial_color = (
            self._plot.line_color(0)
            if hasattr(self._plot, "line_color")
            else self._plot._line[0].get_color()
        )
        ControlPanel.port_idx = 0
        self._multi_ctrl: MultiControlPanel = MultiControlPanel(
            self._pulse[0],
            initial_color,
            self._add_port,
            time_unit=self._time_unit,
            awg_channels=self._qick_awg_channels,
            parent=self,
        )
        self._rf_ports_panel = RfPortsPanel(
            self._pulse[0], time_unit=self._time_unit, parent=self
        )
        self._rf_readout_panel = RfReadoutPanel(
            self._pulse[0], time_unit=self._time_unit, parent=self
        )
        self._experiment_panel = ExperimentPanel(
            fabric_mhz=self._qick_fabric_mhz,
            tproc_mhz=self._qick_tproc_mhz,
            full_scale_mv=self._qick_full_scale_mv,
            awg_channels=self._qick_awg_channels,
            repetitions=self._qick_repetitions_per_sweep,
            bias_t_enabled=self._bias_t_compensation_enabled,
            bias_t_compensation_type=self._bias_t_compensation_type,
            bias_t_compensation_mv=self._bias_t_compensation_voltage_mv,
            bias_t_mode=self._bias_t_compensation_mode,
            bias_t_duration_us=self._bias_t_compensation_duration_us,
            bias_t_filter_tau_us=self._bias_t_filter_tau_us,
            parent=self,
        )
        self._stability_panel = StabilityDiagramPanel(self)
        self._stability_panel.apply_path_settings(
            self._stability_panel.front_panel_values()
        )
        self._sparameter_panel = SParameterSweepPanel(self)
        self._calibration_panel = CalibrationPanel(self)
        self._noise_panel = NoiseAnalysisPanel(
            self,
            default_database_path=DEFAULT_QCODES_DB_PATH,
        )
        self._qick_configuration = None
        self._qick_front_panel_target = None
        self._qcs_front_panel_source_snapshot = None
        self._qcs_front_panel_editor_initialized = False
        self._qcs_front_panel_auto_apply_selection = None
        self._qcs_front_panel_pick_target = None
        self._qcs_front_panel_keep_open_after_selection = False
        self._qcs_m5201_route_context_active = False
        self._qcs_m5201_route_previous_auto_selection = None
        self._qcs_m5201_route_temporary_auto_selection = None
        self._qcs_m5201_route_previous_keep_open = False
        self._preserve_qcs_front_panel_draft_on_output_count_change = False
        # Hardware discovery can remove mappings which no longer exist on the
        # installed chassis.  Such a topology is intentionally not runnable,
        # but it still has to remain the shared front-panel source for every
        # control tab while the user reassigns the affected SMAs.
        self._qcs_front_panel_pending_configuration = None
        self._qcs_front_panel_pending_bindings = None
        self._qcs_mapper_commit_revision = 0
        self._qcs_mapper_commit_thread: Optional[QtCore.QThread] = None
        self._qcs_mapper_commit_worker: Optional[QtCore.QObject] = None
        self._qcs_mapper_commit_active = None
        self._qcs_mapper_commit_queued = None
        self._qcs_mapper_commit_result = None
        self._qcs_mapper_commit_error = None
        self._qcs_mapper_commit_completing = False
        self._qcs_mapper_commit_preparing = False
        self._qcs_front_panel_session_revision = 0
        self._pending_qcs_hardware_inventory = None
        self._pending_qcs_hardware_error = None
        self._qick_front_panel_dialog = QtWidgets.QDialog(self)
        self._qick_front_panel_dialog.setModal(False)
        self._qick_front_panel_dialog.setWindowTitle("QICK Front Panel")
        self._qick_front_panel_dialog.resize(1080, 680)
        front_panel_layout = QtWidgets.QVBoxLayout(self._qick_front_panel_dialog)
        self._qick_front_panel = QickFrontPanelControl(
            self._qick_front_panel_dialog
        )
        front_panel_layout.addWidget(self._qick_front_panel)
        front_panel_buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Close,
            parent=self._qick_front_panel_dialog,
        )
        front_panel_buttons.rejected.connect(self._qick_front_panel_dialog.close)
        front_panel_layout.addWidget(front_panel_buttons)
        self._qick_front_panel.set_path_values(
            self._sparameter_panel.front_panel_values()
        )
        self._qcs_front_panel_dialog = QtWidgets.QDialog(self)
        self._qcs_front_panel_dialog.setModal(False)
        self._qcs_front_panel_dialog.setWindowTitle(
            "Keysight QCS M5000 Front Panel"
        )
        self._qcs_front_panel_dialog.resize(1280, 850)
        qcs_front_panel_layout = QtWidgets.QVBoxLayout(
            self._qcs_front_panel_dialog
        )
        self._qcs_front_panel = QcsFrontPanelControl(
            self._qcs_front_panel_dialog
        )
        self._qcs_front_panel.set_apply_preflight(
            self._validate_qcs_front_panel_source_snapshot
        )
        qcs_front_panel_layout.addWidget(self._qcs_front_panel)
        qcs_front_panel_buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Close,
            parent=self._qcs_front_panel_dialog,
        )
        qcs_front_panel_buttons.rejected.connect(
            self._qcs_front_panel_dialog.close
        )
        self._qcs_front_panel_dialog.finished.connect(
            self._clear_qcs_front_panel_pick_context
        )
        qcs_front_panel_layout.addWidget(qcs_front_panel_buttons)
        self._rf_ports_panel.specs_changed.connect(self._on_rf_specs_changed)
        self._rf_readout_panel.spec_changed.connect(self._on_readout_spec_changed)
        self._experiment_panel.set_ddr_readout_spec(
            self._rf_readout_panel.spec()
        )
        self.sweep_state_changed.connect(self._synchronize_sweep_views)
        self._experiment_panel.run_requested.connect(self._run_experiment)
        self._experiment_panel.stop_requested.connect(self._stop_experiment)
        self._experiment_panel.qcs_hardware_configuration_requested.connect(
            self._show_qcs_front_panel
        )
        self._experiment_panel.execution_backend_changed.connect(
            self._on_execution_backend_changed
        )
        self._experiment_panel.fabric_mhz.valueChanged.connect(
            lambda _value: self._refresh_qcs_waveform_capacity()
        )
        self._experiment_panel.qcs_dc_channel_names.textChanged.connect(
            lambda _text: self._refresh_qcs_waveform_capacity()
        )
        self._experiment_panel.qcs_dc_full_scale_v.valueChanged.connect(
            self._on_qcs_dc_full_scale_changed
        )
        self._experiment_panel.full_scale_mv.valueChanged.connect(
            self._on_qick_full_scale_changed
        )
        self._experiment_panel.qcs_init_time_us.valueChanged.connect(
            lambda _value: self._refresh_qcs_waveform_capacity()
        )
        self._experiment_panel.qcs_init_time_us.valueChanged.connect(
            self._stability_panel.set_qcs_init_time_us
        )
        self._stability_panel.set_qcs_init_time_us(
            self._experiment_panel.qcs_init_time_us.value()
        )
        self._experiment_panel.qcs_mapper_path.textChanged.connect(
            lambda _text: self._refresh_qcs_waveform_capacity()
        )
        self._experiment_panel.qcs_hw_demod.toggled.connect(
            self._rf_readout_panel.set_qcs_hardware_demodulation
        )
        self._experiment_panel.qcs_hw_demod.toggled.connect(
            lambda _checked: self._refresh_qcs_waveform_capacity()
        )
        self._rf_readout_panel.qcs_acquisition_mode_changed.connect(
            self._experiment_panel.qcs_hw_demod.setChecked
        )
        self._experiment_panel.qcs_sample_rate_hz.valueChanged.connect(
            self._rf_readout_panel.set_qcs_acquisition_timing_rate
        )
        self._rf_readout_panel.set_qcs_hardware_demodulation(
            self._experiment_panel.qcs_hw_demod.isChecked()
        )
        self._rf_readout_panel.set_qcs_acquisition_timing_rate(
            self._experiment_panel.qcs_sample_rate_hz.value()
        )
        self._experiment_panel.qcs_output_count_changed.connect(
            self._on_qcs_output_count_changed
        )
        self._experiment_panel.qcs_output_sync_warning.connect(
            lambda message: self.statusBar().showMessage(message, 12000)
        )
        self._experiment_panel.run_state_changed.connect(
            lambda running: self._qcs_front_panel.set_editing_enabled(
                not running
            )
        )
        self._experiment_panel.run_state_changed.connect(
            lambda running: self._rf_readout_panel
            .set_qcs_acquisition_mode_editing_enabled(not running)
        )
        self._experiment_panel.show_program_requested.connect(self._show_program)
        self._experiment_panel.awg_metadata_requested.connect(
            self._show_awg_metadata
        )
        self._experiment_panel.ddr_capacity_refresh_requested.connect(
            self._identify_experiment_qick_configuration
        )
        self._experiment_panel.sweep_axes_changed.connect(
            self._refresh_awg_sweep_map_from_last_result
        )
        self._experiment_panel.sweep_update_requested.connect(
            self._update_sweep_parameter
        )
        self._experiment_panel.sweep_remove_requested.connect(
            self._remove_sweep_parameter
        )
        self._experiment_panel.bias_t_changed.connect(self._on_bias_t_changed)
        self._stability_panel.start_requested.connect(
            lambda: self._run_stability_diagram(continuous=True)
        )
        self._stability_panel.stop_requested.connect(
            self._stop_stability_diagram
        )
        self._stability_panel.single_shot_requested.connect(
            lambda: self._run_stability_diagram(continuous=False)
        )
        self._stability_panel.saved_run_requested.connect(
            self._load_stability_saved_run
        )
        self._stability_panel.front_panel_requested.connect(
            lambda: self._show_active_front_panel(
                "path",
                self._stability_panel,
            )
        )
        self._stability_panel.electrode_front_panel_requested.connect(
            self._show_stability_electrode_front_panel
        )
        self._stability_panel.awg_output_name_changed.connect(
            self._set_awg_output_display_name
        )
        self._sparameter_panel.run_requested.connect(
            self._run_sparameter_sweep
        )
        self._sparameter_panel.stop_requested.connect(
            self._stop_sparameter_sweep
        )
        self._sparameter_panel.load_requested.connect(
            self._load_sparameter_run
        )
        self._sparameter_panel.front_panel_requested.connect(
            lambda: self._show_active_front_panel(
                "path",
                self._sparameter_panel,
            )
        )
        self._rf_ports_panel.front_panel_requested.connect(
            lambda panel: self._show_active_front_panel("output", panel)
        )
        self._rf_ports_panel.lo_frequency_change_requested.connect(
            self._set_qcs_m5300_lo_from_rf_output
        )
        self._multi_ctrl.front_panel_requested.connect(
            lambda target: self._show_active_front_panel("output", target)
        )
        self._multi_ctrl.awg_channel_changed.connect(
            self._set_awg_output_channel
        )
        self._rf_readout_panel.front_panel_requested.connect(
            lambda panel: self._show_active_front_panel("input", panel)
        )
        self._calibration_panel.output_requested.connect(
            lambda: self._run_power_calibration("output")
        )
        self._calibration_panel.input_requested.connect(
            lambda: self._run_power_calibration("input")
        )
        self._calibration_panel.dc_voltage_requested.connect(
            lambda: self._run_power_calibration("dc_voltage")
        )
        self._calibration_panel.qcs_dc_output_requested.connect(
            lambda: self._run_power_calibration("qcs_dc_output")
        )
        self._calibration_panel.qcs_rf_output_requested.connect(
            lambda: self._run_power_calibration("qcs_rf_output")
        )
        self._calibration_panel.front_panel_requested.connect(
            lambda target: self._show_active_front_panel("path", target)
        )
        self._noise_panel.load_requested.connect(self._load_noise_trace)
        self._noise_panel.acquire_requested.connect(
            self._run_noise_acquisition
        )
        self._noise_panel.front_panel_requested.connect(
            lambda target: self._show_active_front_panel("input", target)
        )
        self._sync_shared_qick_controls()
        self._qick_front_panel.identify_requested.connect(
            self._identify_qick_configuration
        )
        self._qick_front_panel.settings_applied.connect(
            self._apply_front_panel_settings
        )
        self._qcs_front_panel.settings_applied.connect(
            self._apply_qcs_front_panel_settings
        )
        self._qcs_front_panel.connector_selected.connect(
            self._on_qcs_front_panel_connector_selected
        )
        self._qcs_front_panel.connector_picked.connect(
            self._on_qcs_front_panel_connector_picked
        )
        self._qcs_front_panel.m5201_route_selection_started.connect(
            self._on_qcs_m5201_route_selection_started
        )
        self._qcs_front_panel.m5201_route_selection_finished.connect(
            self._on_qcs_m5201_route_selection_finished
        )
        self._qcs_front_panel.m5300_lo_frequency_changed.connect(
            self._on_qcs_m5300_lo_frequency_changed
        )
        self._qcs_front_panel.draft_staged.connect(
            self._on_qcs_front_panel_draft_staged
        )
        self._qcs_front_panel.identify_requested.connect(
            self._identify_qcs_hardware_configuration
        )
        self._qcs_front_panel.mapper_saved.connect(
            lambda path: self.statusBar().showMessage(
                f"QCS ChannelMapper saved: {path}", 10000
            )
        )

        self._time_unit_combo = QtWidgets.QComboBox()
        self._time_unit_combo.addItems(tuple(TIME_UNIT_NS))
        self._time_unit_combo.setCurrentText(self._time_unit)
        self._time_unit_combo.currentTextChanged.connect(self._set_time_unit)
        unit_row = QtWidgets.QHBoxLayout()
        unit_row.addWidget(QtWidgets.QLabel("Time unit:"))
        unit_row.addWidget(self._time_unit_combo)
        unit_row.addStretch(1)

        self._awg_tuning_page = QtWidgets.QWidget(self)
        awg_tuning_layout = QtWidgets.QVBoxLayout(self._awg_tuning_page)
        awg_tuning_layout.setContentsMargins(0, 0, 0, 0)
        self._awg_tuning_tabs = QtWidgets.QTabWidget(self._awg_tuning_page)
        self._awg_tuning_tabs.addTab(self._multi_ctrl, "AWG Outputs")
        self._awg_tuning_tabs.addTab(self._rf_ports_panel, "RF Outputs")
        self._awg_tuning_tabs.addTab(self._rf_readout_panel, "RF Readout")
        self._awg_tuning_tabs.addTab(self._experiment_panel, "Experiment")
        self._awg_tuning_tabs.setCurrentWidget(self._multi_ctrl)
        awg_tuning_layout.addWidget(self._awg_tuning_tabs)

        self._control_tabs = QtWidgets.QTabWidget()
        self._control_tabs.addTab(self._awg_tuning_page, "AWG Tuning")
        self._control_tabs.addTab(
            self._stability_panel,
            "Stability Diagram",
        )
        self._control_tabs.addTab(self._sparameter_panel, "RF S-Parameter")
        self._control_tabs.addTab(self._calibration_panel, "Calibration")
        self._control_tabs.addTab(self._noise_panel, "Noise Analysis")
        self._control_tabs.setCurrentWidget(self._awg_tuning_page)
        self._control_tabs.currentChanged.connect(self._on_control_tab_changed)
        control_container = QtWidgets.QWidget(self)
        control_layout = QtWidgets.QVBoxLayout(control_container)
        control_layout.setContentsMargins(4, 4, 4, 4)
        control_layout.addLayout(unit_row)
        control_layout.addWidget(self._control_tabs)
        self.refresh_panel_table()
        self._wire_control_panel(self._multi_ctrl._ctrl_pannels[0])
        self._multi_ctrl.panel_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._multi_ctrl.panel_table.customContextMenuRequested.connect(self._on_port_menu)

        # Create the trace plot only after both X and Y are requested.
        self._trace: Optional[TracePlotWidget] = None
        self._trace_placeholder = QtWidgets.QLabel(
            "Select set_x and set_y ports to create the X/Y trace plot."
        )
        self._trace_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._trace_placeholder.setWordWrap(True)
        self._trace_overlay_selector = StabilityOverlaySelector(self)
        self._trace_overlay_selector.load_requested.connect(
            self._load_trace_stability_overlay
        )
        self._trace_overlay_selector.latest_requested.connect(
            self._use_latest_trace_stability_overlay
        )
        self._trace_overlay_selector.quantity_changed.connect(
            self._set_trace_stability_quantity
        )
        self._trace_container = QtWidgets.QWidget(self)
        self._trace_layout = QtWidgets.QVBoxLayout(self._trace_container)
        self._trace_layout.setContentsMargins(2, 2, 2, 2)
        self._trace_layout.setSpacing(2)
        self._trace_layout.addWidget(self._trace_overlay_selector)
        self._trace_layout.addWidget(self._trace_placeholder, 1)

        self._rf_timelines = []
        self._rf_timeline = None
        self._rf_timeline_container = QtWidgets.QWidget(self)
        self._rf_timeline_layout = QtWidgets.QVBoxLayout(self._rf_timeline_container)
        self._rf_timeline_layout.setContentsMargins(0, 0, 0, 0)
        self._rf_timeline_layout.setSpacing(2)
        self._waveform_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        self._waveform_splitter.addWidget(self._plot)
        self._waveform_splitter.setCollapsible(0, False)
        self._waveform_splitter.setStretchFactor(0, 4)
        if RfPulseTimelineWidget is not None:
            self._waveform_splitter.addWidget(self._rf_timeline_container)
            self._waveform_splitter.setCollapsible(1, True)
            self._waveform_splitter.setStretchFactor(1, 1)
            self._rf_timeline_container.hide()

        self._dock_ctrl  = QtWidgets.QDockWidget("Control Panel", self)
        self._dock_ctrl.setWidget(control_container)

        self._dock_plot = QtWidgets.QDockWidget(
            "Waveform Plot - Virtual (solid) / Physical (dashed)",
            self,
        )
        self._dock_plot.setWidget(self._waveform_splitter)

        self._dock_trace = QtWidgets.QDockWidget("Trace Plot", self)
        self._dock_trace.setWidget(self._trace_container)

        self._stability_plot = self._stability_panel.detach_plot()
        self._dock_stability = QtWidgets.QDockWidget(
            "Stability Diagram - FIR Magnitude / Phase",
            self,
        )
        self._dock_stability.setWidget(self._stability_plot)

        self._awg_sweep_container = QtWidgets.QWidget(self)
        awg_sweep_layout = QtWidgets.QVBoxLayout(
            self._awg_sweep_container
        )
        awg_sweep_layout.setContentsMargins(0, 0, 0, 0)
        awg_sweep_layout.setSpacing(4)
        self._awg_sweep_run_selector = AwgSweepRunSelector(
            self._awg_sweep_container,
            default_database_path=(
                self._experiment_panel.database_path.text().strip()
                or DEFAULT_QCODES_DB_PATH
            ),
        )
        self._awg_sweep_run_selector.load_requested.connect(
            self._load_awg_sweep_saved_run
        )
        self._awg_sweep_run_selector.latest_requested.connect(
            self._use_latest_awg_sweep_result
        )
        awg_sweep_layout.addWidget(self._awg_sweep_run_selector)
        self._awg_sweep_plot = AwgSweepMapPlotWidget(
            self._awg_sweep_container
        )
        awg_sweep_layout.addWidget(self._awg_sweep_plot, 1)
        self._dock_awg_sweep = QtWidgets.QDockWidget(
            "AWG Tuning - I / Q / Magnitude / Angle",
            self,
        )
        self._dock_awg_sweep.setWidget(self._awg_sweep_container)

        self._sparameter_plot = SParameterPlotWidget(self)
        self._dock_sparameter = QtWidgets.QDockWidget(
            "RF S-Parameter", self
        )
        self._dock_sparameter.setWidget(self._sparameter_plot)

        self._noise_plot = self._noise_panel.detach_plot()
        self._dock_noise = QtWidgets.QDockWidget(
            "Noise Analysis - Current / ASD", self
        )
        self._dock_noise.setWidget(self._noise_plot)

        for dock in (
            self._dock_ctrl,
            self._dock_plot,
            self._dock_trace,
            self._dock_stability,
            self._dock_awg_sweep,
            self._dock_sparameter,
            self._dock_noise,
        ):
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable |
                QtWidgets.QDockWidget.DockWidgetFloatable
            )
        self._multi_ctrl.btn_reset.clicked.connect(self._plot._restore_full_intensity)

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._dock_ctrl)
        self.splitDockWidget(self._dock_ctrl, self._dock_plot, QtCore.Qt.Horizontal)
        self.splitDockWidget(self._dock_plot, self._dock_trace, QtCore.Qt.Horizontal)
        self.tabifyDockWidget(self._dock_trace, self._dock_stability)
        self.tabifyDockWidget(self._dock_trace, self._dock_awg_sweep)
        self.tabifyDockWidget(self._dock_trace, self._dock_sparameter)
        self.tabifyDockWidget(self._dock_trace, self._dock_noise)
        self.resizeDocks(
            [self._dock_ctrl, self._dock_plot, self._dock_trace],
            [200, 400, 300],
            QtCore.Qt.Horizontal
        )
        self._dock_plot.raise_()
        self._dock_trace.raise_()

        # status bar
        self.statusBar().showMessage("Ready")

        self._plot.flat_moved.connect(self._flat_update)
        self._plot.point_moved.connect(self._point_update)

        # The curve moves immediately in PyQtGraph.  Table and X/Y trace
        # updates are coalesced to roughly 30 Hz while the mouse is moving.
        self._pending_table_refresh = False
        self._pending_trace_refresh = False
        self._pending_rf_refresh = False
        self._pending_sweep_refresh = False
        self._pending_shared_timing_refresh = False
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(33)
        self._refresh_timer.timeout.connect(self._flush_deferred_refresh)

        self._build_menu()
        self._build_toolbar()
        self._on_execution_backend_changed(
            self._experiment_panel.execution_backend()
        )
        self._set_grid_settings(
            time_step_ns=self._grid_time_ns,
            voltage_step_mv=self._grid_voltage_mv,
            snap_enabled=self._grid_snap_enabled,
            visible=self._grid_visible,
        )
        self._set_time_unit(self._time_unit)
        self._sync_awg_output_display_names()
        self._refresh_sweep_overlay()

    def _wire_control_panel(self, control: ControlPanel) -> None:
        control.add_requested.connect(self._add_segment)
        control.update_plot.connect(self._plot_refresh)
        control.port_is_selected.connect(self._port_select)
        control.hardware_output_edit_requested.connect(
            self._edit_awg_output_hardware
        )
        control.sweep_requested.connect(self._configure_segment_sweep)
        control.sweep_remove_requested.connect(self._remove_segment_sweep)
        control.ramp_sweep_requested.connect(self._configure_ramp_rate_sweep)
        control.ramp_sweep_remove_requested.connect(
            self._remove_ramp_rate_sweep
        )
        control.hold_sweep_requested.connect(
            self._configure_hold_duration_sweep
        )
        control.hold_sweep_remove_requested.connect(
            self._remove_hold_duration_sweep
        )
        control.segment_timing_changed.connect(
            self._on_segment_timing_changed
        )
        control.segment_name_changed.connect(
            self._on_segment_name_changed
        )
        control.segment_structure_changed.connect(self._on_segment_structure_changed)
        control.output_name_changed.connect(
            lambda index, name: self._set_awg_output_display_name(
                f"awg_{index}",
                name,
            )
        )

    def _on_port_menu(self, pos: QtCore.QPoint) -> None:
        row = self._multi_ctrl.panel_table.rowAt(pos.y())
        if row < 0:
            return
        menu = QtWidgets.QMenu(self)
        act_del = menu.addAction("Delete port")
        if len(self._pulse) <= 1:
            act_del.setEnabled(False)     # must keep at least one
        if menu.exec_(
            self._multi_ctrl.panel_table.viewport().mapToGlobal(pos)
        ) == act_del:
            self._delete_port(row)

    def _delete_port(self, idx: int) -> None:
        """Completely remove pulse, plot line and control panel at index idx."""
        if (
            hasattr(self, "_experiment_panel")
            and not self._suspend_qcs_output_sync
        ):
            self._experiment_panel.remove_qcs_output(
                idx,
                output_count=len(self._pulse) - 1,
            )
        sweep_entries = [
            (spec, self._sweep_target_indices(spec)) for spec in self._sweep_specs
        ]
        self._plot.remove_pulse(idx)
        self._pulse.pop(idx)
        self._awg_output_display_names.pop(idx)
        self._qick_awg_channels = tuple(
            channel
            for channel_index, channel in enumerate(self._qick_awg_channels)
            if channel_index != idx
        )
        if hasattr(self, "_experiment_panel"):
            self._experiment_panel.set_qick_values(
                fabric_mhz=self._qick_fabric_mhz,
                tproc_mhz=self._qick_tproc_mhz,
                full_scale_mv=self._qick_full_scale_mv,
                awg_channels=self._qick_awg_channels,
                repetitions=self._qick_repetitions_per_sweep,
            )
        self._cross_capacitance = np.delete(
            np.delete(self._cross_capacitance, idx, axis=0),
            idx,
            axis=1,
        )
        ctrl = self._multi_ctrl._ctrl_pannels.pop(idx)
        ctrl.setParent(None)
        ctrl.deleteLater()
        self._multi_ctrl._color_map.pop(idx)
        ControlPanel.port_idx -= 1  # decrement port index counter
        for i, ctrl in enumerate(self._multi_ctrl._ctrl_pannels):
            ctrl.idx = i
        self._multi_ctrl.set_awg_channels(self._qick_awg_channels)
        self._sync_awg_output_display_names()
        self._multi_ctrl.update_port_strip_geometry()
        pending_preserved = False
        if not self._suspend_qcs_output_sync:
            pending_preserved = (
                self._synchronize_pending_qcs_front_panel_output_count(
                    removed_index=idx,
                )
            )
        if (
            not self._suspend_qcs_output_sync
            and not pending_preserved
            and self._experiment_panel.execution_backend()
            == EXECUTION_BACKEND_QCS
        ):
            self._propagate_qcs_hardware_configuration(
                *self._qcs_bindings_from_settings(
                    self._experiment_panel.qcs_settings_dict()
                ),
                require_complete_rf=False,
            )

        if idx < self._selected_port_idx:
            self._selected_port_idx -= 1
        elif self._selected_port_idx >= len(self._pulse):
            self._selected_port_idx = len(self._pulse) - 1
        self._plot.set_selected_port_idx(self._selected_port_idx)
        self._multi_ctrl.set_selected_port(self._selected_port_idx)

        self._multi_ctrl.refresh_table()
        self.refresh_panel_table()
        updated_sweeps = []
        for spec, target in sweep_entries:
            if isinstance(
                spec,
                (QickRampRateSweepSpec, QickHoldDurationSweepSpec),
            ):
                updated_sweeps.append(spec)
                continue
            if target is None or target[0] == idx:
                continue
            if target[0] > idx:
                spec = QickSweepSpec(
                    segment_name=spec.segment_name,
                    output_name=f"awg_{target[0] - 1}",
                    start=spec.start,
                    stop=spec.stop,
                    count=spec.count,
                )
            updated_sweeps.append(spec)
        self._sweep_specs = updated_sweeps
        self._notify_sweep_state_changed(waveform_changed=True)
        self._refresh_rf_editor()
        if self._trace is not None:
            for attr in ("x_idx", "y_idx"):
                trace_index = getattr(self._trace, attr)
                if trace_index == idx:
                    setattr(self._trace, attr, None)
                elif trace_index is not None and trace_index > idx:
                    setattr(self._trace, attr, trace_index - 1)
            self._trace.refresh_trace(self._pulse)

    def _shared_qick_connection(self) -> QickConnectionConfig:
        connection, _run = self._experiment_panel.connection_values(
            require_run_config=False
        )
        return connection

    def _sync_shared_qick_controls(self) -> None:
        """Keep hidden compatibility fields aligned with the Setup menu."""
        connection = self._shared_qick_connection()
        self._noise_panel.set_connection_values(connection)

    def _apply_shared_qick_setup(
        self,
        connection: QickConnectionConfig,
        fabric_mhz: float,
        tproc_mhz: float,
    ) -> None:
        self._qick_fabric_mhz = float(fabric_mhz)
        self._qick_tproc_mhz = float(tproc_mhz)
        self._experiment_panel.set_connection_values(connection)
        self._experiment_panel.set_qick_values(
            fabric_mhz=self._qick_fabric_mhz,
            tproc_mhz=self._qick_tproc_mhz,
            full_scale_mv=self._qick_full_scale_mv,
            awg_channels=self._qick_awg_channels,
            repetitions=self._qick_repetitions_per_sweep,
        )
        self._noise_panel.set_connection_values(connection)

    def _configure_qick_setup(self) -> None:
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "QICK task running",
                "Wait for the current QICK task to finish before changing Setup.",
            )
            return
        dialog = QickSetupDialog(
            connection=self._shared_qick_connection(),
            fabric_mhz=self._qick_fabric_mhz,
            tproc_mhz=self._qick_tproc_mhz,
            parent=self,
        )
        while dialog.exec_() == QtWidgets.QDialog.Accepted:
            try:
                connection, fabric_mhz, tproc_mhz = dialog.values()
            except (TypeError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid QICK Setup",
                    str(exc),
                )
                continue
            self._apply_shared_qick_setup(
                connection,
                fabric_mhz,
                tproc_mhz,
            )
            self.statusBar().showMessage(
                f"QICK Setup: {connection.host}:{connection.ns_port} / "
                f"AWG {fabric_mhz:g} MHz / tProcessor {tproc_mhz:g} MHz"
            )
            break

    def _qcs_bindings_from_settings(
        self,
        settings: Mapping[str, object],
    ) -> tuple[
        Optional[dict],
        Tuple[str, ...],
        dict[int, str],
        Optional[str],
    ]:
        """Return normalized QCS bindings used by every experiment editor."""
        settings = dict(settings)
        configuration = settings.get("hardware_configuration")
        if configuration is not None:
            configuration = normalize_qcs_hardware_configuration(
                configuration,
                required_dc_count=len(self._pulse),
            )
            dc_names, rf_names, acquisition_name = qcs_role_bindings(
                configuration,
                required_dc_count=len(self._pulse),
            )
            return (
                configuration,
                tuple(str(name) for name in dc_names),
                {
                    int(index): str(name)
                    for index, name in rf_names.items()
                },
                acquisition_name,
            )

        dc_names = tuple(
            str(name).strip()
            for name in settings.get("dc_channel_names", ())
            if str(name).strip()
        )
        if len(dc_names) != len(self._pulse):
            raise ValueError(
                "QCS requires one DC role for every waveform output"
            )
        rf_names = {
            int(index): str(name).strip()
            for index, name in dict(
                settings.get("rf_channel_names", {})
            ).items()
        }
        acquisition_name = settings.get("acquisition_channel_name")
        acquisition_name = (
            None
            if acquisition_name is None
            else str(acquisition_name).strip() or None
        )
        return configuration, dc_names, rf_names, acquisition_name

    def _clear_pending_qcs_front_panel_configuration(self) -> None:
        """Forget a preview-only discovered topology after commit or reload."""

        self._qcs_front_panel_pending_configuration = None
        self._qcs_front_panel_pending_bindings = None

    def _retire_qcs_mapper_commits(self) -> None:
        """Invalidate pending mapper results after an external source change."""

        self._qcs_mapper_commit_revision += 1
        self._qcs_mapper_commit_queued = None

    def _qcs_mapper_commit_pending(self) -> bool:
        """Return whether native mapper work is active or awaiting handoff."""

        return (
            self._qcs_mapper_commit_thread is not None
            or self._qcs_mapper_commit_active is not None
            or self._qcs_mapper_commit_queued is not None
        )

    def _pending_qcs_front_panel_state(self):
        """Return the shared non-runnable topology currently being edited."""

        configuration = self._qcs_front_panel_pending_configuration
        bindings = self._qcs_front_panel_pending_bindings
        if configuration is None or bindings is None:
            return None
        dc_names, rf_names, acquisition_name = bindings
        return (
            configuration,
            tuple(dc_names),
            dict(rf_names),
            acquisition_name,
        )

    def _synchronize_pending_qcs_front_panel_output_count(
        self,
        *,
        removed_index: Optional[int] = None,
    ) -> bool:
        """Keep identified hardware while waveform outputs are added/removed."""

        pending = self._pending_qcs_front_panel_state()
        if pending is None:
            return False
        configuration, _old_dc, rf_names, acquisition_name = pending
        dc_names = tuple(
            name.strip()
            for name in self._experiment_panel.qcs_dc_channel_names.text().split(
                ","
            )
            if name.strip()
        )
        if len(dc_names) != len(self._pulse):
            raise ValueError(
                "QCS requires one DC virtual-channel name for every waveform "
                "output"
            )
        configuration = resize_incomplete_qcs_dc_mappings(
            configuration,
            dc_names,
            removed_index=removed_index,
        )
        self._qcs_front_panel_pending_configuration = configuration
        self._qcs_front_panel_pending_bindings = (
            dc_names,
            dict(rf_names),
            acquisition_name,
        )
        self._experiment_panel.set_qcs_front_panel_draft_pending(True)

        # Rehydrate the editor as well as compact previews.  In particular,
        # deleting an output must remove/reindex its DC role row immediately.
        self._qcs_front_panel.set_settings(
            {
                "mapper_path": self._experiment_panel.qcs_mapper_path.text(),
                "dc_channel_names": list(dc_names),
                "rf_channel_names": {
                    str(index): name for index, name in rf_names.items()
                },
                "acquisition_channel_name": acquisition_name,
                "hardware_configuration": configuration,
                "hardware_configuration_state": QCS_HARDWARE_STATE_DRAFT,
                "hardware_mapper_sha256": None,
            },
            output_count=len(self._pulse),
        )
        self._qcs_front_panel_editor_initialized = True
        if (
            self._experiment_panel.execution_backend()
            == EXECUTION_BACKEND_QCS
        ):
            self._propagate_qcs_hardware_configuration(
                configuration,
                dc_names,
                rf_names,
                acquisition_name,
                require_complete_rf=False,
            )
        self._qcs_front_panel_source_snapshot = (
            self._current_qcs_front_panel_source_snapshot()
        )
        return True

    def _propagate_qcs_hardware_configuration(
        self,
        configuration: Optional[Mapping[str, object]],
        dc_channel_names: Sequence[str],
        rf_channel_names: Mapping[int, str],
        acquisition_channel_name: Optional[str],
        *,
        require_complete_rf: bool = True,
    ) -> None:
        """Apply QCS role identities to all integrated experiment editors."""
        del acquisition_channel_name  # The acquisition role is always index 0.
        rf_binding_error = None
        try:
            self._rf_ports_panel.validate_qcs_rf_channel_names(
                rf_channel_names
            )
        except ValueError as exc:
            if require_complete_rf:
                raise
            rf_binding_error = exc
        self._multi_ctrl.set_hardware_backend(EXECUTION_BACKEND_QCS)
        self._rf_ports_panel.set_hardware_backend(EXECUTION_BACKEND_QCS)
        self._rf_readout_panel.set_hardware_backend(
            EXECUTION_BACKEND_QCS
        )
        self._stability_panel.set_hardware_backend(EXECUTION_BACKEND_QCS)
        self._sparameter_panel.set_hardware_backend(EXECUTION_BACKEND_QCS)
        self._calibration_panel.set_hardware_backend(EXECUTION_BACKEND_QCS)
        self._noise_panel.set_hardware_backend(EXECUTION_BACKEND_QCS)
        self._multi_ctrl.set_qcs_front_panel_configuration(
            configuration,
            dc_channel_names,
        )
        if rf_binding_error is None:
            self._rf_ports_panel.set_qcs_front_panel_configuration(
                configuration,
                rf_channel_names,
            )
        else:
            self._rf_ports_panel.set_qcs_front_panel_preview_configuration(
                configuration
            )
        self._rf_readout_panel.set_qcs_front_panel_configuration(
            configuration
        )
        self._stability_panel.set_qcs_front_panel_configuration(configuration)
        self._sparameter_panel.set_qcs_front_panel_configuration(configuration)
        self._calibration_panel.set_qcs_front_panel_configuration(configuration)
        self._noise_panel.set_qcs_front_panel_configuration(configuration)
        self.refresh_panel_table()
        self._refresh_qcs_waveform_capacity()
        if rf_binding_error is not None:
            self.statusBar().showMessage(
                f"QCS hardware configuration needs attention: "
                f"{rf_binding_error}",
                12000,
            )

    def _active_awg_full_scale_mv(
        self,
        backend: Optional[str] = None,
    ) -> float:
        """Return the physical AWG scale for the selected execution system."""
        if backend is None:
            backend = self._experiment_panel.execution_backend()
        backend = str(backend).strip().lower()
        if backend == EXECUTION_BACKEND_QCS:
            return float(self._experiment_panel.qcs_dc_full_scale_v.value()) * 1000.0
        if backend == EXECUTION_BACKEND_QICK:
            return float(self._experiment_panel.full_scale_mv.value())
        raise ValueError(f"unsupported execution backend {backend!r}")

    def _set_awg_voltage_scale(
        self,
        full_scale_mv: float,
        *,
        backend: Optional[str] = None,
    ) -> None:
        """Change normalization while preserving every physical millivolt."""
        full_scale_mv = float(full_scale_mv)
        if not np.isfinite(full_scale_mv) or full_scale_mv <= 0.0:
            raise ValueError("AWG full scale must be positive")
        old_scale_mv = float(self._awg_voltage_scale_mv)
        if old_scale_mv <= 0.0:
            old_scale_mv = full_scale_mv

        sweep_limit_mv = max(
            (
                max(abs(float(spec.start)), abs(float(spec.stop)))
                * old_scale_mv
                for spec in self._sweep_specs
                if isinstance(spec, QickSweepSpec)
            ),
            default=0.0,
        )
        # Pulse levels are stored directly in millivolts and can remain
        # visible in an intentionally invalid configuration; run validation
        # will then report the exact over-range level.  Normalized sweep
        # coordinates, however, must be convertible before the scale changes
        # or their physical meaning would be lost.
        required_mv = sweep_limit_mv
        if required_mv > full_scale_mv + 1.0e-9:
            label = (
                "QCS"
                if backend == EXECUTION_BACKEND_QCS
                else "QICK"
                if backend == EXECUTION_BACKEND_QICK
                else "selected"
            )
            raise ValueError(
                f"The voltage sweep reaches {required_mv:.6g} mV, "
                f"outside the {label} +/-{full_scale_mv:.6g} mV full scale. "
                "Reduce the programmed voltage before selecting that scale."
            )

        if not np.isclose(
            old_scale_mv,
            full_scale_mv,
            rtol=0.0,
            atol=1.0e-12,
        ):
            converted = []
            for spec in self._sweep_specs:
                if isinstance(spec, QickSweepSpec):
                    converted.append(
                        replace(
                            spec,
                            start=float(spec.start) * old_scale_mv / full_scale_mv,
                            stop=float(spec.stop) * old_scale_mv / full_scale_mv,
                        )
                    )
                else:
                    converted.append(spec)
            self._sweep_specs = converted

        self._awg_voltage_scale_mv = full_scale_mv
        if backend is not None:
            self._awg_voltage_backend = str(backend).strip().lower()
        for pulse in self._pulse:
            pulse.v_bounds = (-full_scale_mv, full_scale_mv)
        self._experiment_panel.set_sweep_voltage_scale(full_scale_mv)

    def _on_qcs_dc_full_scale_changed(self, value_v: float) -> None:
        """Apply a QCS scale edit without clipping existing AWG settings."""
        if self._loading_awg_settings:
            return
        if self._experiment_panel.execution_backend() != EXECUTION_BACKEND_QCS:
            self._refresh_qcs_waveform_capacity()
            return
        try:
            self._set_awg_voltage_scale(
                float(value_v) * 1000.0,
                backend=EXECUTION_BACKEND_QCS,
            )
        except ValueError as exc:
            with QtCore.QSignalBlocker(
                self._experiment_panel.qcs_dc_full_scale_v
            ):
                self._experiment_panel.qcs_dc_full_scale_v.setValue(
                    self._awg_voltage_scale_mv / 1000.0
                )
            self.statusBar().showMessage(
                f"Cannot reduce QCS AWG full scale: {exc}",
                12000,
            )
            return
        self._multi_ctrl.refresh_table()
        self._refresh_sweep_overlay(fit_view=True, sync_rows=True)

    def _on_qick_full_scale_changed(self, value_mv: float) -> None:
        """Apply the legacy QICK scale only while QICK is active."""
        if self._loading_awg_settings:
            return
        if self._experiment_panel.execution_backend() != EXECUTION_BACKEND_QICK:
            return
        try:
            self._set_awg_voltage_scale(
                value_mv,
                backend=EXECUTION_BACKEND_QICK,
            )
        except ValueError as exc:
            with QtCore.QSignalBlocker(self._experiment_panel.full_scale_mv):
                self._experiment_panel.full_scale_mv.setValue(
                    self._awg_voltage_scale_mv
                )
            self.statusBar().showMessage(
                f"Cannot reduce QICK AWG full scale: {exc}",
                12000,
            )
            return
        self._qick_full_scale_mv = float(value_mv)
        self._multi_ctrl.refresh_table()
        self._refresh_sweep_overlay(fit_view=True, sync_rows=True)

    def _on_execution_backend_changed(self, backend: str) -> None:
        """Synchronize every hardware preview with the execution system."""
        backend = str(backend).strip().lower()
        if backend not in EXECUTION_BACKENDS:
            raise ValueError(f"unsupported execution backend {backend!r}")
        if self._loading_awg_settings:
            return
        previous_backend = self._awg_voltage_backend
        try:
            self._set_awg_voltage_scale(
                self._active_awg_full_scale_mv(backend),
                backend=backend,
            )
        except ValueError as exc:
            selector = self._experiment_panel.backend_selector
            previous_index = selector.findData(previous_backend)
            if previous_index >= 0:
                with QtCore.QSignalBlocker(selector):
                    selector.setCurrentIndex(previous_index)
                self._experiment_panel._last_execution_backend = previous_backend
                self._experiment_panel._update_execution_backend_controls()
            self.statusBar().showMessage(
                f"Cannot change AWG execution system: {exc}",
                12000,
            )
            return
        if backend == EXECUTION_BACKEND_QICK:
            self._qick_full_scale_mv = self._active_awg_full_scale_mv(backend)
        if (
            backend != EXECUTION_BACKEND_QCS
            and hasattr(self, "_qcs_mapper_commit_revision")
            and (
                self._qcs_mapper_commit_thread is not None
                or self._qcs_mapper_commit_active is not None
                or self._qcs_mapper_commit_queued is not None
            )
        ):
            # A later explicit backend choice wins over a mapper job started
            # from QCS. The worker may finish its atomic local file write, but
            # its result must not switch the application back to QCS.
            self._retire_qcs_mapper_commits()
        self._multi_ctrl.set_hardware_backend(backend)
        self._rf_ports_panel.set_hardware_backend(backend)
        self._rf_readout_panel.set_hardware_backend(backend)
        self._rf_readout_panel.set_qcs_hardware_demodulation(
            self._experiment_panel.qcs_hw_demod.isChecked()
        )
        self._rf_readout_panel.set_qcs_acquisition_timing_rate(
            self._experiment_panel.qcs_sample_rate_hz.value()
        )
        self._stability_panel.set_hardware_backend(backend)
        self._sparameter_panel.set_hardware_backend(backend)
        self._calibration_panel.set_hardware_backend(backend)
        self._noise_panel.set_hardware_backend(backend)

        if backend == EXECUTION_BACKEND_QCS:
            try:
                bindings = self._pending_qcs_front_panel_state()
                if bindings is None:
                    bindings = self._qcs_bindings_from_settings(
                        self._experiment_panel.qcs_settings_dict()
                    )
                self._propagate_qcs_hardware_configuration(
                    *bindings,
                    require_complete_rf=False,
                )
            except (TypeError, ValueError) as exc:
                # Do not leave a previously valid mapper visible after the
                # current Experiment settings fail validation.
                self._multi_ctrl.set_qcs_front_panel_configuration(None, ())
                self._rf_ports_panel.set_qcs_front_panel_preview_configuration(
                    None
                )
                self._rf_readout_panel.set_qcs_front_panel_configuration(None)
                self._stability_panel.set_qcs_front_panel_configuration(None)
                self._sparameter_panel.set_qcs_front_panel_configuration(None)
                self._calibration_panel.set_qcs_front_panel_configuration(None)
                self._noise_panel.set_qcs_front_panel_configuration(None)
                self.refresh_panel_table()
                self.statusBar().showMessage(
                    f"QCS hardware configuration needs attention: {exc}",
                    12000,
                )
        else:
            self.refresh_panel_table()

        is_qick = backend == EXECUTION_BACKEND_QICK
        if hasattr(self, "_qick_setup_action"):
            self._qick_setup_action.setVisible(is_qick)
        if hasattr(self, "_generate_qick_action"):
            self._generate_qick_action.setVisible(is_qick)
        if hasattr(self, "_preview_qick_action"):
            self._preview_qick_action.setVisible(is_qick)
        if hasattr(self, "_qcs_setup_action"):
            self._qcs_setup_action.setVisible(True)
        if hasattr(self, "_awg_tuning_tabs"):
            readout_index = self._awg_tuning_tabs.indexOf(
                self._rf_readout_panel
            )
            if readout_index >= 0:
                self._awg_tuning_tabs.setTabText(
                    readout_index,
                    "RF Readout" if is_qick else "QCS Acquisition",
                )
        if hasattr(self, "_control_tabs"):
            legacy_tab_names = {
                1: "Stability Diagram",
                2: "RF S-Parameter",
                3: "Calibration",
                4: "Noise Analysis",
            }
            for index, name in legacy_tab_names.items():
                if index >= self._control_tabs.count():
                    continue
                self._control_tabs.setTabEnabled(index, True)
                self._control_tabs.setTabText(index, name)
                self._control_tabs.setTabToolTip(
                    index,
                    (
                        ""
                        if is_qick
                        else (
                            "QCS Noise Analysis records the mapped M5200 "
                            "raw time trace for the requested measurement "
                            "duration; it does not use an IntegrationFilter."
                            if index == 4
                            else (
                                "The front panel shows the shared QCS hardware "
                                "mapping, but measurement execution in this tab "
                                "still uses the legacy QICK path."
                            )
                        )
                    ),
                )
        self._multi_ctrl.refresh_table()
        self._refresh_sweep_overlay(sync_rows=True)

    def _show_active_front_panel(self, scope: str, target=None) -> None:
        """Open the front panel for the currently active execution system."""
        if (
            self._experiment_panel.execution_backend()
            == EXECUTION_BACKEND_QICK
        ):
            self._show_qick_front_panel(scope, target)
            return
        pick_only = target is not None and callable(
            getattr(target, "accept_qcs_front_panel_connector", None)
        )
        self._qcs_front_panel_pick_target = target if pick_only else None
        existing_mapping = (
            target.current_qcs_mapping()
            if pick_only
            and callable(getattr(target, "current_qcs_mapping", None))
            else None
        )
        self._qcs_front_panel.set_connector_pick_only(
            pick_only,
            existing_mapping,
        )
        role = None
        logical_index = 0
        path_selections = None
        if (
            target is self._stability_panel
            or target is self._sparameter_panel
        ) and hasattr(target, "qcs_rf_acquisition_front_panel_selections"):
            path_selections = tuple(
                (str(candidate_role), int(candidate_index))
                for candidate_role, candidate_index in (
                    target.qcs_rf_acquisition_front_panel_selections()
                )
            )
        if target is not None and hasattr(
            target,
            "qcs_front_panel_selection",
        ):
            role, logical_index = target.qcs_front_panel_selection()
        elif isinstance(target, MultiControlPanel):
            role = "dc"
            logical_index = target._selected_index
        elif isinstance(target, RfPulsePortPanel):
            role = "rf"
            logical_index = target.gen_ch.value()
        elif isinstance(target, RfReadoutPanel):
            role = "acquisition"
        shown = self._show_qcs_front_panel(role, logical_index)
        if not shown:
            self._qcs_front_panel_pick_target = None
            self._qcs_front_panel.set_connector_pick_only(False)
        if shown:
            self._qcs_front_panel_keep_open_after_selection = (
                target is self._stability_panel
                or target is self._sparameter_panel
                or target is self._stability_panel.x_axis
                or target is self._stability_panel.y_axis
            )
        if shown and path_selections is not None:
            selections_by_role = dict(path_selections)
            self._qcs_front_panel.focus_rf_acquisition_path(
                selections_by_role["rf"],
                selections_by_role["acquisition"],
            )
            self._qcs_front_panel_auto_apply_selection = frozenset(
                path_selections
            )
            return
        if (
            shown
            and target is not None
            and not pick_only
            and role in {"dc", "rf", "acquisition"}
        ):
            self._qcs_front_panel_auto_apply_selection = (
                role,
                int(logical_index),
            )

    def _edit_awg_output_hardware(self, output_index: int) -> None:
        """Open the full front panel focused on one waveform output."""

        output_index = int(output_index)
        if not 0 <= output_index < len(self._pulse):
            return
        # ``port_is_selected`` normally arrives first from ControlPanel, but
        # keep this slot correct when invoked directly or from a queued signal.
        if self._selected_port_idx != output_index:
            self._port_select(output_index)
        self._show_active_front_panel("output", self._multi_ctrl)

    def _show_stability_electrode_front_panel(self, editor) -> None:
        """Provision and open one of two independent Stability outputs."""

        if (
            self._experiment_panel.execution_backend()
            == EXECUTION_BACKEND_QCS
            and (
                (
                    self._experiment_thread is not None
                    and self._experiment_thread.isRunning()
                )
                or self._experiment_panel._running
            )
        ):
            self._show_active_front_panel("output", editor)
            return

        added_output = False
        if len(self._pulse) < 2:
            preserve_draft = (
                self._qcs_front_panel_editor_initialized
                and self._experiment_panel._qcs_front_panel_draft_pending
            )
            self._preserve_qcs_front_panel_draft_on_output_count_change = (
                preserve_draft
            )
            try:
                self._add_port()
            finally:
                self._preserve_qcs_front_panel_draft_on_output_count_change = (
                    False
                )
            if preserve_draft:
                dc_names_text = (
                    self._experiment_panel.qcs_dc_channel_names.text()
                )
                dc_channel_names = tuple(
                    name.strip()
                    for name in dc_names_text.split(",")
                    if name.strip()
                )
                self._qcs_front_panel.update_source_dc_channels(
                    dc_channel_names,
                    output_count=len(self._pulse),
                )
                self._qcs_front_panel_editor_initialized = True
                # ``_add_port`` synchronizes runnable Experiment settings,
                # which do not contain an incomplete identified topology.
                # Publish the editor draft again so Stability and every other
                # control tab continue to show the discovered chassis.
                self._on_qcs_front_panel_draft_staged(
                    self._qcs_front_panel.working_configuration()
                )
                self._qcs_front_panel_source_snapshot = (
                    self._current_qcs_front_panel_source_snapshot()
                )
                self._experiment_panel.set_qcs_front_panel_draft_pending(True)
            self._refresh_stability_targets()
            added_output = True

        self._show_active_front_panel("output", editor)
        if added_output:
            self.statusBar().showMessage(
                "Added a second waveform output so Stability X and Y use "
                "independent DC channels.",
                10000,
            )

    def _show_qcs_front_panel(
        self,
        role: Optional[str] = None,
        logical_index: int = 0,
    ) -> bool:
        """Open the editable QCS mapper builder for the current experiment."""
        self._qcs_front_panel_auto_apply_selection = None
        self._qcs_front_panel_keep_open_after_selection = False
        if isinstance(role, bool):
            role = None
        if (
            self._experiment_thread is not None
            and self._experiment_thread.isRunning()
        ) or self._experiment_panel._running:
            QtWidgets.QMessageBox.information(
                self,
                "QCS task running",
                "Wait for the current experiment to finish before changing "
                "the QCS hardware configuration.",
            )
            return False
        try:
            source_snapshot = self._current_qcs_front_panel_source_snapshot()
            source_changed = (
                self._qcs_front_panel_editor_initialized
                and self._qcs_front_panel_source_snapshot != source_snapshot
            )
            if source_changed:
                # A genuine Experiment source edit supersedes an unfinished
                # graphical draft.  Ordinary control-tab changes do not alter
                # this snapshot and therefore keep identified hardware alive.
                self._clear_pending_qcs_front_panel_configuration()
                self._retire_qcs_mapper_commits()
                self._experiment_panel.set_qcs_front_panel_draft_pending(False)
            # A closed modeless editor is transient UI state, not an authority.
            # Reopening it must hydrate from the shared pending topology first,
            # or from canonical Experiment settings otherwise.  This also
            # repairs any stale initialized widget tree left by another tab.
            reopen_from_shared = not self._qcs_front_panel_dialog.isVisible()
            if (
                not self._qcs_front_panel_editor_initialized
                or source_changed
                or reopen_from_shared
            ):
                try:
                    settings = self._experiment_panel.qcs_settings_dict()
                except ValueError:
                    if (
                        self._experiment_panel._qcs_hardware_configuration
                        is not None
                    ):
                        raise
                    self._experiment_panel.set_qcs_output_count(
                        len(self._pulse)
                    )
                    settings = self._experiment_panel.qcs_settings_dict()
                pending = self._pending_qcs_front_panel_state()
                if pending is not None:
                    (
                        pending_configuration,
                        pending_dc,
                        pending_rf,
                        pending_acquisition,
                    ) = pending
                    settings = dict(settings)
                    settings.update(
                        {
                            "dc_channel_names": list(pending_dc),
                            "rf_channel_names": {
                                str(index): name
                                for index, name in pending_rf.items()
                            },
                            "acquisition_channel_name": pending_acquisition,
                            "hardware_configuration": pending_configuration,
                            "hardware_configuration_state": (
                                QCS_HARDWARE_STATE_DRAFT
                            ),
                            "hardware_mapper_sha256": None,
                        }
                    )
                self._qcs_front_panel.set_settings(
                    settings,
                    output_count=len(self._pulse),
                )
                self._qcs_front_panel_editor_initialized = True
                # A source change replaces any unfinished graphical draft.
                # Restore every preview from the canonical Experiment state
                # and release the temporary hardware-execution guard only
                # after the editor reload itself has succeeded.
                self._experiment_panel.set_qcs_front_panel_draft_pending(
                    pending is not None
                )
                self._on_execution_backend_changed(
                    self._experiment_panel.execution_backend()
                )
            self._qcs_front_panel.show_front_panel()
            if role is not None:
                self._qcs_front_panel.focus_mapping(
                    role,
                    logical_index,
                )
            else:
                self._qcs_front_panel.clear_mapping_focus()
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid QCS configuration",
                str(exc),
            )
            return False
        self._qcs_front_panel_source_snapshot = (
            self._current_qcs_front_panel_source_snapshot()
        )
        self._qcs_front_panel_session_revision += 1
        self._qcs_front_panel.set_editing_enabled(True)
        self._qcs_front_panel_dialog.show()
        self._qcs_front_panel_dialog.raise_()
        self._qcs_front_panel_dialog.activateWindow()
        return True

    def _hydrate_qcs_front_panel_editor(self) -> None:
        """Load shared QCS state for a non-dialog hardware edit."""

        if (
            self._experiment_thread is not None
            and self._experiment_thread.isRunning()
        ) or self._experiment_panel._running:
            raise ValueError(
                "Wait for the current hardware task before changing the "
                "M5300 LO frequency."
            )
        source_snapshot = self._current_qcs_front_panel_source_snapshot()
        if (
            self._qcs_front_panel_editor_initialized
            and self._qcs_front_panel_dialog.isVisible()
            and self._qcs_front_panel_source_snapshot == source_snapshot
        ):
            # The modeless editor may contain a just-entered value whose
            # editingFinished/draft-staging signal has not run yet.  Reusing
            # that live widget tree preserves the draft when Apply LO is
            # clicked from the RF Outputs tab.
            return
        settings = self._experiment_panel.qcs_settings_dict()
        pending = self._pending_qcs_front_panel_state()
        if pending is not None:
            configuration, dc_names, rf_names, acquisition_name = pending
            settings = dict(settings)
            settings.update(
                {
                    "dc_channel_names": list(dc_names),
                    "rf_channel_names": {
                        str(index): name
                        for index, name in rf_names.items()
                    },
                    "acquisition_channel_name": acquisition_name,
                    "hardware_configuration": configuration,
                    "hardware_configuration_state": QCS_HARDWARE_STATE_DRAFT,
                    "hardware_mapper_sha256": None,
                }
            )
        self._qcs_front_panel.set_settings(
            settings,
            output_count=len(self._pulse),
        )
        self._qcs_front_panel_editor_initialized = True
        self._qcs_front_panel_source_snapshot = source_snapshot

    def _set_qcs_m5300_lo_from_rf_output(
        self,
        logical_index: int,
        lo_frequency_hz: float,
    ) -> None:
        """Apply the RF Outputs LO editor without opening the full panel."""

        try:
            self._hydrate_qcs_front_panel_editor()
            configuration = self._qcs_front_panel.working_configuration()
            mapping = next(
                candidate
                for candidate in configuration["channel_mappings"]
                if (
                    str(candidate["role"]) == "rf"
                    and int(candidate["logical_index"]) == int(logical_index)
                )
            )
            module = next(
                candidate
                for candidate in configuration["modules"]
                if int(candidate["slot"]) == int(mapping["slot"])
            )
            if str(module["model"]) != "M5300A":
                raise ValueError(
                    f"QCS RF output {int(logical_index)} is mapped to "
                    f"{module['model']}, not an M5300A."
                )
            changed = self._qcs_front_panel.set_m5300_lo_frequency(
                int(mapping["slot"]),
                int(mapping["channel"]),
                float(lo_frequency_hz),
            )
            if not changed:
                raise ValueError(self._qcs_front_panel.status.text())
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "M5300 LO was not changed",
                str(exc),
            )

    def _on_qcs_m5201_route_selection_started(self) -> None:
        """Temporarily accept the acquisition endpoint of a cable gesture."""

        if not self._qcs_m5201_route_context_active:
            self._qcs_m5201_route_previous_auto_selection = (
                self._qcs_front_panel_auto_apply_selection
            )
            self._qcs_m5201_route_previous_keep_open = (
                self._qcs_front_panel_keep_open_after_selection
            )
            self._qcs_m5201_route_context_active = True
        automatic = self._qcs_front_panel_auto_apply_selection
        acquisition = ("acquisition", 0)
        if isinstance(automatic, frozenset):
            self._qcs_front_panel_auto_apply_selection = (
                automatic | {acquisition}
            )
        elif automatic is None or automatic == acquisition:
            self._qcs_front_panel_auto_apply_selection = acquisition
        else:
            self._qcs_front_panel_auto_apply_selection = frozenset(
                (automatic, acquisition)
            )
        self._qcs_m5201_route_temporary_auto_selection = (
            self._qcs_front_panel_auto_apply_selection
        )

    def _on_qcs_m5201_route_selection_finished(self) -> None:
        """Restore the channel-selection context that opened the chassis."""

        if not self._qcs_m5201_route_context_active:
            return
        if (
            self._qcs_front_panel_auto_apply_selection
            == self._qcs_m5201_route_temporary_auto_selection
        ):
            self._qcs_front_panel_auto_apply_selection = (
                self._qcs_m5201_route_previous_auto_selection
            )
            self._qcs_front_panel_keep_open_after_selection = (
                self._qcs_m5201_route_previous_keep_open
            )
        self._qcs_m5201_route_context_active = False
        self._qcs_m5201_route_previous_auto_selection = None
        self._qcs_m5201_route_temporary_auto_selection = None
        self._qcs_m5201_route_previous_keep_open = False

    def _on_qcs_m5300_lo_frequency_changed(
        self,
        role: str,
        logical_index: int,
        slot: int,
        channel: int,
        lo_frequency_hz: float,
    ) -> None:
        """Persist a front-panel or RF-tab LO edit through one mapper job."""

        del lo_frequency_hz
        self._qcs_front_panel_auto_apply_selection = (
            str(role),
            int(logical_index),
        )
        self._qcs_front_panel_keep_open_after_selection = (
            self._qcs_front_panel_dialog.isVisible()
        )
        self._on_qcs_front_panel_connector_selected(
            str(role),
            int(logical_index),
            int(slot),
            int(channel),
            True,
        )

    def _on_qcs_front_panel_connector_picked(
        self,
        role: str,
        logical_index: int,
        slot: int,
        channel: int,
        mapping: Mapping[str, object],
    ) -> None:
        """Apply a calibration-local SMA pick without touching Experiment."""

        target = self._qcs_front_panel_pick_target
        if target is None:
            return
        try:
            expected_role, _expected_index = target.qcs_front_panel_selection()
            if str(expected_role) != str(role):
                raise ValueError(
                    f"Expected a QCS {expected_role} connector, not {role}"
                )
            target.accept_qcs_front_panel_connector(mapping)
        except (KeyError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration connector was not selected",
                str(exc),
            )
            return
        self._qcs_front_panel_pick_target = None
        self._qcs_front_panel.set_connector_pick_only(False)
        self._qcs_front_panel_auto_apply_selection = None
        self._qcs_front_panel_dialog.close()
        self.statusBar().showMessage(
            f"Calibration {role} uses slot {int(slot)} CH{int(channel)}; "
            "the Experiment mapper is unchanged",
            10000,
        )

    def _clear_qcs_front_panel_pick_context(self, *_args) -> None:
        self._qcs_front_panel_pick_target = None
        self._qcs_front_panel.set_connector_pick_only(False)

    def _on_qcs_front_panel_connector_selected(
        self,
        role: str,
        logical_index: int,
        slot: int,
        channel: int,
        changed: bool,
    ) -> None:
        """Queue mapper work after the selected SMA can be painted."""

        keep_open = self._qcs_front_panel_keep_open_after_selection
        selection = (str(role), int(logical_index))
        automatic_selection = self._qcs_front_panel_auto_apply_selection
        if isinstance(automatic_selection, frozenset):
            selection_matches = selection in automatic_selection
        else:
            selection_matches = selection == automatic_selection
        if selection_matches and (
            bool(changed)
            or self._experiment_panel._qcs_front_panel_draft_pending
        ):
            # Set the execution guard before yielding back to Qt. Even if a
            # Run click is already queued, it cannot use the previous native
            # mapper after the graphical assignment has changed.
            try:
                working_configuration = (
                    self._qcs_front_panel.working_configuration()
                )
            except (TypeError, ValueError):
                working_configuration = None
            if working_configuration is not None:
                self._stage_qcs_front_panel_pending_configuration(
                    working_configuration,
                    propagate=False,
                )
        # ``connector_selected`` is emitted directly from the mouse event.
        # Defer even the lightweight request capture by one event turn so the
        # selected ring is visibly painted before any follow-up processing.
        self._qcs_front_panel.reference_label.repaint()
        QtCore.QTimer.singleShot(
            0,
            lambda: self._commit_qcs_front_panel_connector_selection(
                role,
                logical_index,
                slot,
                channel,
                changed,
                keep_open=keep_open,
            ),
        )

    def _commit_qcs_front_panel_connector_selection(
        self,
        role: str,
        logical_index: int,
        slot: int,
        channel: int,
        changed: bool,
        *,
        keep_open: bool,
    ) -> None:
        """Commit one contextual SMA selection to the native QCS mapper."""

        selection = (str(role), int(logical_index))
        automatic_selection = self._qcs_front_panel_auto_apply_selection
        if isinstance(automatic_selection, frozenset):
            selection_matches = selection in automatic_selection
        else:
            selection_matches = selection == automatic_selection
        if not selection_matches:
            return
        try:
            working_configuration = (
                self._qcs_front_panel.working_configuration()
            )
        except (TypeError, ValueError):
            return
        module_model = next(
            (
                str(module["model"])
                for module in working_configuration["modules"]
                if int(module["slot"]) == int(slot)
            ),
            "QCS module",
        )
        selection_name = {
            "dc": f"QCS DC output {selection[1]}",
            "rf": f"QCS RF output {selection[1]}",
            "acquisition": "QCS acquisition input",
        }.get(selection[0], f"QCS {selection[0]} channel {selection[1]}")
        physical_name = (
            f"{module_model} slot {int(slot)} CH{int(channel)}"
        )
        if not bool(changed):
            try:
                experiment_settings = self._experiment_panel.qcs_settings_dict()
                run_ready = experiment_settings[
                    "hardware_configuration_state"
                ] in {
                    QCS_HARDWARE_STATE_SAVED,
                    QCS_HARDWARE_STATE_IMPORTED,
                }
                if run_ready:
                    self._experiment_panel.qcs_connection_values(
                        len(self._pulse)
                    )
                    experiment_mapping = next(
                        candidate
                        for candidate in experiment_settings[
                            "hardware_configuration"
                        ]["channel_mappings"]
                        if (
                            str(candidate["role"]) == selection[0]
                            and int(candidate["logical_index"])
                            == selection[1]
                        )
                    )
                    run_ready = (
                        int(experiment_mapping["slot"]),
                        int(experiment_mapping["channel"]),
                    ) == (int(slot), int(channel))
                    run_ready = run_ready and (
                        experiment_settings["hardware_configuration"]
                        == working_configuration
                    )
            except (
                ImportError,
                KeyError,
                OSError,
                StopIteration,
                TypeError,
                ValueError,
            ):
                run_ready = False
            if run_ready:
                self._experiment_panel.set_qcs_front_panel_draft_pending(
                    False
                )
                route_selection = (
                    selection == ("acquisition", 0)
                    and self._qcs_front_panel.m5201_route_selection_active()
                )
                route_from_other_focus = route_selection and (
                    self._qcs_front_panel._focused_mapping != selection
                )
                if route_selection:
                    self._qcs_front_panel.complete_m5201_route_selection()
                if not keep_open and not route_from_other_focus:
                    self._qcs_front_panel_auto_apply_selection = None
                    self._qcs_front_panel_dialog.close()
                self.statusBar().showMessage(
                    f"{selection_name} already uses {physical_name}",
                    10000,
                )
                return
        mapped_dc_indices = {
            int(mapping["logical_index"])
            for mapping in working_configuration["channel_mappings"]
            if str(mapping["role"]) == "dc"
        }
        required_dc_indices = set(range(len(self._pulse)))
        if mapped_dc_indices != required_dc_indices:
            self._on_qcs_front_panel_draft_staged(working_configuration)
            missing = sorted(required_dc_indices - mapped_dc_indices)
            if not keep_open:
                self._qcs_front_panel_auto_apply_selection = None
                self._qcs_front_panel_dialog.close()
            self.statusBar().showMessage(
                f"Mapped {selection_name} to {physical_name}; select SMA "
                "output(s) "
                + ", ".join(str(index) for index in missing)
                + " to finish the automatic mapper",
                12000,
            )
            return
        self._qcs_mapper_commit_preparing = True
        try:
            prepared = self._qcs_front_panel.prepare_connector_selection()
        finally:
            self._qcs_mapper_commit_preparing = False
        if prepared is None:
            # Preserve an invalid automatic-save selection as an editable
            # draft. A later click on the same SMA retries preparation.
            self._on_qcs_front_panel_draft_staged(working_configuration)
            return
        settings = dict(prepared["settings"])
        source_dc, source_rf, source_acquisition = (
            self._qcs_front_panel.working_source_bindings()
        )
        self._stage_qcs_front_panel_pending_configuration(
            working_configuration,
            propagate=False,
        )
        request = {
            "settings": settings,
            "configuration_fingerprint": (
                qcs_hardware_mapper_fingerprint(working_configuration)
            ),
            "mapper_path": str(prepared["mapper_path"]),
            "write_mapper": bool(prepared["write_mapper"]),
            "selection": selection,
            "address": (int(slot), int(channel)),
            "selection_name": selection_name,
            "physical_name": physical_name,
            "keep_open": bool(keep_open),
            "m5201_route_selection": (
                selection == ("acquisition", 0)
                and self._qcs_front_panel.m5201_route_selection_active()
            ),
            "source_snapshot": (
                self._current_qcs_front_panel_source_snapshot()
            ),
            "source_bindings": (
                tuple(str(name) for name in source_dc),
                {
                    int(index): str(name)
                    for index, name in source_rf.items()
                },
                source_acquisition,
            ),
            "output_count": len(self._pulse),
            "backend": self._experiment_panel.execution_backend(),
            "session_revision": self._qcs_front_panel_session_revision,
        }
        self._queue_qcs_mapper_commit(request)
        self.statusBar().showMessage(
            f"Selected {physical_name} for {selection_name}; generating the "
            "QCS mapper in the background",
        )

    def _stage_qcs_front_panel_pending_configuration(
        self,
        configuration: Mapping[str, object],
        *,
        propagate: bool,
    ) -> bool:
        """Make a graphical mapping authoritative while keeping Run blocked."""

        if not self._qcs_front_panel_editor_initialized:
            return False
        try:
            normalized = normalize_qcs_hardware_configuration(configuration)
            dc_names, rf_names, acquisition_name = (
                self._qcs_front_panel.working_source_bindings()
            )
            normalized_dc_names = tuple(str(name) for name in dc_names)
            normalized_rf_names = {
                int(index): str(name) for index, name in rf_names.items()
            }
            self._qcs_front_panel_pending_configuration = normalized
            self._qcs_front_panel_pending_bindings = (
                normalized_dc_names,
                normalized_rf_names,
                acquisition_name,
            )
            self._experiment_panel.set_qcs_front_panel_draft_pending(True)
            if (
                self._experiment_panel.execution_backend()
                != EXECUTION_BACKEND_QCS
            ):
                self._experiment_panel.set_execution_backend(
                    EXECUTION_BACKEND_QCS
                )
            elif propagate:
                self._propagate_qcs_hardware_configuration(
                    normalized,
                    normalized_dc_names,
                    normalized_rf_names,
                    acquisition_name,
                    require_complete_rf=False,
                )
        except (KeyError, TypeError, ValueError) as exc:
            self.statusBar().showMessage(
                f"QCS front-panel draft could not be previewed: {exc}",
                12000,
            )
            return False
        return True

    def _queue_qcs_mapper_commit(self, request: Mapping[str, object]) -> None:
        """Serialize native QCS mapper jobs and retain only the latest edit."""

        self._qcs_mapper_commit_revision += 1
        queued = dict(request)
        queued["revision"] = self._qcs_mapper_commit_revision
        self._qcs_mapper_commit_queued = queued
        if (
            self._qcs_mapper_commit_thread is None
            and self._qcs_mapper_commit_active is None
        ):
            self._start_next_qcs_mapper_commit()

    def _start_next_qcs_mapper_commit(self) -> None:
        request = self._qcs_mapper_commit_queued
        if request is None:
            return
        if (
            self._qcs_mapper_commit_thread is not None
            or self._qcs_mapper_commit_active is not None
        ):
            return
        if (
            self._experiment_thread is not None
            and self._experiment_thread.isRunning()
        ) or self._experiment_panel._running:
            return
        self._qcs_mapper_commit_queued = None
        self._qcs_mapper_commit_active = request
        self._qcs_mapper_commit_result = None
        self._qcs_mapper_commit_error = None
        thread = QtCore.QThread(self)
        worker = QcsMapperSaveWorker(
            int(request["revision"]),
            request["settings"]["hardware_configuration"],
            str(request["mapper_path"]),
            write_mapper=bool(request["write_mapper"]),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._receive_qcs_mapper_commit_result)
        worker.failed.connect(self._receive_qcs_mapper_commit_error)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(
            lambda revision=int(request["revision"]), owner=thread: (
                self._finish_qcs_mapper_commit(revision, owner)
            )
        )
        thread.finished.connect(thread.deleteLater)
        self._qcs_mapper_commit_thread = thread
        self._qcs_mapper_commit_worker = worker
        thread.start()

    def _receive_qcs_mapper_commit_result(
        self,
        revision: int,
        result: Mapping[str, object],
    ) -> None:
        if (
            self._qcs_mapper_commit_active is None
            or int(self._qcs_mapper_commit_active["revision"])
            != int(revision)
        ):
            return
        self._qcs_mapper_commit_result = (int(revision), dict(result))

    def _receive_qcs_mapper_commit_error(
        self,
        revision: int,
        details: str,
    ) -> None:
        if (
            self._qcs_mapper_commit_active is None
            or int(self._qcs_mapper_commit_active["revision"])
            != int(revision)
        ):
            return
        self._qcs_mapper_commit_error = (int(revision), str(details))

    def _qcs_mapper_request_is_current(
        self,
        request: Mapping[str, object],
    ) -> bool:
        if int(request["revision"]) != self._qcs_mapper_commit_revision:
            return False
        if int(request["output_count"]) != len(self._pulse):
            return False
        if request["backend"] != EXECUTION_BACKEND_QCS:
            return False
        if (
            self._experiment_panel.execution_backend()
            != EXECUTION_BACKEND_QCS
        ):
            return False
        if (
            request["source_snapshot"]
            != self._current_qcs_front_panel_source_snapshot()
        ):
            return False
        pending = self._pending_qcs_front_panel_state()
        if pending is None:
            return False
        pending_configuration = pending[0]
        request_configuration = normalize_qcs_hardware_configuration(
            request["settings"]["hardware_configuration"]
        )
        if pending_configuration != request_configuration:
            return False
        try:
            editor_configuration = (
                self._qcs_front_panel.working_configuration()
            )
            source_dc, source_rf, source_acquisition = (
                self._qcs_front_panel.working_source_bindings()
            )
        except (TypeError, ValueError):
            return False
        current_bindings = (
            tuple(str(name) for name in source_dc),
            {int(index): str(name) for index, name in source_rf.items()},
            source_acquisition,
        )
        return (
            editor_configuration == request_configuration
            and current_bindings == request["source_bindings"]
        )

    def _finish_qcs_mapper_commit(
        self,
        revision: int,
        owner_thread: QtCore.QThread,
    ) -> None:
        if (
            self._qcs_mapper_commit_thread is not owner_thread
            or self._qcs_mapper_commit_active is None
            or int(self._qcs_mapper_commit_active["revision"])
            != int(revision)
        ):
            return
        request = self._qcs_mapper_commit_active
        result_entry = self._qcs_mapper_commit_result
        error_entry = self._qcs_mapper_commit_error
        self._qcs_mapper_commit_thread = None
        self._qcs_mapper_commit_worker = None
        self._qcs_mapper_commit_active = None
        self._qcs_mapper_commit_result = None
        self._qcs_mapper_commit_error = None

        if request is not None and self._qcs_mapper_request_is_current(request):
            revision = int(request["revision"])
            if error_entry is not None and error_entry[0] == revision:
                details = error_entry[1]
                self._qcs_front_panel.fail_connector_selection(details)
                lines = [
                    line for line in details.rstrip().splitlines() if line.strip()
                ]
                summary = lines[-1] if lines else "Unknown QCS mapper error"
                self.statusBar().showMessage(
                    f"QCS mapper generation failed: {summary}",
                    12000,
                )
            elif result_entry is not None and result_entry[0] == revision:
                settings = dict(request["settings"])
                settings.update(result_entry[1])
                self._qcs_mapper_commit_completing = True
                try:
                    committed = (
                        self._qcs_front_panel.complete_connector_selection(
                            settings,
                            commit_callback=(
                                self._apply_qcs_front_panel_settings
                            ),
                        )
                    )
                finally:
                    self._qcs_mapper_commit_completing = False
                if committed:
                    self._experiment_panel.set_qcs_front_panel_draft_pending(
                        False
                    )
                    if bool(request.get("m5201_route_selection")):
                        self._qcs_front_panel.complete_m5201_route_selection()
                    if (
                        not bool(request["keep_open"])
                        and self._qcs_mapper_request_owns_dialog(request)
                    ):
                        self._qcs_front_panel_auto_apply_selection = None
                        self._qcs_front_panel_dialog.close()
                    self.statusBar().showMessage(
                        f"Mapped {request['selection_name']} to "
                        f"{request['physical_name']}",
                        10000,
                    )
                else:
                    # The physical edit remains visible and non-runnable so
                    # the user can retry it without reconstructing the panel.
                    self._experiment_panel.set_qcs_front_panel_draft_pending(
                        True
                    )

        # A rapid later click replaces the queued job. Start it only after the
        # previous QCS worker has fully exited, avoiding concurrent library or
        # file access.
        if self._qcs_mapper_commit_queued is not None:
            self._start_next_qcs_mapper_commit()

    def _qcs_mapper_request_owns_dialog(
        self,
        request: Mapping[str, object],
    ) -> bool:
        """Return whether an old completion still owns the visible picker."""

        if (
            int(request["session_revision"])
            != self._qcs_front_panel_session_revision
        ):
            return False
        selection = tuple(request["selection"])
        automatic_selection = self._qcs_front_panel_auto_apply_selection
        if isinstance(automatic_selection, frozenset):
            if selection not in automatic_selection:
                return False
        elif selection != automatic_selection:
            return False
        if self._qcs_front_panel._focused_mapping != selection:
            return False
        try:
            address = self._qcs_front_panel._focused_mapping_address(
                self._qcs_front_panel.working_configuration()
            )
        except (TypeError, ValueError):
            return False
        return address == tuple(request["address"])

    def _on_qcs_front_panel_draft_staged(
        self,
        configuration: Mapping[str, object],
    ) -> None:
        """Preview an incomplete editor draft while keeping it non-executable."""

        # Closing an invalidated editor can emit QLineEdit.editingFinished.
        # Once the editor has been invalidated, that late signal must not
        # resurrect the obsolete topology or its source snapshot.
        if not self._qcs_front_panel_editor_initialized:
            return
        # Manual topology edits and incomplete multi-output assignments still
        # update every compact preview. Complete SMA selections use the same
        # helper with ``propagate=False`` while their mapper is saved.
        self._stage_qcs_front_panel_pending_configuration(
            configuration,
            propagate=True,
        )

    def _identify_qcs_hardware_configuration(
        self,
        ip_address: str,
    ) -> None:
        """Read installed QCS modules without blocking the Qt GUI thread."""

        if self._qcs_mapper_commit_pending():
            QtWidgets.QMessageBox.information(
                self,
                "QCS mapper updating",
                "Wait for the selected SMA assignment to finish saving "
                "before identifying the chassis again.",
            )
            return
        if (
            self._experiment_thread is not None
            and self._experiment_thread.isRunning()
        ) or self._experiment_panel._running:
            QtWidgets.QMessageBox.information(
                self,
                "Hardware task running",
                "Wait for the current hardware task to finish before "
                "identifying the QCS chassis.",
            )
            return

        self._pending_qcs_hardware_inventory = None
        self._pending_qcs_hardware_error = None
        self._qcs_front_panel.set_identifying(
            True,
            f"Connecting to {ip_address} and reading installed modules...",
        )
        self.statusBar().showMessage(
            f"Identifying QCS hardware at {ip_address}"
        )
        thread = QtCore.QThread(self)
        worker = QcsHardwareIdentificationWorker(ip_address)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(
            self._on_qcs_hardware_inventory_received
        )
        worker.failed.connect(
            self._on_qcs_hardware_identification_failed
        )
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(
            self._finish_qcs_hardware_identification
        )
        thread.finished.connect(thread.deleteLater)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _on_qcs_hardware_inventory_received(
        self,
        inventory: Mapping[str, object],
    ) -> None:
        self._pending_qcs_hardware_inventory = dict(inventory)

    def _on_qcs_hardware_identification_failed(
        self,
        details: str,
    ) -> None:
        self._pending_qcs_hardware_error = str(details)

    def _finish_qcs_hardware_identification(self) -> None:
        inventory = self._pending_qcs_hardware_inventory
        details = self._pending_qcs_hardware_error
        self._pending_qcs_hardware_inventory = None
        self._pending_qcs_hardware_error = None
        self._clear_experiment_thread()

        if details is not None:
            lines = [
                line
                for line in details.rstrip().splitlines()
                if line.strip()
            ]
            summary = (
                lines[-1]
                if lines
                else "Unknown QCS hardware identification error"
            )
            self._qcs_front_panel.set_identifying(
                False,
                f"Hardware identification failed: {summary}",
                error=True,
            )
            self.statusBar().showMessage(
                "QCS hardware identification failed"
            )
            dialog = DetailedErrorMessageBox(
                "QCS hardware identification failed",
                summary,
                details,
                self,
            )
            dialog.exec_()
            return

        if inventory is None:
            summary = "QCS hardware identification returned no result"
            self._qcs_front_panel.set_identifying(
                False,
                summary,
                error=True,
            )
            self.statusBar().showMessage(summary)
            return

        try:
            self._validate_qcs_front_panel_source_snapshot()
        except ValueError as exc:
            self._qcs_front_panel.set_identifying(
                False,
                "The QCS inventory was not applied because the experiment "
                f"configuration changed: {exc}",
                error=True,
            )
            self.statusBar().showMessage(
                "QCS inventory ignored because its source settings changed",
                10000,
            )
            return

        self._qcs_front_panel.set_identifying(
            False,
            "QCS inventory received; updating the front panel...",
        )
        try:
            applied = (
                self._qcs_front_panel
                .apply_discovered_hardware_configuration(
                    inventory,
                    commit_callback=self._apply_qcs_front_panel_settings,
                )
            )
        except Exception:
            details = traceback.format_exc()
            lines = [
                line
                for line in details.rstrip().splitlines()
                if line.strip()
            ]
            summary = (
                lines[-1]
                if lines
                else "Unknown QCS inventory application error"
            )
            self._qcs_front_panel.set_identifying(
                False,
                f"Discovered hardware was not applied: {summary}",
                error=True,
            )
            dialog = DetailedErrorMessageBox(
                "QCS hardware inventory could not be applied",
                summary,
                details,
                self,
            )
            dialog.exec_()
            return
        self.statusBar().showMessage(
            "QCS hardware identified and front panel updated"
            if applied
            else "QCS hardware identified; current configuration unchanged",
            10000,
        )

    def _apply_qcs_front_panel_settings(
        self,
        front_panel_settings: Mapping[str, object],
    ) -> bool:
        """Synchronize mapper-builder role bindings with Experiment controls."""
        try:
            self._validate_qcs_front_panel_source_snapshot()
            payload = dict(front_panel_settings)
            configuration = normalize_qcs_hardware_configuration(
                payload.get("hardware_configuration"),
                required_dc_count=len(self._pulse),
            )
            bound_dc, bound_rf, bound_acquisition = qcs_role_bindings(
                configuration,
                required_dc_count=len(self._pulse),
            )
            payload_dc = [
                str(name) for name in payload.get("dc_channel_names", ())
            ]
            payload_rf = {
                str(int(channel)): str(name)
                for channel, name in dict(
                    payload.get("rf_channel_names", {})
                ).items()
            }
            payload_acquisition = payload.get("acquisition_channel_name")
            payload_acquisition = (
                None
                if payload_acquisition is None
                else str(payload_acquisition).strip() or None
            )
            if (
                bound_dc != payload_dc
                or bound_rf != payload_rf
                or bound_acquisition != payload_acquisition
            ):
                raise ValueError(
                    "The front-panel role bindings are inconsistent with "
                    "the current experiment"
                )
            normalized_rf_bindings = {
                int(index): str(name)
                for index, name in bound_rf.items()
            }
            self._rf_ports_panel.validate_qcs_rf_channel_names(
                normalized_rf_bindings
            )
            state = str(
                payload.get(
                    "hardware_configuration_state",
                    QCS_HARDWARE_STATE_DRAFT,
                )
            )
            if state not in QCS_HARDWARE_STATES:
                raise ValueError(
                    f"unsupported QCS hardware configuration state {state!r}"
                )
            payload["hardware_configuration"] = configuration
            payload["hardware_configuration_state"] = state
            settings = self._experiment_panel.qcs_settings_dict()
            for override_key in (
                "dc_channel_names_text",
                "rf_channel_names_text",
                "acquisition_channel_name_text",
            ):
                settings.pop(override_key, None)
            settings.update(payload)
            was_qcs = (
                self._experiment_panel.execution_backend()
                == EXECUTION_BACKEND_QCS
            )
            self._experiment_panel.set_qcs_settings(
                settings,
                len(self._pulse),
            )
            self._clear_pending_qcs_front_panel_configuration()
            self._experiment_panel.set_execution_backend(
                EXECUTION_BACKEND_QCS
            )
            if was_qcs:
                self._propagate_qcs_hardware_configuration(
                    configuration,
                    tuple(bound_dc),
                    normalized_rf_bindings,
                    bound_acquisition,
                )
            self._qcs_front_panel_source_snapshot = (
                self._current_qcs_front_panel_source_snapshot()
            )
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "QCS front panel not applied",
                str(exc),
            )
            return False
        mapping_count = len(
            settings["hardware_configuration"]["channel_mappings"]
        )
        self.statusBar().showMessage(
            f"Applied QCS front panel with {mapping_count} channel "
            "mapping(s)",
            10000,
        )
        return True

    def _current_qcs_front_panel_source_snapshot(self) -> tuple:
        panel = self._experiment_panel
        hardware_configuration = panel._qcs_hardware_configuration
        hardware_fingerprint = (
            None
            if hardware_configuration is None
            else qcs_hardware_mapper_fingerprint(hardware_configuration)
        )
        return (
            panel.qcs_mapper_path.text(),
            panel.qcs_dc_channel_names.text(),
            panel.qcs_rf_channel_names.text(),
            panel.qcs_acquisition_channel_name.text(),
            len(self._pulse),
            panel._qcs_hardware_configuration_state,
            panel._qcs_hardware_mapper_sha256,
            hardware_fingerprint,
        )

    def _validate_qcs_front_panel_source_snapshot(self) -> None:
        if (
            self._qcs_mapper_commit_pending()
            and not self._qcs_mapper_commit_completing
            and not self._qcs_mapper_commit_preparing
        ):
            raise ValueError(
                "The selected QCS SMA assignment is still generating or "
                "validating its native mapper. Wait for that background "
                "update before loading, saving, or applying another mapper."
            )
        if (
            self._experiment_thread is not None
            and self._experiment_thread.isRunning()
        ) or self._experiment_panel._running:
            raise ValueError(
                "The QCS hardware configuration cannot be applied or saved "
                "while a hardware task is running."
            )
        if (
            self._qcs_front_panel_source_snapshot
            != self._current_qcs_front_panel_source_snapshot()
        ):
            raise ValueError(
                "The main QCS mapper path or channel names changed while the "
                "front panel was open. Reopen the front panel before applying "
                "or saving."
            )

    def _on_qcs_output_count_changed(self, output_count: int) -> None:
        """Invalidate a modeless front-panel snapshot after AWG topology edits."""
        # An identified-but-incomplete topology is the authoritative visual
        # source until its missing SMAs are selected.  _add_port/_delete_port
        # resize that pending state after their pulse mutation completes.
        if self._pending_qcs_front_panel_state() is not None:
            return
        if self._preserve_qcs_front_panel_draft_on_output_count_change:
            return
        self._clear_pending_qcs_front_panel_configuration()
        self._retire_qcs_mapper_commits()
        self._qcs_front_panel_source_snapshot = None
        self._qcs_front_panel_editor_initialized = False
        self._experiment_panel.set_qcs_front_panel_draft_pending(False)
        if self._qcs_front_panel_dialog.isVisible():
            self._qcs_front_panel_dialog.close()
            self.statusBar().showMessage(
                "QCS front panel closed because the waveform output count "
                f"changed to {int(output_count)}. Reopen it to continue.",
                10000,
            )

    def _set_awg_output_channel(self, output_index: int, channel: int) -> None:
        """Apply one front-panel generator mapping without duplicate channels."""
        output_index = int(output_index)
        channel = int(channel)
        if not 0 <= output_index < len(self._qick_awg_channels):
            raise IndexError("AWG output index is out of range")
        if channel < 0:
            raise ValueError("AWG generator channel must be nonnegative")
        channels = list(self._qick_awg_channels)
        previous = channels[output_index]
        if channel in channels and channels.index(channel) != output_index:
            other_index = channels.index(channel)
            channels[other_index] = previous
        channels[output_index] = channel
        self._qick_awg_channels = tuple(channels)
        self._multi_ctrl.set_awg_channels(self._qick_awg_channels)
        self._experiment_panel.set_qick_values(
            fabric_mhz=self._qick_fabric_mhz,
            tproc_mhz=self._qick_tproc_mhz,
            full_scale_mv=self._qick_full_scale_mv,
            awg_channels=self._qick_awg_channels,
            repetitions=self._qick_repetitions_per_sweep,
        )
        self.refresh_panel_table()
        self._refresh_stability_targets()

    def _build_menu(self):
        mb          = self.menuBar()

        # File
        m_file      = mb.addMenu("&File")
        save_settings = m_file.addAction("Save Settings JSON...")
        save_settings.setShortcut(QtGui.QKeySequence.Save)
        save_settings.triggered.connect(self._save_json)
        load_settings = m_file.addAction("Load Settings JSON...")
        load_settings.setShortcut(QtGui.QKeySequence.Open)
        load_settings.triggered.connect(self._load_json)
        m_file.addSeparator()
        m_file.addAction("E&xit",       self.close)

        # Setup
        m_setup = mb.addMenu("&Setup")
        self._qick_setup_action = m_setup.addAction(
            "QICK Connection and Clocks..."
        )
        self._qick_setup_action.triggered.connect(
            self._configure_qick_setup
        )
        self._qcs_setup_action = m_setup.addAction(
            "Keysight QCS M5000 Front Panel..."
        )
        self._qcs_setup_action.triggered.connect(
            self._show_qcs_front_panel
        )

        # Pulse
        m_pulse     = mb.addMenu("&Pulse")
        self._generate_qcs_action = m_pulse.addAction(
            "&Generate QCS string…"
        )
        self._generate_qcs_action.setShortcut(
            QtGui.QKeySequence("Ctrl+G")
        )
        self._generate_qcs_action.triggered.connect(self._generate_pulse)
        m_pulse.addSeparator()
        self._generate_qick_action = m_pulse.addAction(
            "Generate QICK program..."
        )
        self._generate_qick_action.setShortcut(
            QtGui.QKeySequence("Ctrl+Shift+G")
        )
        self._generate_qick_action.triggered.connect(
            self._generate_qick_pulse
        )
        self._preview_qick_action = m_pulse.addAction(
            "Preview QICK waveforms..."
        )
        self._preview_qick_action.triggered.connect(
            self._preview_qick_pulse
        )
        sync_timing = m_pulse.addAction("Synchronize port timing from selected port")
        sync_timing.triggered.connect(self._synchronize_port_timing)
        m_pulse.addSeparator()
        cross_capacitance = m_pulse.addAction(
            "Virtual-to-physical cross-capacitance matrix..."
        )
        cross_capacitance.setShortcut(QtGui.QKeySequence("Ctrl+Alt+C"))
        cross_capacitance.triggered.connect(self._configure_cross_capacitance)

        # Sweep
        m_sweep = mb.addMenu("&Sweep")
        clear_sweeps = m_sweep.addAction("Clear all sweeps")
        clear_sweeps.triggered.connect(
            lambda: self._clear_sweep(
                "All voltage, RAMP-rate, and hold-time sweeps removed"
            )
        )

        # Grid
        m_grid = mb.addMenu("&Grid")
        configure_grid = m_grid.addAction("Configure grid...")
        configure_grid.setShortcut(QtGui.QKeySequence("Ctrl+Alt+G"))
        configure_grid.triggered.connect(self._configure_grid)
        m_grid.addSeparator()
        self._grid_snap_action = m_grid.addAction("Snap dragged points")
        self._grid_snap_action.setCheckable(True)
        self._grid_snap_action.setChecked(self._grid_snap_enabled)
        self._grid_snap_action.toggled.connect(self._toggle_grid_snap)
        self._grid_visible_action = m_grid.addAction("Show fixed grid")
        self._grid_visible_action.setCheckable(True)
        self._grid_visible_action.setChecked(self._grid_visible)
        self._grid_visible_action.toggled.connect(self._toggle_grid_visible)

        # View
        m_view = mb.addMenu("&View")
        voltage_view = m_view.addMenu("Voltage traces")
        self._voltage_view_actions = {}
        voltage_view_group = QtWidgets.QActionGroup(self)
        voltage_view_group.setExclusive(True)
        for label, mode in (
            ("Virtual + Physical", "both"),
            ("Virtual only", "virtual"),
            ("Physical only", "physical"),
        ):
            action = voltage_view.addAction(label)
            action.setCheckable(True)
            action.setChecked(mode == "both")
            action.triggered.connect(
                lambda checked=False, selected=mode: self._set_voltage_view(selected)
            )
            voltage_view_group.addAction(action)
            self._voltage_view_actions[mode] = action

    def _set_grid_settings(
        self,
        *,
        time_step_ns: float,
        voltage_step_mv: float,
        snap_enabled: bool,
        visible: bool,
    ) -> None:
        self._grid_time_ns = float(time_step_ns)
        self._grid_voltage_mv = float(voltage_step_mv)
        self._grid_snap_enabled = bool(snap_enabled)
        self._grid_visible = bool(visible)
        self._plot.set_grid(
            time_step_ns=self._grid_time_ns,
            voltage_step_mv=self._grid_voltage_mv,
            snap_enabled=self._grid_snap_enabled,
            visible=self._grid_visible,
        )
        if hasattr(self, "_grid_snap_action"):
            with QtCore.QSignalBlocker(self._grid_snap_action):
                self._grid_snap_action.setChecked(self._grid_snap_enabled)
        if hasattr(self, "_grid_visible_action"):
            with QtCore.QSignalBlocker(self._grid_visible_action):
                self._grid_visible_action.setChecked(self._grid_visible)

    def _set_time_unit(self, unit: str) -> None:
        if unit not in TIME_UNIT_NS:
            raise ValueError(f"unsupported time unit {unit!r}")
        self._time_unit = unit
        self._multi_ctrl.set_time_unit(unit)
        self._rf_ports_panel.set_time_unit(unit)
        self._rf_readout_panel.set_time_unit(unit)
        if hasattr(self._plot, "set_time_unit"):
            self._plot.set_time_unit(unit)
        if self._trace is not None and hasattr(self._trace, "set_time_unit"):
            self._trace.set_time_unit(unit)
        for timeline in self._rf_timelines:
            if hasattr(timeline, "set_time_unit"):
                timeline.set_time_unit(unit)
        self.statusBar().showMessage(f"Time display unit: {unit}")

    def _configure_grid(self) -> None:
        dialog = GridSettingsDialog(
            time_step_ns=self._grid_time_ns,
            voltage_step_mv=self._grid_voltage_mv,
            snap_enabled=(self._grid_snap_enabled if self._grid_configured else True),
            visible=self._grid_visible,
            time_unit=self._time_unit,
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        time_step_ns, voltage_step_mv, snap_enabled, visible = dialog.values()
        self._grid_configured = True
        self._set_grid_settings(
            time_step_ns=time_step_ns,
            voltage_step_mv=voltage_step_mv,
            snap_enabled=snap_enabled,
            visible=visible,
        )
        self.statusBar().showMessage(
            f"Grid: {_time_from_ns(time_step_ns, self._time_unit):.6g} "
            f"{self._time_unit} x {voltage_step_mv:.6g} mV; "
            f"snap {'on' if snap_enabled else 'off'}"
        )

    def _toggle_grid_snap(self, enabled: bool) -> None:
        self._grid_configured = True
        self._set_grid_settings(
            time_step_ns=self._grid_time_ns,
            voltage_step_mv=self._grid_voltage_mv,
            snap_enabled=enabled,
            visible=self._grid_visible,
        )

    def _toggle_grid_visible(self, visible: bool) -> None:
        self._set_grid_settings(
            time_step_ns=self._grid_time_ns,
            voltage_step_mv=self._grid_voltage_mv,
            snap_enabled=self._grid_snap_enabled,
            visible=visible,
        )

    def _configure_cross_capacitance(self) -> None:
        dialog = CrossCapacitanceDialog(
            len(self._pulse),
            self._cross_capacitance,
            self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        self._cross_capacitance = dialog.matrix()
        self._refresh_sweep_overlay(fit_view=True)
        off_diagonal = self._cross_capacitance.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        nonzero = int(np.count_nonzero(np.abs(off_diagonal) > 1.0e-15))
        self.statusBar().showMessage(
            "Virtual-to-physical matrix applied: "
            f"{nonzero} nonzero cross-capacitance terms"
        )

    def _set_voltage_view(self, mode: str) -> None:
        self._plot.set_voltage_view(mode)
        self.statusBar().showMessage(
            {
                "both": "Showing virtual (solid) and physical (dashed) voltages",
                "virtual": "Showing editable virtual-gate voltages",
                "physical": "Showing physical AWG voltages",
            }[mode]
        )

    def _refresh_physical_waveforms(self, *, fit_view: bool = False) -> None:
        full_scale_mv = self._active_awg_full_scale_mv()
        if self._bias_t_compensation_enabled:
            sequence = build_qick_sequence(
                self._pulse,
                output_names=self._qick_output_names(),
                fabric_mhz=self._qick_fabric_mhz,
                full_scale_mv=full_scale_mv,
                sweep=(self._sweep_specs[0] if len(self._sweep_specs) == 1 else None),
                sweeps=(tuple(self._sweep_specs) if len(self._sweep_specs) > 1 else None),
                cross_capacitance=self._cross_capacitance,
                bias_t_compensation_enabled=True,
                bias_t_compensation_type=self._bias_t_compensation_type,
                bias_t_compensation_voltage_mv=self._bias_t_compensation_voltage_mv,
                bias_t_compensation_mode=self._bias_t_compensation_mode,
                bias_t_compensation_duration_us=(
                    self._bias_t_compensation_duration_us
                ),
                bias_t_filter_tau_us=self._bias_t_filter_tau_us,
            )
            cycles, waveforms, _boundaries = (
                sequence.compensated_waveform_vertices(0)
            )
            time_ns = np.asarray(cycles, dtype=float) * 1000.0 / self._qick_fabric_mhz
            physical_mv = np.vstack([
                np.asarray(waveforms[name], dtype=float)
                * full_scale_mv
                for name in self._qick_output_names()
            ])
        else:
            time_ns, _virtual_mv, physical_mv = transform_virtual_waveforms(
                self._pulse,
                self._cross_capacitance,
            )
        self._plot.set_physical_waveforms(time_ns, physical_mv)
        if fit_view:
            self._plot.fit_view()

    def _refresh_qcs_waveform_capacity(self) -> None:
        """Refresh the live M5301 budget without loading a mapper or hardware."""
        if (
            not hasattr(self, "_experiment_panel")
            or self._experiment_panel.execution_backend()
            != EXECUTION_BACKEND_QCS
        ):
            return
        try:
            full_scale_mv = self._active_awg_full_scale_mv()
            arguments = self._experiment_run_arguments(
                require_readout=False,
                require_run_config=False,
                validate_qick_hardware=False,
                validate_qcs_hardware=False,
                for_qcs=True,
            )
            sequence = arguments["sequence"]
            sweep_preview = qcs_sweep_execution_preview(
                sequence,
                hardware_demodulation=(
                    self._experiment_panel.qcs_hw_demod.isChecked()
                ),
                source_full_scale_mv=full_scale_mv,
                dc_full_scale_v=(
                    self._experiment_panel.qcs_dc_full_scale_v.value()
                ),
                fabric_mhz=self._qick_fabric_mhz,
                init_time_s=(
                    self._experiment_panel.qcs_init_time_us.value() * 1.0e-6
                ),
            )
            self._experiment_panel.set_qcs_sweep_execution_status(
                sweep_preview
            )
            if sweep_preview.mode == "hybrid":
                # A mixed sweep can choose a different safe Hold graph for
                # each Python-loop coordinate. The worker performs an
                # exhaustive all-slices capacity preflight before its first
                # hardware submission.
                self._experiment_panel.set_qcs_mixed_waveform_capacity_pending()
                return
            point_indices = (
                None
                if int(sequence.sweep_point_count) <= 256
                else qcs_m5301_capacity_preview_point_indices(sequence)
            )
            report = qcs_m5301_waveform_capacity_report(
                sequence,
                point_indices=point_indices,
                fabric_mhz=self._qick_fabric_mhz,
                amplitude_scale=(
                    full_scale_mv
                    / (
                        self._experiment_panel.qcs_dc_full_scale_v.value()
                        * 1000.0
                    )
                ),
                source_full_scale_mv=full_scale_mv,
                dc_full_scale_v=(
                    self._experiment_panel.qcs_dc_full_scale_v.value()
                ),
            )
            physical_names = tuple(
                name.strip()
                for name in self._experiment_panel.qcs_dc_channel_names.text().split(",")
                if name.strip()
            )
            self._experiment_panel.set_qcs_waveform_capacity(
                report,
                physical_names,
            )
        except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            self._experiment_panel.set_qcs_waveform_capacity_error(str(exc))
            self._experiment_panel.set_qcs_sweep_execution_status(
                QcsSweepExecutionPreview(
                    mode="invalid",
                    reasons=(str(exc),),
                )
            )

    def _sweep_target_indices(
        self,
        spec: Optional[QickSweepAxisSpec] = None,
    ) -> Optional[Tuple[int, int]]:
        if spec is None:
            spec = self._sweep_spec
        if spec is None or not isinstance(spec, QickSweepSpec):
            return None
        try:
            port_index = self._qick_output_names().index(spec.output_name)
            segment_index = int(spec.segment_name.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return None
        if port_index >= len(self._pulse):
            return None
        if (
            segment_index < 0
            or segment_index >= len(self._pulse[port_index].flat_segments())
        ):
            return None
        return port_index, segment_index

    @staticmethod
    def _ramp_sweep_row(
        spec: QickSweepAxisSpec,
    ) -> Optional[int]:
        if not isinstance(spec, QickRampRateSweepSpec):
            return None
        match = re.fullmatch(r"ramp_(\d+)_to_(\d+)", spec.segment_name)
        if match is None:
            return None
        source_row, destination_row = map(int, match.groups())
        if destination_row != source_row + 1:
            return None
        return destination_row

    @staticmethod
    def _hold_sweep_row(
        spec: QickSweepAxisSpec,
    ) -> Optional[int]:
        if not isinstance(spec, QickHoldDurationSweepSpec):
            return None
        match = re.fullmatch(r"set_(\d+)", spec.segment_name)
        return None if match is None else int(match.group(1))

    def _sweep_cartesian_count(self) -> int:
        specs = self._active_map_sweep_specs()
        return prod(spec.count for spec in specs) if specs else 1

    def _sync_sweep_rows(self) -> None:
        rows_by_port = {index: set() for index in range(len(self._pulse))}
        ramp_rows = {
            row
            for row in (
                self._ramp_sweep_row(spec) for spec in self._sweep_specs
            )
            if row is not None
        }
        hold_rows = {
            row
            for row in (
                self._hold_sweep_row(spec) for spec in self._sweep_specs
            )
            if row is not None
        }
        for spec in self._sweep_specs:
            target = self._sweep_target_indices(spec)
            if target is not None:
                rows_by_port[target[0]].add(target[1])
        for port_index, control in enumerate(self._multi_ctrl._ctrl_pannels):
            rows = rows_by_port.get(port_index, set())
            color = None
            if rows:
                color = (
                    self._plot.line_color(port_index)
                    if hasattr(self._plot, "line_color")
                    else QtGui.QColor(self._multi_ctrl._color_map[port_index])
                )
            control.set_sweep_state(rows, ramp_rows, hold_rows, color)

    def _reconcile_sweep_specs(self) -> int:
        """Drop sweep references that no longer resolve to live segments."""
        valid_specs = []
        removed = 0
        for spec in self._sweep_specs:
            if isinstance(spec, QickSweepSpec):
                valid = self._sweep_target_indices(spec) is not None
            elif isinstance(spec, QickRampRateSweepSpec):
                row = self._ramp_sweep_row(spec)
                valid = (
                    row is not None
                    and all(
                        row < len(pulse.flat_segments())
                        for pulse in self._pulse
                    )
                )
            elif isinstance(spec, QickHoldDurationSweepSpec):
                row = self._hold_sweep_row(spec)
                valid = (
                    row is not None
                    and all(
                        row < len(pulse.flat_segments())
                        for pulse in self._pulse
                    )
                )
            else:
                valid = False
            if valid:
                valid_specs.append(spec)
            else:
                removed += 1
        if removed:
            self._sweep_specs = valid_specs
        return removed

    def _notify_sweep_state_changed(
        self,
        *,
        fit_view: bool = False,
        selected_map_keys: Optional[
            Tuple[Tuple[str, str], Tuple[str, str]]
        ] = None,
        selected_parameter_key: Optional[
            Tuple[str, str, str]
        ] = None,
        waveform_changed: bool = False,
        trace_changed: bool = False,
        rf_changed: bool = False,
    ) -> None:
        """Emit one event after a sweep or its segment model changes."""
        self.sweep_state_changed.emit(
            {
                "fit_view": bool(fit_view),
                "selected_map_keys": selected_map_keys,
                "selected_parameter_key": selected_parameter_key,
                "waveform_changed": bool(waveform_changed),
                "trace_changed": bool(trace_changed),
                "rf_changed": bool(rf_changed),
            }
        )

    @QtCore.pyqtSlot(object)
    def _synchronize_sweep_views(self, request) -> None:
        """Synchronize the sweep model with every dependent AWG view."""
        options = request if isinstance(request, Mapping) else {}
        self._reconcile_sweep_specs()
        if options.get("waveform_changed", False):
            self._refresh_waveform_plot()
        self._refresh_sweep_overlay(
            fit_view=bool(options.get("fit_view", False)),
            sync_rows=True,
            selected_map_keys=options.get("selected_map_keys"),
            selected_parameter_key=options.get(
                "selected_parameter_key"
            ),
        )
        if options.get("trace_changed", False):
            self._refresh_trace_if_needed(force=True)
        if options.get("rf_changed", False):
            self._refresh_rf_editor()

    def _refresh_sweep_overlay(
        self,
        *,
        fit_view: bool = False,
        sync_rows: bool = False,
        selected_map_keys: Optional[
            Tuple[Tuple[str, str], Tuple[str, str]]
        ] = None,
        selected_parameter_key: Optional[
            Tuple[str, str, str]
        ] = None,
    ) -> None:
        full_scale_mv = self._active_awg_full_scale_mv()
        if hasattr(self, "_experiment_panel"):
            self._experiment_panel.set_sweep_specs(
                self._active_map_sweep_specs(),
                selected_keys=selected_map_keys,
                selected_parameter_key=selected_parameter_key,
            )
        self._refresh_physical_waveforms()
        envelopes = []
        sweeps_by_segment = {}
        for spec in self._sweep_specs:
            if not isinstance(spec, QickSweepSpec):
                continue
            target = self._sweep_target_indices(spec)
            if target is None:
                continue
            source_port, segment_index = target
            sweeps_by_segment.setdefault(
                (spec.segment_name, segment_index), []
            ).append((spec, source_port))

        for (segment_name, segment_index), segment_sweeps in sweeps_by_segment.items():
            point_index = segment_index * 2
            for destination_port in range(len(self._pulse)):
                if not any(
                    abs(self._cross_capacitance[destination_port, source_port])
                    > 1.0e-15
                    for _spec, source_port in segment_sweeps
                ):
                    continue
                lower_virtual = [pulse.copy() for pulse in self._pulse]
                upper_virtual = [pulse.copy() for pulse in self._pulse]
                for spec, source_port in segment_sweeps:
                    coefficient = float(
                        self._cross_capacitance[destination_port, source_port]
                    )
                    candidates = (
                        (coefficient * spec.start, spec.start),
                        (coefficient * spec.stop, spec.stop),
                    )
                    lower_value = min(candidates, key=lambda item: item[0])[1]
                    upper_value = max(candidates, key=lambda item: item[0])[1]
                    lower_virtual[source_port].v[
                        point_index:point_index + 2
                    ] = lower_value * full_scale_mv
                    upper_virtual[source_port].v[
                        point_index:point_index + 2
                    ] = upper_value * full_scale_mv
                time_ns, _virtual_lower, physical_lower = transform_virtual_waveforms(
                    lower_virtual,
                    self._cross_capacitance,
                )
                upper_time_ns, _virtual_upper, physical_upper = (
                    transform_virtual_waveforms(
                        upper_virtual,
                        self._cross_capacitance,
                    )
                )
                if not np.array_equal(time_ns, upper_time_ns):
                    raise RuntimeError("sweep endpoint time grids do not match")
                envelopes.append(
                    (
                        (f"awg_{destination_port}", segment_name),
                        destination_port,
                        time_ns,
                        physical_lower[destination_port],
                        physical_upper[destination_port],
                    )
                )

        self._plot.set_sweep_envelopes(envelopes)
        if sync_rows:
            self._sync_sweep_rows()
        if fit_view:
            self._plot.fit_view()
        self._refresh_qcs_waveform_capacity()

    def _clear_sweep(self, message: Optional[str] = None) -> None:
        self._sweep_specs = []
        self._notify_sweep_state_changed()
        if message:
            self.statusBar().showMessage(message)

    def _configure_segment_sweep(self, port_index: int, segment_index: int) -> None:
        if port_index < 0 or port_index >= len(self._pulse):
            return
        flat_segments = self._pulse[port_index].flat_segments()
        if segment_index < 0 or segment_index >= len(flat_segments):
            return
        output_name = f"awg_{port_index}"
        segment_name = f"set_{segment_index}"
        point_index = segment_index * 2
        full_scale_mv = self._active_awg_full_scale_mv()
        current_amplitude = float(
            np.clip(
                self._pulse[port_index].v[point_index] / full_scale_mv,
                -1.0,
                1.0,
            )
        )
        initial = next(
            (
                spec
                for spec in self._sweep_specs
                if spec.output_name == output_name and spec.segment_name == segment_name
            ),
            None,
        )
        other_point_count = prod(
            spec.count for spec in self._sweep_specs if spec is not initial
        ) if self._sweep_specs else 1
        dialog = SweepSettingsDialog(
            output_name=output_name,
            segment_name=segment_name,
            current_amplitude=current_amplitude,
            full_scale_mv=full_scale_mv,
            initial=initial,
            cartesian_base_count=other_point_count,
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        new_spec = dialog.value()
        for index, spec in enumerate(self._sweep_specs):
            if spec.output_name == output_name and spec.segment_name == segment_name:
                self._sweep_specs[index] = new_spec
                break
        else:
            self._sweep_specs.append(new_spec)
        self._port_select(port_index)
        self._notify_sweep_state_changed(fit_view=True)
        self.statusBar().showMessage(
            f"Sweep applied to {output_name}/{segment_name}: "
            f"{new_spec.start * full_scale_mv:.6g} mV to "
            f"{new_spec.stop * full_scale_mv:.6g} mV, "
            f"{new_spec.count} axis points; "
            f"{self._sweep_cartesian_count()} Cartesian combinations"
        )

    def _remove_segment_sweep(self, port_index: int, segment_index: int) -> None:
        remaining = [
            spec
            for spec in self._sweep_specs
            if self._sweep_target_indices(spec) != (port_index, segment_index)
        ]
        if len(remaining) != len(self._sweep_specs):
            self._sweep_specs = remaining
            self._notify_sweep_state_changed(fit_view=True)
            self.statusBar().showMessage(
                f"Voltage sweep removed; {self._sweep_cartesian_count()} "
                "Cartesian combinations remain"
            )

    def _configure_ramp_rate_sweep(
        self,
        port_index: int,
        segment_index: int,
    ) -> None:
        if not 0 <= int(port_index) < len(self._pulse):
            return
        if segment_index <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "No incoming RAMP",
                "The first SET has no preceding RAMP to sweep.",
            )
            return
        pulse = self._pulse[int(port_index)]
        if segment_index >= len(pulse.flat_segments()):
            return
        point_index = int(segment_index) * 2
        current_duration_us = (
            float(pulse.t[point_index] - pulse.t[point_index - 1]) / 1000.0
        )
        voltage_delta_mv = float(
            pulse.v[point_index] - pulse.v[point_index - 2]
        )
        segment_name = (
            f"ramp_{int(segment_index) - 1}_to_{int(segment_index)}"
        )
        initial = next(
            (
                spec
                for spec in self._sweep_specs
                if isinstance(spec, QickRampRateSweepSpec)
                and spec.segment_name == segment_name
            ),
            None,
        )
        other_point_count = prod(
            spec.count
            for spec in self._sweep_specs
            if spec is not initial
        )
        dialog = RampRateSweepSettingsDialog(
            segment_name=segment_name,
            current_duration_us=current_duration_us,
            voltage_delta_mv=voltage_delta_mv,
            initial=initial,
            cartesian_base_count=other_point_count,
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        new_spec = dialog.value()
        ramp_specs = [
            spec
            for spec in self._sweep_specs
            if isinstance(spec, QickRampRateSweepSpec)
        ]
        for index, spec in enumerate(ramp_specs):
            if spec.segment_name == segment_name:
                ramp_specs[index] = new_spec
                break
        else:
            ramp_specs.append(new_spec)
        self._sweep_specs = ramp_specs + [
            spec
            for spec in self._sweep_specs
            if not isinstance(spec, QickRampRateSweepSpec)
        ]
        self._port_select(int(port_index))
        self._notify_sweep_state_changed(fit_view=True)
        self.statusBar().showMessage(
            f"RAMP-rate sweep applied to {segment_name}: "
            f"{new_spec.start:.9g} to {new_spec.stop:.9g} us, "
            f"{new_spec.count} axis points; "
            f"{self._sweep_cartesian_count()} Cartesian combinations"
        )

    def _remove_ramp_rate_sweep(
        self,
        _port_index: int,
        segment_index: int,
    ) -> None:
        segment_name = f"ramp_{int(segment_index) - 1}_to_{int(segment_index)}"
        remaining = [
            spec
            for spec in self._sweep_specs
            if not (
                isinstance(spec, QickRampRateSweepSpec)
                and spec.segment_name == segment_name
            )
        ]
        if len(remaining) == len(self._sweep_specs):
            return
        self._sweep_specs = remaining
        self._notify_sweep_state_changed(fit_view=True)
        self.statusBar().showMessage(
            f"RAMP-rate sweep removed; {self._sweep_cartesian_count()} "
            "Cartesian combinations remain"
        )

    def _configure_hold_duration_sweep(
        self,
        port_index: int,
        segment_index: int,
    ) -> None:
        if not 0 <= int(port_index) < len(self._pulse):
            return
        pulse = self._pulse[int(port_index)]
        if not 0 <= int(segment_index) < len(pulse.flat_segments()):
            return
        flat_index = int(segment_index) * 2
        current_duration_us = (
            float(pulse.t[flat_index + 1] - pulse.t[flat_index]) / 1000.0
        )
        segment_name = f"set_{int(segment_index)}"
        initial = next(
            (
                spec
                for spec in self._sweep_specs
                if isinstance(spec, QickHoldDurationSweepSpec)
                and spec.segment_name == segment_name
            ),
            None,
        )
        other_point_count = prod(
            spec.count
            for spec in self._sweep_specs
            if spec is not initial
        )
        dialog = HoldDurationSweepSettingsDialog(
            segment_name=segment_name,
            current_duration_us=current_duration_us,
            initial=initial,
            cartesian_base_count=other_point_count,
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        new_spec = dialog.value()
        ramp_specs = [
            spec
            for spec in self._sweep_specs
            if isinstance(spec, QickRampRateSweepSpec)
        ]
        hold_specs = [
            spec
            for spec in self._sweep_specs
            if isinstance(spec, QickHoldDurationSweepSpec)
        ]
        for index, spec in enumerate(hold_specs):
            if spec.segment_name == segment_name:
                hold_specs[index] = new_spec
                break
        else:
            hold_specs.append(new_spec)
        self._sweep_specs = ramp_specs + hold_specs + [
            spec
            for spec in self._sweep_specs
            if not isinstance(
                spec,
                (QickRampRateSweepSpec, QickHoldDurationSweepSpec),
            )
        ]
        self._port_select(int(port_index))
        self._notify_sweep_state_changed(fit_view=True)
        self.statusBar().showMessage(
            f"Hold-time sweep applied to {segment_name}: "
            f"{new_spec.start:.9g} to {new_spec.stop:.9g} us, "
            f"{new_spec.count} axis points; "
            f"{self._sweep_cartesian_count()} Cartesian combinations"
        )

    def _remove_hold_duration_sweep(
        self,
        _port_index: int,
        segment_index: int,
    ) -> None:
        segment_name = f"set_{int(segment_index)}"
        remaining = [
            spec
            for spec in self._sweep_specs
            if not (
                isinstance(spec, QickHoldDurationSweepSpec)
                and spec.segment_name == segment_name
            )
        ]
        if len(remaining) == len(self._sweep_specs):
            return
        self._sweep_specs = remaining
        self._notify_sweep_state_changed(fit_view=True)
        self.statusBar().showMessage(
            f"Hold-time sweep removed; {self._sweep_cartesian_count()} "
            "Cartesian combinations remain"
        )

    def _mirror_segment_structure_edit(
        self,
        source_port: int,
        operation: str,
        segment_index: int,
    ) -> None:
        """Apply one insert/delete to every AWG output except the source."""
        source_port = int(source_port)
        segment_index = int(segment_index)
        source = self._pulse[source_port]
        if operation == "insert":
            timing = self._segment_timing_ns(source, segment_index)
            if timing is None:
                raise RuntimeError("inserted segment timing is unavailable")
            ramp_ns, flat_ns = timing
            segment_name = source.segment_name(segment_index)
        elif operation == "delete":
            ramp_ns = flat_ns = None
            segment_name = None
        else:
            raise ValueError(
                f"unsupported segment structure operation {operation!r}"
            )

        for port_index, pulse in enumerate(self._pulse):
            if port_index == source_port:
                continue
            if operation == "insert":
                ok = pulse.insert_flat_ramp(
                    segment_index * 2,
                    ramp_ns=ramp_ns,
                    flat_ns=flat_ns,
                )
                if ok:
                    pulse.rename_segment(segment_index, segment_name)
            else:
                ok = pulse.delete_flat_ramp(segment_index * 2)
            if not ok:
                raise RuntimeError(
                    f"could not {operation} segment {segment_index + 1} "
                    f"on AWG output {port_index + 1}"
                )
            self._multi_ctrl._ctrl_pannels[
                port_index
            ]._remap_structure_markers(operation, segment_index)

    def _on_segment_structure_changed(
        self,
        port_index: int,
        operation: str,
        segment_index: int,
    ) -> None:
        self._mirror_segment_structure_edit(
            int(port_index),
            operation,
            int(segment_index),
        )
        self._share_matching_port_timings(int(port_index))
        self._share_matching_segment_names(int(port_index))
        previous_map_keys = (
            self._experiment_panel.selected_sweep_axis_keys()
            if hasattr(self, "_experiment_panel")
            else None
        )
        previous_parameter_key = (
            self._experiment_panel._selected_sweep_parameter_key()
            if hasattr(self, "_experiment_panel")
            else None
        )
        sweep_key_remap = {}
        updated_sweeps = []
        remapped_count = 0
        removed_count = 0
        for spec in self._sweep_specs:
            if isinstance(spec, QickSweepSpec):
                old_key = (
                    str(spec.output_name),
                    str(spec.segment_name),
                )
                remapped_name = _remap_set_segment_name(
                    spec.segment_name,
                    operation,
                    segment_index,
                )
                if remapped_name is None:
                    removed_count += 1
                    sweep_key_remap[old_key] = None
                    continue
                if remapped_name != spec.segment_name:
                    spec = replace(spec, segment_name=remapped_name)
                    remapped_count += 1
                updated_sweeps.append(spec)
                sweep_key_remap[old_key] = (
                    str(spec.output_name),
                    str(spec.segment_name),
                )
                continue

            if isinstance(spec, QickHoldDurationSweepSpec):
                old_key = (
                    str(spec.output_name),
                    str(spec.segment_name),
                )
                remapped_name = _remap_set_segment_name(
                    spec.segment_name,
                    operation,
                    segment_index,
                )
                if remapped_name is None:
                    removed_count += 1
                    sweep_key_remap[old_key] = None
                    continue
                if remapped_name != spec.segment_name:
                    spec = replace(spec, segment_name=remapped_name)
                    remapped_count += 1
                updated_sweeps.append(spec)
                sweep_key_remap[old_key] = (
                    str(spec.output_name),
                    str(spec.segment_name),
                )
                continue

            if isinstance(spec, QickRampRateSweepSpec):
                old_key = (
                    str(spec.output_name),
                    str(spec.segment_name),
                )
                remapped_name = _remap_ramp_segment_name(
                    spec.segment_name,
                    operation,
                    segment_index,
                )
                if remapped_name is None:
                    removed_count += 1
                    sweep_key_remap[old_key] = None
                    continue
                if remapped_name != spec.segment_name:
                    spec = replace(spec, segment_name=remapped_name)
                    remapped_count += 1
                updated_sweeps.append(spec)
                sweep_key_remap[old_key] = (
                    str(spec.output_name),
                    str(spec.segment_name),
                )
                continue

            updated_sweeps.append(spec)

        self._sweep_specs = updated_sweeps

        if hasattr(self, "_rf_ports_panel"):
            rf_key_remap = self._rf_ports_panel.remap_segment_references(
                self._pulse[0],
                operation,
                segment_index,
            )
            sweep_key_remap.update(rf_key_remap)
            self._rf_pulse_specs = list(self._rf_ports_panel.specs())
            remapped_count += sum(
                new_key is not None and new_key != old_key
                for old_key, new_key in rf_key_remap.items()
            )
            removed_count += sum(
                new_key is None for new_key in rf_key_remap.values()
            )

        acquisition_was_enabled = self._rf_readout_panel.isChecked()
        acquisition_old_name, acquisition_new_name = (
            self._rf_readout_panel.remap_segment_reference(
                self._pulse[0],
                operation,
                segment_index,
            )
        )
        if acquisition_new_name is None and acquisition_was_enabled:
            removed_count += 1
        elif acquisition_new_name != acquisition_old_name:
            remapped_count += 1

        selected_map_keys = None
        if previous_map_keys is not None:
            remapped_keys = []
            for key in previous_map_keys:
                remapped_key = sweep_key_remap.get(tuple(key), tuple(key))
                if remapped_key is None:
                    remapped_keys = []
                    break
                remapped_keys.append(remapped_key)
            available_keys = {
                (str(spec.output_name), str(spec.segment_name))
                for spec in self._active_map_sweep_specs()
            }
            if (
                len(remapped_keys) == 2
                and remapped_keys[0] != remapped_keys[1]
                and all(key in available_keys for key in remapped_keys)
            ):
                selected_map_keys = tuple(remapped_keys)

        selected_parameter_key = None
        if previous_parameter_key is not None:
            axis_kind, output_name, segment_name = previous_parameter_key
            remapped_target = sweep_key_remap.get(
                (output_name, segment_name),
                (output_name, segment_name),
            )
            if remapped_target is not None:
                selected_parameter_key = (
                    axis_kind,
                    str(remapped_target[0]),
                    str(remapped_target[1]),
                )

        self._notify_sweep_state_changed(
            fit_view=True,
            selected_map_keys=selected_map_keys,
            selected_parameter_key=selected_parameter_key,
            waveform_changed=True,
            trace_changed=True,
            rf_changed=True,
        )
        action = "inserted" if operation == "insert" else "deleted"
        self.statusBar().showMessage(
            f"Segment {segment_index + 1} {action} on all AWG outputs: "
            f"remapped {remapped_count} sweep target(s), removed "
            f"{removed_count} invalid sweep(s)"
        )
        self._refresh_stability_targets()

    def _build_toolbar(self):
        tb          = self.addToolBar("Tools")
        btn_fit     = QtWidgets.QAction("Fit", self)
        btn_fit.setToolTip("Show the full pulse (keyboard: F)")
        btn_fit.triggered.connect(self._fit_view)
        tb.addAction(btn_fit)
        # keyboard shortcut
        QtWidgets.QShortcut(QtGui.QKeySequence("F"), self, self._fit_view)

    def _fit_view(self):
        """Fit the view to the pulse data."""
        self._plot.fit_view()
        rf_ranges = []
        for spec in self._rf_pulse_specs:
            try:
                start_us, end_us, _ = rf_pulse_absolute_times_us(self._pulse[0], spec)
            except ValueError:
                continue
            rf_ranges.append((start_us * 1000.0, end_us * 1000.0))
        if rf_ranges:
            x_range = self._plot.getPlotItem().vb.viewRange()[0]
            x_min = min(float(x_range[0]), *(start for start, _end in rf_ranges))
            x_max = max(float(x_range[1]), *(end for _start, end in rf_ranges))
            margin = max(1.0, 0.03 * max(1.0, x_max - x_min))
            self._plot.setXRange(x_min - margin, x_max + margin, padding=0.0)
        if self._trace is not None:
            self._trace.fit_view()
        self._stability_panel.plot.fit_view()
        self._awg_sweep_plot.fit_view()
        self._sparameter_plot.fit_view()
        self._noise_plot.fit_view()

    def _show_rf_editor(self) -> None:
        """Compatibility helper: reveal the always-present RF Outputs tab."""
        self._control_tabs.setCurrentWidget(self._awg_tuning_page)
        self._awg_tuning_tabs.setCurrentWidget(self._rf_ports_panel)
        self._dock_ctrl.show()
        self._dock_ctrl.raise_()

    def _on_control_tab_changed(self, _index: int) -> None:
        current = self._control_tabs.currentWidget()
        if current is self._stability_panel:
            self._dock_stability.show()
            self._dock_stability.raise_()
        elif current is self._sparameter_panel:
            self._dock_sparameter.show()
            self._dock_sparameter.raise_()
        elif current is self._noise_panel:
            self._dock_noise.show()
            self._dock_noise.raise_()
        else:
            self._dock_plot.show()
            self._dock_plot.raise_()

    def _apply_rf_spec(self, spec: QickRfPulseSpec) -> None:
        self._rf_pulse_specs = [spec]
        self._rf_pulse_spec = spec
        self._refresh_rf_timeline(fit_view=True)
        self.statusBar().showMessage(
            f"RF pulse applied: {spec.frequency_mhz:.6g} MHz, gain {spec.gain}, "
            f"ATT1/ATT2 {spec.att1_db:.2f}/{spec.att2_db:.2f} dB, "
            f"{spec.filter_type} filter"
        )

    def _remove_rf_spec(self) -> None:
        self._rf_pulse_specs = []
        self._rf_pulse_spec = None
        self._refresh_rf_timeline()
        self.statusBar().showMessage("RF pulse disabled")

    def _on_rf_specs_changed(self, specs) -> None:
        self._rf_pulse_specs = list(specs)
        self._rf_pulse_spec = self._rf_pulse_specs[0] if self._rf_pulse_specs else None
        self._notify_sweep_state_changed()
        self._refresh_rf_timeline()

    def _active_map_sweep_specs(self) -> tuple:
        rf_specs = tuple(getattr(self, "_rf_pulse_specs", ()))
        if (
            self._experiment_panel.execution_backend()
            == EXECUTION_BACKEND_QCS
        ):
            rf_specs = tuple(
                replace(
                    spec,
                    power_sweep_enabled=False,
                    power_calibration_enabled=False,
                )
                for spec in rf_specs
            )
        return tuple(self._sweep_specs) + tuple(
            axis
            for spec in rf_specs
            for axis in spec.sweep_axes
        )

    def _update_sweep_parameter(
        self,
        spec,
        start: float,
        stop: float,
        count: int,
    ) -> None:
        key = ExperimentPanel._sweep_parameter_key(spec)
        axis_kind = key[0]
        try:
            if axis_kind in {
                "rf_duration",
                "rf_frequency",
                "rf_power",
            }:
                panel = next(
                    (
                        candidate
                        for candidate in self._rf_ports_panel._panels
                        if candidate.isChecked()
                        and candidate.gen_ch.value() == int(spec.gen_ch)
                        and str(candidate.segment.currentData())
                        == str(spec.segment_name)
                    ),
                    None,
                )
                if panel is None:
                    raise ValueError(
                        "the selected RF sweep no longer has an "
                        "enabled RF Output editor"
                    )
                with QtCore.QSignalBlocker(panel):
                    if axis_kind == "rf_duration":
                        panel.duration_sweep_enabled.setChecked(True)
                        panel.duration_sweep_start.setValue(
                            _time_from_ns(
                                float(start) * 1000.0,
                                panel._time_unit,
                            )
                        )
                        panel.duration_sweep_stop.setValue(
                            _time_from_ns(
                                float(stop) * 1000.0,
                                panel._time_unit,
                            )
                        )
                        panel.duration_sweep_count.setValue(int(count))
                        panel._update_duration_sweep_controls()
                    elif axis_kind == "rf_frequency":
                        panel.frequency_sweep_enabled.setChecked(True)
                        panel.frequency_sweep_start_mhz.setValue(float(start))
                        panel.frequency_sweep_stop_mhz.setValue(float(stop))
                        panel.frequency_sweep_count.setValue(int(count))
                        panel._update_frequency_sweep_controls()
                    else:
                        panel.power_calibration_group.setChecked(True)
                        panel.power_sweep_enabled.setChecked(True)
                        panel.power_sweep_start_dbm.setValue(float(start))
                        panel.power_sweep_stop_dbm.setValue(float(stop))
                        panel.power_sweep_count.setValue(int(count))
                        panel._update_power_calibration_controls()
                        panel._update_power_sweep_controls()
                panel.changed.emit()
                self.statusBar().showMessage(
                    f"{axis_kind.replace('_', ' ')} updated: "
                    f"{start:.9g} to {stop:.9g} "
                    f"{spec.coordinate_unit}, {count} points"
                )
                return

            replacement = None
            if axis_kind == "ramp_duration":
                replacement = replace(
                    spec,
                    start=float(start),
                    stop=float(stop),
                    count=int(count),
                )
            elif axis_kind == "hold_duration":
                replacement = replace(
                    spec,
                    start=float(start),
                    stop=float(stop),
                    count=int(count),
                )
            elif axis_kind == "amplitude":
                full_scale_mv = self._active_awg_full_scale_mv()
                replacement = replace(
                    spec,
                    start=float(start) / full_scale_mv,
                    stop=float(stop) / full_scale_mv,
                    count=int(count),
                )
            else:
                raise ValueError(f"unsupported sweep type {axis_kind!r}")

            replacement_index = next(
                (
                    index
                    for index, current in enumerate(self._sweep_specs)
                    if ExperimentPanel._sweep_parameter_key(current) == key
                ),
                None,
            )
            if replacement_index is None:
                raise ValueError(
                    "the selected sweep no longer exists in the AWG settings"
                )
            self._sweep_specs[replacement_index] = replacement
            self._notify_sweep_state_changed(fit_view=True)
            _kind, target, display_start, display_stop, unit = (
                self._experiment_panel._sweep_parameter_values(replacement)
            )
            self.statusBar().showMessage(
                f"Sweep updated for {target}: {display_start:.9g} to "
                f"{display_stop:.9g} {unit}, {count} points"
            )
        except (TypeError, ValueError) as exc:
            self._experiment_panel.set_sweep_specs(
                self._active_map_sweep_specs()
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid sweep parameters",
                str(exc),
            )

    def _remove_sweep_parameter(self, spec) -> None:
        key = ExperimentPanel._sweep_parameter_key(spec)
        if key[0] in {"rf_duration", "rf_frequency", "rf_power"}:
            panel = next(
                (
                    candidate
                    for candidate in self._rf_ports_panel._panels
                    if candidate.isChecked()
                    and candidate.gen_ch.value() == int(spec.gen_ch)
                    and str(candidate.segment.currentData())
                    == str(spec.segment_name)
                ),
                None,
            )
            if panel is None:
                self._experiment_panel.set_sweep_specs(
                    self._active_map_sweep_specs()
                )
                return
            with QtCore.QSignalBlocker(panel):
                if key[0] == "rf_duration":
                    panel.duration_sweep_enabled.setChecked(False)
                    panel._update_duration_sweep_controls()
                elif key[0] == "rf_frequency":
                    panel.frequency_sweep_enabled.setChecked(False)
                    panel._update_frequency_sweep_controls()
                else:
                    panel.power_sweep_enabled.setChecked(False)
                    panel._update_power_sweep_controls()
            panel.changed.emit()
            self.statusBar().showMessage(
                f"{key[0].replace('_', ' ')} removed from RF gen "
                f"{spec.gen_ch} / "
                f"{spec.segment_name}"
            )
            return

        remaining = [
            current
            for current in self._sweep_specs
            if ExperimentPanel._sweep_parameter_key(current) != key
        ]
        if len(remaining) == len(self._sweep_specs):
            self._experiment_panel.set_sweep_specs(
                self._active_map_sweep_specs()
            )
            return
        self._sweep_specs = remaining
        self._notify_sweep_state_changed(fit_view=True)
        self.statusBar().showMessage(
            f"Sweep removed; {self._sweep_cartesian_count()} Cartesian "
            "combinations remain"
        )

    def _on_readout_spec_changed(self, spec) -> None:
        self._ddr_readout_spec = spec
        self._experiment_panel.set_ddr_readout_spec(spec)
        is_qcs = (
            self._experiment_panel.execution_backend()
            == EXECUTION_BACKEND_QCS
        )
        if spec is None:
            self.statusBar().showMessage(
                "QCS acquisition disabled"
                if is_qcs
                else "RF readout disabled"
            )
        elif is_qcs:
            result_detail = (
                "one I/Q value/shot from "
                f"{spec.samples_per_trigger:,} total averaging samples"
                if self._experiment_panel.qcs_hw_demod.isChecked()
                else f"{spec.samples_per_trigger:,} trace samples/shot"
            )
            self.statusBar().showMessage(
                f"QCS acquisition: {result_detail}, "
                f"{spec.segment_name} + {spec.delay_us:g} us pre-delay"
            )
        elif spec.effective_measurement_representation == "current":
            self.statusBar().showMessage(
                f"DC current readout {spec.ro_ch}: "
                f"{spec.samples_per_trigger} stored FIR samples, "
                f"gain {spec.dc_measure_gain_v_per_a:g} V/A"
            )
        elif spec.effective_measurement_representation == "voltage":
            run_label = (
                "latest matching"
                if spec.dc_voltage_calibration_run_id == 0
                else f"Run {spec.dc_voltage_calibration_run_id}"
            )
            calibration_label = (
                run_label
                if spec.dc_voltage_calibration_enabled
                else "identity ADC-to-voltage"
            )
            self.statusBar().showMessage(
                f"DC voltage readout {spec.ro_ch}: "
                f"{spec.samples_per_trigger} stored FIR samples, "
                f"{calibration_label}"
            )
        else:
            self.statusBar().showMessage(
                f"RF readout {spec.ro_ch}: "
                f"{spec.samples_per_trigger} stored FIR samples"
            )
        self._refresh_qcs_waveform_capacity()

    def _on_bias_t_changed(
        self,
        enabled: bool,
        compensation_type: str,
        compensation_mv: float,
        mode: str,
        duration_us: float,
        filter_tau_us: float,
    ) -> None:
        self._bias_t_compensation_enabled = bool(enabled)
        self._bias_t_compensation_type = str(compensation_type)
        self._bias_t_compensation_voltage_mv = float(compensation_mv)
        self._bias_t_compensation_mode = str(mode)
        self._bias_t_compensation_duration_us = float(duration_us)
        self._bias_t_filter_tau_us = float(filter_tau_us)
        try:
            self._refresh_physical_waveforms(fit_view=False)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            self.statusBar().showMessage(f"Bias-T preview unavailable: {exc}")
            self._refresh_qcs_waveform_capacity()
            return
        self._refresh_qcs_waveform_capacity()
        if enabled:
            if compensation_type == "filter":
                self.statusBar().showMessage(
                    "Bias-T filter compensation enabled; flat-segment slew is "
                    f"target/tau with tau={filter_tau_us:.6g} us"
                )
            elif mode == "fixed_time":
                self.statusBar().showMessage(
                    "Bias-T compensation enabled for "
                    f"{duration_us:.6g} us; voltage follows pulse area"
                )
            else:
                self.statusBar().showMessage(
                    f"Bias-T compensation enabled at {compensation_mv:.6g} mV"
                )
        else:
            self.statusBar().showMessage("Bias-T compensation disabled")

    def _experiment_run_arguments(
        self,
        *,
        require_readout: bool = True,
        require_run_config: bool = True,
        validate_qick_hardware: bool = True,
        validate_qcs_hardware: bool = True,
        for_qcs: bool = False,
    ) -> dict:
        values = self._experiment_panel.values(
            len(self._pulse),
            require_run_config=require_run_config,
            validate_qcs_hardware=validate_qcs_hardware,
        )
        self._qick_fabric_mhz = values["fabric_mhz"]
        self._qick_tproc_mhz = values["tproc_mhz"]
        self._qick_full_scale_mv = values["full_scale_mv"]
        self._qick_awg_channels = values["awg_channels"]
        self._multi_ctrl.set_awg_channels(self._qick_awg_channels)
        self._qick_repetitions_per_sweep = values["repetitions_per_sweep"]
        self._bias_t_compensation_enabled = values[
            "bias_t_compensation_enabled"
        ]
        self._bias_t_compensation_type = values["bias_t_compensation_type"]
        self._bias_t_compensation_voltage_mv = values[
            "bias_t_compensation_voltage_mv"
        ]
        self._bias_t_compensation_mode = values["bias_t_compensation_mode"]
        self._bias_t_compensation_duration_us = values[
            "bias_t_compensation_duration_us"
        ]
        self._bias_t_filter_tau_us = values["bias_t_filter_tau_us"]
        source_full_scale_mv = self._active_awg_full_scale_mv(
            values["execution_backend"]
        )
        if validate_qick_hardware:
            for panel in self._rf_ports_panel._panels:
                panel.validate_power_calibration()
        rf_specs = self._rf_ports_panel.specs(for_qcs=for_qcs)
        readout_spec = self._rf_readout_panel.spec()
        if require_readout and readout_spec is None:
            raise ValueError(
                (
                    "enable QCS Acquisition before running so acquisition "
                    "results can be saved"
                    if (
                        self._experiment_panel.execution_backend()
                        == EXECUTION_BACKEND_QCS
                    )
                    else (
                        "enable RF Readout before running so the FIR IQ trace "
                        "can be saved"
                    )
                )
            )
        if validate_qick_hardware:
            rf_channels = {spec.gen_ch for spec in rf_specs}
            overlap = rf_channels.intersection(self._qick_awg_channels)
            if overlap:
                raise ValueError(
                    "RF and AWG generator indices overlap: "
                    + ", ".join(str(channel) for channel in sorted(overlap))
                )

        sweep = self._sweep_specs[0] if len(self._sweep_specs) == 1 else None
        sweeps = tuple(self._sweep_specs) if len(self._sweep_specs) > 1 else None
        sequence = build_qick_sequence(
            tuple(pulse.copy() for pulse in self._pulse),
            output_names=self._qick_output_names(),
            fabric_mhz=self._qick_fabric_mhz,
            full_scale_mv=source_full_scale_mv,
            sweep=sweep,
            sweeps=sweeps,
            rf_pulse_specs=rf_specs,
            cross_capacitance=self._cross_capacitance.copy(),
            bias_t_compensation_enabled=self._bias_t_compensation_enabled,
            bias_t_compensation_type=self._bias_t_compensation_type,
            bias_t_compensation_voltage_mv=self._bias_t_compensation_voltage_mv,
            bias_t_compensation_mode=self._bias_t_compensation_mode,
            bias_t_compensation_duration_us=self._bias_t_compensation_duration_us,
            bias_t_filter_tau_us=self._bias_t_filter_tau_us,
        )
        gui_settings = self._settings_to_dict() if require_run_config else None
        return {
            "connection_config": values["connection"],
            "run_config": values["run"],
            "sequence": sequence,
            "source_full_scale_mv": source_full_scale_mv,
            "awg_channels": self._qick_awg_channels,
            "repetitions_per_sweep": self._qick_repetitions_per_sweep,
            "iq_repetition_policy": (
                self._experiment_panel.iq_repetition_policy_value(
                    effective=True
                )
            ),
            "compile_validation_mode": values["compile_validation_mode"],
            "rf_specs": rf_specs,
            "readout_spec": readout_spec,
            "gui_settings": gui_settings,
            "progress": False,
        }

    def _qcs_experiment_run_arguments(self) -> dict:
        qick_arguments = self._experiment_run_arguments(
            require_readout=False,
            validate_qick_hardware=False,
            validate_qcs_hardware=False,
            for_qcs=True,
        )
        sweep_preview = qcs_sweep_execution_preview(
            qick_arguments["sequence"],
            hardware_demodulation=(
                self._experiment_panel.qcs_hw_demod.isChecked()
            ),
            source_full_scale_mv=qick_arguments["source_full_scale_mv"],
            dc_full_scale_v=(
                self._experiment_panel.qcs_dc_full_scale_v.value()
            ),
            fabric_mhz=self._qick_fabric_mhz,
            init_time_s=(
                self._experiment_panel.qcs_init_time_us.value() * 1.0e-6
            ),
        )
        self._experiment_panel.set_qcs_sweep_execution_status(sweep_preview)
        sequence = qick_arguments["sequence"]
        # Run-button preflight executes on the Qt thread.  Keep large-grid
        # capacity feedback bounded here; QCS compilation still validates the
        # complete sweep before submitting anything to hardware.
        if sweep_preview.mode == "hybrid":
            # The exact mixed partition can use a different legal Hold graph
            # in each outer coordinate. The worker validates every coordinate
            # before executing any of them.
            self._experiment_panel.set_qcs_mixed_waveform_capacity_pending()
        else:
            capacity_point_indices = (
                None
                if int(sequence.sweep_point_count) <= 256
                else qcs_m5301_capacity_preview_point_indices(sequence)
            )
            capacity_report = validate_qcs_m5301_waveform_capacity(
                sequence,
                point_indices=capacity_point_indices,
                fabric_mhz=self._qick_fabric_mhz,
                amplitude_scale=(
                    qick_arguments["source_full_scale_mv"]
                    / (
                        self._experiment_panel.qcs_dc_full_scale_v.value()
                        * 1000.0
                    )
                ),
                source_full_scale_mv=qick_arguments["source_full_scale_mv"],
                dc_full_scale_v=(
                    self._experiment_panel.qcs_dc_full_scale_v.value()
                ),
            )
            physical_names = tuple(
                name.strip()
                for name in self._experiment_panel.qcs_dc_channel_names.text().split(",")
                if name.strip()
            )
            self._experiment_panel.set_qcs_waveform_capacity(
                capacity_report,
                physical_names,
            )
        if qick_arguments["readout_spec"] is None:
            raise ValueError(
                "enable QCS Acquisition before running so acquisition "
                "results can be saved"
            )
        connection = self._experiment_panel.qcs_connection_values(
            len(self._pulse)
        )
        mapper_path = Path(connection.mapper_path).expanduser()
        if not mapper_path.is_file():
            raise ValueError(
                f"QCS ChannelMapper file not found: {mapper_path}"
            )
        rf_specs = tuple(qick_arguments["rf_specs"])
        missing_rf_channels = sorted(
            {
                int(spec.gen_ch)
                for spec in rf_specs
                if int(spec.gen_ch) not in connection.rf_channel_names
            }
        )
        if missing_rf_channels:
            raise ValueError(
                "QCS RF channel map is missing generator "
                + ", ".join(str(channel) for channel in missing_rf_channels)
            )
        active_rf_panels = tuple(
            panel
            for panel in self._rf_ports_panel._panels
            if panel.isChecked()
        )
        if len(active_rf_panels) != len(rf_specs):
            raise RuntimeError(
                "QCS RF output editor state changed while preparing the run"
            )
        rf_pulses = tuple(
            QcsRfPulseConfig(
                gen_ch=int(spec.gen_ch),
                at_segment=str(spec.segment_name),
                duration_s=float(spec.duration_us) * 1.0e-6,
                amplitude=float(panel.qcs_amplitude.value()),
                frequency_hz=float(spec.frequency_mhz) * 1.0e6,
                phase_rad=float(np.deg2rad(spec.phase_degrees)),
                delay_s=float(spec.delay_us) * 1.0e-6,
                envelope="constant",
                require_within_segment=bool(spec.require_within_segment),
                power_calibration=panel.qcs_power_calibration_config(),
            )
            for panel, spec in zip(active_rf_panels, rf_specs)
        )

        readout_spec = qick_arguments["readout_spec"]
        if readout_spec is None:
            acquisition = None
        else:
            # QICK-only readout values remain serialized so switching back to
            # QICK is lossless. They are intentionally dormant here: QCS
            # acquisition only translates the shared segment, pre-delay,
            # integration/trace length, and optional IntegrationFilter
            # frequency controls.
            if connection.acquisition_channel_name is None:
                raise ValueError(
                    "set a QCS acquisition virtual channel before running "
                    "with RF Readout enabled"
                )
            acquisition = QcsAcquisitionConfig(
                at_segment=str(readout_spec.segment_name),
                duration_s=(
                    int(readout_spec.samples_per_trigger)
                    / self._experiment_panel.qcs_sample_rate_hz.value()
                ),
                pre_delay_s=float(readout_spec.delay_us) * 1.0e-6,
                sample_rate_hz=(
                    self._experiment_panel.qcs_sample_rate_hz.value()
                ),
                sample_count=int(readout_spec.samples_per_trigger),
                integration_filter=None,
                frequency_hz=(
                    (
                        float(readout_spec.readout_frequency_mhz) * 1.0e6
                    )
                    if connection.hw_demod
                    else 0.0
                ),
                phase_rad=0.0,
                envelope="constant",
            )

        return {
            "connection_config": connection,
            "run_config": qick_arguments["run_config"],
            "sequence": qick_arguments["sequence"],
            "repetitions_per_sweep": qick_arguments[
                "repetitions_per_sweep"
            ],
            "iq_repetition_policy": qick_arguments[
                "iq_repetition_policy"
            ],
            "fabric_mhz": self._qick_fabric_mhz,
            "source_full_scale_mv": qick_arguments["source_full_scale_mv"],
            "rf_pulses": rf_pulses,
            "acquisition": acquisition,
            "gui_settings": qick_arguments["gui_settings"],
        }

    def _qcs_stability_connection_values(
        self,
    ) -> tuple[QcsConnectionConfig, Mapping[str, object]]:
        """Build the mapper owned by Stability from the shared inventory.

        The graphical chassis inventory is shared by all tabs.  Native
        ChannelMapper files are not: validating the Experiment/AWG mapper
        here made a complete Stability selection fail whenever another tab
        had an unrelated draft.  A content-addressed Stability mapper keeps
        the physical endpoint selections isolated while retaining the same
        installed-module view.
        """

        if self._qcs_mapper_commit_pending():
            raise ValueError(
                "The selected QCS SMA assignment is still being applied. "
                "Wait for the background front-panel update to finish, then "
                "run Stability again."
            )
        self._experiment_panel._commit_qcs_inter_iteration_delay()
        mapper_configuration = (
            self._stability_panel.qcs_mapper_configuration()
        )
        dc_names, rf_names, acquisition_name = qcs_role_bindings(
            mapper_configuration,
            required_dc_count=len(self._pulse),
        )
        mapper_path = qcs_workflow_mapper_output_path(
            mapper_configuration,
            "stability",
        )
        mapper_digest = (
            qcs_mapper_file_sha256(mapper_path)
            if mapper_path.is_file()
            else None
        )
        connection = QcsConnectionConfig(
            mapper_path=str(mapper_path),
            mapper_sha256=mapper_digest,
            dc_channel_names=tuple(dc_names),
            dc_full_scale_v=float(
                self._experiment_panel.qcs_dc_full_scale_v.value()
            ),
            rf_channel_names={
                int(index): str(name)
                for index, name in rf_names.items()
            },
            acquisition_channel_name=acquisition_name,
            hw_demod=True,
            init_time_s=(
                float(self._experiment_panel.qcs_init_time_us.value())
                * 1.0e-6
            ),
            blocking=bool(self._experiment_panel._qcs_blocking),
        )
        return connection, mapper_configuration

    def _stability_run_arguments(self, *, save: bool) -> dict:
        database_path = (
            self._stability_panel.database_path_value() if save else None
        )
        backend = self._experiment_panel.execution_backend()
        if backend == EXECUTION_BACKEND_QCS:
            # Synchronize the Stability selectors before deriving its scoped
            # mapper.  Otherwise an output added or removed immediately before
            # Run could leave qcs_mapper_configuration() with a stale count.
            self._stability_panel.refresh_targets(
                self._qick_output_names(),
                tuple(range(len(self._pulse))),
                display_names=self._awg_output_display_names,
            )
            connection, mapper_configuration = (
                self._qcs_stability_connection_values()
            )
            run_config = self._experiment_panel.run_config_values(
                require_run_config=save,
                database_path=database_path,
            )
            qcs_full_scale_mv = float(connection.dc_full_scale_v) * 1000.0
            stability_config = self._stability_panel.config(
                full_scale_mv=qcs_full_scale_mv
            )
            return self._qcs_stability_run_arguments(
                connection=connection,
                run_config=run_config,
                stability_config=stability_config,
                save=save,
                mapper_configuration=mapper_configuration,
            )

        values = self._experiment_panel.values(
            len(self._pulse),
            require_run_config=save,
            database_path=database_path,
        )
        self._qick_fabric_mhz = values["fabric_mhz"]
        self._qick_tproc_mhz = values["tproc_mhz"]
        self._qick_full_scale_mv = values["full_scale_mv"]
        self._qick_awg_channels = values["awg_channels"]
        self._multi_ctrl.set_awg_channels(self._qick_awg_channels)
        self._refresh_stability_targets()
        stability_config = self._stability_panel.config(
            full_scale_mv=self._qick_full_scale_mv
        )
        path = dict(self._stability_panel.front_panel_values())
        anchor_segment = stability_config.x_axis.segment_name
        settle_time_us = float(stability_config.settle_time_us)
        identified_rate_hz = getattr(
            self._qick_configuration,
            "fir_sample_rate_hz",
            None,
        )
        identified_sample_period_us = (
            1.0
            if identified_rate_hz is None
            else 1_000_000.0 / float(identified_rate_hz)
        )
        modulation_duration_us = max(
            identified_sample_period_us,
            float(stability_config.trace_samples_per_point)
            * identified_sample_period_us,
        )
        rf_specs = (
            QickRfPulseSpec(
                gen_ch=int(path["output_ch"]),
                segment_name=anchor_segment,
                delay_us=settle_time_us,
                duration_us=modulation_duration_us,
                frequency_mhz=stability_config.modulation_frequency_mhz,
                gain=stability_config.modulation_gain,
                att1_db=float(path["output_att1_db"]),
                att2_db=float(path["output_att2_db"]),
                phase_degrees=0.0,
                nqz=int(path["output_nqz"]),
                require_within_segment=False,
                filter_type=str(path["output_filter_type"]),
                filter_cutoff=float(path["output_filter_cutoff_ghz"]),
                filter_bandwidth=float(path["output_filter_bandwidth_ghz"]),
                output_board_type=str(path["output_board_type"]),
            ),
        )
        readout_spec = QickDdrReadoutSpec(
            ro_ch=int(path["readout_ch"]),
            segment_name=anchor_segment,
            delay_us=settle_time_us,
            samples_per_trigger=stability_config.trace_samples_per_point,
            readout_frequency_mhz=stability_config.modulation_frequency_mhz,
            margin_input_samples=1024,
            force_overwrite=True,
            post_run_read_delay_seconds=0.1,
            attenuation_db=float(path["readout_attenuation_db"]),
            filter_type=str(path["readout_filter_type"]),
            filter_cutoff=float(path["readout_filter_cutoff_ghz"]),
            filter_bandwidth=float(path["readout_filter_bandwidth_ghz"]),
            input_board_type=str(path["input_board_type"]),
            dc_gain_db=float(path["readout_dc_gain_db"]),
            dc_measure_mode=self._stability_panel.dc_measure_mode.isChecked(),
            dc_measure_gain_v_per_a=(
                self._stability_panel.dc_measure_gain_v_per_a.value()
            ),
            measurement_representation=str(
                self._stability_panel.measurement_unit.currentData()
            ),
            dc_voltage_calibration_enabled=(
                self._stability_panel.dc_calibration_group.isChecked()
            ),
            dc_voltage_calibration_database_path=(
                self._stability_panel.dc_calibration_path.text().strip()
            ),
            dc_voltage_calibration_run_id=(
                self._stability_panel.dc_calibration_run_id.value()
            ),
            nqz=int(path["readout_nqz"]),
            fpga_trigger_delay_us=(
                stability_config.fpga_trigger_delay_us
            ),
        )
        overlap = {spec.gen_ch for spec in rf_specs}.intersection(
            self._qick_awg_channels
        )
        if overlap:
            raise ValueError(
                "RF and AWG generator indices overlap: "
                + ", ".join(str(channel) for channel in sorted(overlap))
            )

        sequence = build_stability_hold_sequence(
            stability_config,
            output_names=self._qick_output_names(),
            fabric_mhz=self._qick_fabric_mhz,
            full_scale_mv=self._qick_full_scale_mv,
            cross_capacitance=self._cross_capacitance.copy(),
            sample_period_us=identified_sample_period_us,
        )
        return {
            "connection_config": values["connection"],
            "run_config": values["run"],
            "gui_settings": self._settings_to_dict() if save else None,
            "stability_config": stability_config,
            "stability_fabric_mhz": self._qick_fabric_mhz,
            "full_scale_mv": self._qick_full_scale_mv,
            "sequence": sequence,
            "awg_channels": self._qick_awg_channels,
            "repetitions_per_sweep": stability_config.repetitions_per_point,
            "tproc_mhz": self._qick_tproc_mhz,
            "rf_specs": rf_specs,
            "readout_spec": readout_spec,
            "progress": False,
        }

    def _qcs_stability_run_arguments(
        self,
        *,
        connection: QcsConnectionConfig,
        run_config: Optional[QcodesRunConfig],
        stability_config,
        save: bool,
        mapper_configuration: Mapping[str, object],
    ) -> dict:
        """Translate Stability controls into one native QCS X/Y sweep."""
        # Acquisition mode belongs to the measurement, not to the shared QCS
        # hardware connection.  AWG Tuning may independently request a raw
        # trace, while Stability always returns hardware-demodulated I/Q.
        connection = replace(connection, hw_demod=True)
        if not connection.blocking:
            raise ValueError(
                "QCS Stability hardware sweep requires blocking execution"
            )
        if stability_config.bias_t_compensation_enabled and (
            stability_config.bias_t_compensation_type != "dc"
            or stability_config.bias_t_compensation_mode != "fixed_time"
        ):
            raise ValueError(
                "QCS Stability supports only fixed-time DC Bias-T "
                "compensation"
            )

        rf_gen_ch = (
            self._stability_panel.path_diagram
            .qcs_output_mapping_selector.currentData()
        )
        if rf_gen_ch is None:
            raise ValueError(
                "Select a mapped QCS RF output module for Stability"
            )
        rf_gen_ch = int(rf_gen_ch)
        if rf_gen_ch not in connection.rf_channel_names:
            raise ValueError(
                f"QCS RF channel map has no logical output {rf_gen_ch}"
            )
        if connection.acquisition_channel_name is None:
            raise ValueError(
                "Select a mapped M5200 acquisition channel for Stability"
            )

        sample_rate_hz = float(
            self._experiment_panel.qcs_sample_rate_hz.value()
        )
        qcs_full_scale_mv = float(connection.dc_full_scale_v) * 1000.0
        qcs_sequence_fabric_mhz = 300.0
        acquisition_duration_s = (
            int(stability_config.trace_samples_per_point) / sample_rate_hz
        )
        stability_timing_block_samples = int(
            2 * QCS_M5200_INTEGRATION_BLOCK_SAMPLES
        )
        averaging_plan = plan_qcs_total_iq_averaging(
            acquisition_duration_s,
            sample_rate_hz=sample_rate_hz,
            block_samples=stability_timing_block_samples,
            max_pass_samples=QCS_SPARAMETER_MAX_INTEGRATION_SAMPLES,
        )
        pass_sample_count = int(averaging_plan.pass_sample_counts[0])
        pass_acquisition_duration_s = pass_sample_count / sample_rate_hz
        integration_segment_sample_counts = (
            qcs_stability_integration_segment_sample_counts(
                pass_sample_count
            )
        )
        integration_dead_time_s = (
            max(0, len(integration_segment_sample_counts) - 1)
            * QCS_STABILITY_INTER_SEGMENT_DELAY_S
        )
        settle_s = float(stability_config.settle_time_us) * 1.0e-6
        # The M5301 needs a minimum direct ramp before Hold can retain a swept
        # Scalar.  Put that ramp before the user-requested settle interval, and
        # reserve a falling ramp plus explicit zero after the full-level guard.
        # Thus "settle" continues to mean time spent at the requested voltage.
        readout_delay_s = settle_s + QCS_STABILITY_DC_RAMP_S
        # The public configuration records the requested total averaging
        # time.  One physical Stability program must cover only one bounded
        # pass; the backend reuses that program and combines pass results.
        # Building the DC hold from the total sample count would incorrectly
        # stretch every pass to as much as 100 ms.
        pass_stability_config = replace(
            stability_config,
            trace_samples_per_point=pass_sample_count,
        )
        sequence = build_stability_hold_sequence(
            pass_stability_config,
            output_names=self._qick_output_names(),
            fabric_mhz=qcs_sequence_fabric_mhz,
            full_scale_mv=qcs_full_scale_mv,
            cross_capacitance=self._cross_capacitance.copy(),
            sample_period_us=1.0e6 / sample_rate_hz,
            target_edge_padding_us=(
                (
                    QCS_STABILITY_DC_EDGE_PADDING_S
                    + integration_dead_time_s
                )
                * 1.0e6
            ),
        )
        power_calibration_values = (
            self._stability_panel.qcs_power_calibration_values()
        )
        power_calibration = (
            None
            if power_calibration_values is None
            else QcsRfPowerCalibrationConfig(**power_calibration_values)
        )
        rf_pulses = (
            QcsRfPulseConfig(
                gen_ch=rf_gen_ch,
                at_segment=stability_config.x_axis.segment_name,
                duration_s=pass_acquisition_duration_s,
                amplitude=float(
                    self._stability_panel
                    .qcs_modulation_amplitude.value()
                ),
                frequency_hz=(
                    float(stability_config.modulation_frequency_mhz)
                    * 1.0e6
                ),
                phase_rad=0.0,
                delay_s=readout_delay_s,
                envelope="constant",
                require_within_segment=False,
                power_calibration=power_calibration,
            ),
        )
        acquisition = QcsAcquisitionConfig(
            at_segment=stability_config.x_axis.segment_name,
            duration_s=acquisition_duration_s,
            pre_delay_s=readout_delay_s,
            sample_rate_hz=sample_rate_hz,
            sample_count=int(
                stability_config.trace_samples_per_point
            ),
            integration_filter=None,
            frequency_hz=(
                float(stability_config.modulation_frequency_mhz)
                * 1.0e6
            ),
            phase_rad=0.0,
            envelope="constant",
        )
        gui_settings = None
        if save:
            stability_mapper_saved = connection.mapper_sha256 is not None
            qcs_settings_override = {
                "mapper_path": connection.mapper_path,
                "dc_channel_names": list(connection.dc_channel_names),
                "dc_full_scale_v": float(connection.dc_full_scale_v),
                "rf_channel_names": {
                    str(index): name
                    for index, name in connection.rf_channel_names.items()
                },
                "acquisition_channel_name": (
                    connection.acquisition_channel_name
                ),
                "hw_demod": True,
                "sample_rate_hz": sample_rate_hz,
                "init_time_s": float(connection.init_time_s),
                "blocking": bool(connection.blocking),
                "hardware_configuration": dict(mapper_configuration),
                "hardware_configuration_state": (
                    QCS_HARDWARE_STATE_SAVED
                    if stability_mapper_saved
                    else QCS_HARDWARE_STATE_DRAFT
                ),
                "hardware_mapper_sha256": connection.mapper_sha256,
            }
            gui_settings = self._settings_to_dict(
                qcs_settings_override=qcs_settings_override,
            )
        return {
            "connection_config": connection,
            "run_config": run_config,
            "gui_settings": gui_settings,
            "stability_config": stability_config,
            "full_scale_mv": qcs_full_scale_mv,
            "sequence": sequence,
            "repetitions_per_point": (
                stability_config.repetitions_per_point
            ),
            "fabric_mhz": qcs_sequence_fabric_mhz,
            "rf_pulses": rf_pulses,
            "acquisition": acquisition,
            "readout_spec": None,
            "mapper_configuration": mapper_configuration,
        }

    def _sparameter_run_arguments(self) -> dict:
        if (
            self._experiment_panel.execution_backend()
            == EXECUTION_BACKEND_QCS
        ):
            connection = self._experiment_panel.qcs_connection_values(
                len(self._pulse)
            )
            # RF S-parameter is intrinsically a hardware-demodulated I/Q
            # measurement.  Do not inherit AWG Tuning's independent choice
            # between raw trace and Single I/Q acquisition.
            connection = replace(connection, hw_demod=True)
            config = self._sparameter_panel.config()
            rf_gen_ch = (
                self._sparameter_panel.path_diagram
                .qcs_output_mapping_selector.currentData()
            )
            if rf_gen_ch is None:
                raise ValueError(
                    "Select a mapped M5300 RF output for S-parameter."
                )
            rf_gen_ch = int(rf_gen_ch)
            if rf_gen_ch not in connection.rf_channel_names:
                raise ValueError(
                    f"QCS RF channel map has no logical output {rf_gen_ch}."
                )
            if connection.acquisition_channel_name is None:
                raise ValueError(
                    "Select a mapped M5200 acquisition input for S-parameter."
                )
            return {
                "connection_config": connection,
                "run_config": self._experiment_panel.run_config_values(
                    require_run_config=True,
                    database_path=(
                        self._sparameter_panel.database_path_value()
                    ),
                ),
                "sweep_config": config,
                "rf_gen_ch": rf_gen_ch,
                "rf_amplitude": float(
                    self._sparameter_panel.qcs_amplitude.value()
                ),
                "repetitions_per_point": int(
                    self._experiment_panel.repetitions.value()
                ),
            }
        connection, run = self._experiment_panel.connection_values(
            database_path=self._sparameter_panel.database_path_value()
        )
        return {
            "connection_config": connection,
            "run_config": run,
            "sweep_config": self._sparameter_panel.config(),
            "tproc_mhz": self._experiment_panel.tproc_mhz.value(),
        }

    def _show_qick_front_panel(self, scope: str, target=None) -> None:
        """Open the live front-panel selector for the requesting editor."""
        self._qick_front_panel_target = target
        self._qick_front_panel.set_scope(scope)
        titles = {
            "path": "QICK Front Panel - RF Measurement Path",
            "output": "QICK Front Panel - RF Output",
            "input": "QICK Front Panel - RF Readout",
        }
        self._qick_front_panel_dialog.setWindowTitle(titles[scope])
        if scope == "output" and isinstance(target, MultiControlPanel):
            self._qick_front_panel_dialog.setWindowTitle(
                "QICK Front Panel - AWG Output"
            )
            self._qick_front_panel.apply_button.setText("Update AWG Output")
        elif scope == "output" and not isinstance(target, RfPulsePortPanel):
            self._qick_front_panel_dialog.setWindowTitle(
                "QICK Front Panel - Stability Electrode"
            )
            self._qick_front_panel.apply_button.setText(
                "Update Stability Electrode"
            )

        if scope == "output":
            if hasattr(target, "front_panel_values"):
                values = dict(target.front_panel_values())
            else:
                spec = target.configured_spec()
                values = {
                    "output_ch": spec.gen_ch,
                    "output_board_type": spec.output_board_type,
                    "output_nqz": spec.nqz,
                    "output_att1_db": spec.att1_db,
                    "output_att2_db": spec.att2_db,
                    "output_filter_type": spec.filter_type,
                    "output_filter_cutoff_ghz": spec.filter_cutoff,
                    "output_filter_bandwidth_ghz": spec.filter_bandwidth,
                }
        elif scope == "input":
            spec = target.configured_spec()
            values = {
                "readout_ch": spec.ro_ch,
                "input_board_type": spec.input_board_type,
                "readout_nqz": spec.nqz,
                "readout_attenuation_db": spec.attenuation_db,
                "readout_dc_gain_db": spec.dc_gain_db,
                "readout_filter_type": spec.filter_type,
                "readout_filter_cutoff_ghz": spec.filter_cutoff,
                "readout_filter_bandwidth_ghz": spec.filter_bandwidth,
            }
        else:
            if target is None or not hasattr(target, "front_panel_values"):
                raise RuntimeError("RF path front-panel target is not available")
            values = dict(target.front_panel_values())
        self._qick_front_panel.set_path_values(values)
        if self._qick_configuration is not None:
            self._qick_front_panel.set_configuration(self._qick_configuration)
        self._qick_front_panel_dialog.show()
        self._qick_front_panel_dialog.raise_()
        self._qick_front_panel_dialog.activateWindow()
        # A tab-local connection (currently Noise Analysis) must identify its
        # own HWH even when another tab already populated the shared preview.
        if self._qick_configuration is None or (
            target is not None and hasattr(target, "connection_config")
        ):
            QtCore.QTimer.singleShot(0, self._identify_qick_configuration)

    def _identify_qick_configuration(self) -> None:
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "QICK task running",
                "Wait for the current QICK task to finish.",
            )
            return
        try:
            target = self._qick_front_panel_target
            if target is not None and hasattr(target, "connection_config"):
                connection = target.connection_config()
            else:
                connection, _run = self._experiment_panel.connection_values(
                    require_run_config=False
                )
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot identify QICK",
                str(exc),
            )
            return

        self._qick_front_panel.set_identifying(
            True,
            "Connecting and reading HWH configuration...",
        )
        self.statusBar().showMessage("Identifying QICK front-panel configuration")
        thread = QtCore.QThread(self)
        worker = QickConfigurationWorker(connection)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_qick_configuration_identified)
        worker.failed.connect(self._on_qick_configuration_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _identify_experiment_qick_configuration(self) -> None:
        """Refresh Experiment memory capacity using the shared QICK endpoint."""
        self._qick_front_panel_target = None
        self._identify_qick_configuration()

    def _on_qick_configuration_identified(self, configuration) -> None:
        self._qick_configuration = configuration
        self._experiment_panel.set_ddr_memory_configuration(configuration)
        self._qick_front_panel.set_configuration(configuration)
        self._multi_ctrl.set_front_panel_configuration(configuration)
        self._sparameter_panel.set_front_panel_configuration(configuration)
        self._stability_panel.set_front_panel_configuration(configuration)
        self._rf_ports_panel.set_front_panel_configuration(configuration)
        self._rf_readout_panel.set_front_panel_configuration(configuration)
        self._calibration_panel.set_front_panel_configuration(configuration)
        self._noise_panel.set_front_panel_configuration(configuration)
        self._qick_front_panel.set_identifying(
            False,
            f"{configuration.mapped_output_count} DAC / "
            f"{configuration.mapped_input_count} ADC ports mapped | "
            f"{configuration.fir_rate_label}",
        )
        self.statusBar().showMessage(
            f"QICK identified: {configuration.mapped_output_count} DAC and "
            f"{configuration.mapped_input_count} ADC front-panel mappings | "
            f"{configuration.fir_rate_label}"
        )

    def _on_qick_configuration_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown QICK identification error"
        self._qick_front_panel.set_identifying(False, f"Failed: {summary}")
        self.statusBar().showMessage("QICK front-panel identification failed")
        dialog = DetailedErrorMessageBox(
            "QICK configuration identification failed",
            summary,
            details,
            self,
        )
        dialog.exec_()

    def _apply_front_panel_settings(self, values: Mapping[str, object]) -> None:
        """Commit graphical SMA selections through the existing RF path model."""
        scope = str(values.get("selection_scope", "path"))
        if scope == "output":
            target = self._qick_front_panel_target
            if target is None or not hasattr(target, "apply_front_panel_settings"):
                raise RuntimeError("RF output front-panel target is no longer available")
            if (
                isinstance(target, RfPulsePortPanel)
                and self._experiment_thread is not None
                and self._experiment_thread.isRunning()
            ):
                QtWidgets.QMessageBox.information(
                    self,
                    "QICK task running",
                    "Wait for the current QICK task before updating RF output hardware.",
                )
                return
            try:
                if target in (
                    self._stability_panel.x_axis,
                    self._stability_panel.y_axis,
                ):
                    self._apply_qick_stability_output_selection(target, values)
                else:
                    target.apply_front_panel_settings(values)
            except (TypeError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid output selection",
                    str(exc),
                )
                return
            if isinstance(target, RfPulsePortPanel):
                try:
                    connection, _run = self._experiment_panel.connection_values(
                        require_run_config=False
                    )
                except (TypeError, ValueError) as exc:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Cannot update RF output",
                        str(exc),
                    )
                    return
                self._start_rf_output_hardware_update(
                    connection,
                    target.configured_spec(),
                )
                self._qick_front_panel_dialog.close()
                return
            self._qick_front_panel_dialog.close()
            target_name = (
                f"RF output {target._index + 1}"
                if isinstance(target, RfPulsePortPanel)
                else (
                    f"AWG output {target._selected_index + 1}"
                    if isinstance(target, MultiControlPanel)
                    else "Stability electrode"
                )
            )
            self.statusBar().showMessage(
                f"{target_name} mapped to "
                f"DAC{values['output_panel_port']} / generator {values['output_ch']}"
            )
            return
        if scope == "input":
            target = self._qick_front_panel_target
            if target is None or not hasattr(target, "apply_front_panel_settings"):
                raise RuntimeError("RF readout front-panel target is no longer available")
            try:
                target.apply_front_panel_settings(values)
            except (TypeError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid input selection",
                    str(exc),
                )
                return
            self._qick_front_panel_dialog.close()
            self.statusBar().showMessage(
                f"Readout mapped to ADC{values['input_panel_port']} / "
                f"readout {values['readout_ch']}"
            )
            return

        target = self._qick_front_panel_target
        if target is None or not hasattr(target, "apply_front_panel_settings"):
            raise RuntimeError("RF path front-panel target is no longer available")
        target.apply_front_panel_settings(values)
        self._qick_front_panel_dialog.close()
        self.statusBar().showMessage(
            f"Front-panel RF path applied only to {target.__class__.__name__}"
        )

    def _apply_qick_stability_output_selection(
        self,
        target,
        values: Mapping[str, object],
    ) -> None:
        """Map the selected Stability electrode itself to a QICK DAC."""

        generator = int(values["output_ch"])
        configuration = target._front_panel_configuration
        if configuration is not None:
            port_index = QickFrontPanelControl._find_port_for_channel(
                configuration.outputs,
                generator,
            )
            if port_index is None:
                raise ValueError(
                    f"generator {generator} is not mapped to a front-panel DAC"
                )
            port = configuration.port("output", port_index)
            channel_position = port.qick_channels.index(generator)
            if "axis_awg_tuning_v1" not in (
                port.block_paths[channel_position].lower()
            ):
                raise ValueError(
                    f"generator {generator} is not an axis_awg_tuning_v1 "
                    "channel"
                )

        output_name = str(target.output.currentData() or "")
        try:
            output_index = self._qick_output_names().index(output_name)
        except ValueError as exc:
            raise ValueError(
                "the selected Stability electrode is no longer present"
            ) from exc
        self._set_awg_output_channel(output_index, generator)

    def _start_rf_output_hardware_update(
        self,
        connection: QickConnectionConfig,
        spec: QickRfPulseSpec,
    ) -> None:
        """Apply committed RF output settings in a non-blocking worker."""
        self.statusBar().showMessage(
            f"Applying RF output {spec.gen_ch} attenuator and filter settings"
        )
        thread = QtCore.QThread(self)
        worker = QickRfOutputConfigurationWorker(connection, spec)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_rf_output_hardware_updated)
        worker.failed.connect(self._on_rf_output_hardware_update_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _on_rf_output_hardware_updated(
        self,
        details: Mapping[str, object],
    ) -> None:
        gen_ch = int(details["gen_ch"])
        if bool(details["attenuators_present"]):
            message = (
                f"RF output {gen_ch} applied: ATT1/ATT2 "
                f"{float(details['commanded_att1_db']):.2f}/"
                f"{float(details['commanded_att2_db']):.2f} dB, "
                f"{details['filter_type']} filter"
            )
        else:
            message = f"DC output {gen_ch} enabled"
        self.statusBar().showMessage(message)

    def _on_rf_output_hardware_update_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown RF output configuration error"
        self.statusBar().showMessage("RF output hardware update failed")
        dialog = DetailedErrorMessageBox(
            "RF output hardware update failed",
            summary,
            details,
            self,
        )
        dialog.exec_()

    def _legacy_qick_workflow_available(self, workflow: str) -> bool:
        """Guard auxiliary acquisition engines that have not migrated to QCS."""
        if (
            self._experiment_panel.execution_backend()
            == EXECUTION_BACKEND_QICK
        ):
            return True
        QtWidgets.QMessageBox.information(
            self,
            f"{workflow} execution unavailable",
            f"{workflow} now displays the shared QCS front-panel mapping, "
            "but its hardware execution engine still uses QICK. Select the "
            "QICK execution system to run this workflow.",
        )
        return False

    def _run_sparameter_sweep(self) -> None:
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "QICK task running",
                "Wait for the current QICK task to finish.",
            )
            return
        try:
            arguments = self._sparameter_run_arguments()
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self, "Cannot run RF sweep", str(exc)
            )
            return

        config = arguments["sweep_config"]
        is_qcs = (
            self._experiment_panel.execution_backend()
            == EXECUTION_BACKEND_QCS
        )
        if is_qcs:
            repetitions = int(arguments["repetitions_per_point"])
            sample_summary = (
                f"{config.scan_time_us:g} us requested total I/Q averaging, "
                f"{repetitions:,} repetition(s) per frequency"
            )
            power_count = 1
        else:
            identified_rate = getattr(
                self._qick_configuration,
                "fir_sample_rate_hz",
                None,
            )
            if identified_rate is None:
                sample_summary = (
                    f"{config.scan_time_us:g} us requested; HWH selects the "
                    "stored FIR sample count"
                )
            else:
                sample_count = max(
                    1,
                    int(np.ceil(config.scan_time_us * identified_rate / 1_000_000.0)),
                )
                actual_time_us = sample_count * 1_000_000.0 / identified_rate
                sample_summary = (
                    f"{sample_count:,} samples at "
                    f"{format_sample_rate_hz(identified_rate)} "
                    f"({actual_time_us:g} us actual)"
                )
            power_count = int(config.power_gains.size)
        self._sparameter_panel.set_running(
            True,
            (
                f"0% - Preparing {power_count:,} power point(s) x "
                f"{config.frequency_points:,} frequency points, "
                f"{sample_summary}"
            ),
        )
        self.statusBar().showMessage("RF S-parameter sweep running")
        thread = QtCore.QThread(self)
        worker = (
            QcsSParameterSweepWorker(arguments)
            if is_qcs
            else SParameterSweepWorker(arguments)
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_sparameter_finished)
        worker.failed.connect(self._on_sparameter_failed)
        worker.progress_changed.connect(self._on_sparameter_progress)
        if is_qcs:
            worker.partial_result.connect(self._on_qcs_sparameter_partial)
            worker.stopped.connect(self._on_sparameter_stopped)
            worker.stopped.connect(thread.quit)
            worker.stopped.connect(worker.deleteLater)
        else:
            worker.partial_result.connect(self._on_sparameter_partial)
            worker.warning_raised.connect(self._on_sparameter_warning)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _stop_sparameter_sweep(self) -> None:
        worker = self._experiment_worker
        if not isinstance(worker, QcsSParameterSweepWorker):
            return
        first_request = worker.request_stop()
        if not worker.is_stop_requested():
            self._sparameter_panel.status.setText(
                "QCS execution already completed; finalizing result"
            )
            return
        self._sparameter_panel.set_stopping()
        if first_request:
            self.statusBar().showMessage(
                "Stopping QCS RF S-parameter sweep..."
            )

    def _load_sparameter_run(self, run_id: int) -> None:
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "QICK task running",
                "Wait for the current QICK task to finish.",
            )
            return
        try:
            database_path = self._sparameter_panel.database_path_value()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Cannot load RF sweep", str(exc)
            )
            return
        self._sparameter_panel.set_running(
            True,
            "Loading saved RF S-parameter run...",
        )
        thread = QtCore.QThread(self)
        worker = SParameterLoadWorker(database_path, run_id)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_sparameter_finished)
        worker.failed.connect(self._on_sparameter_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _on_sparameter_progress(self, percent: int, message: str) -> None:
        self._sparameter_panel.update_progress(percent, message)
        self.statusBar().showMessage(f"RF sweep {percent}%: {message}")

    def _on_sparameter_warning(self, message: str) -> None:
        self._sparameter_panel.status.setText(f"Warning: {message}")
        self.statusBar().showMessage(f"RF sweep warning: {message}")
        QtWidgets.QMessageBox.warning(
            self,
            "Input calibration warning",
            message,
        )

    def _on_sparameter_partial(self, stored) -> None:
        self._sparameter_panel.show_partial_result(stored)
        self._sparameter_plot.set_result(stored.result)
        self._dock_sparameter.show()
        self._dock_sparameter.raise_()
        power_count = int(getattr(stored.result, "power_count", 1))
        self.statusBar().showMessage(
            f"RF sweep run {stored.run_id}: {power_count} power point(s) saved"
        )

    def _on_qcs_sparameter_partial(self, execution) -> None:
        """Display the cumulative I/Q average from completed QCS passes."""

        iq = np.asarray(execution.iq)
        if iq.ndim != 4 or iq.shape[-2:] != (1, 2):
            raise ValueError(
                "QCS S-parameter partial I/Q must have shape "
                "(frequency, repetition, 1, 2)"
            )
        frequencies_mhz = (
            np.asarray(execution.frequencies_hz, dtype=float) / 1.0e6
        )
        result = SParameterSweepResult.from_iq(
            frequencies_mhz,
            frequencies_mhz,
            iq[:, :, 0, :],
            sample_rate_hz=float(
                execution.program_summary.get(
                    "sample_rate_hz",
                    QCS_M5200_SAMPLE_RATE_HZ,
                )
            ),
            actual_output_powers_dbm=getattr(
                execution,
                "dut_input_powers_dbm",
                None,
            ),
            input_powers_dbm=getattr(
                execution,
                "dut_output_powers_dbm",
                None,
            ),
            s21_reference=(
                "qcs_m5300_m5200_calibration_thru"
                if getattr(execution, "dut_input_powers_dbm", None) is not None
                else None
            ),
        )
        self._sparameter_plot.set_result(result)
        self._dock_sparameter.show()
        summary = execution.program_summary
        completed = int(summary.get("completed_iq_averaging_passes", 0))
        planned = int(summary.get("iq_averaging_pass_count", 0))
        effective_ms = (
            float(summary.get("effective_integration_duration_s", 0.0))
            * 1.0e3
        )
        message = (
            f"QCS RF sweep: {completed:,}/{planned:,} averaging passes, "
            f"{effective_ms:.9g} ms effective I/Q time"
        )
        self._sparameter_panel.status.setText(message)
        self.statusBar().showMessage(message)

    def _on_sparameter_stopped(self, message: str) -> None:
        self._sparameter_panel.set_running(False, str(message))
        self.statusBar().showMessage(str(message))

    def _on_sparameter_finished(self, stored) -> None:
        self._sparameter_panel.show_result(stored)
        self._sparameter_plot.set_result(stored.result)
        self._dock_sparameter.show()
        self._dock_sparameter.raise_()
        self.statusBar().showMessage(
            f"RF S-parameter run {stored.run_id} loaded from {stored.database_path}"
        )

    def _on_sparameter_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown RF S-parameter error"
        self._sparameter_panel.set_running(False, f"Failed: {summary}")
        self.statusBar().showMessage("RF S-parameter sweep failed")
        dialog = DetailedErrorMessageBox(
            "RF S-parameter sweep failed", summary, details, self
        )
        dialog.exec_()

    def _run_noise_acquisition(self, request) -> None:
        """Run one independent QICK FIR trace or QCS M5200 raw trace."""
        backend = self._experiment_panel.execution_backend()
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Hardware task running",
                "Wait for the current hardware task to finish.",
            )
            return

        if backend == EXECUTION_BACKEND_QCS:
            try:
                duration_s = float(request.duration_s)
                connection = replace(
                    self._experiment_panel.qcs_connection_values(
                        len(self._pulse)
                    ),
                    # Noise Analysis always returns a time trace. QCS cannot
                    # expose get_trace() from a hardware-demodulated backend.
                    hw_demod=False,
                    blocking=True,
                )
                config = QcsNoiseTraceConfig(
                    connection_config=connection,
                    duration_s=duration_s,
                )
            except (AttributeError, OSError, TypeError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Cannot acquire QCS noise trace",
                    str(exc),
                )
                return
            worker = QcsNoiseAcquisitionWorker(config)
            starting_message = "Loading QCS mapper for M5200 raw trace..."
        else:
            config = getattr(request, "qick_config", request)
            if config is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Cannot acquire QICK noise trace",
                    "The QICK noise-acquisition settings are missing.",
                )
                return
            worker = NoiseAcquisitionWorker(config)
            starting_message = (
                "Connecting for independent QICK FIR-DDR acquisition..."
            )

        self._noise_panel.set_acquiring(
            True,
            starting_message,
        )
        self.statusBar().showMessage("Starting independent noise acquisition")
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress_changed.connect(
            self._on_noise_acquisition_progress
        )
        worker.finished.connect(self._on_noise_acquisition_finished)
        worker.failed.connect(self._on_noise_acquisition_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _on_noise_acquisition_progress(
        self,
        percent: int,
        message: str,
    ) -> None:
        self._noise_panel.update_acquisition_progress(percent, message)
        self.statusBar().showMessage(
            f"Noise acquisition {int(percent)}%: {message}"
        )

    def _on_noise_acquisition_finished(self, collection) -> None:
        self._noise_panel.set_acquiring(False)
        self._noise_panel.set_collection(collection)
        self._dock_noise.show()
        self._dock_noise.raise_()
        self.statusBar().showMessage(
            f"Noise acquisition completed: {collection.sample_count:,} "
            "time-trace samples analyzed"
        )

    def _on_noise_acquisition_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown noise-acquisition error"
        self._noise_panel.show_acquisition_error(summary)
        self.statusBar().showMessage("Noise trace acquisition failed")
        dialog = DetailedErrorMessageBox(
            "Noise trace acquisition failed",
            summary,
            details,
            self,
        )
        dialog.exec_()

    def _load_noise_trace(self, database_path: str, run_id: int) -> None:
        """Load one saved experiment's I traces outside the GUI thread."""
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "QICK task running",
                "Wait for the current QICK task to finish.",
            )
            return
        self._noise_panel.set_loading(
            True,
            "Loading saved QCoDeS I traces...",
        )
        self.statusBar().showMessage("Loading I traces for noise analysis")
        thread = QtCore.QThread(self)
        worker = NoiseTraceLoadWorker(database_path, run_id)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_noise_trace_loaded)
        worker.failed.connect(self._on_noise_trace_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _on_noise_trace_loaded(self, collection) -> None:
        self._noise_panel.set_collection(collection)
        self._dock_noise.show()
        self._dock_noise.raise_()
        self.statusBar().showMessage(
            f"Noise analysis loaded {collection.source}: "
            f"{collection.sample_count:,} I samples per trace"
        )

    def _on_noise_trace_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown noise-analysis load error"
        self._noise_panel.show_load_error(summary)
        self.statusBar().showMessage("Noise-analysis trace load failed")
        dialog = DetailedErrorMessageBox(
            "Cannot load noise-analysis trace",
            summary,
            details,
            self,
        )
        dialog.exec_()

    def _run_power_calibration(self, mode: str) -> None:
        is_qcs = (
            self._experiment_panel.execution_backend()
            == EXECUTION_BACKEND_QCS
        )
        if is_qcs and mode not in {"qcs_rf_output", "qcs_dc_output"}:
            return
        if not is_qcs and mode in {"qcs_rf_output", "qcs_dc_output"}:
            return
        if not is_qcs and not self._legacy_qick_workflow_available(
            "Calibration"
        ):
            return
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Hardware task running",
                "Wait for the current hardware task to finish.",
            )
            return
        try:
            if is_qcs:
                if mode == "qcs_rf_output":
                    mapper_configuration = (
                        self._calibration_panel.qcs_rf_mapper_configuration()
                    )
                    mapper_path = qcs_workflow_mapper_output_path(
                        mapper_configuration,
                        "calibration_rf",
                    )
                    calibration_config = self._calibration_panel.qcs_rf_output_config(
                        mapper_path=mapper_path,
                        init_time_s=(
                            self._experiment_panel.qcs_init_time_us.value()
                            * 1.0e-6
                        ),
                    )
                else:
                    calibration_config = (
                        self._calibration_panel.qcs_dc_output_config(
                            nominal_full_scale_v=(
                                self._experiment_panel.qcs_dc_full_scale_v.value()
                            )
                        )
                    )
            else:
                connection, _run = self._experiment_panel.connection_values(
                    database_path=(
                        self._calibration_panel.database_path_value()
                    )
                )
                calibration_config = (
                    self._calibration_panel.output_config()
                    if mode == "output"
                    else (
                        self._calibration_panel.input_config()
                        if mode == "input"
                        else self._calibration_panel.dc_voltage_config()
                    )
                )
            if mode == "output" and not (
                calibration_config.oscilloscope.visa_resource.strip()
            ):
                raise ValueError("Oscilloscope VISA resource must not be empty")
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot run calibration",
                str(exc),
            )
            return
        arguments = {"calibration_config": calibration_config}
        if not is_qcs:
            arguments["connection_config"] = connection
        elif mode == "qcs_rf_output":
            arguments["mapper_configuration"] = mapper_configuration
        if mode in ("input", "dc_voltage"):
            arguments["tproc_mhz"] = self._experiment_panel.tproc_mhz.value()
        label = {
            "output": "oscilloscope output",
            "input": "FIR-DDR input",
            "dc_voltage": "0 MHz DC voltage",
            "qcs_rf_output": "M5300A / M5200A RF power",
            "qcs_dc_output": "M5301A / 1 Mohm scope DC output",
        }[mode]
        self._calibration_panel.set_running(
            True,
            f"0% - Preparing {label} calibration",
        )
        system_name = "QCS" if is_qcs else "QICK"
        self.statusBar().showMessage(
            f"{system_name} {label} calibration running"
        )
        thread = QtCore.QThread(self)
        worker = CalibrationWorker(mode, arguments)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_calibration_finished)
        worker.failed.connect(self._on_calibration_failed)
        worker.progress_changed.connect(self._on_calibration_progress)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _on_calibration_progress(self, percent: int, message: str) -> None:
        self._calibration_panel.update_progress(percent, message)
        self.statusBar().showMessage(f"Calibration {percent}%: {message}")

    def _on_calibration_finished(self, stored) -> None:
        self._calibration_panel.show_result(stored)
        calibration = getattr(stored, "calibration", None)
        if calibration is not None and hasattr(
            calibration,
            "full_scale_power_dbm",
        ):
            self._stability_panel.adopt_qcs_power_calibration(
                str(stored.database_path),
                run_id=int(stored.run_id),
                enable=True,
            )
        if calibration is not None and hasattr(
            calibration,
            "corrected_maximum_abs_voltage_v",
        ):
            corrected_maximum = float(
                calibration.corrected_maximum_abs_voltage_v
            )
            self._experiment_panel.qcs_dc_full_scale_v.setValue(
                corrected_maximum
            )
            self._refresh_qcs_waveform_capacity()
        qcodes_run_id = int(getattr(stored, "qcodes_run_id", 0) or 0)
        if qcodes_run_id:
            saved_label = (
                f"QCoDeS Run {qcodes_run_id} "
                f"(calibration record {int(stored.run_id)})"
            )
        else:
            saved_label = f"Calibration Run {stored.run_id}"
        self.statusBar().showMessage(
            f"{saved_label} saved to {stored.database_path}"
            + (
                f"; QCS DC full scale set to +/-{corrected_maximum:.9g} V"
                if calibration is not None
                and hasattr(
                    calibration,
                    "corrected_maximum_abs_voltage_v",
                )
                else ""
            )
        )

    def _on_calibration_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown calibration error"
        self._calibration_panel.set_running(False, f"Failed: {summary}")
        system_name = (
            "QCS"
            if self._experiment_panel.execution_backend()
            == EXECUTION_BACKEND_QCS
            else "QICK"
        )
        self.statusBar().showMessage(f"{system_name} calibration failed")
        dialog = DetailedErrorMessageBox(
            f"{system_name} calibration failed", summary, details, self
        )
        dialog.exec_()

    def _load_stability_saved_run(
        self,
        database_path: str,
        run_id: int,
    ) -> None:
        thread = self._stability_plot_load_thread
        if thread is not None and thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Stability Diagram loading",
                "Wait for the selected saved diagram to finish loading.",
            )
            return
        self._stability_panel.set_saved_run_loading(True, run_id=run_id)
        self.statusBar().showMessage(
            f"Loading Stability Diagram QCoDeS Run {run_id}"
        )
        thread = QtCore.QThread(self)
        worker = StabilityOverlayLoadWorker(database_path, run_id)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_stability_saved_run_loaded)
        worker.failed.connect(self._on_stability_saved_run_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_stability_plot_load_thread)
        self._stability_plot_load_thread = thread
        self._stability_plot_load_worker = worker
        thread.start()

    def _on_stability_saved_run_loaded(self, result) -> None:
        self._stability_panel.show_loaded_run(result)
        self._dock_stability.raise_()
        self.statusBar().showMessage(
            f"Displaying {result.source_label} from {result.database_path}"
        )

    def _on_stability_saved_run_failed(self, details: str) -> None:
        self._stability_panel.set_saved_run_loading(False)
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown Stability Diagram load error"
        self._stability_panel.saved_run_status.setText(f"Failed: {summary}")
        self.statusBar().showMessage("Saved Stability Diagram load failed")
        dialog = DetailedErrorMessageBox(
            "Saved Stability Diagram load failed",
            summary,
            details,
            self,
        )
        dialog.exec_()

    def _clear_stability_plot_load_thread(self) -> None:
        self._stability_plot_load_thread = None
        self._stability_plot_load_worker = None

    def _run_stability_diagram(self, *, continuous: bool) -> None:
        backend = self._experiment_panel.execution_backend()
        if (
            self._stability_plot_load_thread is not None
            and self._stability_plot_load_thread.isRunning()
        ):
            QtWidgets.QMessageBox.information(
                self,
                "Stability Diagram loading",
                "Wait for the selected saved diagram to finish loading.",
            )
            return
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Hardware task running",
                "Wait for the current hardware task to finish.",
            )
            return
        try:
            arguments = self._stability_run_arguments(save=not continuous)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot run stability diagram",
                str(exc),
            )
            return

        config = arguments["stability_config"]
        mode = "continuous" if continuous else "single shot"
        acquisition_summary = (
            f"{config.trace_samples_per_point / QCS_M5200_SAMPLE_RATE_HZ * 1.0e6:.12g} "
            "us QCS integration time"
            if backend == EXECUTION_BACKEND_QCS
            else (
                f"{config.trace_samples_per_point:,} FIR samples / trace"
            )
        )
        self._stability_panel.set_running(
            True,
            (
                f"Preparing {mode} scan: {config.x_axis.points} x "
                f"{config.y_axis.points} points, "
                f"{config.repetitions_per_point} repetitions / point, "
                f"{acquisition_summary}"
            ),
        )
        self._experiment_panel.set_running(
            True,
            (
                f"{backend.upper()} Stability Diagram owns the active "
                "hardware configuration"
            ),
            show_progress=False,
        )
        self.statusBar().showMessage(
            f"{backend.upper()} stability diagram {mode} running"
        )
        thread = QtCore.QThread(self)
        worker_type = (
            QcsStabilityDiagramWorker
            if backend == EXECUTION_BACKEND_QCS
            else StabilityDiagramWorker
        )
        worker = worker_type(arguments, continuous=continuous)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.scan_ready.connect(self._on_stability_scan_ready)
        worker.progress_changed.connect(self._on_stability_progress)
        worker.single_finished.connect(self._on_stability_single_finished)
        worker.stopped.connect(self._on_stability_stopped)
        worker.failed.connect(self._on_stability_failed)
        worker.single_finished.connect(thread.quit)
        worker.stopped.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.single_finished.connect(worker.deleteLater)
        worker.stopped.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _stop_stability_diagram(self) -> None:
        worker = self._experiment_worker
        if not isinstance(
            worker,
            (StabilityDiagramWorker, QcsStabilityDiagramWorker),
        ):
            return
        worker.request_stop()
        self._stability_panel.set_stopping()
        self.statusBar().showMessage(
            "Stopping the active QCS pass; completed Stability averages "
            "will remain displayed"
            if isinstance(worker, QcsStabilityDiagramWorker)
            else "Stopping after the active stability scan completes"
        )

    def _on_stability_progress(self, percent: int, message: str) -> None:
        self._stability_panel.update_progress(percent, message)
        self.statusBar().showMessage(
            f"Stability diagram {percent}%: {message}"
        )

    def _active_trace_stability_result(self):
        if getattr(self, "_trace_overlay_pinned", False):
            return getattr(self, "_trace_overlay_result", None)
        return self._last_stability_result

    def _apply_trace_stability_overlay(self, *, fit: bool = False) -> None:
        if self._trace is None:
            return
        self._trace.set_stability_overlay(
            MainWindow._active_trace_stability_result(self),
            getattr(self, "_trace_overlay_quantity", "magnitude"),
        )
        if fit:
            self._trace.fit_view()

    def _set_trace_stability_quantity(self, quantity: str) -> None:
        if quantity not in {"i", "q", "magnitude", "phase"}:
            return
        self._trace_overlay_quantity = quantity
        self._apply_trace_stability_overlay()

    def _use_latest_trace_stability_overlay(self) -> None:
        self._trace_overlay_pinned = False
        self._trace_overlay_result = None
        self._trace_overlay_selector.show_latest_result(
            self._last_stability_result
        )
        self._apply_trace_stability_overlay(fit=True)
        self.statusBar().showMessage(
            "Trace Plot now follows the latest Stability Diagram scan"
        )

    def _restore_trace_stability_overlay_settings(self, settings: dict) -> None:
        """Restore the Trace Plot overlay selector and reload a saved run."""
        selector = self._trace_overlay_selector
        database_path = str(settings["database_path"])
        run_id = int(settings["run_id"])
        quantity = str(settings["quantity"])

        selector.database_path.setText(database_path)
        quantity_index = selector.quantity_combo.findData(quantity)
        with QtCore.QSignalBlocker(selector.quantity_combo):
            selector.quantity_combo.setCurrentIndex(quantity_index)
        self._trace_overlay_quantity = quantity

        selector.run_combo.clear()
        if run_id > 0:
            selector.run_combo.addItem(
                f"Run {run_id} (from loaded settings)",
                run_id,
            )

        if settings["mode"] == "saved":
            self._trace_overlay_pinned = False
            self._trace_overlay_result = None
            self._apply_trace_stability_overlay()
            self._load_trace_stability_overlay(
                database_path,
                run_id,
                quantity,
            )
        else:
            self._use_latest_trace_stability_overlay()

    def _load_trace_stability_overlay(
        self,
        database_path: str,
        run_id: int,
        quantity: str,
    ) -> None:
        thread = self._stability_overlay_thread
        if thread is not None and thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Stability overlay loading",
                "Wait for the selected Stability Diagram run to finish loading.",
            )
            return
        self._trace_overlay_quantity = quantity
        self._trace_overlay_selector.set_loading(True, run_id=run_id)
        self.statusBar().showMessage(
            f"Loading Stability Diagram QCoDeS Run {run_id}"
        )
        thread = QtCore.QThread(self)
        worker = StabilityOverlayLoadWorker(database_path, run_id)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_trace_stability_overlay_loaded)
        worker.failed.connect(self._on_trace_stability_overlay_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_stability_overlay_thread)
        self._stability_overlay_thread = thread
        self._stability_overlay_worker = worker
        thread.start()

    def _on_trace_stability_overlay_loaded(self, result) -> None:
        self._trace_overlay_result = result
        self._trace_overlay_pinned = True
        self._trace_overlay_selector.show_loaded_result(result)
        self._apply_trace_stability_overlay(fit=True)
        self.statusBar().showMessage(
            f"Trace Plot pinned to {result.source_label} from "
            f"{result.database_path}"
        )

    def _on_trace_stability_overlay_failed(self, details: str) -> None:
        self._trace_overlay_selector.set_loading(False)
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown Stability overlay error"
        self._trace_overlay_selector.status.setText(f"Failed: {summary}")
        self.statusBar().showMessage("Stability Diagram overlay load failed")
        dialog = DetailedErrorMessageBox(
            "Stability Diagram overlay load failed",
            summary,
            details,
            self,
        )
        dialog.exec_()

    def _clear_stability_overlay_thread(self) -> None:
        self._stability_overlay_thread = None
        self._stability_overlay_worker = None

    def _on_stability_scan_ready(self, result) -> None:
        result = MainWindow._stability_result_with_display_names(self, result)
        self._last_stability_result = result
        self._stability_panel.show_result(result)
        if not getattr(self, "_trace_overlay_pinned", False):
            MainWindow._apply_trace_stability_overlay(self, fit=True)
            selector = getattr(self, "_trace_overlay_selector", None)
            if selector is not None:
                selector.show_latest_result(result)
        self.statusBar().showMessage(
            f"Stability diagram scan {result.iteration} complete"
        )

    def _on_stability_single_finished(self, stored) -> None:
        diagram = MainWindow._stability_result_with_display_names(
            self,
            stored.diagram,
        )
        if diagram is not stored.diagram:
            try:
                stored = replace(stored, diagram=diagram)
            except TypeError:
                # Preserve the callback's long-standing duck-typed contract
                # for external workers and lightweight integrations.
                stored.diagram = diagram
        self._last_stability_result = stored.diagram
        self._stability_panel.show_saved_result(stored)
        self._experiment_panel.set_running(
            False,
            "Stability Diagram acquisition completed",
            show_progress=False,
        )
        selector = getattr(self, "_trace_overlay_selector", None)
        if selector is not None:
            selector.database_path.setText(str(stored.database_path))
            selector.refresh_runs()
        if not getattr(self, "_trace_overlay_pinned", False):
            MainWindow._apply_trace_stability_overlay(self, fit=True)
            if selector is not None:
                selector.show_latest_result(stored.diagram)
        self.statusBar().showMessage(
            f"Stability diagram QCoDeS Run {stored.run_id} saved to "
            f"{stored.database_path}"
        )

    def _on_stability_stopped(self) -> None:
        self._stability_panel.set_running(False, "Continuous acquisition stopped")
        self._experiment_panel.set_running(
            False,
            "Stability Diagram acquisition stopped",
            show_progress=False,
        )
        self.statusBar().showMessage("Stability diagram acquisition stopped")

    def _on_stability_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown stability-diagram error"
        self._stability_panel.set_running(False, f"Failed: {summary}")
        self._experiment_panel.set_running(
            False,
            f"Stability Diagram failed: {summary}",
            show_progress=False,
        )
        backend = self._experiment_panel.execution_backend().upper()
        self.statusBar().showMessage(f"{backend} stability diagram failed")
        dialog = DetailedErrorMessageBox(
            f"{backend} stability diagram failed",
            summary,
            details,
            self,
        )
        dialog.exec_()

    def _run_experiment(self) -> None:
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Experiment running",
                "Wait for the current hardware experiment to finish.",
            )
            return
        backend = self._experiment_panel.execution_backend()
        backend_label = backend.upper()
        self._active_experiment_backend = backend
        self._experiment_panel.start_run_timeline(
            f"{backend_label} run requested; validating Experiment settings"
        )
        self._experiment_panel.record_run_event(
            "validation",
            "started",
            "Validating waveform, sweep, RF readout, and hardware settings",
        )
        try:
            if backend == EXECUTION_BACKEND_QICK:
                arguments = self._experiment_run_arguments()
            else:
                arguments = self._qcs_experiment_run_arguments()
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            self._experiment_panel.record_run_event(
                "validation",
                "failed",
                str(exc),
            )
            self._experiment_panel.finish_run_timeline(
                f"Run validation failed: {exc}",
                success=False,
            )
            QtWidgets.QMessageBox.warning(self, "Cannot run experiment", str(exc))
            return
        self._experiment_panel.record_run_event(
            "validation",
            "completed",
            "Experiment settings validated",
        )

        sweep_points = arguments["sequence"].sweep_point_count
        repetitions = arguments["repetitions_per_sweep"]
        if backend == EXECUTION_BACKEND_QICK:
            samples_per_acquisition = int(
                arguments["readout_spec"].samples_per_trigger
            )
            worker_class = QickExperimentWorker
        else:
            acquisition = arguments["acquisition"]
            samples_per_acquisition = (
                0
                if acquisition is None
                else (
                    1
                    if arguments["connection_config"].hw_demod
                    else int(
                        acquisition.sample_count
                        if acquisition.sample_count is not None
                        else round(
                            acquisition.duration_s
                            * acquisition.sample_rate_hz
                        )
                    )
                )
            )
            worker_class = QcsExperimentWorker
        iq_repetition_policy = normalize_iq_repetition_policy(
            arguments["iq_repetition_policy"]
        )
        stored_repetitions = (
            1
            if (
                iq_repetition_policy
                == IQ_REPETITION_POLICY_COHERENT_AVERAGE
            )
            else repetitions
        )
        expected_rows = (
            sweep_points * stored_repetitions * samples_per_acquisition
        )
        storage_summary = (
            "; coherent average saves one I/Q value per sweep point"
            if (
                iq_repetition_policy
                == IQ_REPETITION_POLICY_COHERENT_AVERAGE
            )
            else "; every repetition is saved"
        )
        self._experiment_panel.set_running(
            True,
            (
                f"0% - Preparing {sweep_points:,} sweep points x "
                f"{repetitions:,} repetitions "
                f"({sweep_points * repetitions:,} acquisitions, "
                f"{expected_rows:,} IQ sample rows{storage_summary})"
            ),
            allow_stop=backend == EXECUTION_BACKEND_QCS,
        )
        self.statusBar().showMessage(f"{backend_label} experiment running")
        thread = QtCore.QThread(self)
        worker = worker_class(arguments)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_experiment_finished)
        worker.failed.connect(self._on_experiment_failed)
        worker.progress_changed.connect(self._on_experiment_progress)
        worker.event_changed.connect(self._on_experiment_event)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        if isinstance(worker, QcsExperimentWorker):
            self._active_qcs_partial_result = None
            worker.partial_result.connect(self._on_experiment_partial)
            worker.stopped.connect(self._on_experiment_stopped)
            worker.stop_status_changed.connect(
                self._on_experiment_stop_status
            )
            worker.stopped.connect(thread.quit)
            worker.stopped.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _stop_experiment(self) -> None:
        worker = self._experiment_worker
        if not isinstance(worker, QcsExperimentWorker):
            return
        first_request = worker.request_stop()
        if not worker.is_stop_requested():
            self._experiment_panel.set_stop_available(False)
            self._experiment_panel.run_status.setText(
                "QCS hardware execution already completed; finalizing "
                "the acquired result"
            )
            self.statusBar().showMessage(
                "QCS hardware execution already completed; finalizing result"
            )
            return
        self._experiment_panel.set_stopping(
            "Stopping QCS experiment and resetting mapped DC outputs..."
        )
        if first_request:
            self._experiment_panel.record_run_event(
                "cancellation",
                "started",
                "User requested QCS program abort",
            )
        self.statusBar().showMessage(
            "Stopping QCS experiment; waiting for hardware abort and DC reset"
        )

    def _show_awg_metadata(self) -> None:
        try:
            arguments = self._experiment_run_arguments(
                require_readout=False,
                require_run_config=False,
            )
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot build AWG metadata",
                str(exc),
            )
            return
        dialog = AwgMetadataDialog(
            arguments["sequence"],
            fabric_mhz=self._qick_fabric_mhz,
            full_scale_mv=arguments["source_full_scale_mv"],
            parent=self,
        )
        dialog.exec_()

    def _show_program(self) -> None:
        """Show the backend-appropriate Program without running hardware."""

        if (
            self._experiment_panel.execution_backend()
            == EXECUTION_BACKEND_QCS
        ):
            self._show_qcs_program()
        else:
            self._show_qick_program()

    def _show_qcs_program(self) -> None:
        """Show a pure source snapshot without importing or invoking QCS."""

        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "QCS task running",
                "Wait for the current QCS task to finish.",
            )
            return
        try:
            code = self._generate_qcs_code()
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot show QCS program",
                str(exc),
            )
            return
        self.statusBar().showMessage(
            "Showing offline QCS source; no mapper, compile, or hardware call"
        )
        dialog = QcsProgramCodeDialog(code, self)
        dialog.exec_()

    def _show_qick_program(self) -> None:
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "QICK task running",
                "Wait for the current QICK task to finish.",
            )
            return
        try:
            arguments = self._experiment_run_arguments(
                require_readout=False, require_run_config=False
            )
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot show QICK program", str(exc))
            return

        program_kwargs = {
            "sequence": arguments["sequence"],
            "awg_channels": arguments["awg_channels"],
            "repetitions_per_sweep": arguments["repetitions_per_sweep"],
            "tproc_mhz": self._qick_tproc_mhz,
            "rf_specs": arguments["rf_specs"],
            "readout_spec": arguments["readout_spec"],
            "compile_validation_mode": arguments[
                "compile_validation_mode"
            ],
        }
        self._experiment_panel.set_running(
            True,
            "Connecting to QICK and compiling tProcessor assembly...",
            show_progress=False,
        )
        self.statusBar().showMessage("Compiling QICK tProcessor assembly")
        thread = QtCore.QThread(self)
        worker = QickProgramWorker(
            arguments["connection_config"], program_kwargs
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_qick_program_ready)
        worker.failed.connect(self._on_qick_program_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _on_qick_program_ready(self, result: dict) -> None:
        instruction_count = int(result.get("instruction_count", 0))
        self._experiment_panel.set_running(
            False,
            f"QICK program compiled: {instruction_count:,} instructions",
        )
        self.statusBar().showMessage(
            f"QICK tProcessor program compiled ({instruction_count:,} instructions)"
        )
        dialog = QickAssemblyDialog(result, self)
        dialog.exec_()

    def _on_qick_program_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown QICK program compilation error"
        self._experiment_panel.set_running(False, f"Failed: {summary}")
        self.statusBar().showMessage("QICK program compilation failed")
        dialog = DetailedErrorMessageBox(
            "Cannot show QICK program", summary, details, self
        )
        dialog.exec_()

    def _on_experiment_progress(self, percent: int, message: str) -> None:
        self._experiment_panel.update_progress(percent, message)
        backend = self._active_experiment_backend.upper()
        self.statusBar().showMessage(
            f"{backend} experiment {percent}%: {message}"
        )

    def _on_experiment_event(
        self,
        key: str,
        state: str,
        message: str,
    ) -> None:
        self._experiment_panel.record_run_event(key, state, message)
        if (
            self._active_experiment_backend == EXECUTION_BACKEND_QCS
            and str(key) == "program_build"
        ):
            lowered_message = str(message).lower()
            lowered_state = str(state).lower()
            if (
                lowered_state == "started"
                and "fixed numeric" in lowered_message
            ):
                self._experiment_panel.set_qcs_sweep_execution_status(
                    QcsSweepExecutionPreview(
                        mode="software",
                        reasons=(str(message),),
                    )
                )
            elif (
                lowered_state == "completed"
                and "native qcs hardware sweep" in lowered_message
            ):
                self._experiment_panel.set_qcs_sweep_execution_status(
                    QcsSweepExecutionPreview(
                        mode="hardware",
                        reasons=(str(message),),
                    )
                )
            elif "hardware sweep inside software loop" in lowered_message:
                self._experiment_panel.set_qcs_sweep_execution_status(
                    QcsSweepExecutionPreview(
                        mode="hybrid",
                        reasons=(str(message),),
                    )
                )
            elif (
                lowered_state == "completed"
                and (
                    "software" in lowered_message
                    or "fixed numeric" in lowered_message
                )
            ):
                self._experiment_panel.set_qcs_sweep_execution_status(
                    QcsSweepExecutionPreview(
                        mode="software",
                        reasons=(str(message),),
                    )
                )
        hardware_finished = (
            str(key) == "acquisition"
            and str(state).lower() == "completed"
        )
        save_started = (
            str(key) == "qcodes_save"
            and str(state).lower() == "started"
        )
        if hardware_finished or save_started:
            # Once acquisition completes there is no instrument program left
            # to abort. Keep result normalization and the database transaction
            # atomic instead of presenting a misleading late Stop action.
            self._experiment_panel.set_stop_available(False)

    def _on_experiment_stop_status(self, message: str) -> None:
        self._experiment_panel.run_status.setText(str(message))
        self._experiment_panel.record_run_event(
            "cancellation",
            "info",
            str(message),
        )

    def _load_awg_sweep_saved_run(
        self,
        database_path: str,
        run_id: int,
    ) -> None:
        thread = self._awg_sweep_load_thread
        if thread is not None and thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "AWG sweep loading",
                "Wait for the selected AWG sweep run to finish loading.",
            )
            return
        self._awg_sweep_run_selector.set_loading(True, run_id=run_id)
        self.statusBar().showMessage(
            f"Loading AWG sweep QCoDeS Run {run_id}"
        )
        thread = QtCore.QThread(self)
        worker = AwgSweepMapLoadWorker(database_path, run_id)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_awg_sweep_saved_run_loaded)
        worker.failed.connect(self._on_awg_sweep_saved_run_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_awg_sweep_load_thread)
        self._awg_sweep_load_thread = thread
        self._awg_sweep_load_worker = worker
        thread.start()

    def _on_awg_sweep_saved_run_loaded(self, result) -> None:
        self._awg_sweep_plot.set_result(result)
        self._awg_sweep_run_selector.show_loaded_result(result)
        self._dock_awg_sweep.show()
        self._dock_awg_sweep.raise_()
        self.statusBar().showMessage(
            f"Displaying {result.source_label} from {result.database_path}"
        )

    def _on_awg_sweep_saved_run_failed(self, details: str) -> None:
        self._awg_sweep_run_selector.set_loading(False)
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown AWG sweep load error"
        self._awg_sweep_run_selector.status.setText(f"Failed: {summary}")
        self.statusBar().showMessage("Saved AWG sweep load failed")
        dialog = DetailedErrorMessageBox(
            "Saved AWG sweep load failed",
            summary,
            details,
            self,
        )
        dialog.exec_()

    def _clear_awg_sweep_load_thread(self) -> None:
        self._awg_sweep_load_thread = None
        self._awg_sweep_load_worker = None

    def _use_latest_awg_sweep_result(self) -> None:
        result = self._last_awg_sweep_map_result
        self._awg_sweep_run_selector.show_latest_result(result)
        if result is None:
            self.statusBar().showMessage(
                "No in-memory AWG 2D sweep result is available"
            )
            return
        self._awg_sweep_plot.set_result(result)
        self._dock_awg_sweep.show()
        self._dock_awg_sweep.raise_()
        self.statusBar().showMessage(
            f"Displaying {result.source_label or 'latest AWG experiment'}"
        )

    def _refresh_awg_sweep_map_from_last_result(
        self,
        result=None,
        *,
        source_label: Optional[str] = None,
        report_error: bool = True,
        raise_dock: bool = True,
    ) -> Optional[str]:
        if result is None:
            result = self._last_experiment_result
        if result is None:
            return None
        selected = self._experiment_panel.selected_sweep_axis_keys()
        if selected is None:
            return None
        try:
            result_full_scale_mv = float(
                dict(getattr(result, "program_summary", {}) or {}).get(
                    "source_full_scale_mv",
                    self._active_awg_full_scale_mv(),
                )
            )
            iq_values, unit, mode, _metadata = measurement_iq_values(
                result.ddr_result.iq,
                result.rf_settings,
            )
            map_result = reduce_awg_sweep_map(
                result.ddr_result,
                x_axis_key=selected[0],
                y_axis_key=selected[1],
                full_scale_mv=result_full_scale_mv,
                iq_values=iq_values,
                value_unit=unit,
                measurement_mode=mode,
                output_name_mapping=self._awg_output_display_name_map(),
            )
            run_id = int(getattr(result, "run_id", 0) or 0)
            database_path = str(getattr(result, "database_path", "") or "")
            if source_label is None:
                source_label = (
                    f"QCoDeS Run {run_id}"
                    if run_id > 0
                    else "Live QCS mixed sweep"
                )
            map_result = replace(
                map_result,
                source_label=source_label,
                database_path=database_path,
                run_id=run_id,
                source=(
                    None
                    if map_result.source is None
                    else replace(
                        map_result.source,
                        source_label=source_label,
                        database_path=database_path,
                        run_id=run_id,
                    )
                ),
            )
        except Exception as exc:
            if report_error:
                run_id = int(getattr(result, "run_id", 0) or 0)
                prefix = (
                    f"QCoDeS run {run_id} saved"
                    if run_id > 0
                    else "Live QCS partial result"
                )
                self.statusBar().showMessage(
                    f"{prefix}; AWG 2D map unavailable: {exc}"
                )
            return str(exc)
        self._last_awg_sweep_map_result = map_result
        self._awg_sweep_plot.set_result(map_result)
        self._awg_sweep_run_selector.show_latest_result(map_result)
        self._dock_awg_sweep.show()
        if raise_dock:
            self._dock_awg_sweep.raise_()
        return None

    def _on_experiment_partial(self, result) -> None:
        """Expose every completed mixed-sweep block without ending the run."""

        self._active_qcs_partial_result = result
        self._last_experiment_result = result
        summary = getattr(result, "program_summary", {}) or {}
        completed = int(summary.get("completed_points", 0))
        planned = int(summary.get("planned_points", completed))
        label = f"Live QCS mixed sweep ({completed:,}/{planned:,} points)"
        self._refresh_awg_sweep_map_from_last_result(
            result,
            source_label=label,
            report_error=False,
            raise_dock=False,
        )
        self._experiment_panel.run_status.setText(
            f"Mixed QCS sweep: {completed:,}/{planned:,} acquired point(s) "
            "available; Stop remains active"
        )
        self.statusBar().showMessage(
            f"{label}; completed data retained in memory"
        )

    def _on_experiment_finished(self, result) -> None:
        self._active_qcs_partial_result = None
        self._last_experiment_result = result
        self._experiment_panel.show_result(result)
        map_error = self._refresh_awg_sweep_map_from_last_result()
        message = (
            f"QCoDeS run {result.run_id} saved to {result.database_path}"
        )
        if map_error:
            message += f"; AWG 2D map unavailable: {map_error}"
        self._experiment_panel.finish_run_timeline(
            f"QCoDeS Run {result.run_id} completed",
            success=True,
        )
        self.statusBar().showMessage(message)

    def _on_experiment_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        backend = self._active_experiment_backend.upper()
        summary = lines[-1] if lines else f"Unknown {backend} experiment error"
        self._experiment_panel.set_running(False, f"Failed: {summary}")
        self._experiment_panel.finish_run_timeline(
            f"Run failed: {summary}",
            success=False,
        )
        self.statusBar().showMessage(f"{backend} experiment failed")
        dialog = DetailedErrorMessageBox(
            f"{backend} experiment failed", summary, details, self
        )
        dialog.exec_()

    def _on_experiment_stopped(self, message: str) -> None:
        message = str(message).strip() or "Stopped by user"
        partial = self._active_qcs_partial_result
        if partial is not None:
            summary = getattr(partial, "program_summary", {}) or {}
            completed = int(summary.get("completed_points", 0))
            planned = int(summary.get("planned_points", completed))
            run_id = int(getattr(partial, "run_id", 0) or 0)
            database_path = str(
                getattr(partial, "database_path", "") or ""
            )
            if run_id > 0:
                message += (
                    f"; retained {completed:,}/{planned:,} completed point(s) "
                    f"in QCoDeS Run {run_id} at {database_path}"
                )
            else:
                message += (
                    f"; retained {completed:,}/{planned:,} completed point(s) "
                    "in memory"
                )
        self._experiment_panel.set_running(False, message)
        self._experiment_panel.finish_run_timeline(
            message,
            success=False,
            stopped=True,
        )
        self.statusBar().showMessage(message)

    def _clear_experiment_thread(self) -> None:
        self._experiment_thread = None
        self._experiment_worker = None
        if self._qcs_mapper_commit_queued is not None:
            QtCore.QTimer.singleShot(0, self._start_next_qcs_mapper_commit)

    def _refresh_rf_timeline(self, *, fit_view: bool = False) -> None:
        if RfPulseTimelineWidget is None:
            return
        valid_specs = []
        valid_ranges = []
        for spec in self._rf_pulse_specs:
            try:
                start_us, end_us, _ = rf_pulse_absolute_times_us(self._pulse[0], spec)
            except ValueError:
                continue
            valid_specs.append(spec)
            valid_ranges.append((start_us, end_us))
        while len(self._rf_timelines) < len(valid_specs):
            timeline = RfPulseTimelineWidget(self._rf_timeline_container)
            if hasattr(timeline, "set_time_unit"):
                timeline.set_time_unit(self._time_unit)
            timeline.getPlotItem().getViewBox().setXLink(
                self._plot.getPlotItem().getViewBox()
            )
            self._rf_timeline_layout.addWidget(timeline)
            self._rf_timelines.append(timeline)
        while len(self._rf_timelines) > len(valid_specs):
            timeline = self._rf_timelines.pop()
            self._rf_timeline_layout.removeWidget(timeline)
            timeline.deleteLater()
        self._rf_timeline = self._rf_timelines[0] if self._rf_timelines else None
        if not valid_specs:
            self._rf_timeline_container.hide()
            return
        for timeline, spec, (start_us, end_us) in zip(
            self._rf_timelines, valid_specs, valid_ranges
        ):
            timeline.set_pulse(
                gen_ch=spec.gen_ch,
                start_ns=start_us * 1000.0,
                duration_ns=(end_us - start_us) * 1000.0,
                frequency_mhz=spec.frequency_mhz,
                gain=spec.gain,
                phase_degrees=spec.phase_degrees,
                att1_db=spec.att1_db,
                att2_db=spec.att2_db,
            )
            timeline.show()
        self._rf_timeline_container.show()
        total_height = max(360, self._waveform_splitter.height())
        rf_height = min(520, max(130, 130 * len(valid_specs)))
        self._waveform_splitter.setSizes([max(220, total_height - rf_height), rf_height])
        if fit_view:
            self._fit_view()

    def _refresh_rf_editor(self) -> None:
        self._refresh_rf_timeline()
        self._rf_ports_panel.refresh_segments(self._pulse[0])
        self._rf_readout_panel.refresh_segments(self._pulse[0])
        if self._rf_panel is None:
            return
        self._rf_panel.refresh_segments(self._pulse[0])
        if self._rf_pulse_spec is not None:
            self._rf_panel.load_spec(self._rf_pulse_spec)

    def _add_segment(self, ramp: float, flat: float, v: float) -> None:
        try:
            source_port = int(self._selected_port_idx)
            source = self._pulse[source_port]
            source.add_flat_ramp(
                ramp,
                flat,
                v
            )
            segment_index = len(source.flat_segments()) - 1
            segment_name = source.segment_name(segment_index)
            for port_index, pulse in enumerate(self._pulse):
                if port_index == source_port:
                    continue
                pulse.add_flat_ramp(
                    ramp,
                    flat,
                    float(pulse.v[-1]),
                )
                pulse.rename_segment(segment_index, segment_name)
            self._share_segment_timing(
                source_port,
                segment_index,
            )
            self._share_segment_name(
                source_port,
                segment_index,
            )
            self._plot.set_selected_port_idx(source_port)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid input", str(exc))
            return
        self._plot.refresh()
        self._plot.fit_view()
        self._multi_ctrl.refresh_table()
        self._refresh_stability_targets()
        self._refresh_trace_if_needed(force=True)
        self._refresh_sweep_overlay()
        self._refresh_rf_editor()

    def _flat_update(self, i0: int, i1: int, new_v: float) -> None:
        self.statusBar().showMessage(f"Flat {i0}-{i1} moved to {new_v:.6g} mV")
        self._schedule_deferred_refresh()

    def _point_update(self, i0, i1, new_t):
        self._share_segment_timing(
            self._selected_port_idx,
            int(i0) // 2,
        )
        self.statusBar().showMessage(
            f"Point {i0}-{i1} moved to "
            f"{_time_from_ns(new_t, self._time_unit):.6g} {self._time_unit}"
        )
        self._schedule_deferred_refresh(
            timing_changed=True,
            shared_timing_changed=True,
        )

    def _schedule_deferred_refresh(
        self,
        *,
        timing_changed: bool = False,
        shared_timing_changed: bool = False,
    ) -> None:
        self._pending_table_refresh = True
        self._pending_shared_timing_refresh |= shared_timing_changed
        if (
            self._trace is not None
            and (
                shared_timing_changed
                or self._trace.uses_port(self._selected_port_idx)
            )
        ):
            self._pending_trace_refresh = True
        self._pending_rf_refresh |= timing_changed
        sweep_targets = {
            target
            for target in (
                self._sweep_target_indices(spec) for spec in self._sweep_specs
            )
            if target is not None
        }
        self._pending_sweep_refresh |= (
            shared_timing_changed
            or bool(self._sweep_specs)
            or any(target[0] == self._selected_port_idx for target in sweep_targets)
        )
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def _flush_deferred_refresh(self) -> None:
        if self._pending_table_refresh:
            if self._pending_shared_timing_refresh:
                self._multi_ctrl.refresh_table()
                self._refresh_waveform_plot()
            else:
                self._multi_ctrl._ctrl_pannels[
                    self._selected_port_idx
                ].refresh_table()
        if self._pending_trace_refresh and self._trace is not None:
            self._trace.refresh_trace(self._pulse)
        if self._pending_rf_refresh:
            self._refresh_rf_editor()
        if self._pending_sweep_refresh:
            self._refresh_sweep_overlay()
        elif self._pending_table_refresh:
            self._refresh_physical_waveforms()
            self._refresh_qcs_waveform_capacity()
        self._pending_table_refresh = False
        self._pending_trace_refresh = False
        self._pending_rf_refresh = False
        self._pending_sweep_refresh = False
        self._pending_shared_timing_refresh = False

    def _ensure_trace_widget(self) -> TracePlotWidget:
        if self._trace is None:
            self._trace = TracePlotWidget(self)
            self._trace.hold_edit_requested.connect(
                self._edit_trace_hold
            )
            self._trace.ramp_edit_requested.connect(
                self._edit_trace_ramp
            )
            if hasattr(self._trace, "set_time_unit"):
                self._trace.set_time_unit(self._time_unit)
            if hasattr(self._trace, "set_output_names"):
                self._trace.set_output_names(
                    self._awg_output_display_names
                )
            if hasattr(self._trace, "set_stability_overlay"):
                self._trace.set_stability_overlay(
                    self._active_trace_stability_result(),
                    self._trace_overlay_quantity,
                )
            self._trace_layout.replaceWidget(
                self._trace_placeholder,
                self._trace,
            )
            self._trace_placeholder.hide()
            self._trace_placeholder.deleteLater()
        return self._trace

    def _trace_selection(self) -> Tuple[int, int]:
        if self._trace is None or not self._trace.has_selection:
            raise ValueError("select both X and Y AWG outputs first")
        return int(self._trace.x_idx), int(self._trace.y_idx)

    def _refresh_after_trace_segment_edit(self) -> None:
        self._multi_ctrl.refresh_table()
        self._refresh_waveform_plot()
        self._refresh_trace_if_needed(force=True)
        self._refresh_rf_editor()
        self._refresh_sweep_overlay()

    def _apply_trace_hold_edit(
        self,
        segment_index: int,
        x_mv: float,
        y_mv: float,
        hold_ns: float,
        segment_name: Optional[str] = None,
    ) -> None:
        x_output, y_output = self._trace_selection()
        segment_index = int(segment_index)
        flat_index = 2 * segment_index
        pulse_x = self._pulse[x_output]
        pulse_y = self._pulse[y_output]
        if (
            segment_index >= len(pulse_x.flat_segments())
            or segment_index >= len(pulse_y.flat_segments())
        ):
            raise ValueError("the selected trace point no longer exists")
        if x_output == y_output and not np.isclose(
            float(x_mv),
            float(y_mv),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "X and Y use the same AWG output and must have one voltage"
            )
        if not pulse_x.edit_voltage(flat_index, float(x_mv)):
            raise ValueError("invalid X hold point")
        if x_output != y_output and not pulse_y.edit_voltage(
            flat_index,
            float(y_mv),
        ):
            raise ValueError("invalid Y hold point")
        if not pulse_x.edit_flat(flat_index, float(hold_ns)):
            raise ValueError("hold duration must be positive")
        if segment_name is not None:
            pulse_x.rename_segment(segment_index, segment_name)
            self._share_segment_name(x_output, segment_index)
        self._share_segment_timing(x_output, segment_index)
        self._refresh_after_trace_segment_edit()
        display_name = pulse_x.segment_name(segment_index)
        self.statusBar().showMessage(
            f"{display_name} updated: X {float(x_mv):.6g} mV, "
            f"Y {float(y_mv):.6g} mV, hold "
            f"{_time_from_ns(float(hold_ns), self._time_unit):.6g} "
            f"{self._time_unit}"
        )

    def _apply_trace_ramp_edit(
        self,
        segment_index: int,
        ramp_ns: float,
    ) -> None:
        x_output, _y_output = self._trace_selection()
        segment_index = int(segment_index)
        if segment_index <= 0:
            raise ValueError("the initial hold has no incoming ramp")
        pulse = self._pulse[x_output]
        if segment_index >= len(pulse.flat_segments()):
            raise ValueError("the selected ramp no longer exists")
        if not pulse.edit_ramp(2 * segment_index, float(ramp_ns)):
            raise ValueError("ramp duration must be positive")
        self._share_segment_timing(x_output, segment_index)
        self._refresh_after_trace_segment_edit()
        self.statusBar().showMessage(
            f"Ramp P{segment_index - 1} to P{segment_index} updated to "
            f"{_time_from_ns(float(ramp_ns), self._time_unit):.6g} "
            f"{self._time_unit}"
        )

    def _edit_trace_hold(self, segment_index: int) -> None:
        try:
            x_output, y_output = self._trace_selection()
            segment_index = int(segment_index)
            flat_index = 2 * segment_index
            pulse_x = self._pulse[x_output]
            pulse_y = self._pulse[y_output]
            timing = self._segment_timing_ns(pulse_x, segment_index)
            if (
                timing is None
                or segment_index >= len(pulse_y.flat_segments())
            ):
                raise ValueError("the selected trace point no longer exists")
            dialog = TraceHoldEditDialog(
                segment_index=segment_index,
                x_output=x_output,
                y_output=y_output,
                x_mv=float(pulse_x.v[flat_index]),
                y_mv=float(pulse_y.v[flat_index]),
                x_bounds=tuple(pulse_x.v_bounds),
                y_bounds=tuple(pulse_y.v_bounds),
                hold_ns=timing[1],
                segment_name=pulse_x.segment_name(segment_index),
                time_unit=self._time_unit,
                parent=self,
            )
            if dialog.exec_() != QtWidgets.QDialog.Accepted:
                return
            self._apply_trace_hold_edit(
                segment_index,
                *dialog.values(),
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Trace hold edit failed",
                str(exc),
            )

    def _edit_trace_ramp(self, segment_index: int) -> None:
        try:
            x_output, _y_output = self._trace_selection()
            segment_index = int(segment_index)
            timing = self._segment_timing_ns(
                self._pulse[x_output],
                segment_index,
            )
            if timing is None or segment_index <= 0:
                raise ValueError("the selected ramp no longer exists")
            dialog = TraceRampEditDialog(
                segment_index=segment_index,
                ramp_ns=timing[0],
                time_unit=self._time_unit,
                parent=self,
            )
            if dialog.exec_() != QtWidgets.QDialog.Accepted:
                return
            self._apply_trace_ramp_edit(
                segment_index,
                dialog.value_ns(),
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Trace ramp edit failed",
                str(exc),
            )

    def _refresh_trace_if_needed(self, *, force: bool = False) -> None:
        if self._trace is None:
            return
        if force or self._trace.uses_port(self._selected_port_idx):
            self._trace.refresh_trace(self._pulse)

    def _trace_stability_overlay_settings(self) -> dict:
        """Return the reloadable Trace Plot overlay selection."""
        selector = self._trace_overlay_selector
        quantity = str(
            getattr(self, "_trace_overlay_quantity", selector.quantity)
        )
        database_path = selector.database_path.text().strip()
        selected_run_id = selector.run_combo.currentData()
        run_id = int(selected_run_id) if selected_run_id is not None else 0
        mode = "latest"

        result = getattr(self, "_trace_overlay_result", None)
        if getattr(self, "_trace_overlay_pinned", False) and result is not None:
            result_database = str(
                getattr(result, "database_path", "")
            ).strip()
            result_run_id = int(getattr(result, "run_id", 0) or 0)
            if result_database and result_run_id > 0:
                mode = "saved"
                database_path = result_database
                run_id = result_run_id

        return {
            "mode": mode,
            "database_path": database_path or DEFAULT_STABILITY_DB_PATH,
            "run_id": run_id,
            "quantity": quantity,
        }

    def _settings_to_dict(
        self,
        *,
        qcs_settings_override: Optional[Mapping[str, object]] = None,
    ) -> dict:
        """Return every user-editable experiment setting in canonical units."""
        experiment_values = self._experiment_panel.values(
            len(self._pulse),
            validate_qcs_hardware=False,
        )
        connection = experiment_values["connection"]
        run = experiment_values["run"]
        self._noise_panel.set_connection_values(connection)
        self._qick_fabric_mhz = experiment_values["fabric_mhz"]
        self._qick_tproc_mhz = experiment_values["tproc_mhz"]
        self._qick_full_scale_mv = experiment_values["full_scale_mv"]
        self._qick_awg_channels = experiment_values["awg_channels"]
        self._multi_ctrl.set_awg_channels(self._qick_awg_channels)
        self._qick_repetitions_per_sweep = experiment_values[
            "repetitions_per_sweep"
        ]
        self._bias_t_compensation_enabled = experiment_values[
            "bias_t_compensation_enabled"
        ]
        self._bias_t_compensation_type = experiment_values[
            "bias_t_compensation_type"
        ]
        self._bias_t_compensation_voltage_mv = experiment_values[
            "bias_t_compensation_voltage_mv"
        ]
        self._bias_t_compensation_mode = experiment_values[
            "bias_t_compensation_mode"
        ]
        self._bias_t_compensation_duration_us = experiment_values[
            "bias_t_compensation_duration_us"
        ]
        self._bias_t_filter_tau_us = experiment_values["bias_t_filter_tau_us"]
        stability_settings = self._stability_panel.settings_dict()
        sweep_map_settings = (
            self._awg_sweep_plot.axis_selection_settings()
            or self._experiment_panel.sweep_map_settings()
        )
        sweep_map_settings.setdefault("slice_axes", [])
        sweep_map_settings["color_ranges"] = (
            self._awg_sweep_plot.color_range_settings()
        )
        sweep_map_settings["visible_data"] = list(
            self._awg_sweep_plot.visible_data()
        )
        return {
            "schema": SETTINGS_SCHEMA,
            "version": SETTINGS_VERSION,
            "display": {
                "time_unit": self._time_unit,
                "voltage_view": self._plot.voltage_view,
                "selected_awg_output": self._selected_port_idx,
                "selected_control_tab": self._control_tabs.currentIndex(),
                "selected_awg_tuning_tab": self._awg_tuning_tabs.currentIndex(),
                "trace_stability_overlay": (
                    self._trace_stability_overlay_settings()
                ),
            },
            "grid": {
                "time_step_ns": self._grid_time_ns,
                "voltage_step_mv": self._grid_voltage_mv,
                "snap_enabled": self._grid_snap_enabled,
                "visible": self._grid_visible,
                "configured": self._grid_configured,
            },
            "awg": {
                "outputs": [pulse.to_dict() for pulse in self._pulse],
                "output_names": list(self._awg_output_display_names),
                "output_name_mapping": list(
                    self._awg_output_name_mapping()
                ),
                "cross_capacitance": self._cross_capacitance.tolist(),
                "voltage_coordinate_full_scale_mv": (
                    self._awg_voltage_scale_mv
                ),
                "sweeps": [
                    {
                        "axis_kind": spec.axis_kind,
                        "segment_name": spec.segment_name,
                        "output_name": spec.output_name,
                        "start": spec.start,
                        "stop": spec.stop,
                        "count": spec.count,
                    }
                    for spec in self._sweep_specs
                ],
            },
            "qick": {
                "fabric_mhz": self._qick_fabric_mhz,
                "tproc_mhz": self._qick_tproc_mhz,
                "full_scale_mv": self._qick_full_scale_mv,
                "awg_channels": list(self._qick_awg_channels),
                "repetitions_per_sweep": self._qick_repetitions_per_sweep,
                "awg_metadata_mode": experiment_values["awg_metadata_mode"],
                "compile_validation_mode": experiment_values[
                    "compile_validation_mode"
                ],
                "bias_t_compensation": {
                    "enabled": self._bias_t_compensation_enabled,
                    "type": self._bias_t_compensation_type,
                    "mode": self._bias_t_compensation_mode,
                    "voltage_mv": self._bias_t_compensation_voltage_mv,
                    "duration_us": self._bias_t_compensation_duration_us,
                    "filter_tau_us": self._bias_t_filter_tau_us,
                },
            },
            "qcs": (
                self._experiment_panel.qcs_settings_dict()
                if qcs_settings_override is None
                else dict(qcs_settings_override)
            ),
            "experiment": {
                "execution_backend": experiment_values["execution_backend"],
                "qick_host": connection.host,
                "ns_port": connection.ns_port,
                "proxy_name": connection.proxy_name,
                "database_path": run.database_path,
                "experiment_name": run.experiment_name,
                "sample_name": run.sample_name,
                "notes": run.notes,
                "iq_repetition_policy": experiment_values[
                    "iq_repetition_policy"
                ],
                "sweep_map": sweep_map_settings,
            },
            "rf_outputs": list(self._rf_ports_panel.settings()),
            "rf_readout": self._rf_readout_panel.settings_dict(),
            "stability_diagram": stability_settings,
            "s_parameter": dict(self._sparameter_panel.settings_dict()),
            "calibration": dict(self._calibration_panel.settings_dict()),
            "noise_analysis": dict(self._noise_panel.settings_dict()),
        }

    @staticmethod
    def _json_bool(value, name: str) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
        return value

    @staticmethod
    def _json_finite_float(value, name: str, *, positive: bool = False) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric")
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} must be finite")
        if positive and result <= 0.0:
            raise ValueError(f"{name} must be positive")
        return result

    @staticmethod
    def _json_int(value, name: str, *, minimum: int = 0) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        result = int(value)
        if result < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return result

    def _decode_rf_output_entries(
        self,
        raw_entries,
        *,
        set_names: set,
        label: str,
    ) -> Tuple[dict, ...]:
        """Validate one tab's independently stored RF output editors."""
        if not isinstance(raw_entries, list) or len(raw_entries) > 8:
            raise ValueError(f"{label} must contain at most eight entries")
        decoded = []
        for index, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, dict):
                raise TypeError(f"each {label} entry must be a JSON object")
            defaults = dict(DEFAULT_RF_OUTPUT_SETTINGS)
            defaults["enabled"] = True
            defaults["gen_ch"] = DEFAULT_QSTL_RF_CHANNELS[index]
            entry = {**defaults, **raw_entry}
            enabled = self._json_bool(entry["enabled"], f"{label} enabled")
            require_within = self._json_bool(
                entry["require_within_segment"],
                f"{label} require_within_segment",
            )
            duration_sweep_enabled = self._json_bool(
                entry["duration_sweep_enabled"],
                f"{label} duration_sweep_enabled",
            )
            frequency_sweep_enabled = self._json_bool(
                entry["frequency_sweep_enabled"],
                f"{label} frequency_sweep_enabled",
            )
            power_sweep_enabled = self._json_bool(
                entry["power_sweep_enabled"],
                f"{label} power_sweep_enabled",
            )
            power_calibration_enabled = self._json_bool(
                entry["power_calibration_enabled"],
                f"{label} power_calibration_enabled",
            )
            power_calibration_database_path = str(
                entry["power_calibration_database_path"]
            )
            power_calibration_run_id = self._json_int(
                entry["power_calibration_run_id"],
                f"{label} power_calibration_run_id",
            )
            target_output_power_dbm = self._json_finite_float(
                entry["target_output_power_dbm"],
                f"{label} target_output_power_dbm",
            )
            qcs_power_calibration_enabled = self._json_bool(
                entry["qcs_power_calibration_enabled"],
                f"{label} qcs_power_calibration_enabled",
            )
            qcs_power_calibration_database_path = str(
                entry["qcs_power_calibration_database_path"]
            )
            qcs_power_calibration_run_id = self._json_int(
                entry["qcs_power_calibration_run_id"],
                f"{label} qcs_power_calibration_run_id",
            )
            qcs_target_output_power_dbm = self._json_finite_float(
                entry["qcs_target_output_power_dbm"],
                f"{label} qcs_target_output_power_dbm",
            )
            if (
                qcs_power_calibration_enabled
                and not qcs_power_calibration_database_path.strip()
            ):
                raise ValueError(
                    f"{label} QCS power-calibration database path is required"
                )
            spec = QickRfPulseSpec(
                gen_ch=self._json_int(entry["gen_ch"], f"{label} generator channel"),
                segment_name=_resolve_stored_set_segment_name(
                    entry["segment_name"],
                    set_names,
                    label=label,
                ),
                delay_us=float(entry["delay_us"]),
                duration_us=float(entry["duration_us"]),
                frequency_mhz=float(entry["frequency_mhz"]),
                gain=int(entry["gain"]),
                att1_db=float(entry["att1_db"]),
                att2_db=float(entry["att2_db"]),
                phase_degrees=float(entry["phase_degrees"]),
                nqz=int(entry["nqz"]),
                require_within_segment=require_within,
                filter_type=str(entry["filter_type"]),
                filter_cutoff=float(entry["filter_cutoff"]),
                filter_bandwidth=float(entry["filter_bandwidth"]),
                output_board_type=str(entry["output_board_type"]),
                duration_sweep_enabled=duration_sweep_enabled,
                duration_sweep_start_us=float(
                    entry["duration_sweep_start_us"]
                ),
                duration_sweep_stop_us=float(
                    entry["duration_sweep_stop_us"]
                ),
                duration_sweep_count=int(
                    entry["duration_sweep_count"]
                ),
                segment_length_mode=str(entry["segment_length_mode"]),
                frequency_sweep_enabled=frequency_sweep_enabled,
                frequency_sweep_start_mhz=float(
                    entry["frequency_sweep_start_mhz"]
                ),
                frequency_sweep_stop_mhz=float(
                    entry["frequency_sweep_stop_mhz"]
                ),
                frequency_sweep_count=int(
                    entry["frequency_sweep_count"]
                ),
                power_sweep_enabled=power_sweep_enabled,
                power_sweep_start_dbm=float(
                    entry["power_sweep_start_dbm"]
                ),
                power_sweep_stop_dbm=float(
                    entry["power_sweep_stop_dbm"]
                ),
                power_sweep_count=int(entry["power_sweep_count"]),
                power_calibration_enabled=power_calibration_enabled,
                power_calibration_database_path=(
                    power_calibration_database_path
                ),
                power_calibration_run_id=power_calibration_run_id,
                target_output_power_dbm=target_output_power_dbm,
            )
            if spec.segment_name not in set_names:
                raise ValueError(f"unknown {label} anchor {spec.segment_name!r}")
            decoded.append(
                {
                    "enabled": enabled,
                    "gen_ch": spec.gen_ch,
                    "segment_name": spec.segment_name,
                    "delay_us": spec.delay_us,
                    "duration_us": spec.duration_us,
                    "frequency_mhz": spec.frequency_mhz,
                    "gain": spec.gain,
                    "output_board_type": spec.output_board_type,
                    "att1_db": spec.att1_db,
                    "att2_db": spec.att2_db,
                    "filter_type": spec.filter_type,
                    "filter_cutoff": spec.filter_cutoff,
                    "filter_bandwidth": spec.filter_bandwidth,
                    "phase_degrees": spec.phase_degrees,
                    "nqz": spec.nqz,
                    "require_within_segment": spec.require_within_segment,
                    "duration_sweep_enabled": spec.duration_sweep_enabled,
                    "duration_sweep_start_us": spec.duration_sweep_start_us,
                    "duration_sweep_stop_us": spec.duration_sweep_stop_us,
                    "duration_sweep_count": spec.duration_sweep_count,
                    "segment_length_mode": spec.segment_length_mode,
                    "frequency_sweep_enabled": (
                        spec.frequency_sweep_enabled
                    ),
                    "frequency_sweep_start_mhz": (
                        spec.frequency_sweep_start_mhz
                    ),
                    "frequency_sweep_stop_mhz": (
                        spec.frequency_sweep_stop_mhz
                    ),
                    "frequency_sweep_count": spec.frequency_sweep_count,
                    "power_sweep_enabled": spec.power_sweep_enabled,
                    "power_sweep_start_dbm": spec.power_sweep_start_dbm,
                    "power_sweep_stop_dbm": spec.power_sweep_stop_dbm,
                    "power_sweep_count": spec.power_sweep_count,
                    "power_calibration_enabled": (
                        spec.power_calibration_enabled
                    ),
                    "power_calibration_database_path": (
                        spec.power_calibration_database_path
                    ),
                    "power_calibration_run_id": (
                        spec.power_calibration_run_id
                    ),
                    "target_output_power_dbm": (
                        spec.target_output_power_dbm
                    ),
                    "qcs_power_calibration_enabled": (
                        qcs_power_calibration_enabled
                    ),
                    "qcs_power_calibration_database_path": (
                        qcs_power_calibration_database_path
                    ),
                    "qcs_power_calibration_run_id": (
                        qcs_power_calibration_run_id
                    ),
                    "qcs_target_output_power_dbm": (
                        qcs_target_output_power_dbm
                    ),
                }
            )
        return tuple(decoded)

    def _decode_rf_readout_entry(
        self,
        raw_entry,
        *,
        set_names: set,
        label: str,
    ) -> dict:
        """Validate one tab's independently stored FIR DDR readout editor."""
        if raw_entry is None:
            raw_entry = {}
        if not isinstance(raw_entry, dict):
            raise TypeError(f"{label} must be a JSON object")
        representation_explicit = "measurement_representation" in raw_entry
        entry = {**DEFAULT_RF_READOUT_SETTINGS, **raw_entry}
        if not representation_explicit:
            entry["measurement_representation"] = (
                "current"
                if bool(entry.get("dc_measure_mode", False))
                else (
                    "voltage"
                    if bool(
                        entry.get(
                            "dc_voltage_calibration_enabled",
                            False,
                        )
                    )
                    else "adc"
                )
            )
        enabled = self._json_bool(entry["enabled"], f"{label} enabled")
        spec = QickDdrReadoutSpec(
            ro_ch=self._json_int(entry["ro_ch"], f"{label} channel"),
            segment_name=_resolve_stored_set_segment_name(
                entry["segment_name"],
                set_names,
                label=label,
            ),
            delay_us=float(entry["delay_us"]),
            samples_per_trigger=self._json_int(
                entry["samples_per_trigger"],
                f"{label} samples_per_trigger",
                minimum=1,
            ),
            readout_frequency_mhz=float(entry["readout_frequency_mhz"]),
            margin_input_samples=self._json_int(
                entry["margin_input_samples"],
                f"{label} margin_input_samples",
            ),
            fpga_trigger_delay_us=(
                None
                if entry.get("fpga_trigger_delay_us") is None
                else self._json_finite_float(
                    entry["fpga_trigger_delay_us"],
                    f"{label} fpga_trigger_delay_us",
                )
            ),
            force_overwrite=self._json_bool(
                entry["force_overwrite"],
                f"{label} force_overwrite",
            ),
            post_run_read_delay_seconds=self._json_finite_float(
                entry.get("post_run_read_delay_seconds", 0.1),
                f"{label} post_run_read_delay_seconds",
            ),
            attenuation_db=float(entry["attenuation_db"]),
            filter_type=str(entry["filter_type"]),
            filter_cutoff=float(entry["filter_cutoff"]),
            filter_bandwidth=float(entry["filter_bandwidth"]),
            input_board_type=str(entry["input_board_type"]),
            dc_gain_db=float(entry["dc_gain_db"]),
            dc_measure_mode=self._json_bool(
                entry["dc_measure_mode"],
                f"{label} dc_measure_mode",
            ),
            dc_measure_gain_v_per_a=self._json_finite_float(
                entry["dc_measure_gain_v_per_a"],
                f"{label} dc_measure_gain_v_per_a",
                positive=True,
            ),
            measurement_representation=str(
                entry.get("measurement_representation", "auto")
            ),
            dc_voltage_calibration_enabled=self._json_bool(
                entry.get("dc_voltage_calibration_enabled", False),
                f"{label} dc_voltage_calibration_enabled",
            ),
            dc_voltage_calibration_database_path=str(
                entry.get("dc_voltage_calibration_database_path", "")
            ),
            dc_voltage_calibration_run_id=self._json_int(
                entry.get("dc_voltage_calibration_run_id", 0),
                f"{label} dc_voltage_calibration_run_id",
            ),
            nqz=self._json_int(entry["nqz"], f"{label} nqz", minimum=1),
        )
        if spec.segment_name not in set_names:
            raise ValueError(f"unknown {label} anchor {spec.segment_name!r}")
        return {
            "enabled": enabled,
            "ro_ch": spec.ro_ch,
            "segment_name": spec.segment_name,
            "delay_us": spec.delay_us,
            "samples_per_trigger": spec.samples_per_trigger,
            "readout_frequency_mhz": spec.readout_frequency_mhz,
            "margin_input_samples": spec.margin_input_samples,
            "fpga_trigger_delay_us": spec.fpga_trigger_delay_us,
            "force_overwrite": spec.force_overwrite,
            "post_run_read_delay_seconds": spec.post_run_read_delay_seconds,
            "input_board_type": spec.input_board_type,
            "attenuation_db": spec.attenuation_db,
            "dc_gain_db": spec.dc_gain_db,
            "dc_measure_mode": spec.dc_measure_mode,
            "dc_measure_gain_v_per_a": spec.dc_measure_gain_v_per_a,
            "measurement_representation": (
                spec.effective_measurement_representation
            ),
            "dc_voltage_calibration_enabled": (
                spec.dc_voltage_calibration_enabled
            ),
            "dc_voltage_calibration_database_path": (
                spec.dc_voltage_calibration_database_path
            ),
            "dc_voltage_calibration_run_id": spec.dc_voltage_calibration_run_id,
            "filter_type": spec.filter_type,
            "filter_cutoff": spec.filter_cutoff,
            "filter_bandwidth": spec.filter_bandwidth,
            "nqz": spec.nqz,
        }

    def _decode_settings(self, data: dict) -> dict:
        """Validate a versioned settings document without changing the GUI."""
        if not isinstance(data, dict):
            raise TypeError("settings JSON root must be an object")
        if data.get("schema") != SETTINGS_SCHEMA:
            raise ValueError(f"unsupported settings schema {data.get('schema')!r}")
        settings_version = data.get("version", 1)
        if settings_version not in SUPPORTED_SETTINGS_VERSIONS:
            supported = ", ".join(map(str, SUPPORTED_SETTINGS_VERSIONS))
            raise ValueError(
                f"unsupported settings version {settings_version!r}; "
                f"expected one of {supported}"
            )

        display = data.get("display", {})
        grid = data.get("grid", {})
        awg = data.get("awg")
        qick = data.get("qick", {})
        qcs = data.get("qcs", {})
        if qcs is None:
            qcs = {}
        if not all(
            isinstance(item, dict) for item in (display, grid, awg, qick, qcs)
        ):
            raise TypeError(
                "display, grid, awg, qick, and qcs must be JSON objects"
            )

        time_unit = str(display.get("time_unit", DEFAULT_TIME_UNIT))
        if time_unit not in TIME_UNIT_NS:
            raise ValueError(f"unsupported time unit {time_unit!r}")
        voltage_view = str(display.get("voltage_view", "both"))
        if voltage_view not in {"both", "virtual", "physical"}:
            raise ValueError(f"unsupported voltage view {voltage_view!r}")
        raw_trace_overlay = display.get("trace_stability_overlay", {})
        if raw_trace_overlay is None:
            raw_trace_overlay = {}
        if not isinstance(raw_trace_overlay, dict):
            raise TypeError(
                "display trace_stability_overlay must be a JSON object"
            )
        trace_overlay_mode = str(
            raw_trace_overlay.get("mode", "latest")
        ).strip().lower()
        if trace_overlay_mode not in {"latest", "saved"}:
            raise ValueError(
                "Trace Plot Stability overlay mode must be latest or saved"
            )
        trace_overlay_database_path = str(
            raw_trace_overlay.get(
                "database_path",
                DEFAULT_STABILITY_DB_PATH,
            )
        ).strip()
        if not trace_overlay_database_path:
            raise ValueError(
                "Trace Plot Stability overlay database path must not be empty"
            )
        trace_overlay_run_id = self._json_int(
            raw_trace_overlay.get("run_id", 0),
            "Trace Plot Stability overlay run_id",
        )
        trace_overlay_quantity = str(
            raw_trace_overlay.get("quantity", "magnitude")
        ).strip().lower()
        if trace_overlay_quantity not in {"i", "q", "magnitude", "phase"}:
            raise ValueError(
                "Trace Plot Stability overlay quantity must be "
                "i, q, magnitude, or phase"
            )
        if trace_overlay_mode == "saved" and trace_overlay_run_id < 1:
            raise ValueError(
                "a saved Trace Plot Stability overlay requires run_id >= 1"
            )

        raw_outputs = awg.get("outputs")
        if not isinstance(raw_outputs, list) or not 1 <= len(raw_outputs) <= 8:
            raise ValueError("AWG outputs must contain between one and eight waveforms")
        pulses = tuple(PulseSequence.from_dict(entry) for entry in raw_outputs)
        raw_output_names = awg.get(
            "output_names",
            [f"awg_{index}" for index in range(len(pulses))],
        )
        if (
            not isinstance(raw_output_names, list)
            or len(raw_output_names) != len(pulses)
        ):
            raise ValueError(
                "AWG output_names must contain one name per waveform output"
            )
        awg_output_names = tuple(
            str(name).strip() for name in raw_output_names
        )
        if any(not name for name in awg_output_names):
            raise ValueError("AWG output names must not be empty")
        if any(len(name) > 128 for name in awg_output_names):
            raise ValueError("AWG output names must not exceed 128 characters")
        if any(
            any(ord(character) < 32 for character in name)
            for name in awg_output_names
        ):
            raise ValueError("AWG output names must not contain control characters")
        if len(set(awg_output_names)) != len(awg_output_names):
            raise ValueError("AWG output names must be unique")
        selected_output = self._json_int(
            display.get("selected_awg_output", 0),
            "selected AWG output",
        )
        if selected_output >= len(pulses):
            raise ValueError("selected AWG output is out of range")
        stored_selected_tab = self._json_int(
            display.get("selected_control_tab", 0),
            "selected control tab",
        )
        if settings_version < 12:
            if stored_selected_tab <= 3:
                selected_tab = 0
                selected_awg_tuning_tab = stored_selected_tab
            elif stored_selected_tab in {4, 6}:
                # Version 10 briefly stored the former full-size Front Panel
                # tab at index 6; restore that view to RF S-Parameter.
                selected_tab = self._control_tabs.indexOf(self._sparameter_panel)
                selected_awg_tuning_tab = 0
            elif stored_selected_tab == 5:
                selected_tab = self._control_tabs.indexOf(self._calibration_panel)
                selected_awg_tuning_tab = 0
            else:
                raise ValueError("selected control tab is out of range")
        elif settings_version == 12:
            selected_awg_tuning_tab = self._json_int(
                display.get("selected_awg_tuning_tab", 0),
                "selected AWG Tuning tab",
            )
            if stored_selected_tab == 0:
                selected_tab = self._control_tabs.indexOf(self._awg_tuning_page)
            elif stored_selected_tab == 1:
                selected_tab = self._control_tabs.indexOf(self._sparameter_panel)
            elif stored_selected_tab == 2:
                selected_tab = self._control_tabs.indexOf(self._calibration_panel)
            else:
                raise ValueError("selected control tab is out of range")
        else:
            selected_tab = stored_selected_tab
            selected_awg_tuning_tab = self._json_int(
                display.get("selected_awg_tuning_tab", 0),
                "selected AWG Tuning tab",
            )
        if selected_tab >= self._control_tabs.count():
            raise ValueError("selected control tab is out of range")
        if selected_awg_tuning_tab >= self._awg_tuning_tabs.count():
            raise ValueError("selected AWG Tuning tab is out of range")

        expected_shape = (len(pulses), len(pulses))
        if "cross_capacitance" in awg:
            matrix = np.asarray(awg["cross_capacitance"], dtype=float)
        else:
            matrix = np.eye(len(pulses), dtype=float)
        if matrix.shape != expected_shape or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"cross-capacitance matrix must be a finite {expected_shape} array"
            )
        if not np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("cross-capacitance diagonal entries must equal 1")

        raw_sweeps = awg.get("sweeps", [])
        if not isinstance(raw_sweeps, list):
            raise TypeError("AWG sweeps must be a JSON array")
        decoded_sweeps = []
        for entry in raw_sweeps:
            if not isinstance(entry, dict):
                raise TypeError("each AWG sweep must be a JSON object")
            axis_kind = str(entry.get("axis_kind", "amplitude"))
            if axis_kind == "ramp_duration":
                decoded_sweeps.append(QickRampRateSweepSpec(
                    segment_name=str(entry["segment_name"]),
                    start=float(entry["start"]),
                    stop=float(entry["stop"]),
                    count=self._json_int(
                        entry["count"],
                        "RAMP duration sweep count",
                        minimum=1,
                    ),
                ))
            elif axis_kind == "hold_duration":
                decoded_sweeps.append(QickHoldDurationSweepSpec(
                    segment_name=str(entry["segment_name"]),
                    start=float(entry["start"]),
                    stop=float(entry["stop"]),
                    count=self._json_int(
                        entry["count"],
                        "SET hold sweep count",
                        minimum=1,
                    ),
                ))
            elif axis_kind in {"amplitude", "voltage"}:
                decoded_sweeps.append(QickSweepSpec(
                    segment_name=str(entry["segment_name"]),
                    output_name=str(entry["output_name"]),
                    start=float(entry["start"]),
                    stop=float(entry["stop"]),
                    count=self._json_int(
                        entry["count"],
                        "sweep count",
                        minimum=1,
                    ),
                ))
            else:
                raise ValueError(f"unknown AWG sweep axis_kind {axis_kind!r}")
        ramp_sweeps = [
            spec
            for spec in decoded_sweeps
            if isinstance(spec, QickRampRateSweepSpec)
        ]
        ramp_targets = [spec.segment_name for spec in ramp_sweeps]
        if len(set(ramp_targets)) != len(ramp_targets):
            raise ValueError(
                "each RAMP segment may have only one duration/rate sweep"
            )
        hold_sweeps = [
            spec
            for spec in decoded_sweeps
            if isinstance(spec, QickHoldDurationSweepSpec)
        ]
        hold_targets = [spec.segment_name for spec in hold_sweeps]
        if len(set(hold_targets)) != len(hold_targets):
            raise ValueError(
                "each SET segment may have only one hold-duration sweep"
            )
        sweeps = (
            tuple(ramp_sweeps)
            + tuple(hold_sweeps)
            + tuple(
                spec
                for spec in decoded_sweeps
                if isinstance(spec, QickSweepSpec)
            )
        )
        sweep_targets = set()
        for spec in sweeps:
            if isinstance(spec, QickRampRateSweepSpec):
                ramp_row = self._ramp_sweep_row(spec)
                if ramp_row is None:
                    raise ValueError(
                        f"invalid RAMP sweep segment {spec.segment_name!r}"
                    )
                if any(ramp_row >= pulse.set_count for pulse in pulses):
                    raise ValueError(
                        f"unknown RAMP sweep segment {spec.segment_name!r}"
                    )
                continue
            if isinstance(spec, QickHoldDurationSweepSpec):
                hold_row = self._hold_sweep_row(spec)
                if hold_row is None:
                    raise ValueError(
                        f"invalid SET hold sweep segment "
                        f"{spec.segment_name!r}"
                    )
                if any(hold_row >= pulse.set_count for pulse in pulses):
                    raise ValueError(
                        f"unknown SET hold sweep segment "
                        f"{spec.segment_name!r}"
                    )
                continue
            target = (spec.output_name, spec.segment_name)
            if target in sweep_targets:
                raise ValueError("each AWG output/SET may have only one sweep")
            sweep_targets.add(target)
            try:
                output_index = int(spec.output_name.rsplit("_", 1)[1])
                segment_index = int(spec.segment_name.rsplit("_", 1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid sweep target {target!r}") from exc
            if spec.output_name != f"awg_{output_index}":
                raise ValueError(f"invalid sweep output {spec.output_name!r}")
            if not 0 <= output_index < len(pulses):
                raise ValueError(f"unknown sweep output {spec.output_name!r}")
            if not 0 <= segment_index < pulses[output_index].set_count:
                raise ValueError(f"unknown sweep SET {spec.segment_name!r}")

        fabric_mhz = self._json_finite_float(
            qick.get("fabric_mhz", DEFAULT_QICK_FABRIC_MHZ),
            "QICK fabric_mhz",
            positive=True,
        )
        tproc_mhz = self._json_finite_float(
            qick.get("tproc_mhz", DEFAULT_QICK_TPROC_MHZ),
            "QICK tproc_mhz",
            positive=True,
        )
        full_scale_mv = self._json_finite_float(
            qick.get("full_scale_mv", DEFAULT_QICK_FULL_SCALE_MV),
            "QICK full_scale_mv",
            positive=True,
        )
        awg_metadata_mode = normalize_awg_metadata_mode(
            qick.get("awg_metadata_mode", DEFAULT_AWG_METADATA_MODE)
        )
        compile_validation_mode = normalize_compile_validation_mode(
            qick.get(
                "compile_validation_mode",
                DEFAULT_GUI_COMPILE_VALIDATION_MODE,
            )
        )
        raw_stability_settings = data.get("stability_diagram")
        if raw_stability_settings is None:
            raw_stability_settings = {}
        elif isinstance(raw_stability_settings, dict):
            raw_stability_settings = dict(raw_stability_settings)
        if (
            isinstance(raw_stability_settings, dict)
            and "qcs_power_calibration_enabled"
            not in raw_stability_settings
        ):
            # Settings written before Stability had its own calibrated-power
            # controls already contain the Calibration tab's database.  If
            # that database has an M5300 record, adopt it instead of silently
            # retaining the old 20000/32767 relative-amplitude behavior.
            raw_calibration_settings = data.get("calibration", {})
            calibration_database_path = (
                str(raw_calibration_settings.get("database_path", "")).strip()
                if isinstance(raw_calibration_settings, dict)
                else ""
            )
            if _qcs_rf_calibration_database_has_records(
                calibration_database_path
            ):
                raw_stability_settings.update(
                    {
                        "qcs_power_calibration_enabled": True,
                        "qcs_power_calibration_database_path": (
                            calibration_database_path
                        ),
                        "qcs_power_calibration_run_id": 0,
                    }
                )
        stability_settings = normalize_stability_settings(
            raw_stability_settings,
            output_names=tuple(f"awg_{index}" for index in range(len(pulses))),
        )
        noise_analysis_settings = normalize_noise_analysis_settings(
            data.get("noise_analysis")
        )
        raw_bias_t = qick.get("bias_t_compensation", {})
        if raw_bias_t is None:
            raw_bias_t = {}
        if not isinstance(raw_bias_t, dict):
            raise TypeError("QICK bias_t_compensation must be a JSON object")
        bias_t_enabled = self._json_bool(
            raw_bias_t.get("enabled", False),
            "QICK Bias-T compensation enabled",
        )
        bias_t_type = str(raw_bias_t.get("type", "dc"))
        if bias_t_type not in BIAS_T_COMPENSATION_TYPES:
            raise ValueError(
                "QICK Bias-T compensation type must be one of "
                f"{BIAS_T_COMPENSATION_TYPES}"
            )
        bias_t_mode = str(raw_bias_t.get("mode", "fixed_voltage"))
        if bias_t_mode not in BIAS_T_COMPENSATION_MODES:
            raise ValueError(
                f"QICK Bias-T compensation mode must be one of "
                f"{BIAS_T_COMPENSATION_MODES}"
            )
        bias_t_compensation_mv = self._json_finite_float(
            raw_bias_t.get(
                "voltage_mv",
                full_scale_mv * DEFAULT_BIAS_T_COMPENSATION_FRACTION,
            ),
            "QICK Bias-T compensation voltage_mv",
            positive=True,
        )
        if bias_t_compensation_mv > full_scale_mv:
            raise ValueError(
                "QICK Bias-T compensation voltage exceeds full_scale_mv"
            )
        bias_t_duration_us = self._json_finite_float(
            raw_bias_t.get(
                "duration_us",
                DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
            ),
            "QICK Bias-T compensation duration_us",
            positive=True,
        )
        bias_t_filter_tau_us = self._json_finite_float(
            raw_bias_t.get("filter_tau_us", DEFAULT_BIAS_T_FILTER_TAU_US),
            "QICK Bias-T filter_tau_us",
            positive=True,
        )
        repetitions = self._json_int(
            qick.get("repetitions_per_sweep", 1),
            "QICK repetitions_per_sweep",
            minimum=1,
        )
        raw_channels = qick.get(
            "awg_channels", list(DEFAULT_QSTL_AWG_CHANNELS[:len(pulses)])
        )
        if not isinstance(raw_channels, list) or len(raw_channels) != len(pulses):
            raise ValueError("QICK awg_channels must match the AWG output count")
        awg_channels = tuple(
            self._json_int(channel, "QICK AWG channel") for channel in raw_channels
        )
        if len(set(awg_channels)) != len(awg_channels):
            raise ValueError("QICK AWG channels must be unique")

        experiment = data.get("experiment", {})
        if not isinstance(experiment, dict):
            raise TypeError("experiment must be a JSON object")
        execution_backend = str(
            experiment.get(
                "execution_backend",
                (
                    EXECUTION_BACKEND_QICK
                    if settings_version < 35
                    else DEFAULT_EXECUTION_BACKEND
                ),
            )
        ).strip().lower()
        if execution_backend not in EXECUTION_BACKENDS:
            raise ValueError(
                f"unsupported execution backend {execution_backend!r}"
            )
        qcs_mapper_path = str(
            qcs.get("mapper_path", DEFAULT_QCS_MAPPER_PATH)
        ).strip()
        if execution_backend == EXECUTION_BACKEND_QCS and not qcs_mapper_path:
            raise ValueError(
                "QCS mapper_path must not be empty when QCS is selected"
            )
        raw_qcs_dc_names = qcs.get(
            "dc_channel_names",
            [f"dc_ch_{index + 1}" for index in range(len(pulses))],
        )
        if not isinstance(raw_qcs_dc_names, list):
            raise TypeError("QCS dc_channel_names must be an array")
        qcs_dc_channel_names = tuple(
            str(name).strip() for name in raw_qcs_dc_names
        )
        if any(not name for name in qcs_dc_channel_names):
            raise ValueError(
                "QCS dc_channel_names entries must be nonempty"
            )
        if len(set(qcs_dc_channel_names)) != len(qcs_dc_channel_names):
            raise ValueError("QCS dc_channel_names must be unique")
        if (
            execution_backend == EXECUTION_BACKEND_QCS
            and len(qcs_dc_channel_names) != len(pulses)
        ):
            raise ValueError(
                "QCS dc_channel_names must contain one name per AWG output"
            )
        raw_qcs_rf_names = qcs.get("rf_channel_names", {})
        if not isinstance(raw_qcs_rf_names, dict):
            raise TypeError("QCS rf_channel_names must be an object")
        qcs_rf_channel_names = {}
        for raw_channel, raw_name in raw_qcs_rf_names.items():
            try:
                channel = int(raw_channel)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "QCS rf_channel_names keys must be generator integers"
                ) from exc
            name = str(raw_name).strip()
            if channel < 0 or not name:
                raise ValueError(
                    "QCS RF mappings require nonnegative generators and "
                    "nonempty virtual channel names"
                )
            if channel in qcs_rf_channel_names:
                raise ValueError(
                    f"QCS RF generator {channel} is mapped more than once"
                )
            qcs_rf_channel_names[channel] = name
        if len(set(qcs_rf_channel_names.values())) != len(qcs_rf_channel_names):
            raise ValueError("QCS RF virtual channel names must be unique")
        raw_acquisition_name = qcs.get("acquisition_channel_name")
        qcs_acquisition_channel_name = (
            None
            if raw_acquisition_name is None
            else str(raw_acquisition_name).strip() or None
        )
        qcs_init_time_s = self._json_finite_float(
            qcs.get("init_time_s", DEFAULT_QCS_INIT_TIME_US * 1.0e-6),
            "QCS init_time_s",
        )
        if qcs_init_time_s < 0.0:
            raise ValueError("QCS init_time_s must be nonnegative")
        if settings_version < 39 and np.isclose(
            qcs_init_time_s,
            100.0e-6,
            rtol=0.0,
            atol=1.0e-15,
        ):
            # This was the old QCS library default, exposed under the
            # misleading label "Backend initialization time". It is really
            # paid between every sweep point/repetition, so migrate only that
            # exact inherited value to the validated video-mode default.
            qcs_init_time_s = DEFAULT_QCS_INIT_TIME_US * 1.0e-6
        qcs_sample_rate_hz = self._json_finite_float(
            qcs.get("sample_rate_hz", DEFAULT_QCS_SAMPLE_RATE_HZ),
            "QCS sample_rate_hz",
        )
        if qcs_sample_rate_hz <= 0.0:
            raise ValueError("QCS sample_rate_hz must be positive")
        # Keep accepting the legacy serialized field, but never use it as a
        # configurable QCS clock. The M5200 rate is defined by QCS/hardware.
        qcs_sample_rate_hz = DEFAULT_QCS_SAMPLE_RATE_HZ
        qcs_dc_full_scale_v = self._json_finite_float(
            qcs.get("dc_full_scale_v", DEFAULT_QCS_FULL_SCALE_V),
            "QCS dc_full_scale_v",
        )
        if qcs_dc_full_scale_v <= 0.0:
            raise ValueError("QCS dc_full_scale_v must be positive")
        sweep_voltage_scale_mv = self._json_finite_float(
            awg.get(
                "voltage_coordinate_full_scale_mv",
                # Versions through 39 always normalized AWG voltage sweeps
                # with the QICK field, even when QCS was selected.
                full_scale_mv,
            ),
            "AWG voltage_coordinate_full_scale_mv",
            positive=True,
        )
        stability_full_scale_mv = (
            qcs_dc_full_scale_v * 1000.0
            if execution_backend == EXECUTION_BACKEND_QCS
            else full_scale_mv
        )
        for axis_name in ("x_axis", "y_axis"):
            axis = stability_settings[axis_name]
            if (
                max(abs(axis["start_mv"]), abs(axis["stop_mv"]))
                > stability_full_scale_mv
            ):
                raise ValueError(
                    f"{axis_name} stability sweep exceeds "
                    f"+/-{stability_full_scale_mv:g} mV "
                    f"{execution_backend.upper()} AWG full scale"
                )
        raw_qcs_hardware_configuration = qcs.get("hardware_configuration")
        qcs_mapper_sha256 = normalize_qcs_mapper_sha256(
            qcs.get("hardware_mapper_sha256")
        )
        raw_qcs_hardware_state = str(
            qcs.get(
                "hardware_configuration_state",
                (
                    QCS_HARDWARE_STATE_EXTERNAL
                    if raw_qcs_hardware_configuration is None
                    else QCS_HARDWARE_STATE_DRAFT
                ),
            )
        )
        if raw_qcs_hardware_state not in QCS_HARDWARE_STATES:
            raise ValueError(
                "QCS hardware_configuration_state must be one of "
                f"{QCS_HARDWARE_STATES}"
            )
        qcs_hardware_configuration = (
            None
            if raw_qcs_hardware_configuration is None
            else normalize_qcs_hardware_configuration(
                raw_qcs_hardware_configuration,
                required_dc_count=len(pulses),
            )
        )
        if qcs_hardware_configuration is None:
            if raw_qcs_hardware_state != QCS_HARDWARE_STATE_EXTERNAL:
                raise ValueError(
                    "QCS hardware_configuration_state must be external when "
                    "hardware_configuration is null"
                )
            qcs_hardware_state = QCS_HARDWARE_STATE_EXTERNAL
            qcs_mapper_sha256 = None
        else:
            qcs_hardware_state = (
                QCS_HARDWARE_STATE_DRAFT
                if raw_qcs_hardware_state == QCS_HARDWARE_STATE_EXTERNAL
                else raw_qcs_hardware_state
            )
            bound_dc, bound_rf, bound_acquisition = qcs_role_bindings(
                qcs_hardware_configuration,
                required_dc_count=len(pulses),
            )
            expected_rf = {
                str(channel): name
                for channel, name in qcs_rf_channel_names.items()
            }
            if (
                bound_dc != list(qcs_dc_channel_names)
                or bound_rf != expected_rf
                or bound_acquisition != qcs_acquisition_channel_name
            ):
                raise ValueError(
                    "QCS hardware_configuration role bindings must exactly "
                    "match dc_channel_names, rf_channel_names, and "
                    "acquisition_channel_name"
                )
            if (
                qcs_hardware_state == QCS_HARDWARE_STATE_SAVED
                and qcs_mapper_sha256 is None
            ):
                qcs_hardware_state = QCS_HARDWARE_STATE_DRAFT
            elif (
                qcs_hardware_state == QCS_HARDWARE_STATE_IMPORTED
                and qcs_mapper_sha256 is None
            ):
                qcs_hardware_state = QCS_HARDWARE_STATE_IMPORTED_DIRTY
            if qcs_hardware_state not in {
                QCS_HARDWARE_STATE_SAVED,
                QCS_HARDWARE_STATE_IMPORTED,
            }:
                qcs_mapper_sha256 = None

        raw_qcs_dc_names_text = qcs.get("dc_channel_names_text")
        raw_qcs_rf_names_text = qcs.get("rf_channel_names_text")
        raw_qcs_acquisition_name_text = qcs.get(
            "acquisition_channel_name_text"
        )
        if (
            raw_qcs_dc_names_text is not None
            and not isinstance(raw_qcs_dc_names_text, str)
        ):
            raise TypeError("QCS dc_channel_names_text must be a string")
        if (
            raw_qcs_rf_names_text is not None
            and not isinstance(raw_qcs_rf_names_text, str)
        ):
            raise TypeError("QCS rf_channel_names_text must be a string")
        if (
            raw_qcs_acquisition_name_text is not None
            and not isinstance(raw_qcs_acquisition_name_text, str)
        ):
            raise TypeError(
                "QCS acquisition_channel_name_text must be a string"
            )
        qcs_settings = {
            "mapper_path": qcs_mapper_path,
            "dc_channel_names": list(qcs_dc_channel_names),
            "dc_full_scale_v": qcs_dc_full_scale_v,
            "rf_channel_names": {
                str(channel): name
                for channel, name in qcs_rf_channel_names.items()
            },
            "acquisition_channel_name": qcs_acquisition_channel_name,
            "hw_demod": self._json_bool(
                qcs.get("hw_demod", True),
                "QCS hw_demod",
            ),
            "sample_rate_hz": qcs_sample_rate_hz,
            "init_time_s": qcs_init_time_s,
            "blocking": self._json_bool(
                qcs.get("blocking", True),
                "QCS blocking",
            ),
            "hardware_configuration": qcs_hardware_configuration,
            "hardware_configuration_state": qcs_hardware_state,
            "hardware_mapper_sha256": qcs_mapper_sha256,
        }
        if raw_qcs_dc_names_text is not None:
            qcs_settings["dc_channel_names_text"] = raw_qcs_dc_names_text
        if raw_qcs_rf_names_text is not None:
            qcs_settings["rf_channel_names_text"] = raw_qcs_rf_names_text
        if raw_qcs_acquisition_name_text is not None:
            qcs_settings["acquisition_channel_name_text"] = (
                raw_qcs_acquisition_name_text
            )
        raw_sweep_map = experiment.get("sweep_map", {})
        if raw_sweep_map is None:
            raw_sweep_map = {}
        if not isinstance(raw_sweep_map, dict):
            raise TypeError("experiment sweep_map must be a JSON object")
        sweep_map_color_ranges = normalize_awg_sweep_color_ranges(
            raw_sweep_map.get("color_ranges")
        )
        sweep_map_visible_data = normalize_awg_sweep_visible_data(
            raw_sweep_map.get("visible_data")
        )
        def decode_sweep_map_settings(available_sweep_axes):
            """Restore display axes after every AWG and RF sweep is decoded."""
            available_sweep_axes = tuple(dict.fromkeys(available_sweep_axes))

            def decode_axis(name: str):
                raw_axis = raw_sweep_map.get(name)
                if raw_axis is None:
                    return None
                if not isinstance(raw_axis, dict):
                    raise TypeError(
                        f"experiment sweep_map {name} must be an object"
                    )
                key = (
                    str(raw_axis.get("output_name", "")),
                    str(raw_axis.get("segment_name", "")),
                )
                # Axis selection is a display preference. A removed sweep must
                # not make the complete experiment settings document unloadable.
                return key if key in available_sweep_axes else None

            sweep_map_x = decode_axis("x_axis")
            sweep_map_y = decode_axis("y_axis")
            if len(available_sweep_axes) >= 2:
                if sweep_map_x is None:
                    sweep_map_x = available_sweep_axes[0]
                if sweep_map_y is None or sweep_map_y == sweep_map_x:
                    sweep_map_y = next(
                        key for key in available_sweep_axes if key != sweep_map_x
                    )
                sweep_map_axes = (sweep_map_x, sweep_map_y)
            else:
                sweep_map_axes = None

            raw_slice_axes = raw_sweep_map.get("slice_axes", [])
            if not isinstance(raw_slice_axes, list):
                raise TypeError("experiment sweep_map slice_axes must be an array")
            sweep_map_slices = []
            seen_slice_axes = set()
            for entry in raw_slice_axes:
                if not isinstance(entry, dict):
                    raise TypeError(
                        "each experiment sweep_map slice axis must be an object"
                    )
                key = (
                    str(entry.get("output_name", "")),
                    str(entry.get("segment_name", "")),
                )
                # Ignore stale visualization entries left by a removed AWG/RF
                # sweep. Active RF axes are included in available_sweep_axes.
                if key not in available_sweep_axes:
                    continue
                if key in seen_slice_axes:
                    raise ValueError(
                        f"experiment sweep_map slice axis {key!r} is duplicated"
                    )
                if sweep_map_axes is not None and key in sweep_map_axes:
                    continue
                seen_slice_axes.add(key)
                mode = str(entry.get("mode", "average")).strip().lower()
                decoded_slice = {
                    "output_name": key[0],
                    "segment_name": key[1],
                    "mode": mode,
                }
                if mode == "value":
                    value = float(entry["value"])
                    if not np.isfinite(value):
                        raise ValueError(
                            "experiment sweep_map slice value must be finite"
                        )
                    decoded_slice["value"] = value
                elif mode != "average":
                    raise ValueError(
                        "experiment sweep_map slice mode must be average or value"
                    )
                sweep_map_slices.append(decoded_slice)
            return sweep_map_axes, sweep_map_slices
        connection_config = QickConnectionConfig(
            host=str(experiment.get("qick_host", DEFAULT_QICK_HOST)),
            ns_port=self._json_int(
                experiment.get("ns_port", DEFAULT_QICK_NS_PORT),
                "QICK nameserver port",
                minimum=1,
            ),
            proxy_name=str(
                experiment.get("proxy_name", DEFAULT_QICK_PROXY_NAME)
            ),
        )
        run_config = QcodesRunConfig(
            database_path=str(
                experiment.get("database_path", DEFAULT_QCODES_DB_PATH)
            ),
            experiment_name=str(
                experiment.get("experiment_name", "QICK pulse experiment")
            ),
            sample_name=str(experiment.get("sample_name", "PulseGenerator")),
            notes=str(experiment.get("notes", "")),
        )
        iq_repetition_policy = normalize_iq_repetition_policy(
            experiment.get(
                "iq_repetition_policy",
                IQ_REPETITION_POLICY_PRESERVE,
            )
        )

        raw_sparameter = data.get("s_parameter", {})
        if raw_sparameter is None:
            raw_sparameter = {}
        if not isinstance(raw_sparameter, dict):
            raise TypeError("s_parameter must be a JSON object")
        sparameter_settings = {
            **DEFAULT_SPARAMETER_SETTINGS,
            **raw_sparameter,
        }
        sparameter_database_path = str(
            sparameter_settings.pop("database_path")
        ).strip()
        if not sparameter_database_path:
            raise ValueError("RF S-parameter database path must not be empty")
        sparameter_qcs_amplitude = self._json_finite_float(
            sparameter_settings.pop("qcs_amplitude", 0.005),
            "QCS S-parameter amplitude",
        )
        if not -1.0 <= sparameter_qcs_amplitude <= 1.0:
            raise ValueError("QCS S-parameter amplitude must be in [-1, 1]")
        sparameter_config = SParameterSweepConfig(**sparameter_settings)

        raw_calibration = data.get("calibration", {})
        if raw_calibration is None:
            raw_calibration = {}
        if not isinstance(raw_calibration, dict):
            raise TypeError("calibration must be a JSON object")
        calibration_defaults = default_calibration_settings()
        calibration_database_path = str(
            raw_calibration.get(
                "database_path",
                calibration_defaults["database_path"],
            )
        ).strip()
        if not calibration_database_path:
            raise ValueError("calibration database path must not be empty")
        raw_output_calibration = raw_calibration.get("output", {})
        raw_input_calibration = raw_calibration.get("input", {})
        raw_dc_voltage_calibration = raw_calibration.get("dc_voltage", {})
        dc_application_explicit = "dc_voltage_application" in raw_calibration
        raw_dc_application = raw_calibration.get(
            "dc_voltage_application",
            calibration_defaults["dc_voltage_application"],
        )
        if not isinstance(raw_output_calibration, dict) or not isinstance(
            raw_input_calibration,
            dict,
        ):
            raise TypeError("calibration output and input must be JSON objects")
        if not isinstance(raw_dc_voltage_calibration, dict):
            raise TypeError("calibration dc_voltage must be a JSON object")
        if not isinstance(raw_dc_application, dict):
            raise TypeError(
                "calibration dc_voltage_application must be a JSON object"
            )
        output_calibration_values = {
            **calibration_defaults["output"],
            **raw_output_calibration,
        }
        scope_values = {
            **calibration_defaults["output"]["oscilloscope"],
            **dict(output_calibration_values.pop("oscilloscope", {})),
        }
        output_calibration = OutputPowerCalibrationConfig(
            database_path=calibration_database_path,
            oscilloscope=OscilloscopeConfig(**scope_values),
            **output_calibration_values,
        )
        input_calibration = InputPowerCalibrationConfig(
            database_path=calibration_database_path,
            **{
                **calibration_defaults["input"],
                **raw_input_calibration,
            },
        )
        dc_voltage_calibration = DcVoltageCalibrationConfig(
            database_path=calibration_database_path,
            **{
                **calibration_defaults["dc_voltage"],
                **raw_dc_voltage_calibration,
            },
        )
        calibration_paths = normalize_calibration_paths(
            raw_calibration,
            output_calibration,
            input_calibration,
            dc_voltage_calibration,
        )
        calibration_selected_tab = self._json_int(
            raw_calibration.get(
                "selected_tab",
                calibration_defaults["selected_tab"],
            ),
            "calibration selected tab",
        )
        if calibration_selected_tab > 4:
            raise ValueError(
                "calibration selected tab must be 0, 1, 2, 3, or 4"
            )
        raw_input_calibration_plot = raw_calibration.get("input_plot", {})
        if not isinstance(raw_input_calibration_plot, dict):
            raise TypeError("calibration input_plot must be a JSON object")
        input_calibration_plot = {
            **calibration_defaults["input_plot"],
            **raw_input_calibration_plot,
        }
        for axis_name in ("x_scale", "y_scale"):
            axis_scale = str(input_calibration_plot[axis_name]).lower()
            if axis_scale not in ("linear", "log"):
                raise ValueError(
                    f"calibration input plot {axis_name} must be linear or log"
                )
            input_calibration_plot[axis_name] = axis_scale
        dc_application_enabled = self._json_bool(
            raw_dc_application.get("enabled", False),
            "DC voltage calibration application enabled",
        )
        dc_application_database_path = str(
            raw_dc_application.get(
                "database_path",
                calibration_database_path,
            )
        ).strip()
        if dc_application_enabled and not dc_application_database_path:
            raise ValueError(
                "DC voltage calibration application database path must not be empty"
            )
        dc_application_run_id = self._json_int(
            raw_dc_application.get("run_id", 0),
            "DC voltage calibration application Run ID",
        )
        raw_qcs_rf_calibration = raw_calibration.get("qcs_rf_output", {})
        if not isinstance(raw_qcs_rf_calibration, dict):
            raise TypeError(
                "calibration qcs_rf_output must be a JSON object"
            )
        qcs_rf_calibration = {
            **calibration_defaults["qcs_rf_output"],
            **raw_qcs_rf_calibration,
        }
        qcs_rf_calibration["output_endpoint"] = (
            normalize_qcs_calibration_endpoint_settings(
                qcs_rf_calibration.get("output_endpoint"),
                label="QCS RF calibration output_endpoint",
                allow_lo_frequency=True,
            )
        )
        qcs_rf_calibration["input_endpoint"] = (
            normalize_qcs_calibration_endpoint_settings(
                qcs_rf_calibration.get("input_endpoint"),
                label="QCS RF calibration input_endpoint",
                allow_lo_frequency=False,
            )
        )
        for key in ("output_logical_index", "input_logical_index"):
            qcs_rf_calibration[key] = self._json_int(
                qcs_rf_calibration[key],
                f"QCS RF calibration {key}",
            )
        for key in (
            "frequency_points",
            "amplitude_points",
            "repetitions",
        ):
            qcs_rf_calibration[key] = self._json_int(
                qcs_rf_calibration[key],
                f"QCS RF calibration {key}",
                minimum=2 if key != "repetitions" else 1,
            )
        for key in (
            "frequency_start_mhz",
            "frequency_end_mhz",
            "amplitude_start",
            "amplitude_end",
            "integration_duration_us",
        ):
            qcs_rf_calibration[key] = self._json_finite_float(
                qcs_rf_calibration[key],
                f"QCS RF calibration {key}",
            )
        if not (
            0.0 < qcs_rf_calibration["amplitude_start"]
            <= qcs_rf_calibration["amplitude_end"]
            <= 1.0
        ):
            raise ValueError(
                "QCS RF calibration amplitudes must satisfy "
                "0 < start <= stop <= 1"
            )
        if qcs_rf_calibration["integration_duration_us"] <= 0.0:
            raise ValueError(
                "QCS RF calibration integration_duration_us must be positive"
            )
        # Versions through v22 exposed user-supplied I/Q-to-dBm slope,
        # intercept, and nominal scale controls. QCS M5200 results are already
        # voltage-scaled, so these legacy settings are intentionally ignored
        # and removed during normalization.
        for legacy_key in (
            "reference_mode",
            "reference_slope",
            "reference_intercept_dbm",
            "nominal_volts_per_iq_unit",
            "reference_uncertainty_db",
            "reference_source",
            "acknowledge_nominal_scaling",
            "path_loss_db",
        ):
            qcs_rf_calibration.pop(legacy_key, None)
        raw_qcs_dc_calibration = raw_calibration.get("qcs_dc_output", {})
        if not isinstance(raw_qcs_dc_calibration, dict):
            raise TypeError(
                "calibration qcs_dc_output must be a JSON object"
            )
        qcs_dc_defaults = calibration_defaults["qcs_dc_output"]
        qcs_dc_calibration = {
            **qcs_dc_defaults,
            **raw_qcs_dc_calibration,
        }
        qcs_dc_calibration["endpoint"] = (
            normalize_qcs_calibration_endpoint_settings(
                qcs_dc_calibration.get("endpoint"),
                label="QCS DC calibration endpoint",
                allow_lo_frequency=False,
            )
        )
        qcs_dc_calibration["logical_index"] = self._json_int(
            qcs_dc_calibration["logical_index"],
            "QCS DC calibration logical index",
        )
        for key in (
            "voltage_start_v",
            "voltage_stop_v",
            "waveform_duration_us",
            "minimum_r_squared",
            "scope_settle_s",
            "scope_interval_s",
        ):
            qcs_dc_calibration[key] = self._json_finite_float(
                qcs_dc_calibration[key],
                f"QCS DC calibration {key}",
            )
        qcs_dc_calibration["voltage_points"] = self._json_int(
            qcs_dc_calibration["voltage_points"],
            "QCS DC calibration voltage points",
            minimum=3,
        )
        qcs_dc_calibration["scope_channel"] = self._json_int(
            qcs_dc_calibration["scope_channel"],
            "QCS DC calibration scope channel",
            minimum=1,
        )
        qcs_dc_calibration["scope_averages"] = self._json_int(
            qcs_dc_calibration["scope_averages"],
            "QCS DC calibration scope averages",
            minimum=1,
        )
        if qcs_dc_calibration["waveform_duration_us"] <= 0.0:
            raise ValueError(
                "QCS DC calibration waveform_duration_us must be positive"
            )
        if not 0.0 <= qcs_dc_calibration["minimum_r_squared"] <= 1.0:
            raise ValueError(
                "QCS DC calibration minimum_r_squared must be in [0, 1]"
            )
        for key in ("experiment_name", "sample_name", "scope_resource"):
            qcs_dc_calibration[key] = str(qcs_dc_calibration[key])
        output_calibration_settings = asdict(output_calibration)
        input_calibration_settings = asdict(input_calibration)
        dc_voltage_calibration_settings = asdict(dc_voltage_calibration)
        output_calibration_settings.pop("database_path")
        input_calibration_settings.pop("database_path")
        dc_voltage_calibration_settings.pop("database_path")
        calibration_settings = {
            "database_path": calibration_database_path,
            "selected_tab": calibration_selected_tab,
            "output": output_calibration_settings,
            "input": input_calibration_settings,
            "dc_voltage": dc_voltage_calibration_settings,
            "paths": calibration_paths,
            "dc_voltage_application": {
                "enabled": dc_application_enabled,
                "database_path": dc_application_database_path,
                "run_id": dc_application_run_id,
            },
            "qcs_rf_output": qcs_rf_calibration,
            "qcs_dc_output": qcs_dc_calibration,
            "input_plot": input_calibration_plot,
        }

        set_names = {f"set_{index}" for index in range(pulses[0].set_count)}
        if "rf_outputs" in data:
            raw_rf_outputs = data["rf_outputs"]
        else:
            raw_rf_outputs = [dict(DEFAULT_RF_OUTPUT_SETTINGS)]
        if not isinstance(raw_rf_outputs, list) or len(raw_rf_outputs) > 8:
            raise ValueError("rf_outputs must contain at most eight entries")
        rf_outputs = []
        active_rf_output_specs = []
        for index, entry in enumerate(raw_rf_outputs):
            if not isinstance(entry, dict):
                raise TypeError("each RF output setting must be a JSON object")
            defaults = dict(DEFAULT_RF_OUTPUT_SETTINGS)
            defaults["enabled"] = True
            defaults["gen_ch"] = DEFAULT_QSTL_RF_CHANNELS[index]
            entry = {**defaults, **entry}
            enabled = self._json_bool(entry["enabled"], "RF output enabled")
            require_within = self._json_bool(
                entry["require_within_segment"],
                "RF require_within_segment",
            )
            duration_sweep_enabled = self._json_bool(
                entry["duration_sweep_enabled"],
                "RF duration_sweep_enabled",
            )
            frequency_sweep_enabled = self._json_bool(
                entry["frequency_sweep_enabled"],
                "RF frequency_sweep_enabled",
            )
            power_sweep_enabled = self._json_bool(
                entry["power_sweep_enabled"],
                "RF power_sweep_enabled",
            )
            power_calibration_enabled = self._json_bool(
                entry["power_calibration_enabled"],
                "RF power_calibration_enabled",
            )
            power_calibration_database_path = str(
                entry["power_calibration_database_path"]
            )
            power_calibration_run_id = self._json_int(
                entry["power_calibration_run_id"],
                "RF power_calibration_run_id",
            )
            target_output_power_dbm = self._json_finite_float(
                entry["target_output_power_dbm"],
                "RF target_output_power_dbm",
            )
            qcs_power_calibration_enabled = self._json_bool(
                entry["qcs_power_calibration_enabled"],
                "RF qcs_power_calibration_enabled",
            )
            qcs_power_calibration_database_path = str(
                entry["qcs_power_calibration_database_path"]
            )
            qcs_power_calibration_run_id = self._json_int(
                entry["qcs_power_calibration_run_id"],
                "RF qcs_power_calibration_run_id",
            )
            qcs_target_output_power_dbm = self._json_finite_float(
                entry["qcs_target_output_power_dbm"],
                "RF qcs_target_output_power_dbm",
            )
            if (
                qcs_power_calibration_enabled
                and not qcs_power_calibration_database_path.strip()
            ):
                raise ValueError(
                    "RF QCS power-calibration database path is required"
                )
            spec = QickRfPulseSpec(
                gen_ch=self._json_int(entry["gen_ch"], "RF generator channel"),
                segment_name=_resolve_stored_set_segment_name(
                    entry["segment_name"],
                    set_names,
                    label="RF output",
                ),
                delay_us=float(entry["delay_us"]),
                duration_us=float(entry["duration_us"]),
                frequency_mhz=float(entry["frequency_mhz"]),
                gain=int(entry["gain"]),
                att1_db=float(entry["att1_db"]),
                att2_db=float(entry["att2_db"]),
                phase_degrees=float(entry["phase_degrees"]),
                nqz=int(entry["nqz"]),
                require_within_segment=require_within,
                filter_type=str(entry["filter_type"]),
                filter_cutoff=float(entry["filter_cutoff"]),
                filter_bandwidth=float(entry["filter_bandwidth"]),
                output_board_type=str(entry["output_board_type"]),
                duration_sweep_enabled=duration_sweep_enabled,
                duration_sweep_start_us=float(
                    entry["duration_sweep_start_us"]
                ),
                duration_sweep_stop_us=float(
                    entry["duration_sweep_stop_us"]
                ),
                duration_sweep_count=int(
                    entry["duration_sweep_count"]
                ),
                segment_length_mode=str(entry["segment_length_mode"]),
                frequency_sweep_enabled=frequency_sweep_enabled,
                frequency_sweep_start_mhz=float(
                    entry["frequency_sweep_start_mhz"]
                ),
                frequency_sweep_stop_mhz=float(
                    entry["frequency_sweep_stop_mhz"]
                ),
                frequency_sweep_count=int(
                    entry["frequency_sweep_count"]
                ),
                power_sweep_enabled=power_sweep_enabled,
                power_sweep_start_dbm=float(
                    entry["power_sweep_start_dbm"]
                ),
                power_sweep_stop_dbm=float(
                    entry["power_sweep_stop_dbm"]
                ),
                power_sweep_count=int(entry["power_sweep_count"]),
                power_calibration_enabled=power_calibration_enabled,
                power_calibration_database_path=(
                    power_calibration_database_path
                ),
                power_calibration_run_id=power_calibration_run_id,
                target_output_power_dbm=target_output_power_dbm,
            )
            if spec.segment_name not in set_names:
                raise ValueError(f"unknown RF output anchor {spec.segment_name!r}")
            if enabled:
                active_rf_output_specs.append(spec)
            rf_outputs.append({"enabled": enabled, **{
                "gen_ch": spec.gen_ch,
                "segment_name": spec.segment_name,
                "delay_us": spec.delay_us,
                "duration_us": spec.duration_us,
                "frequency_mhz": spec.frequency_mhz,
                "gain": spec.gain,
                "output_board_type": spec.output_board_type,
                "att1_db": spec.att1_db,
                "att2_db": spec.att2_db,
                "filter_type": spec.filter_type,
                "filter_cutoff": spec.filter_cutoff,
                "filter_bandwidth": spec.filter_bandwidth,
                "phase_degrees": spec.phase_degrees,
                "nqz": spec.nqz,
                "require_within_segment": spec.require_within_segment,
                "duration_sweep_enabled": spec.duration_sweep_enabled,
                "duration_sweep_start_us": spec.duration_sweep_start_us,
                "duration_sweep_stop_us": spec.duration_sweep_stop_us,
                "duration_sweep_count": spec.duration_sweep_count,
                "segment_length_mode": spec.segment_length_mode,
                "frequency_sweep_enabled": spec.frequency_sweep_enabled,
                "frequency_sweep_start_mhz": (
                    spec.frequency_sweep_start_mhz
                ),
                "frequency_sweep_stop_mhz": (
                    spec.frequency_sweep_stop_mhz
                ),
                "frequency_sweep_count": spec.frequency_sweep_count,
                "power_sweep_enabled": spec.power_sweep_enabled,
                "power_sweep_start_dbm": spec.power_sweep_start_dbm,
                "power_sweep_stop_dbm": spec.power_sweep_stop_dbm,
                "power_sweep_count": spec.power_sweep_count,
                "power_calibration_enabled": (
                    spec.power_calibration_enabled
                ),
                "power_calibration_database_path": (
                    spec.power_calibration_database_path
                ),
                "power_calibration_run_id": (
                    spec.power_calibration_run_id
                ),
                "target_output_power_dbm": (
                    spec.target_output_power_dbm
                ),
                "qcs_power_calibration_enabled": (
                    qcs_power_calibration_enabled
                ),
                "qcs_power_calibration_database_path": (
                    qcs_power_calibration_database_path
                ),
                "qcs_power_calibration_run_id": (
                    qcs_power_calibration_run_id
                ),
                "qcs_target_output_power_dbm": (
                    qcs_target_output_power_dbm
                ),
            }})

        available_sweep_axes = tuple(
            (spec.output_name, spec.segment_name) for spec in sweeps
        ) + tuple(
            (axis.output_name, axis.segment_name)
            for spec in active_rf_output_specs
            for axis in spec.sweep_axes
        )
        sweep_map_axes, sweep_map_slices = decode_sweep_map_settings(
            available_sweep_axes
        )

        raw_readout = data.get("rf_readout", {})
        if raw_readout is None:
            raw_readout = {}
        if not isinstance(raw_readout, dict):
            raise TypeError("rf_readout must be a JSON object")
        qcs_duration_explicit = (
            "qcs_acquisition_duration_s" in raw_readout
        )
        representation_explicit = (
            "measurement_representation" in raw_readout
        )
        raw_readout = {**DEFAULT_RF_READOUT_SETTINGS, **raw_readout}
        if not representation_explicit:
            raw_readout["measurement_representation"] = (
                "current"
                if bool(raw_readout.get("dc_measure_mode", False))
                else (
                    "voltage"
                    if bool(
                        raw_readout.get(
                            "dc_voltage_calibration_enabled",
                            False,
                        )
                    )
                    else "adc"
                )
            )
        readout_enabled = self._json_bool(
            raw_readout["enabled"], "RF readout enabled"
        )
        force_overwrite = self._json_bool(
            raw_readout["force_overwrite"],
            "RF readout force_overwrite",
        )
        readout_spec = QickDdrReadoutSpec(
            ro_ch=self._json_int(raw_readout["ro_ch"], "RF readout channel"),
            segment_name=_resolve_stored_set_segment_name(
                raw_readout["segment_name"],
                set_names,
                label="RF readout",
            ),
            delay_us=float(raw_readout["delay_us"]),
            samples_per_trigger=self._json_int(
                raw_readout["samples_per_trigger"],
                "RF readout samples_per_trigger",
                minimum=1,
            ),
            readout_frequency_mhz=float(raw_readout["readout_frequency_mhz"]),
            margin_input_samples=self._json_int(
                raw_readout["margin_input_samples"],
                "RF readout margin_input_samples",
            ),
            fpga_trigger_delay_us=(
                None
                if raw_readout.get("fpga_trigger_delay_us") is None
                else self._json_finite_float(
                    raw_readout["fpga_trigger_delay_us"],
                    "RF readout fpga_trigger_delay_us",
                )
            ),
            force_overwrite=force_overwrite,
            post_run_read_delay_seconds=self._json_finite_float(
                raw_readout.get("post_run_read_delay_seconds", 0.1),
                "RF readout post_run_read_delay_seconds",
            ),
            attenuation_db=float(raw_readout["attenuation_db"]),
            filter_type=str(raw_readout["filter_type"]),
            filter_cutoff=float(raw_readout["filter_cutoff"]),
            filter_bandwidth=float(raw_readout["filter_bandwidth"]),
            input_board_type=str(raw_readout["input_board_type"]),
            dc_gain_db=float(raw_readout["dc_gain_db"]),
            dc_measure_mode=self._json_bool(
                raw_readout["dc_measure_mode"],
                "RF readout dc_measure_mode",
            ),
            dc_measure_gain_v_per_a=self._json_finite_float(
                raw_readout["dc_measure_gain_v_per_a"],
                "RF readout dc_measure_gain_v_per_a",
                positive=True,
            ),
            measurement_representation=str(
                raw_readout.get("measurement_representation", "auto")
            ),
            dc_voltage_calibration_enabled=self._json_bool(
                raw_readout.get("dc_voltage_calibration_enabled", False),
                "RF readout dc_voltage_calibration_enabled",
            ),
            dc_voltage_calibration_database_path=str(
                raw_readout.get("dc_voltage_calibration_database_path", "")
            ),
            dc_voltage_calibration_run_id=self._json_int(
                raw_readout.get("dc_voltage_calibration_run_id", 0),
                "RF readout dc_voltage_calibration_run_id",
            ),
            nqz=self._json_int(raw_readout["nqz"], "RF readout nqz", minimum=1),
        )
        if readout_spec.segment_name not in set_names:
            raise ValueError(
                f"unknown RF readout anchor {readout_spec.segment_name!r}"
            )
        qcs_acquisition_duration_s = (
            self._json_finite_float(
                raw_readout["qcs_acquisition_duration_s"],
                "QCS acquisition duration",
                positive=True,
            )
            if qcs_duration_explicit
            else (
                readout_spec.samples_per_trigger
                / QCS_M5200_SAMPLE_RATE_HZ
            )
        )
        rf_readout = {
            "enabled": readout_enabled,
            "ro_ch": readout_spec.ro_ch,
            "segment_name": readout_spec.segment_name,
            "delay_us": readout_spec.delay_us,
            "samples_per_trigger": readout_spec.samples_per_trigger,
            "qcs_acquisition_duration_s": qcs_acquisition_duration_s,
            "readout_frequency_mhz": readout_spec.readout_frequency_mhz,
            "margin_input_samples": readout_spec.margin_input_samples,
            "fpga_trigger_delay_us": readout_spec.fpga_trigger_delay_us,
            "force_overwrite": readout_spec.force_overwrite,
            "post_run_read_delay_seconds": (
                readout_spec.post_run_read_delay_seconds
            ),
            "input_board_type": readout_spec.input_board_type,
            "attenuation_db": readout_spec.attenuation_db,
            "dc_gain_db": readout_spec.dc_gain_db,
            "dc_measure_mode": readout_spec.dc_measure_mode,
            "dc_measure_gain_v_per_a": readout_spec.dc_measure_gain_v_per_a,
            "measurement_representation": (
                readout_spec.effective_measurement_representation
            ),
            "dc_voltage_calibration_enabled": (
                readout_spec.dc_voltage_calibration_enabled
            ),
            "dc_voltage_calibration_database_path": (
                readout_spec.dc_voltage_calibration_database_path
            ),
            "dc_voltage_calibration_run_id": (
                readout_spec.dc_voltage_calibration_run_id
            ),
            "filter_type": readout_spec.filter_type,
            "filter_cutoff": readout_spec.filter_cutoff,
            "filter_bandwidth": readout_spec.filter_bandwidth,
            "nqz": readout_spec.nqz,
        }
        # Version 20 and earlier stored a second full RF output/readout editor
        # inside Stability Diagram.  Fold those values into the compact path
        # and shared modulation controls when loading an old JSON file.
        if isinstance(raw_stability_settings, Mapping):
            legacy_outputs = raw_stability_settings.get("rf_outputs")
            legacy_readout = raw_stability_settings.get("rf_readout")
            path = dict(stability_settings["rf_path"])
            if legacy_outputs:
                decoded_outputs = self._decode_rf_output_entries(
                    legacy_outputs,
                    set_names=set_names,
                    label="stability RF outputs",
                )
                if decoded_outputs:
                    output = decoded_outputs[0]
                    path.update(
                        {
                            "output_ch": output["gen_ch"],
                            "output_board_type": output["output_board_type"],
                            "output_nqz": output["nqz"],
                            "output_att1_db": output["att1_db"],
                            "output_att2_db": output["att2_db"],
                            "output_filter_type": output["filter_type"],
                            "output_filter_cutoff_ghz": output["filter_cutoff"],
                            "output_filter_bandwidth_ghz": output["filter_bandwidth"],
                        }
                    )
                    if "modulation_frequency_mhz" not in raw_stability_settings:
                        stability_settings["modulation_frequency_mhz"] = output[
                            "frequency_mhz"
                        ]
                    if "modulation_gain" not in raw_stability_settings:
                        stability_settings["modulation_gain"] = output["gain"]
            if legacy_readout is not None:
                readout = self._decode_rf_readout_entry(
                    legacy_readout,
                    set_names=set_names,
                    label="stability RF readout",
                )
                path.update(
                    {
                        "readout_ch": readout["ro_ch"],
                        "input_board_type": readout["input_board_type"],
                        "readout_nqz": readout["nqz"],
                        "readout_attenuation_db": readout["attenuation_db"],
                        "readout_dc_gain_db": readout["dc_gain_db"],
                        "readout_filter_type": readout["filter_type"],
                        "readout_filter_cutoff_ghz": readout["filter_cutoff"],
                        "readout_filter_bandwidth_ghz": readout["filter_bandwidth"],
                    }
                )
                if (
                    "modulation_frequency_mhz" not in raw_stability_settings
                    and not legacy_outputs
                ):
                    stability_settings["modulation_frequency_mhz"] = readout[
                        "readout_frequency_mhz"
                    ]
            stability_settings["rf_path"] = path
        if not dc_application_explicit:
            calibration_settings["dc_voltage_application"] = {
                "enabled": readout_spec.dc_voltage_calibration_enabled,
                "database_path": (
                    readout_spec.dc_voltage_calibration_database_path
                    or calibration_database_path
                ),
                "run_id": readout_spec.dc_voltage_calibration_run_id,
            }

        return {
            "pulses": pulses,
            "cross_capacitance": matrix,
            "sweeps": sweeps,
            "time_unit": time_unit,
            "voltage_view": voltage_view,
            "selected_output": selected_output,
            "selected_tab": selected_tab,
            "selected_awg_tuning_tab": selected_awg_tuning_tab,
            "trace_stability_overlay": {
                "mode": trace_overlay_mode,
                "database_path": trace_overlay_database_path,
                "run_id": trace_overlay_run_id,
                "quantity": trace_overlay_quantity,
            },
            "grid_time_ns": self._json_finite_float(
                grid.get("time_step_ns", 1000.0),
                "grid time_step_ns",
                positive=True,
            ),
            "grid_voltage_mv": self._json_finite_float(
                grid.get("voltage_step_mv", 100.0),
                "grid voltage_step_mv",
                positive=True,
            ),
            "grid_snap": self._json_bool(
                grid.get("snap_enabled", False), "grid snap_enabled"
            ),
            "grid_visible": self._json_bool(
                grid.get("visible", True), "grid visible"
            ),
            "grid_configured": self._json_bool(
                grid.get("configured", False), "grid configured"
            ),
            "fabric_mhz": fabric_mhz,
            "tproc_mhz": tproc_mhz,
            "full_scale_mv": full_scale_mv,
            "sweep_voltage_scale_mv": sweep_voltage_scale_mv,
            "awg_output_names": awg_output_names,
            "awg_channels": awg_channels,
            "repetitions": repetitions,
            "iq_repetition_policy": iq_repetition_policy,
            "awg_metadata_mode": awg_metadata_mode,
            "compile_validation_mode": compile_validation_mode,
            "bias_t_enabled": bias_t_enabled,
            "bias_t_type": bias_t_type,
            "bias_t_mode": bias_t_mode,
            "bias_t_compensation_mv": bias_t_compensation_mv,
            "bias_t_duration_us": bias_t_duration_us,
            "bias_t_filter_tau_us": bias_t_filter_tau_us,
            "connection_config": connection_config,
            "execution_backend": execution_backend,
            "qcs_settings": qcs_settings,
            "run_config": run_config,
            "sweep_map_axes": sweep_map_axes,
            "sweep_map_slices": sweep_map_slices,
            "sweep_map_color_ranges": sweep_map_color_ranges,
            "sweep_map_visible_data": sweep_map_visible_data,
            "rf_outputs": tuple(rf_outputs),
            "rf_readout": rf_readout,
            "stability_diagram": stability_settings,
            "s_parameter": {
                "database_path": sparameter_database_path,
                "qcs_amplitude": sparameter_qcs_amplitude,
                **asdict(sparameter_config),
            },
            "calibration": calibration_settings,
            "noise_analysis": noise_analysis_settings,
        }

    def _apply_decoded_settings(self, settings: dict) -> None:
        """Apply a fully validated settings object to all GUI panels."""
        if self._qcs_mapper_commit_pending():
            raise RuntimeError(
                "Wait for the background QCS mapper update to finish before "
                "loading application settings."
            )
        if hasattr(self, "_qcs_front_panel_dialog"):
            self._qcs_front_panel_dialog.close()
            self._clear_pending_qcs_front_panel_configuration()
            self._qcs_front_panel_source_snapshot = None
            self._qcs_front_panel_editor_initialized = False
        pulses = settings["pulses"]
        self._sweep_specs = []
        self._suspend_qcs_output_sync = True
        try:
            while len(self._pulse) > len(pulses):
                self._delete_port(len(self._pulse) - 1)
            while len(self._pulse) < len(pulses):
                self._add_port()
        finally:
            self._suspend_qcs_output_sync = False
        self._awg_output_display_names = list(settings["awg_output_names"])
        self._sync_awg_output_display_names()
        for target, source in zip(self._pulse, pulses):
            target.t = source.t.copy()
            target.v = source.v.copy()
            target.v_bounds = tuple(source.v_bounds)
            target.segment_names = list(source.segment_names)
        self._share_matching_port_timings(0)
        self._share_matching_segment_names(0)

        self._cross_capacitance = settings["cross_capacitance"].copy()
        self._sweep_specs = list(settings["sweeps"])
        self._awg_voltage_scale_mv = settings["sweep_voltage_scale_mv"]
        self._qick_fabric_mhz = settings["fabric_mhz"]
        self._qick_tproc_mhz = settings["tproc_mhz"]
        self._qick_full_scale_mv = settings["full_scale_mv"]
        self._qick_awg_channels = tuple(settings["awg_channels"])
        self._qick_repetitions_per_sweep = settings["repetitions"]
        self._bias_t_compensation_enabled = settings["bias_t_enabled"]
        self._bias_t_compensation_type = settings["bias_t_type"]
        self._bias_t_compensation_voltage_mv = settings[
            "bias_t_compensation_mv"
        ]
        self._bias_t_compensation_mode = settings["bias_t_mode"]
        self._bias_t_compensation_duration_us = settings["bias_t_duration_us"]
        self._bias_t_filter_tau_us = settings["bias_t_filter_tau_us"]
        self._loading_awg_settings = True
        try:
            self._experiment_panel.load_settings(
                settings["connection_config"],
                settings["run_config"],
                fabric_mhz=self._qick_fabric_mhz,
                tproc_mhz=self._qick_tproc_mhz,
                full_scale_mv=self._qick_full_scale_mv,
                awg_channels=self._qick_awg_channels,
                repetitions=self._qick_repetitions_per_sweep,
                iq_repetition_policy=settings["iq_repetition_policy"],
                awg_metadata_mode=settings["awg_metadata_mode"],
                compile_validation_mode=settings["compile_validation_mode"],
                bias_t_enabled=self._bias_t_compensation_enabled,
                bias_t_compensation_type=self._bias_t_compensation_type,
                bias_t_compensation_mv=self._bias_t_compensation_voltage_mv,
                bias_t_mode=self._bias_t_compensation_mode,
                bias_t_duration_us=self._bias_t_compensation_duration_us,
                bias_t_filter_tau_us=self._bias_t_filter_tau_us,
                execution_backend=settings["execution_backend"],
                qcs_settings=settings["qcs_settings"],
            )
        finally:
            self._loading_awg_settings = False
        self._multi_ctrl.set_awg_channels(self._qick_awg_channels)
        self._refresh_stability_targets()
        self._stability_panel.load_settings(settings["stability_diagram"])

        with QtCore.QSignalBlocker(self._time_unit_combo):
            self._time_unit_combo.setCurrentText(settings["time_unit"])
        self._set_time_unit(settings["time_unit"])
        self._grid_configured = settings["grid_configured"]
        self._set_grid_settings(
            time_step_ns=settings["grid_time_ns"],
            voltage_step_mv=settings["grid_voltage_mv"],
            snap_enabled=settings["grid_snap"],
            visible=settings["grid_visible"],
        )

        self._rf_ports_panel.refresh_segments(self._pulse[0])
        self._rf_ports_panel.load_settings(settings["rf_outputs"])
        self._rf_readout_panel.refresh_segments(self._pulse[0])
        self._rf_readout_panel.load_settings(settings["rf_readout"])
        self._rf_pulse_specs = list(self._rf_ports_panel.specs())
        self._rf_pulse_spec = self._rf_pulse_specs[0] if self._rf_pulse_specs else None
        self._experiment_panel.set_sweep_specs(
            self._active_map_sweep_specs(),
            selected_keys=settings["sweep_map_axes"],
        )
        sweep_map_axis_settings = {
            "slice_axes": settings["sweep_map_slices"],
        }
        if settings["sweep_map_axes"] is not None:
            x_axis, y_axis = settings["sweep_map_axes"]
            sweep_map_axis_settings.update({
                "x_axis": {
                    "output_name": x_axis[0],
                    "segment_name": x_axis[1],
                },
                "y_axis": {
                    "output_name": y_axis[0],
                    "segment_name": y_axis[1],
                },
            })
        self._awg_sweep_plot.load_axis_selection_settings(
            sweep_map_axis_settings
        )
        self._awg_sweep_plot.load_color_range_settings(
            settings["sweep_map_color_ranges"]
        )
        self._awg_sweep_plot.load_visible_data(
            settings["sweep_map_visible_data"]
        )
        self._ddr_readout_spec = self._rf_readout_panel.spec()
        self._experiment_panel.set_ddr_readout_spec(self._ddr_readout_spec)
        self._sparameter_panel.load_settings(settings["s_parameter"])
        self._calibration_panel.load_settings(settings["calibration"])
        self._noise_panel.load_settings(settings["noise_analysis"])
        self._sync_shared_qick_controls()
        self._qick_front_panel.set_path_values(
            self._sparameter_panel.front_panel_values()
        )

        self._selected_port_idx = settings["selected_output"]
        self._plot.set_selected_port_idx(self._selected_port_idx)
        self._control_tabs.setCurrentIndex(settings["selected_tab"])
        self._awg_tuning_tabs.setCurrentIndex(
            settings["selected_awg_tuning_tab"]
        )
        self._on_execution_backend_changed(
            self._experiment_panel.execution_backend()
        )
        with QtCore.QSignalBlocker(self._voltage_view_actions[settings["voltage_view"]]):
            self._voltage_view_actions[settings["voltage_view"]].setChecked(True)
        self._set_voltage_view(settings["voltage_view"])

        self._plot.refresh()
        self._multi_ctrl.refresh_table()
        self.refresh_panel_table()
        self._refresh_trace_if_needed(force=True)
        self._refresh_rf_timeline(fit_view=True)
        self._refresh_sweep_overlay(fit_view=True, sync_rows=True)
        self._restore_trace_stability_overlay_settings(
            settings["trace_stability_overlay"]
        )

    def _apply_legacy_settings(self, data: dict) -> None:
        """Read the original single-waveform JSON format."""
        if self._qcs_mapper_commit_pending():
            raise RuntimeError(
                "Wait for the background QCS mapper update to finish before "
                "loading application settings."
            )
        loaded = self._pulse[0].copy()
        if "initial_voltage" in data:
            loaded.v[0:2] = float(data["initial_voltage"])
        if "voltage_bounds" in data and len(data["voltage_bounds"]) == 2:
            loaded.v_bounds = tuple(map(float, data["voltage_bounds"]))
        if "time_ns" in data and "voltage_mv" in data:
            loaded = PulseSequence.from_dict(data)
        loaded.validate()
        self._pulse[0].t = loaded.t.copy()
        self._pulse[0].v = loaded.v.copy()
        self._pulse[0].v_bounds = tuple(loaded.v_bounds)
        self._pulse[0].segment_names = list(loaded.segment_names)
        self._plot.refresh()
        self._multi_ctrl.refresh_table()
        self._refresh_trace_if_needed(force=True)
        self._refresh_rf_editor()
        self._refresh_sweep_overlay(fit_view=True, sync_rows=True)

    def _save_settings_json(self, path) -> Path:
        output_path = Path(path)
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")
        text = json.dumps(
            self._settings_to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        output_path.write_text(text + "\n", encoding="utf-8")
        self._settings_path = output_path
        return output_path

    def _load_settings_json(self, path) -> Path:
        input_path = Path(path)
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema") == SETTINGS_SCHEMA:
            self._apply_decoded_settings(self._decode_settings(data))
        else:
            if not isinstance(data, dict):
                raise TypeError("legacy settings JSON root must be an object")
            self._apply_legacy_settings(data)
        self._settings_path = input_path
        return input_path

    def _save_json(self) -> None:
        default_path = self._settings_path or (
            Path(__file__).resolve().parent / "pulse_generator_settings.json"
        )
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save settings JSON",
            str(default_path),
            "JSON files (*.json)",
        )
        if not path:
            return
        try:
            saved_path = self._save_settings_json(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Save failed", f"Could not write JSON:\n{exc}"
            )
            return
        self.statusBar().showMessage(f"Saved settings to {saved_path}")

    def _load_json(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open settings JSON",
            str(self._settings_path.parent if self._settings_path else ""),
            "JSON files (*.json)",
        )
        if not path:
            return
        try:
            loaded_path = self._load_settings_json(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Load failed", f"Could not read JSON:\n{exc}"
            )
            return
        self.statusBar().showMessage(f"Loaded settings from {loaded_path}")

    def _generate_pulse(self):
        try:
            code_str = self._generate_qcs_code()
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "QCS export failed", str(exc))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save generated QCS module",
            str(Path(__file__).resolve().parent / "qcs_dc_waveforms.py"),
            "Python files (*.py)",
        )
        if not path:
            return
        Path(path).write_text(code_str, encoding="utf-8")
        self._show_generated_code("Generated QCS pulse", path, code_str)

    def _generate_qcs_code(self) -> str:
        """Generate QCS code locally from the current AWG Tuning settings."""
        readout_spec = self._rf_readout_panel.spec()
        acquisition_arguments = {}
        if readout_spec is not None:
            acquisition_channel_name = (
                self._experiment_panel.qcs_acquisition_channel_name.text().strip()
            )
            if not acquisition_channel_name:
                raise ValueError(
                    "set a QCS acquisition virtual channel before generating "
                    "a program with QCS Acquisition enabled"
                )
            sample_rate_hz = self._experiment_panel.qcs_sample_rate_hz.value()
            hardware_demodulation = (
                self._experiment_panel.qcs_hw_demod.isChecked()
            )
            acquisition_arguments = {
                "acquisition_channel_name": acquisition_channel_name,
                "acquisition_segment_name": str(readout_spec.segment_name),
                "acquisition_duration_s": (
                    int(readout_spec.samples_per_trigger) / sample_rate_hz
                ),
                "acquisition_pre_delay_s": (
                    float(readout_spec.delay_us) * 1.0e-6
                ),
                "acquisition_hardware_demodulation": hardware_demodulation,
                "acquisition_frequency_hz": (
                    float(readout_spec.readout_frequency_mhz) * 1.0e6
                    if hardware_demodulation
                    else 0.0
                ),
                "acquisition_phase_rad": 0.0,
            }
        return generate_qcs_program_code(
            self._pulse,
            channel_names=(
                self._experiment_panel._parse_qcs_dc_channel_names(
                    len(self._pulse)
                )
            ),
            full_scale_v=(
                self._experiment_panel.qcs_dc_full_scale_v.value()
            ),
            cross_capacitance=self._cross_capacitance,
            fabric_mhz=self._qick_fabric_mhz,
            bias_t_compensation_enabled=(
                self._experiment_panel.bias_t_group.isChecked()
            ),
            bias_t_compensation_type=str(
                self._experiment_panel.bias_t_type.currentData()
            ),
            bias_t_compensation_voltage_mv=(
                self._experiment_panel.bias_t_compensation_mv.value()
            ),
            bias_t_compensation_mode=str(
                self._experiment_panel.bias_t_mode.currentData()
            ),
            bias_t_compensation_duration_us=(
                self._experiment_panel.bias_t_duration_us.value()
            ),
            bias_t_filter_tau_us=(
                self._experiment_panel.bias_t_filter_tau_us.value()
            ),
            **acquisition_arguments,
        )

    def _qick_output_names(self) -> Tuple[str, ...]:
        return tuple(f"awg_{index}" for index in range(len(self._pulse)))

    def _awg_output_name_mapping(self) -> Tuple[dict, ...]:
        return tuple(
            {
                "original_name": original_name,
                "display_name": display_name,
            }
            for original_name, display_name in zip(
                self._qick_output_names(),
                self._awg_output_display_names,
            )
        )

    def _awg_output_display_name_map(self) -> dict:
        return {
            entry["original_name"]: entry["display_name"]
            for entry in self._awg_output_name_mapping()
        }

    def _stability_result_with_display_names(self, result):
        display_names = tuple(
            getattr(self, "_awg_output_display_names", ())
        )
        if not display_names:
            return result
        name_map = {
            f"awg_{index}": name
            for index, name in enumerate(display_names)
        }
        x_display_name = name_map.get(
            str(result.x_axis_label),
            str(result.x_axis_label),
        )
        y_display_name = name_map.get(
            str(result.y_axis_label),
            str(result.y_axis_label),
        )
        if (
            x_display_name == str(result.x_axis_label)
            and y_display_name == str(result.y_axis_label)
            and not result.x_axis_display_label
            and not result.y_axis_display_label
        ):
            return result
        return replace(
            result,
            x_axis_display_label=x_display_name,
            y_axis_display_label=y_display_name,
        )

    def _set_awg_output_display_name(
        self,
        original_name: str,
        requested_name: str,
    ) -> None:
        original_names = self._qick_output_names()
        try:
            index = original_names.index(str(original_name))
        except ValueError:
            return
        name = str(requested_name).strip()
        invalid_reason = ""
        if not name:
            invalid_reason = "AWG output name must not be empty."
        elif any(ord(character) < 32 for character in name):
            invalid_reason = "AWG output name must not contain control characters."
        elif any(
            other_index != index and other_name == name
            for other_index, other_name in enumerate(
                self._awg_output_display_names
            )
        ):
            invalid_reason = f"AWG output name {name!r} is already in use."
        if invalid_reason:
            self._sync_awg_output_display_names()
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid AWG output name",
                invalid_reason,
            )
            return
        if self._awg_output_display_names[index] == name:
            self._sync_awg_output_display_names()
            return
        self._awg_output_display_names[index] = name
        self._sync_awg_output_display_names()
        self._refresh_awg_sweep_map_from_last_result(report_error=False)
        self.statusBar().showMessage(
            f"{original_name} display name changed to {name!r}",
            8000,
        )

    def _sync_awg_output_display_names(self) -> None:
        names = tuple(self._awg_output_display_names)
        if len(names) != len(self._pulse):
            return
        if hasattr(self, "_multi_ctrl"):
            self._multi_ctrl.set_output_display_names(names)
        if hasattr(self, "_plot") and hasattr(self._plot, "set_output_names"):
            self._plot.set_output_names(names)
        if self._trace is not None and hasattr(self._trace, "set_output_names"):
            self._trace.set_output_names(names)
            self._trace.refresh_trace(self._pulse)
        self._refresh_stability_targets()
        stability_result = getattr(self, "_last_stability_result", None)
        if stability_result is not None:
            stability_result = MainWindow._stability_result_with_display_names(
                self,
                stability_result,
            )
            self._last_stability_result = stability_result
            self._stability_panel.plot.set_result(stability_result)
            if not getattr(self, "_trace_overlay_pinned", False):
                MainWindow._apply_trace_stability_overlay(self, fit=False)
        panel_table = getattr(getattr(self, "_multi_ctrl", None), "panel_table", None)
        if panel_table is not None:
            for index, (original_name, display_name) in enumerate(
                zip(self._qick_output_names(), names)
            ):
                item = panel_table.item(index, 0)
                if item is not None:
                    item.setText(
                        display_name
                        if display_name == original_name
                        else f"{display_name}\n({original_name})"
                    )
                    item.setToolTip(
                        f"Display name: {display_name}\nOriginal name: {original_name}"
                    )

    def _next_qick_awg_channel(self) -> int:
        used = set(self._qick_awg_channels)
        for channel in DEFAULT_QSTL_AWG_CHANNELS:
            if channel not in used:
                return channel
        channel = 0
        while channel in used:
            channel += 1
        return channel

    def _qick_export_settings(self) -> Optional[dict]:
        if len(self._pulse) > 8:
            QtWidgets.QMessageBox.warning(
                self,
                "QICK export unavailable",
                "axis_awg_tuning_v1 supports at most eight outputs.",
            )
            return None
        try:
            experiment_values = self._experiment_panel.values(len(self._pulse))
            self._qick_fabric_mhz = experiment_values["fabric_mhz"]
            self._qick_tproc_mhz = experiment_values["tproc_mhz"]
            self._qick_full_scale_mv = experiment_values["full_scale_mv"]
            self._qick_awg_channels = experiment_values["awg_channels"]
            self._qick_repetitions_per_sweep = experiment_values[
                "repetitions_per_sweep"
            ]
            self._bias_t_compensation_enabled = experiment_values[
                "bias_t_compensation_enabled"
            ]
            self._bias_t_compensation_type = experiment_values[
                "bias_t_compensation_type"
            ]
            self._bias_t_compensation_voltage_mv = experiment_values[
                "bias_t_compensation_voltage_mv"
            ]
            self._bias_t_compensation_mode = experiment_values[
                "bias_t_compensation_mode"
            ]
            self._bias_t_compensation_duration_us = experiment_values[
                "bias_t_compensation_duration_us"
            ]
            self._bias_t_filter_tau_us = experiment_values[
                "bias_t_filter_tau_us"
            ]
            set_names = qick_set_segment_names(self._pulse[0])
            self._rf_pulse_specs = list(self._rf_ports_panel.specs())
            self._rf_pulse_spec = (
                self._rf_pulse_specs[0] if self._rf_pulse_specs else None
            )
            self._ddr_readout_spec = self._rf_readout_panel.spec()
            self._experiment_panel.set_ddr_readout_spec(
                self._ddr_readout_spec
            )
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid waveform", str(exc))
            return None
        flat_durations = [
            self._pulse[0].t[end] - self._pulse[0].t[start]
            for start, end in self._pulse[0].flat_segments()
        ]
        default_set_index = int(np.argmax(flat_durations))
        dialog = QickExportDialog(
            len(self._pulse),
            set_names,
            self,
            default_set_index=default_set_index,
            initial_rf_specs=tuple(self._rf_pulse_specs),
            initial_ddr_readout_spec=self._ddr_readout_spec,
            initial_sweeps=tuple(self._sweep_specs),
            initial_cross_capacitance=self._cross_capacitance,
            initial_fabric_mhz=self._qick_fabric_mhz,
            initial_tproc_mhz=self._qick_tproc_mhz,
            initial_full_scale_mv=self._qick_full_scale_mv,
            initial_awg_channels=self._qick_awg_channels,
            initial_repetitions=self._qick_repetitions_per_sweep,
            initial_bias_t_enabled=self._bias_t_compensation_enabled,
            initial_bias_t_compensation_type=self._bias_t_compensation_type,
            initial_bias_t_compensation_mv=self._bias_t_compensation_voltage_mv,
            initial_bias_t_mode=self._bias_t_compensation_mode,
            initial_bias_t_duration_us=self._bias_t_compensation_duration_us,
            initial_bias_t_filter_tau_us=self._bias_t_filter_tau_us,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None
        settings = dialog.values()
        self._rf_pulse_specs = list(settings["rf_pulse_specs"])
        self._rf_pulse_spec = self._rf_pulse_specs[0] if self._rf_pulse_specs else None
        self._ddr_readout_spec = settings["ddr_readout_spec"]
        self._experiment_panel.set_ddr_readout_spec(self._ddr_readout_spec)
        if settings["sweeps"] is not None:
            self._sweep_specs = list(settings["sweeps"])
        elif settings["sweep"] is not None:
            self._sweep_specs = [settings["sweep"]]
        else:
            self._sweep_specs = []
        self._qick_fabric_mhz = settings["fabric_mhz"]
        self._qick_tproc_mhz = settings["tproc_mhz"]
        self._qick_full_scale_mv = settings["full_scale_mv"]
        self._qick_awg_channels = tuple(settings["awg_channels"])
        self._multi_ctrl.set_awg_channels(self._qick_awg_channels)
        self._qick_repetitions_per_sweep = settings["repetitions_per_sweep"]
        self._bias_t_compensation_enabled = settings[
            "bias_t_compensation_enabled"
        ]
        self._bias_t_compensation_type = settings["bias_t_compensation_type"]
        self._bias_t_compensation_voltage_mv = settings[
            "bias_t_compensation_voltage_mv"
        ]
        self._bias_t_compensation_mode = settings["bias_t_compensation_mode"]
        self._bias_t_compensation_duration_us = settings[
            "bias_t_compensation_duration_us"
        ]
        self._bias_t_filter_tau_us = settings["bias_t_filter_tau_us"]
        self._experiment_panel.set_qick_values(
            fabric_mhz=self._qick_fabric_mhz,
            tproc_mhz=self._qick_tproc_mhz,
            full_scale_mv=self._qick_full_scale_mv,
            awg_channels=self._qick_awg_channels,
            repetitions=self._qick_repetitions_per_sweep,
        )
        self._experiment_panel.set_bias_t_values(
            enabled=self._bias_t_compensation_enabled,
            compensation_type=self._bias_t_compensation_type,
            compensation_mv=self._bias_t_compensation_voltage_mv,
            mode=self._bias_t_compensation_mode,
            duration_us=self._bias_t_compensation_duration_us,
            filter_tau_us=self._bias_t_filter_tau_us,
        )
        self._cross_capacitance = np.asarray(
            settings["cross_capacitance"], dtype=float
        )
        self._refresh_rf_timeline(fit_view=True)
        self._refresh_sweep_overlay(fit_view=True, sync_rows=True)
        return settings

    def _generate_qick_pulse(self) -> None:
        settings = self._qick_export_settings()
        if settings is None:
            return
        try:
            code_str = generate_qick_program_code(
                self._pulse,
                output_names=self._qick_output_names(),
                **settings,
            )
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "QICK export failed", str(exc))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save generated QICK module",
            str(Path(__file__).resolve().parent / "qick_dc_waveforms.py"),
            "Python files (*.py)",
        )
        if not path:
            return
        Path(path).write_text(code_str, encoding="utf-8")
        self._show_generated_code("Generated QICK pulse", path, code_str)

    def _preview_qick_pulse(self) -> None:
        settings = self._qick_export_settings()
        if settings is None:
            return
        sequence_settings = {
            name: settings[name]
            for name in (
                "fabric_mhz",
                "full_scale_mv",
                "sweep",
                "sweeps",
                "cross_capacitance",
                "bias_t_compensation_enabled",
                "bias_t_compensation_type",
                "bias_t_compensation_voltage_mv",
                "bias_t_compensation_mode",
                "bias_t_compensation_duration_us",
                "bias_t_filter_tau_us",
            )
        }
        try:
            sequence = build_qick_sequence(
                self._pulse,
                output_names=self._qick_output_names(),
                **sequence_settings,
            )
            # FineTuneSequence defaults to first/middle/last sweep points,
            # keeping preview cost bounded for large sweeps.
            sequence.plot_preview(point_indices=None, show=True)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "QICK preview failed", str(exc))

    def _show_generated_code(self, title: str, path: str, code_str: str) -> None:
        self.statusBar().showMessage(f"Saved {path}")
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle(title)
        dlg.setText(f"Generated module saved to:\n{path}")
        dlg.setDetailedText(code_str)
        dlg.exec_()

    @staticmethod
    def _segment_timing_ns(
        pulse: PulseSequence,
        segment_index: int,
    ) -> Optional[Tuple[float, float]]:
        segments = pulse.flat_segments()
        if not 0 <= int(segment_index) < len(segments):
            return None
        flat_start, flat_end = segments[int(segment_index)]
        ramp_ns = (
            float(pulse.t[flat_start] - pulse.t[flat_start - 1])
            if flat_start > 0
            else 0.0
        )
        flat_ns = float(pulse.t[flat_end] - pulse.t[flat_start])
        return ramp_ns, flat_ns

    def _share_segment_timing(
        self,
        source_port: int,
        segment_index: int,
    ) -> bool:
        """Copy one segment's RAMP/hold durations to matching AWG rows."""
        if not 0 <= int(source_port) < len(self._pulse):
            return False
        timing = self._segment_timing_ns(
            self._pulse[int(source_port)],
            int(segment_index),
        )
        if timing is None:
            return False
        ramp_ns, flat_ns = timing
        changed = False
        for port_index, pulse in enumerate(self._pulse):
            if port_index == int(source_port):
                continue
            segments = pulse.flat_segments()
            if not 0 <= int(segment_index) < len(segments):
                continue
            flat_start, _flat_end = segments[int(segment_index)]
            target_timing = self._segment_timing_ns(pulse, int(segment_index))
            if target_timing is None:
                continue
            target_ramp_ns, target_flat_ns = target_timing
            if flat_start > 0 and not np.isclose(
                target_ramp_ns,
                ramp_ns,
                rtol=0.0,
                atol=1.0e-9,
            ):
                pulse.edit_ramp(flat_start, ramp_ns)
                changed = True
            if not np.isclose(
                target_flat_ns,
                flat_ns,
                rtol=0.0,
                atol=1.0e-9,
            ):
                pulse.edit_flat(flat_start, flat_ns)
                changed = True
        return changed

    def _share_matching_port_timings(self, source_port: int) -> bool:
        """Share every row that exists on the source and destination ports."""
        if not 0 <= int(source_port) < len(self._pulse):
            return False
        changed = False
        for segment_index in range(
            len(self._pulse[int(source_port)].flat_segments())
        ):
            changed |= self._share_segment_timing(
                int(source_port),
                segment_index,
            )
        return changed

    def _share_segment_name(
        self,
        source_port: int,
        segment_index: int,
    ) -> bool:
        """Copy one user-facing segment name to matching AWG rows."""
        if not 0 <= int(source_port) < len(self._pulse):
            return False
        source = self._pulse[int(source_port)]
        if not 0 <= int(segment_index) < source.set_count:
            return False
        name = source.segment_name(int(segment_index))
        changed = False
        for port_index, pulse in enumerate(self._pulse):
            if (
                port_index == int(source_port)
                or int(segment_index) >= pulse.set_count
            ):
                continue
            if pulse.segment_name(int(segment_index)) != name:
                pulse.rename_segment(int(segment_index), name)
                changed = True
        return changed

    def _share_matching_segment_names(self, source_port: int) -> bool:
        """Share all user-facing names present on the source AWG output."""
        if not 0 <= int(source_port) < len(self._pulse):
            return False
        changed = False
        for segment_index in range(self._pulse[int(source_port)].set_count):
            changed |= self._share_segment_name(
                int(source_port),
                segment_index,
            )
        return changed

    def _on_segment_timing_changed(
        self,
        source_port: int,
        segment_index: int,
    ) -> None:
        self._share_segment_timing(source_port, segment_index)
        self._multi_ctrl.refresh_table()
        self._refresh_waveform_plot()
        self._refresh_trace_if_needed(force=True)
        self._refresh_rf_editor()
        self._refresh_sweep_overlay()
        self.statusBar().showMessage(
            f"Segment {int(segment_index) + 1} RAMP/hold timing shared "
            "across AWG outputs"
        )

    def _on_segment_name_changed(
        self,
        source_port: int,
        segment_index: int,
    ) -> None:
        name = self._pulse[int(source_port)].segment_name(int(segment_index))
        self._share_segment_name(source_port, segment_index)
        self._multi_ctrl.refresh_table()
        self._refresh_trace_if_needed(force=True)
        self._refresh_rf_editor()
        self.statusBar().showMessage(
            f"Segment {int(segment_index) + 1} renamed to {name!r} "
            "across AWG outputs"
        )

    def _synchronize_port_timing(self) -> None:
        """Copy the selected SET/RAMP timing grid to every other port."""
        source = self._pulse[self._selected_port_idx]
        for index, pulse in enumerate(self._pulse):
            if index == self._selected_port_idx:
                continue
            levels = [float(np.interp(source.t[set_idx], pulse.t, pulse.v))
                      for set_idx in range(0, len(source.t) - 1, 2)]
            pulse.t = source.t.copy()
            pulse.v = np.empty_like(source.v)
            for set_number, level in enumerate(levels):
                pulse.v[2 * set_number:2 * set_number + 2] = level
            pulse.segment_names = list(source.segment_names)
        self._plot.refresh()
        self._plot.fit_view()
        self._multi_ctrl.refresh_table()
        self._refresh_trace_if_needed(force=True)
        self._refresh_rf_editor()
        self._refresh_sweep_overlay(fit_view=True, sync_rows=True)
        self.statusBar().showMessage("Port timing synchronized for QICK export")

    def _add_port(self):
        """Add a new control panel for a new PulseSequence."""
        if (
            hasattr(self, "_experiment_panel")
            and not self._suspend_qcs_output_sync
        ):
            self._experiment_panel.set_qcs_output_count(
                len(self._pulse) + 1
            )
        # Start with the existing timing grid so multi-output QICK export is
        # immediately valid; levels remain independently editable per port.
        new_pulse       = self._pulse[0].copy()
        new_pulse.v[:]  = DEFAULT_INITIAL_VOLTAGE_MV
        self._plot.add_pulse(new_pulse)
        color = (
            self._plot.line_color(len(self._plot._line) - 1)
            if hasattr(self._plot, "line_color")
            else self._plot._line[-1].get_color()
        )
        self._multi_ctrl._color_map.append(color)
        self._pulse.append(new_pulse)
        self._awg_output_display_names.append(
            f"awg_{len(self._pulse) - 1}"
        )
        self._qick_awg_channels = (
            *self._qick_awg_channels,
            self._next_qick_awg_channel(),
        )
        if hasattr(self, "_experiment_panel"):
            self._experiment_panel.set_qick_values(
                fabric_mhz=self._qick_fabric_mhz,
                tproc_mhz=self._qick_tproc_mhz,
                full_scale_mv=self._qick_full_scale_mv,
                awg_channels=self._qick_awg_channels,
                repetitions=self._qick_repetitions_per_sweep,
            )
        old_count = self._cross_capacitance.shape[0]
        expanded_coupling = np.eye(old_count + 1, dtype=float)
        expanded_coupling[:old_count, :old_count] = self._cross_capacitance
        self._cross_capacitance = expanded_coupling
        new_ctrl = ControlPanel(new_pulse, time_unit=self._time_unit)
        self._wire_control_panel(new_ctrl)
        self._multi_ctrl._ctrl_pannels.append(new_ctrl)
        self._multi_ctrl.splitter.insertWidget(len(self._multi_ctrl._ctrl_pannels) - 1, new_ctrl)
        self._multi_ctrl.set_awg_channels(self._qick_awg_channels)
        self._sync_awg_output_display_names()
        self._multi_ctrl.update_port_strip_geometry()
        pending_preserved = False
        if not self._suspend_qcs_output_sync:
            pending_preserved = (
                self._synchronize_pending_qcs_front_panel_output_count()
            )
        if (
            not self._suspend_qcs_output_sync
            and not pending_preserved
            and self._experiment_panel.execution_backend()
            == EXECUTION_BACKEND_QCS
        ):
            self._propagate_qcs_hardware_configuration(
                *self._qcs_bindings_from_settings(
                    self._experiment_panel.qcs_settings_dict()
                ),
                require_complete_rf=False,
            )
        self._multi_ctrl.refresh_table()
        self.refresh_panel_table()
        self._port_select(len(self._pulse) - 1)
        self._notify_sweep_state_changed(fit_view=True)

    def _port_select(self, idx: int) -> None:
        """Activate the selected control panel."""
        self._selected_port_idx         = idx
        self._plot._selected_port_idx   = idx
        self._plot.set_selected_port_idx(idx)
        self._multi_ctrl.set_selected_port(idx)
        self._refresh_waveform_plot(idx)

    def _refresh_waveform_plot(self, index: Optional[int] = None) -> None:
        """Refresh either plot backend without mixing their call signatures."""
        if _USE_PYQTGRAPH:
            self._plot.refresh(index)
        else:
            self._plot.refresh()

    def _plot_refresh(self):
        self._refresh_waveform_plot(self._selected_port_idx)
        self._refresh_trace_if_needed()
        self._refresh_sweep_overlay()
        if self._selected_port_idx == 0:
            self._refresh_rf_editor()

    def _make_row_slot(self, real_slot):
        def wrapper(checked=False):
            btn   = self.sender()
            row   = self._multi_ctrl.panel_table.indexAt(btn.pos()).row()
            real_slot(row)
        return wrapper
    def refresh_panel_table(self):
        """Refresh the control panel table with current pulse data."""
        backend = (
            self._experiment_panel.execution_backend()
            if hasattr(self, "_experiment_panel")
            else DEFAULT_EXECUTION_BACKEND
        )
        is_qcs = backend == EXECUTION_BACKEND_QCS
        header_item = self._multi_ctrl.panel_table.horizontalHeaderItem(2)
        if header_item is not None:
            header_item.setText(
                "QCS channel" if is_qcs else "QICK output"
            )
        qcs_dc_names: Tuple[str, ...] = ()
        qcs_dc_mappings = {}
        if is_qcs and hasattr(self, "_experiment_panel"):
            try:
                pending = self._pending_qcs_front_panel_state()
                if pending is None:
                    qcs_settings = self._experiment_panel.qcs_settings_dict()
                    qcs_dc_names = tuple(
                        str(name)
                        for name in qcs_settings.get(
                            "dc_channel_names",
                            (),
                        )
                    )
                    configuration = qcs_settings.get(
                        "hardware_configuration"
                    )
                else:
                    configuration, qcs_dc_names, _rf, _acquisition = pending
                if configuration is not None:
                    normalized = normalize_qcs_hardware_configuration(
                        configuration
                    )
                    qcs_dc_mappings = {
                        int(mapping["logical_index"]): mapping
                        for mapping in normalized["channel_mappings"]
                        if mapping["role"] == "dc"
                    }
            except (TypeError, ValueError):
                qcs_dc_names = tuple(
                    name.strip()
                    for name in (
                        self._experiment_panel.qcs_dc_channel_names.text()
                    ).split(",")
                    if name.strip()
                )
        self._multi_ctrl.panel_table.setRowCount(len(self._multi_ctrl._ctrl_pannels))
        for idx, _ in enumerate(self._multi_ctrl._ctrl_pannels):
            original_name = f"awg_{idx}"
            display_name = self._awg_output_display_names[idx]
            item_idx = QtWidgets.QTableWidgetItem(
                display_name
                if display_name == original_name
                else f"{display_name}\n({original_name})"
            )
            item_idx.setTextAlignment(QtCore.Qt.AlignCenter)
            item_idx.setToolTip(
                f"Display name: {display_name}\nOriginal name: {original_name}"
            )
            self._multi_ctrl.panel_table.setItem(idx, 0, item_idx)

            color = self._multi_ctrl._color_map[idx]
            item_color = QtWidgets.QTableWidgetItem()
            if isinstance(color, QtGui.QColor):
                table_color = QtGui.QColor(color)
            elif isinstance(color, tuple):
                rgba = tuple(color) + (1.0,) * (4 - len(color))
                table_color = QtGui.QColor.fromRgbF(*rgba[:4])
            else:
                table_color = QtGui.QColor(str(color))
            item_color.setBackground(table_color)
            self._multi_ctrl.panel_table.setItem(idx, 1, item_color)

            if is_qcs:
                channel_text = (
                    qcs_dc_names[idx]
                    if idx < len(qcs_dc_names)
                    else f"DC {idx}: not mapped"
                )
                mapping = qcs_dc_mappings.get(idx)
                if mapping is not None:
                    channel_text += (
                        f" | slot {int(mapping['slot'])} "
                        f"ch{int(mapping['channel'])}"
                    )
            else:
                channel_text = f"gen {self._qick_awg_channels[idx]}"
            item_hardware = QtWidgets.QTableWidgetItem(channel_text)
            item_hardware.setTextAlignment(QtCore.Qt.AlignCenter)
            self._multi_ctrl.panel_table.setItem(
                idx,
                2,
                item_hardware,
            )

            btn_x = QtWidgets.QPushButton("set_x")
            btn_x.clicked.connect(self._make_row_slot(self._set_x))
            self._multi_ctrl.panel_table.setCellWidget(idx, 3, btn_x)

            btn_y = QtWidgets.QPushButton("set_y")
            btn_y.clicked.connect(self._make_row_slot(self._set_y))
            self._multi_ctrl.panel_table.setCellWidget(idx, 4, btn_y)

        self._refresh_stability_targets()

    def _refresh_stability_targets(self) -> None:
        if not hasattr(self, "_stability_panel"):
            return
        channels = (
            tuple(range(len(self._pulse)))
            if (
                hasattr(self, "_experiment_panel")
                and self._experiment_panel.execution_backend()
                == EXECUTION_BACKEND_QCS
            )
            else self._qick_awg_channels
        )
        self._stability_panel.refresh_targets(
            self._qick_output_names(),
            channels,
            display_names=self._awg_output_display_names,
        )

    def _set_x(self, idx):
        trace = self._ensure_trace_widget()
        trace.x_idx = idx
        trace.refresh_trace(self._pulse)
        trace.fit_view()

    def _set_y(self, idx):
        trace = self._ensure_trace_widget()
        trace.y_idx = idx
        trace.refresh_trace(self._pulse)
        trace.fit_view()

    def closeEvent(self, event) -> None:
        if self._qcs_mapper_commit_pending():
            QtWidgets.QMessageBox.warning(
                self,
                "QCS mapper updating",
                "Wait for the background QCS mapper update to finish before "
                "closing.",
            )
            event.ignore()
            return
        if (
            self._awg_sweep_load_thread is not None
            and self._awg_sweep_load_thread.isRunning()
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "AWG sweep loading",
                "Wait for the saved AWG sweep run to finish loading.",
            )
            event.ignore()
            return
        if (
            self._stability_plot_load_thread is not None
            and self._stability_plot_load_thread.isRunning()
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Stability Diagram loading",
                "Wait for the saved Stability Diagram run to finish loading.",
            )
            event.ignore()
            return
        if (
            self._stability_overlay_thread is not None
            and self._stability_overlay_thread.isRunning()
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Stability overlay loading",
                "Wait for the saved Stability Diagram run to finish loading.",
            )
            event.ignore()
            return
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.warning(
                self,
                "Hardware task running",
                "The hardware task is still running. Wait for it to finish before closing.",
            )
            event.ignore()
            return
        if hasattr(self, "_awg_sweep_plot"):
            self._awg_sweep_plot.close()
        super().closeEvent(event)

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1500, 650)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
