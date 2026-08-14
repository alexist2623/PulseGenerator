"""Headless tests for two-electrode stability-diagram acquisition.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
import sqlite3
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtWidgets
import pytest

from qick_fine_tune_sweep import AmplitudeSweep, FineTuneDdrResult
from qick_qcodes_experiment import (
    QCODES_STAGING_ENV,
    QcodesRunConfig,
    QickConnectionConfig,
    store_qick_result,
)
from qcs_qcodes_experiment import QcsAcquisitionConfig, QcsConnectionConfig
import qcs_front_panel as front_panel
import stability_diagram as stability
from fir_ddr_profile import FirDdrProfile


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _fir_profile(*, is_50_ksps: bool = False):
    sample_rate_hz = 50_000.0 if is_50_ksps else 1_000_000.0
    trigger_delay_samples = 50 if is_50_ksps else 0
    sample_period_us = 1_000_000.0 / sample_rate_hz
    return FirDdrProfile(
        name="50_ksps" if is_50_ksps else "1_msps",
        sample_rate_hz=sample_rate_hz,
        sample_rate_msps=sample_rate_hz / 1_000_000.0,
        sample_period_us=sample_period_us,
        decimation=6000 if is_50_ksps else 300,
        input_rate_mhz=300.0,
        group_delay_input_samples=296_677.0 if is_50_ksps else 8_677.0,
        uses_fpga_trigger_delay=is_50_ksps,
        trigger_delay_samples=trigger_delay_samples,
        software_warmup_compensation=not is_50_ksps,
        config={},
        trigger_delay_units=(
            "valid_input_samples" if is_50_ksps else "none"
        ),
    )


def _config() -> stability.StabilityDiagramConfig:
    return stability.StabilityDiagramConfig(
        x_axis=stability.StabilitySweepAxis(
            "awg_0", -100.0, 100.0, 2
        ),
        y_axis=stability.StabilitySweepAxis(
            "awg_1", -50.0, 50.0, 2
        ),
        repetitions_per_point=2,
        trace_samples_per_point=3,
    )


def _ddr_result():
    axes = (
        SimpleNamespace(
            output_name="awg_0",
            segment_name=stability.STABILITY_HOLD_SEGMENT,
            start=-1.0,
            stop=1.0,
            count=2,
        ),
        SimpleNamespace(
            output_name="awg_1",
            segment_name=stability.STABILITY_HOLD_SEGMENT,
            start=-0.5,
            stop=0.5,
            count=2,
        ),
    )
    coordinates = np.asarray(
        (
            (-1.0, -0.5),
            (-1.0, 0.5),
            (1.0, -0.5),
            (1.0, 0.5),
        ),
        dtype=float,
    )
    iq = np.empty((4, 2, 3, 2), dtype=np.int32)
    for point in range(4):
        for repetition in range(2):
            for sample in range(3):
                iq[point, repetition, sample, 0] = (
                    10 * (point + 1) + repetition + sample
                )
                iq[point, repetition, sample, 1] = (
                    -5 * (point + 1) - repetition - sample
                )
    return SimpleNamespace(
        sweep_points=coordinates,
        sweep_axes=axes,
        iq=iq,
    )


@dataclass(frozen=True)
class _FakeReadoutSpec:
    fpga_trigger_delay_samples: int = 50
    fpga_trigger_delay_us: float | None = None


def _worker_kwargs(tmp_path=None):
    return {
        "connection_config": QickConnectionConfig(
            host="127.0.0.1",
            ns_port=8888,
            proxy_name="testqick",
        ),
        "run_config": (
            None
            if tmp_path is None
            else QcodesRunConfig(database_path=str(tmp_path / "stability.db"))
        ),
        "gui_settings": (
            None
            if tmp_path is None
            else {
                "qick": {"fabric_mhz": 300.0, "full_scale_mv": 100.0},
                "awg": {"outputs": []},
            }
        ),
        "stability_config": _config(),
        "full_scale_mv": 100.0,
        "sequence": SimpleNamespace(),
        "awg_channels": (1, 3),
        "repetitions_per_sweep": 2,
        "tproc_mhz": 300.0,
        "rf_specs": (),
        "readout_spec": _FakeReadoutSpec(),
        "progress": False,
    }


def test_stability_config_requires_two_outputs_and_respects_full_scale():
    with pytest.raises(ValueError, match="different AWG outputs"):
        stability.StabilityDiagramConfig(
            x_axis=stability.StabilitySweepAxis(
                "awg_0", -10.0, 10.0, 2
            ),
            y_axis=stability.StabilitySweepAxis(
                "awg_0", -10.0, 10.0, 2
            ),
        )

    config = _config()
    config.validate_full_scale(100.0)
    with pytest.raises(ValueError, match="stability sweep exceeds"):
        config.validate_full_scale(49.0)

    compensated = stability.StabilityDiagramConfig(
        x_axis=stability.StabilitySweepAxis(
            "awg_0", -100.0, 100.0, 2
        ),
        y_axis=stability.StabilitySweepAxis(
            "awg_1", -50.0, 50.0, 2
        ),
        bias_t_compensation_enabled=True,
        bias_t_compensation_voltage_mv=125.0,
    )
    with pytest.raises(ValueError, match="compensation voltage exceeds"):
        compensated.validate_full_scale(100.0)


def test_stability_settings_add_backward_compatible_bias_t_defaults():
    legacy = stability.default_stability_settings(("awg_0", "awg_1"), ("set_0",))
    legacy.pop("bias_t_compensation")
    legacy.pop("color_ranges")
    legacy.pop("visible_data")
    legacy.pop("saved_plot_database_path")
    legacy.pop("saved_plot_run_id")
    legacy.pop("saved_plot_data")

    normalized = stability.normalize_stability_settings(
        legacy,
        output_names=("awg_0", "awg_1"),
        segment_names=("set_0",),
    )

    assert normalized["bias_t_compensation"] == {
        "enabled": False,
        "type": "dc",
        "mode": "fixed_voltage",
        "voltage_mv": stability.DEFAULT_STABILITY_BIAS_T_COMPENSATION_MV,
        "duration_us": 1.0,
        "filter_tau_us": 100.0,
    }
    assert normalized["settle_time_us"] == stability.DEFAULT_STABILITY_SETTLE_US
    assert "segment_name" not in normalized["x_axis"]
    assert normalized["color_ranges"] == {
        "i": {
            "auto": True,
            "minimum": -1.0,
            "maximum": 1.0,
        },
        "q": {
            "auto": True,
            "minimum": -1.0,
            "maximum": 1.0,
        },
        "magnitude": {
            "auto": True,
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "phase": {
            "auto": False,
            "minimum": -180.0,
            "maximum": 180.0,
        },
    }
    assert normalized["visible_data"] == ["magnitude", "phase"]
    assert normalized["saved_plot_database_path"] == normalized["database_path"]
    assert normalized["saved_plot_run_id"] == 0
    assert normalized["saved_plot_data"] == "magnitude"

    invalid = dict(normalized)
    invalid["color_ranges"] = {
        "magnitude": {
            "auto": False,
            "minimum": 10.0,
            "maximum": 10.0,
        },
        "phase": normalized["color_ranges"]["phase"],
    }
    with pytest.raises(ValueError, match="minimum must be below maximum"):
        stability.normalize_stability_settings(
            invalid,
            output_names=("awg_0", "awg_1"),
        )


def test_stability_settings_repair_duplicate_axes_but_allow_one_output():
    duplicate = stability.default_stability_settings(
        ("awg_0", "awg_1", "awg_2")
    )
    duplicate["y_axis"]["output_name"] = "awg_0"

    normalized = stability.normalize_stability_settings(
        duplicate,
        output_names=("awg_0", "awg_1", "awg_2"),
    )

    assert normalized["x_axis"]["output_name"] == "awg_0"
    assert normalized["y_axis"]["output_name"] == "awg_1"

    single_output = stability.normalize_stability_settings(
        stability.default_stability_settings(("awg_0",)),
        output_names=("awg_0",),
    )
    assert single_output["x_axis"]["output_name"] == "awg_0"
    assert single_output["y_axis"]["output_name"] == "awg_0"


def test_stability_builds_dedicated_set_hold_sequence_without_awg_waveform():
    config = _config()
    sequence = stability.build_stability_hold_sequence(
        config,
        output_names=("awg_0", "awg_1", "awg_2"),
        fabric_mhz=300.0,
        full_scale_mv=100.0,
        cross_capacitance=np.eye(3),
    )

    assert len(sequence.segments) == 1
    segment = sequence.segments[0]
    assert segment.name == stability.STABILITY_HOLD_SEGMENT
    assert segment.kind == "set"
    assert segment.amplitudes == (0.0, 0.0, 0.0)
    assert segment.duration_cycles == int(
        np.ceil(
            (
                config.settle_time_us
                + config.trace_samples_per_point
                + stability.DEFAULT_STABILITY_POINT_GUARD_US
            )
            * 300.0
        )
    )
    assert [item.segment_name for item in sequence.sweeps] == ["set_0", "set_0"]
    assert [item.output_name for item in sequence.sweeps] == ["awg_0", "awg_1"]


def test_stability_50ksps_hold_covers_only_immediate_trace():
    config = _config()
    sequence = stability.build_stability_hold_sequence(
        config,
        output_names=("awg_0", "awg_1"),
        fabric_mhz=300.0,
        full_scale_mv=100.0,
        sample_period_us=20.0,
    )

    expected_hold_us = (
        config.settle_time_us
        + config.trace_samples_per_point * 20.0
        + stability.DEFAULT_STABILITY_POINT_GUARD_US
    )
    assert sequence.segments[0].duration_cycles == int(
        np.ceil(expected_hold_us * 300.0)
    )


def test_reduce_fir_result_restores_voltage_grid_and_coherent_iq_mean():
    raw = _ddr_result()
    result = stability.reduce_fir_stability_result(
        raw,
        _config(),
        full_scale_mv=100.0,
        iteration=7,
    )

    expected_per_point = raw.iq.astype(float).mean(axis=(1, 2))
    assert result.x_voltage_mv.tolist() == [-100.0, 100.0]
    assert result.y_voltage_mv.tolist() == [-50.0, 50.0]
    assert result.i_mean.tolist() == [
        [expected_per_point[0, 0], expected_per_point[2, 0]],
        [expected_per_point[1, 0], expected_per_point[3, 0]],
    ]
    assert result.q_mean.tolist() == [
        [expected_per_point[0, 1], expected_per_point[2, 1]],
        [expected_per_point[1, 1], expected_per_point[3, 1]],
    ]
    assert np.allclose(result.magnitude, np.hypot(result.i_mean, result.q_mean))
    assert np.allclose(
        result.phase_deg,
        np.degrees(np.arctan2(result.q_mean, result.i_mean)),
    )
    assert result.iteration == 7
    assert result.x_axis_label == "awg_0"
    assert result.y_axis_label == "awg_1"
    assert result.repetition_count == 2
    assert result.samples_per_trace == 3
    assert result.value_unit == "ADC units"
    assert result.measurement_mode == "raw_iq"


@pytest.mark.skipif(stability.pg is None, reason="pyqtgraph is not installed")
def test_stability_plot_exposes_and_applies_color_ranges():
    app = _application()
    result = stability.reduce_fir_stability_result(
        _ddr_result(),
        _config(),
        full_scale_mv=100.0,
        iteration=3,
    )
    plot = stability.StabilityDiagramPlotWidget()
    plot.set_result(result)
    app.processEvents()

    magnitude_levels = plot._levels(result.magnitude)
    np.testing.assert_allclose(
        plot.magnitude_image.getLevels(),
        magnitude_levels,
    )
    np.testing.assert_allclose(
        plot.phase_image.getLevels(),
        (-180.0, 180.0),
    )
    assert plot.magnitude_range_control.auto_range.isChecked() is True
    assert plot.phase_range_control.auto_range.isChecked() is False
    assert plot.magnitude_color_bar is not None
    assert plot.phase_color_bar is not None
    assert plot.magnitude_color_bar.interactive is True
    assert plot.phase_color_bar.interactive is True
    assert plot.visible_data() == ("magnitude", "phase")
    assert plot.plot_cells["i"].isHidden() is True
    assert plot.plot_cells["q"].isHidden() is True
    assert plot.plot_cells["magnitude"].isHidden() is False
    assert plot.plot_cells["phase"].isHidden() is False
    assert "Applied:" in plot.magnitude_range_control.range_status.text()
    assert "Data:" in plot.magnitude_range_control.range_status.text()

    plot.magnitude_color_bar.setLevels((7.5, 42.5))
    plot.magnitude_color_bar.sigLevelsChanged.emit(plot.magnitude_color_bar)
    app.processEvents()
    assert plot.magnitude_range_control.auto_range.isChecked() is False
    np.testing.assert_allclose(
        (
            plot.magnitude_range_control.minimum.value(),
            plot.magnitude_range_control.maximum.value(),
        ),
        (7.5, 42.5),
    )
    np.testing.assert_allclose(
        plot.magnitude_image.getLevels(),
        (7.5, 42.5),
    )

    plot.load_color_range_settings({
        "magnitude": {
            "auto": False,
            "minimum": 5.0,
            "maximum": 50.0,
        },
        "phase": {
            "auto": False,
            "minimum": -45.0,
            "maximum": 90.0,
        },
    })
    app.processEvents()
    np.testing.assert_allclose(plot.magnitude_image.getLevels(), (5.0, 50.0))
    np.testing.assert_allclose(plot.phase_image.getLevels(), (-45.0, 90.0))
    assert plot.color_range_settings() == {
        "i": {
            "auto": True,
            "minimum": plot._symmetric_levels(result.i_mean)[0],
            "maximum": plot._symmetric_levels(result.i_mean)[1],
        },
        "q": {
            "auto": True,
            "minimum": plot._symmetric_levels(result.q_mean)[0],
            "maximum": plot._symmetric_levels(result.q_mean)[1],
        },
        "magnitude": {
            "auto": False,
            "minimum": 5.0,
            "maximum": 50.0,
        },
        "phase": {
            "auto": False,
            "minimum": -45.0,
            "maximum": 90.0,
        },
    }
    plot.load_visible_data(["i", "angle"])
    app.processEvents()
    assert plot.visible_data() == ("i", "phase")
    assert plot.plot_cells["i"].isHidden() is False
    assert plot.plot_cells["q"].isHidden() is True
    assert plot.plot_cells["magnitude"].isHidden() is True
    assert plot.plot_cells["phase"].isHidden() is False
    np.testing.assert_allclose(plot.images["i"].image, result.i_mean)
    np.testing.assert_allclose(plot.images["phase"].image, result.phase_deg)

    plot.data_selectors["i"].setChecked(False)
    plot.data_selectors["phase"].setChecked(False)
    app.processEvents()
    assert len(plot.visible_data()) == 1

    plot.close()
    plot.deleteLater()
    app.processEvents()


def test_reduce_fir_result_converts_dc_input_iq_to_current():
    raw = _ddr_result()
    raw_result = stability.reduce_fir_stability_result(
        raw,
        _config(),
        full_scale_mv=100.0,
    )
    current_result = stability.reduce_fir_stability_result(
        raw,
        _config(),
        full_scale_mv=100.0,
        readout_spec=SimpleNamespace(
            input_board_type="DC_In",
            dc_measure_mode=True,
            dc_measure_gain_v_per_a=2.0,
        ),
    )

    np.testing.assert_allclose(current_result.i_mean, raw_result.i_mean / 2.0)
    np.testing.assert_allclose(current_result.q_mean, raw_result.q_mean / 2.0)
    np.testing.assert_allclose(
        current_result.magnitude,
        raw_result.magnitude / 2.0,
    )
    np.testing.assert_allclose(current_result.phase_deg, raw_result.phase_deg)
    assert current_result.value_unit == "A"
    assert current_result.measurement_mode == "dc_current_iq"


def test_reduce_fir_result_auto_scales_nanoamp_current():
    raw = _ddr_result()
    raw.iq = raw.iq.astype(np.float64) * 1.0e-9
    current_result = stability.reduce_fir_stability_result(
        raw,
        _config(),
        full_scale_mv=100.0,
        readout_spec=SimpleNamespace(
            input_board_type="DC_In",
            effective_measurement_representation="current",
            dc_measure_gain_v_per_a=1.0,
            dc_voltage_calibration_enabled=False,
        ),
    )

    assert current_result.value_unit == "nA"
    assert current_result.base_value_unit == "A"
    assert current_result.display_scale == 1.0e9
    assert current_result.measurement_mode == "dc_current_iq"
    assert np.nanmax(np.abs(current_result.i_mean)) > 1.0


def test_stability_panel_measurement_representation_round_trip():
    app = _application()
    panel = stability.StabilityDiagramPanel()
    panel.refresh_targets(("awg_0", "awg_1"), (1, 3))
    panel.apply_path_settings({
        **stability.DEFAULT_STABILITY_RF_PATH,
        "input_board_type": "DC_In",
    })
    panel.measurement_unit.setCurrentIndex(
        panel.measurement_unit.findData("current")
    )
    panel.dc_measure_gain_v_per_a.setValue(1.0e8)
    settings = panel.settings_dict()

    assert settings["measurement_representation"] == "current"
    assert panel.dc_measure_mode.isChecked() is True
    restored = stability.StabilityDiagramPanel()
    restored.refresh_targets(("awg_0", "awg_1"), (1, 3))
    restored.load_settings(settings)
    app.processEvents()
    assert restored.measurement_unit.currentData() == "current"
    assert restored.dc_measure_mode.isChecked() is True
    assert restored.dc_measure_gain_v_per_a.value() == 1.0e8

    restored.close()
    panel.close()


def test_stability_qcs_picker_uses_current_logical_output_channel():
    app = _application()
    panel = stability.StabilityDiagramPanel()
    panel.refresh_targets(("awg_2", "awg_0"), (2, 0))
    panel.set_hardware_backend("qcs")
    panel.set_qcs_front_panel_configuration(
        front_panel.default_qcs_hardware_configuration(
            ("zero", "one", "two"),
            {},
            None,
        )
    )

    panel.x_axis.output.setCurrentIndex(0)
    assert panel.x_axis.qcs_front_panel_selection() == ("dc", 2)
    assert "two /" in panel.x_axis.front_panel_status.text()
    panel.x_axis.output.setCurrentIndex(1)
    assert panel.x_axis.qcs_front_panel_selection() == ("dc", 0)
    assert "zero /" in panel.x_axis.front_panel_status.text()
    assert panel.y_axis.qcs_front_panel_selection() == ("dc", 2)
    assert "two /" in panel.y_axis.front_panel_status.text()
    assert (
        panel.x_axis.output.currentData()
        != panel.y_axis.output.currentData()
    )
    app.processEvents()
    panel.close()


def test_stability_electrodes_show_independent_clickable_qcs_previews():
    app = _application()
    panel = stability.StabilityDiagramPanel()
    panel.refresh_targets(
        ("awg_0", "awg_1", "awg_2"),
        (0, 1, 2),
    )
    panel.set_hardware_backend("qcs")
    configuration = front_panel.default_qcs_hardware_configuration(
        ("x_gate", "y_gate", "spare_gate"),
        {},
        None,
    )
    panel.set_qcs_front_panel_configuration(configuration)
    panel.resize(760, 1200)
    panel.show()
    app.processEvents()

    x_preview = panel.x_axis.front_panel_preview
    y_preview = panel.y_axis.front_panel_preview
    mappings = {
        int(mapping["logical_index"]): (
            int(mapping["slot"]),
            int(mapping["channel"]),
        )
        for mapping in configuration["channel_mappings"]
        if mapping["role"] == "dc"
    }
    assert x_preview is not y_preview
    assert x_preview.currentWidget() is x_preview.qcs_preview
    assert y_preview.currentWidget() is y_preview.qcs_preview
    assert x_preview.isVisible() is True
    assert y_preview.isVisible() is True
    assert x_preview.qcs_preview._selected_address() == mappings[0]
    assert y_preview.qcs_preview._selected_address() == mappings[1]
    assert "x_gate" in x_preview.qcs_preview.binding_label.text()
    assert "y_gate" in y_preview.qcs_preview.binding_label.text()

    requested = []
    panel.electrode_front_panel_requested.connect(requested.append)
    x_preview.activated.emit()
    y_preview.activated.emit()
    assert requested == [panel.x_axis, panel.y_axis]

    panel.x_axis.output.setCurrentIndex(
        panel.x_axis.output.findData("awg_2")
    )
    app.processEvents()
    assert x_preview.qcs_preview._selected_address() == mappings[2]
    assert y_preview.qcs_preview._selected_address() == mappings[1]
    panel.close()


def test_stability_axis_selectors_swap_instead_of_sharing_an_output():
    app = _application()
    panel = stability.StabilityDiagramPanel()
    panel.refresh_targets(
        ("awg_0", "awg_1", "awg_2"),
        (0, 1, 2),
    )

    panel.x_axis.output.setCurrentIndex(
        panel.x_axis.output.findData("awg_1")
    )
    assert panel.x_axis.output.currentData() == "awg_1"
    assert panel.y_axis.output.currentData() == "awg_0"

    panel.y_axis.output.setCurrentIndex(
        panel.y_axis.output.findData("awg_1")
    )
    assert panel.y_axis.output.currentData() == "awg_1"
    assert panel.x_axis.output.currentData() == "awg_0"
    assert (
        panel.x_axis.qcs_front_panel_selection()
        != panel.y_axis.qcs_front_panel_selection()
    )
    assert panel.config(full_scale_mv=1000.0).x_axis.output_name == "awg_0"
    app.processEvents()
    panel.close()


def test_stability_one_output_leaves_y_unassigned_and_measurement_disabled():
    app = _application()
    panel = stability.StabilityDiagramPanel()
    panel.refresh_targets(("awg_0",), (0,))

    assert panel.x_axis.output.currentData() == "awg_0"
    assert panel.y_axis.output.currentData() is None
    assert panel.x_axis.front_panel_preview.isHidden() is False
    assert panel.y_axis.front_panel_preview.isHidden() is False
    assert (
        panel.y_axis.front_panel_preview.qcs_preview._selected_address()
        is None
    )
    assert "no logical channel selected" in (
        panel.y_axis.front_panel_preview.qcs_preview.binding_label.text()
    )
    assert panel.y_axis.settings_dict()["output_name"] == "awg_0"
    assert panel.start_button.isEnabled() is False
    with pytest.raises(ValueError, match="at least two AWG outputs"):
        panel.config(full_scale_mv=1000.0)
    app.processEvents()
    panel.close()


def test_continuous_worker_repeats_without_qcodes_storage(monkeypatch):
    calls = {"execute": 0, "store": 0}

    monkeypatch.setattr(
        stability,
        "connect_qick",
        lambda *_args, **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        stability,
        "resolve_fir_ddr_profile",
        lambda *_args, **_kwargs: _fir_profile(),
    )

    class FakeProgram:
        def summary(self):
            return {}

    def fake_execute(*_args, **_kwargs):
        calls["execute"] += 1
        return FakeProgram(), _ddr_result(), {}

    def forbidden_store(*_args, **_kwargs):
        calls["store"] += 1
        raise AssertionError("continuous mode must not write QCoDeS")

    monkeypatch.setattr(stability, "execute_qick_sequence", fake_execute)
    monkeypatch.setattr(stability, "store_qick_result", forbidden_store)
    worker = stability.StabilityDiagramWorker(
        _worker_kwargs(),
        continuous=True,
    )
    scans = []
    stopped = []
    worker.scan_ready.connect(
        lambda result: (scans.append(result), worker.request_stop())
    )
    worker.stopped.connect(lambda: stopped.append(True))

    worker.run()

    assert calls == {"execute": 1, "store": 0}
    assert len(scans) == 1
    assert stopped == [True]


def test_qcs_continuous_worker_runs_one_native_grid_without_storage(
    monkeypatch,
):
    calls = {"execute": 0, "store": 0}
    config = _config()
    sequence = stability.build_stability_hold_sequence(
        config,
        output_names=("awg_0", "awg_1"),
        fabric_mhz=300.0,
        full_scale_mv=100.0,
    )
    compiled = SimpleNamespace(
        program=object(),
        sweep_shape=(2, 2),
        acquisition_duration_s=3e-6,
    )
    execution = SimpleNamespace(
        ddr_result=_ddr_result(),
        programs=(compiled.program,),
        raw_results=("raw",),
        program_summary={"backend": "qcs", "hardware_sweep": True},
        rf_settings={"backend": "qcs"},
    )
    monkeypatch.setattr(
        stability,
        "load_qcs_channel_mapper",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        stability,
        "compile_qcs_stability_hardware_sweep",
        lambda *_args, **_kwargs: compiled,
    )
    monkeypatch.setattr(
        stability,
        "build_qcs_executor",
        lambda *_args, **_kwargs: object(),
    )

    def fake_execute(**_kwargs):
        calls["execute"] += 1
        return execution

    def forbidden_store(*_args, **_kwargs):
        calls["store"] += 1
        raise AssertionError("continuous QCS scan must not write QCoDeS")

    monkeypatch.setattr(
        stability,
        "execute_qcs_stability_hardware_sweep",
        fake_execute,
    )
    monkeypatch.setattr(
        stability,
        "store_experiment_result",
        forbidden_store,
    )
    worker = stability.QcsStabilityDiagramWorker(
        {
            "connection_config": QcsConnectionConfig(
                mapper_path="unused.qcs",
                dc_channel_names=("dc_x", "dc_y"),
                acquisition_channel_name="digitizer",
            ),
            "run_config": None,
            "gui_settings": None,
            "stability_config": config,
            "full_scale_mv": 100.0,
            "sequence": sequence,
            "repetitions_per_point": 2,
            "fabric_mhz": 300.0,
            "rf_pulses": (),
            "acquisition": QcsAcquisitionConfig(
                at_segment="set_0",
                duration_s=32 / 4.8e9,
                sample_rate_hz=4.8e9,
                sample_count=32,
            ),
            "readout_spec": None,
        },
        continuous=True,
    )
    scans = []
    stopped = []
    worker.scan_ready.connect(
        lambda result: (scans.append(result), worker.request_stop())
    )
    worker.stopped.connect(lambda: stopped.append(True))

    worker.run()

    assert calls == {"execute": 1, "store": 0}
    assert len(scans) == 1
    assert scans[0].magnitude.shape == (2, 2)
    assert stopped == [True]


def test_qcs_single_worker_persists_effective_scale_and_monotonic_progress(
    tmp_path,
    monkeypatch,
):
    config = replace(
        _config(),
        bias_t_compensation_enabled=True,
        bias_t_compensation_type="dc",
        bias_t_compensation_mode="fixed_time",
        bias_t_compensation_duration_us=2.5,
    )
    full_scale_mv = 2500.0
    sequence = stability.build_stability_hold_sequence(
        config,
        output_names=("awg_0", "awg_1"),
        fabric_mhz=300.0,
        full_scale_mv=full_scale_mv,
    )
    iq = np.zeros((4, 2, 1, 2), dtype=float)
    ddr_result = FineTuneDdrResult(
        sweep_points=np.asarray(sequence.sweep_points),
        iq=iq,
        sweep_axes=tuple(sequence.sweep_axes),
        sweep_shape=tuple(sequence.sweep_shape),
        cross_capacitance=np.eye(2),
        sample_rate_hz=2.0e6,
        fir_rate_profile="qcs_hardware_demod",
    )
    compiled = SimpleNamespace(
        program=object(),
        sweep_shape=(2, 2),
        acquisition_duration_s=1.5e-6,
        bias_t_compensation_duration_s=2.5e-6,
    )
    execution = SimpleNamespace(
        ddr_result=ddr_result,
        programs=(compiled.program,),
        raw_results=("raw",),
        program_summary={
            "backend": "qcs",
            "hardware_sweep": True,
            "source_full_scale_mv": full_scale_mv,
        },
        rf_settings={"backend": "qcs"},
    )
    captured = {}

    monkeypatch.setattr(
        stability,
        "load_qcs_channel_mapper",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        stability,
        "compile_qcs_stability_hardware_sweep",
        lambda *_args, **_kwargs: compiled,
    )
    monkeypatch.setattr(
        stability,
        "build_qcs_executor",
        lambda *_args, **_kwargs: object(),
    )

    def fake_execute(**kwargs):
        kwargs["progress_callback"](0, "execute start")
        kwargs["progress_callback"](100, "execute complete")
        return execution

    def fake_store(_result, **kwargs):
        captured.update(kwargs)
        kwargs["progress_callback"](kwargs["progress_start"], "save start")
        kwargs["progress_callback"](kwargs["progress_end"], "save complete")
        return SimpleNamespace(run_id=7, guid="qcs-guid"), 8

    monkeypatch.setattr(
        stability,
        "execute_qcs_stability_hardware_sweep",
        fake_execute,
    )
    monkeypatch.setattr(
        stability,
        "store_experiment_result",
        fake_store,
    )
    worker = stability.QcsStabilityDiagramWorker(
        {
            "connection_config": QcsConnectionConfig(
                mapper_path="unused.qcs",
                dc_channel_names=("dc_x", "dc_y"),
                acquisition_channel_name="digitizer",
            ),
            "run_config": QcodesRunConfig(
                database_path=str(tmp_path / "qcs_stability.db")
            ),
            "gui_settings": {
                "qick": {
                    "fabric_mhz": 300.0,
                    "full_scale_mv": 800.0,
                },
                "stability_diagram": {},
                "awg": {"outputs": []},
            },
            "stability_config": config,
            "full_scale_mv": full_scale_mv,
            "sequence": sequence,
            "repetitions_per_point": 2,
            "fabric_mhz": 300.0,
            "rf_pulses": (),
            "acquisition": QcsAcquisitionConfig(
                at_segment="set_0",
                duration_s=32 / 4.8e9,
                sample_rate_hz=4.8e9,
                sample_count=32,
            ),
            "readout_spec": None,
        },
        continuous=False,
    )
    progress = []
    finished = []
    worker.progress_changed.connect(
        lambda percent, _message: progress.append(percent)
    )
    worker.single_finished.connect(finished.append)

    worker.run()

    assert len(finished) == 1
    assert progress == sorted(progress)
    assert progress[-1] == 100
    assert captured["progress_start"] == 65
    assert captured["progress_end"] == 100
    stored = captured["gui_settings"]
    assert stored["qick"]["full_scale_mv"] == 800.0
    assert (
        stored["stability_diagram"]["coordinate_full_scale_mv"]
        == full_scale_mv
    )
    assert (
        stored["stability_diagram"]["bias_t_compensation_applied"]
        is True
    )
    assert stored["stability_diagram"]["bias_t_compensation_type"] == "dc"
    assert (
        stored["stability_diagram"]["bias_t_compensation_mode"]
        == "fixed_time"
    )
    assert (
        stored["stability_diagram"]["bias_t_compensation_duration_us"]
        == pytest.approx(2.5)
    )
    assert stored["stability_diagram"]["hardware_sweep_shape"] == [4]
    assert stored["stability_diagram"]["stability_grid_shape"] == [2, 2]
    assert (
        stored["stability_diagram"]["acquisition_result_type"]
        == "integrated_iq"
    )
    metadata = {
        "gui_settings": stored,
        "program_summary": execution.program_summary,
    }
    np.testing.assert_allclose(
        stability._stored_coordinate_mv(
            np.asarray([-0.04, 0.04]),
            {"unit": "normalized"},
            metadata,
        ),
        [-100.0, 100.0],
    )


def test_worker_rebuilds_50ksps_sequence_and_rf_hold(monkeypatch):
    profile = _fir_profile(is_50_ksps=True)
    monkeypatch.setattr(
        stability,
        "connect_qick",
        lambda *_args, **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        stability,
        "resolve_fir_ddr_profile",
        lambda *_args, **_kwargs: profile,
    )

    template = SimpleNamespace(
        output_names=("awg_0", "awg_1"),
        cross_capacitance=np.eye(2),
    )
    kwargs = _worker_kwargs()
    kwargs["sequence"] = template
    kwargs["stability_fabric_mhz"] = 300.0
    @dataclass(frozen=True)
    class FakeRfSpec:
        duration_us: float

    kwargs["rf_specs"] = (FakeRfSpec(duration_us=1.0),)
    captured = {}

    class FakeProgram:
        def summary(self):
            return {}

    def fake_execute(*_args, **runtime_kwargs):
        captured.update(runtime_kwargs)
        worker.request_stop()
        return FakeProgram(), _ddr_result(), {}

    monkeypatch.setattr(stability, "execute_qick_sequence", fake_execute)
    monkeypatch.setattr(
        stability,
        "store_qick_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("continuous mode must not store")
        ),
    )
    worker = stability.StabilityDiagramWorker(kwargs, continuous=True)
    worker.run()

    sequence = captured["sequence"]
    expected_hold_us = (
        _config().settle_time_us
        + _config().trace_samples_per_point * profile.sample_period_us
        + stability.DEFAULT_STABILITY_POINT_GUARD_US
    )
    assert sequence.segments[0].duration_cycles == int(
        np.ceil(expected_hold_us * 300.0)
    )
    assert captured["rf_specs"][0].duration_us == (
        _config().trace_samples_per_point * profile.sample_period_us
    )
    assert captured["readout_spec"].fpga_trigger_delay_samples is None
    assert captured["readout_spec"].fpga_trigger_delay_us is None


def test_single_shot_worker_saves_exactly_once(monkeypatch, tmp_path):
    calls = {"store": 0}

    monkeypatch.setattr(
        stability,
        "connect_qick",
        lambda *_args, **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        stability,
        "resolve_fir_ddr_profile",
        lambda *_args, **_kwargs: _fir_profile(),
    )

    class FakeProgram:
        def summary(self):
            return {"cartesian_point_count": 4}

    monkeypatch.setattr(
        stability,
        "execute_qick_sequence",
        lambda *_args, **_kwargs: (FakeProgram(), _ddr_result(), {"rf": "ok"}),
    )

    dataset = SimpleNamespace(run_id=23, guid="test-guid")

    def fake_store(*_args, **_kwargs):
        calls["store"] += 1
        return dataset, 24

    monkeypatch.setattr(stability, "store_qick_result", fake_store)
    worker = stability.StabilityDiagramWorker(
        _worker_kwargs(tmp_path),
        continuous=False,
    )
    finished = []
    worker.single_finished.connect(finished.append)

    worker.run()

    assert calls["store"] == 1
    assert len(finished) == 1
    assert finished[0].run_id == 23
    assert finished[0].experiment.row_count == 24
    assert finished[0].diagram.magnitude.shape == (2, 2)


def test_stability_front_panel_remains_visible_while_controls_scroll():
    app = _application()
    panel = stability.StabilityDiagramPanel()
    panel.resize(640, 900)
    panel.show()
    app.processEvents()
    app.processEvents()

    scrollbar = panel.controls_scroll.verticalScrollBar()
    scrollbar.setValue(scrollbar.maximum())
    assert scrollbar.value() > 0
    app.processEvents()

    assert panel.front_panel_preview is panel.path_diagram.front_panel_preview
    assert panel.front_panel_preview.parentWidget() is panel
    assert panel.front_panel_preview.isVisible()
    assert panel.front_panel_preview.y() < panel.controls_scroll.y()
    panel.close()


def test_stability_qcs_rf_path_uses_modules_and_restores_qick_settings():
    app = _application()
    panel = stability.StabilityDiagramPanel()
    panel.refresh_targets(("awg_0", "awg_1"), (1, 3))
    panel.resize(900, 1200)
    panel.show()
    app.processEvents()

    legacy_path = {
        **stability.DEFAULT_STABILITY_RF_PATH,
        "output_board_type": "RF_Out",
        "input_board_type": "RF_In",
        "output_nqz": 2,
        "readout_nqz": 2,
        "output_att1_db": 7.25,
        "output_att2_db": 12.5,
        "readout_attenuation_db": 18.75,
    }
    panel.apply_path_settings(legacy_path)
    panel.trace_samples.setValue(4321)
    panel.override_fpga_trigger_delay.setChecked(True)
    panel.fpga_trigger_delay_us.setValue(17.5)
    panel.modulation_frequency_mhz.setValue(123.25)
    panel.modulation_gain.setValue(12345)
    panel.dc_calibration_path.setText("legacy_dc_calibration.db")
    panel.dc_calibration_run_id.setValue(29)
    panel.dc_calibration_group.setChecked(True)
    legacy_settings = panel.settings_dict()

    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_gate",),
        {0: "rf_drive"},
        "digitizer",
    )
    panel.set_qcs_front_panel_configuration(configuration)
    panel.set_hardware_backend("qcs")
    app.processEvents()

    path = panel.path_diagram
    module_items = [
        (
            path.qcs_output_module_model.itemText(index),
            path.qcs_output_module_model.itemData(index),
        )
        for index in range(path.qcs_output_module_model.count())
    ]
    assert module_items == [
        ("M5300A RF AWG", "M5300A"),
        ("M5301A Precision AWG", "M5301A"),
    ]
    assert path.qcs_output_mapping_selector.currentData() == 0
    assert path.qcs_front_panel_selection() == ("rf", 0)
    assert panel.qcs_rf_acquisition_front_panel_selections() == (
        ("rf", 0),
        ("acquisition", 0),
    )
    assert path.qcs_output_module_model.currentData() == "M5300A"
    assert "M5300A, slot 3, SMA CH 1" in (
        path.qcs_output_mapping_selector.currentText()
    )
    assert path.qcs_acquisition_mapping_selector.currentData() == 0
    assert path.qcs_acquisition_module_model.currentData() == "M5200A"
    assert "M5200A, slot 5, SMA CH 1" in (
        path.qcs_acquisition_mapping_selector.currentText()
    )

    assert path.qcs_output_endpoint.isVisible() is True
    assert path.qcs_input_endpoint.isVisible() is True
    output_nodes, input_nodes = path._active_arrow_nodes()
    assert output_nodes == (
        path.qcs_output_endpoint,
        path.loss1_component,
    )
    assert input_nodes == (
        path.loss2_component,
        path.amplifier_component,
        path.qcs_input_endpoint,
    )
    assert (
        path.minimumSizeHint().width()
        <= panel.controls_scroll.viewport().width()
    )
    assert path.width() <= panel.controls_scroll.viewport().width()
    assert path.qcs_output_endpoint.geometry().right() <= path.rect().right()
    assert path.qcs_input_endpoint.geometry().right() <= path.rect().right()
    for legacy_widget in (
        path.output_endpoint,
        path.input_endpoint,
        path.output_board_type,
        path.input_board_type,
        path.output_nqz,
        path.readout_nqz,
        path.output_att1_component,
        path.output_att2_component,
        path.input_condition,
    ):
        assert legacy_widget.isVisible() is False

    assert panel.qcs_modulation_amplitude.isVisible() is True
    assert panel.qcs_modulation_amplitude.value() == pytest.approx(
        12345 / 32767,
        abs=0.5e-9,
    )
    assert panel.modulation_gain.isVisible() is False
    assert panel.fpga_delay_widget.isVisible() is False
    assert panel.fir_profile_status.isVisible() is False
    assert panel.measurement_unit.isVisible() is False
    assert panel.dc_measure_gain_v_per_a.isVisible() is False
    assert panel.dc_calibration_group.isVisible() is False
    assert panel.trace_samples.isVisible() is False
    assert panel.qcs_integration_duration_us.isVisible() is True
    assert (
        panel.qcs_integration_duration_label.text()
        == "Total I/Q averaging time:"
    )
    assert "calculates the sample count automatically" in (
        panel.qcs_integration_duration_us.toolTip()
    )
    assert panel.qcs_integration_duration_us.suffix() == " us"
    assert panel.qcs_integration_duration_us.singleStep() == pytest.approx(
        2.0 / 300.0
    )
    assert "6.666667 ns" in panel.qcs_integration_note.text()
    assert "64 M5200 samples" in panel.qcs_integration_note.text()
    assert panel.modulation_frequency_label.text() == "RF frequency:"
    assert (
        panel._acquisition_form.labelForField(
            panel.qcs_modulation_amplitude
        ).text()
        == "Relative amplitude:"
    )
    for label in (
        panel.repetitions_label,
        panel.qcs_integration_duration_label,
        panel.settle_time_label,
        panel.modulation_frequency_label,
        panel.qcs_modulation_amplitude_label,
        panel.point_count_label,
    ):
        assert label.isVisible() is True
        assert label.width() > 0
    assert panel.dc_measure_mode.isVisible() is False
    visible_labels = " ".join(
        label.text()
        for label in panel.findChildren(QtWidgets.QLabel)
        if label.isVisible()
    )
    for qick_term in ("Output board", "Input board", "Nyquist", "HWH", "FIR"):
        assert qick_term not in visible_labels
    assert panel.start_button.isEnabled() is True
    assert panel.single_shot_button.isEnabled() is True
    assert panel.bias_t_group.isVisible() is True
    assert "QCS fixed-time" in panel.bias_t_group.title()
    assert panel.bias_t_type.isVisible() is False
    assert panel.bias_t_mode.isVisible() is False
    assert panel.bias_t_compensation_mv.isVisible() is False
    assert panel.bias_t_filter_tau_us.isVisible() is False
    assert panel.bias_t_duration_us.isVisible() is True
    assert panel.bias_t_duration_us.isEnabled() is False
    assert (
        panel._bias_t_form.labelForField(panel.bias_t_duration_us).text()
        == "Compensation duration:"
    )
    assert not hasattr(panel, "backend_warning")
    panel.set_running(True, "Running QCS hardware sweep")
    assert panel.start_button.isEnabled() is False
    assert panel.single_shot_button.isEnabled() is False
    panel.set_running(False, "Ready")
    assert panel.start_button.isEnabled() is True
    assert panel.single_shot_button.isEnabled() is True
    panel.set_saved_run_loading(True, run_id=1)
    assert panel.start_button.isEnabled() is False
    panel.set_saved_run_loading(False)
    assert panel.start_button.isEnabled() is True

    moved_mappings = []
    for mapping in configuration["channel_mappings"]:
        mapping = dict(mapping)
        if mapping["role"] == "rf":
            mapping.update(
                slot=2,
                channel=3,
                absolute_phase=False,
                lo_frequency_hz=None,
            )
        moved_mappings.append(mapping)
    moved_configuration = front_panel.normalize_qcs_hardware_configuration(
        {
            **configuration,
            "channel_mappings": moved_mappings,
        }
    )
    panel.set_qcs_front_panel_configuration(moved_configuration)
    app.processEvents()

    assert path.qcs_output_module_model.currentData() == "M5301A"
    assert "M5301A, slot 2, SMA CH 3" in (
        path.qcs_output_mapping_selector.currentText()
    )
    assert path.qcs_acquisition_module_model.currentData() == "M5200A"

    panel.set_hardware_backend("qick")
    app.processEvents()

    assert path.output_endpoint.isVisible() is True
    assert path.input_endpoint.isVisible() is True
    assert path.qcs_output_endpoint.isVisible() is False
    assert path.qcs_input_endpoint.isVisible() is False
    assert path.output_board_type.currentText() == "RF_Out"
    assert path.input_board_type.currentText() == "RF_In"
    assert path.output_nqz.value() == 2
    assert path.readout_nqz.value() == 2
    assert path.output_att1_db.value() == pytest.approx(7.25)
    assert path.output_att2_db.value() == pytest.approx(12.5)
    assert path.readout_attenuation_db.value() == pytest.approx(18.75)
    assert path.output_att1_component.isVisible() is True
    assert path.output_att2_component.isVisible() is True
    assert path.input_condition.isVisible() is True
    output_nodes, input_nodes = path._active_arrow_nodes()
    assert output_nodes == (
        path.output_endpoint,
        path.output_att1_component,
        path.output_att2_component,
        path.loss1_component,
    )
    assert input_nodes == (
        path.loss2_component,
        path.amplifier_component,
        path.input_condition,
        path.input_endpoint,
    )
    assert panel.modulation_gain.isVisible() is True
    assert panel.modulation_gain.value() == 12345
    assert panel.qcs_modulation_amplitude.isVisible() is False
    assert panel.fpga_delay_widget.isVisible() is True
    assert panel.override_fpga_trigger_delay.isVisible() is True
    assert panel.override_fpga_trigger_delay.isChecked() is True
    assert panel.fpga_trigger_delay_us.value() == pytest.approx(17.5)
    assert panel.fir_profile_status.isVisible() is True
    assert panel.fir_profile_label.text() == "HWH FIR DDR:"
    assert panel.trace_samples.isVisible() is True
    assert panel.qcs_integration_duration_us.isVisible() is False
    assert panel.qcs_integration_note.isVisible() is False
    assert panel.trace_samples_label.text() == "FIR trace samples / point:"
    assert panel.modulation_frequency_label.text() == "Modulation frequency:"
    assert panel.measurement_unit.isVisible() is True
    assert panel.dc_measure_gain_v_per_a.isVisible() is True
    assert panel.dc_calibration_group.isVisible() is True
    assert panel.dc_calibration_group.isChecked() is True
    assert panel.dc_calibration_path.text() == "legacy_dc_calibration.db"
    assert panel.dc_calibration_run_id.value() == 29
    assert panel.settings_dict() == legacy_settings
    assert panel.start_button.isEnabled() is True
    assert panel.single_shot_button.isEnabled() is True

    panel.close()


def test_stability_qcs_integration_time_rounds_up_and_reports_effective_time():
    app = _application()
    panel = stability.StabilityDiagramPanel()
    panel.refresh_targets(("awg_0", "awg_1"), (1, 3))
    panel.set_hardware_backend("qcs")
    panel.show()
    app.processEvents()

    panel.qcs_integration_duration_us.setValue(0.0101)
    assert "Requested 0.0101 us" in panel.qcs_integration_note.text()
    assert "adjusted upward to 0.0133333333333 us" in (
        panel.qcs_integration_note.text()
    )
    panel.qcs_integration_duration_us.editingFinished.emit()

    assert panel.qcs_integration_duration_us.value() == pytest.approx(
        0.013333333333,
        abs=0.5e-12,
    )
    assert panel.config(full_scale_mv=2500.0).trace_samples_per_point == 64
    settings = panel.settings_dict()
    assert settings["qcs_integration_duration_s"] == pytest.approx(
        64 / stability.QCS_M5200_SAMPLE_RATE_HZ
    )

    panel.qcs_integration_duration_us.setValue(0.020)
    panel.qcs_integration_duration_us.editingFinished.emit()
    assert panel.qcs_integration_duration_us.value() == pytest.approx(0.020)
    assert panel.config(full_scale_mv=2500.0).trace_samples_per_point == 96
    assert "programmed as 0.02 us total I/Q averaging time" in (
        panel.qcs_integration_note.text()
    )

    assert panel.qcs_integration_duration_us.maximum() == pytest.approx(
        100_000.0
    )
    panel.qcs_integration_duration_us.setValue(100_000.0)
    panel.qcs_integration_duration_us.editingFinished.emit()
    config = panel.config(full_scale_mv=2500.0)
    assert config.trace_samples_per_point == 480_000_000
    assert "1,000 bounded QCS passes" in panel.qcs_integration_note.text()
    assert "running sample-weighted complex I/Q average" in (
        panel.qcs_integration_note.text()
    )

    panel.close()


def test_stability_qcs_timeline_reports_complete_target_and_compensation():
    app = _application()
    panel = stability.StabilityDiagramPanel()
    panel.refresh_targets(("awg_0", "awg_1"), (1, 3))
    panel.set_hardware_backend("qcs")
    panel.settle_time_us.setValue(10.0)
    panel.qcs_integration_duration_us.setValue(
        10_000 / stability.QCS_M5200_SAMPLE_RATE_HZ * 1.0e6
    )
    panel.qcs_integration_duration_us.editingFinished.emit()
    panel.bias_t_group.setChecked(True)
    panel.bias_t_duration_us.setValue(50.0)
    panel.x_axis.points.setValue(101)
    panel.y_axis.points.setValue(101)
    panel.set_qcs_init_time_us(100.0)
    app.processEvents()

    note = panel.qcs_point_timing_note.text()
    assert "10,016 M5200 samples" in panel.qcs_integration_note.text()
    assert "target DC interval 0-13.1266666667 us" in note
    assert "10.0133333333 to 12.1 us" in note
    assert "after 10 us at the full target voltage" in note
    assert "post-readout full-level guard is 1 us" in note
    assert "Compensation: 13.1266666667-63.1266666667 us" in note
    assert "HCL inter-iteration delay: 100 us" in note
    assert "10,201 scheduled point/repetition iterations" in note
    assert "0.643955 s active program" in note
    assert "1.02 s inter-iteration delay" in note
    assert "dominates scan time" in note
    assert "explicit terminal zero" in note

    panel.set_qcs_init_time_us(0.07)
    assert "HCL inter-iteration delay: 0.07 us" in (
        panel.qcs_point_timing_note.text()
    )

    # RF/acquisition delay rounds to the nearest fabric cycle, whereas the
    # complete target and compensation intervals round upward.  The displayed
    # contract must use those independently programmed values.
    panel.settle_time_us.setValue(10.001)
    panel.qcs_integration_duration_us.setValue(0.020)
    panel.qcs_integration_duration_us.editingFinished.emit()
    panel.bias_t_duration_us.setValue(50.000001)
    note = panel.qcs_point_timing_note.text()
    assert "target DC interval 0-11.0633333333 us" in note
    assert "10.0133333333 to 10.0333333333 us" in note
    assert "1.00333333333 us" in note
    assert "Compensation: 11.0633333333-61.0666666667 us" in note
    panel.close()


def test_stability_legacy_sample_setting_derives_qcs_time_then_aligns_it():
    app = _application()
    panel = stability.StabilityDiagramPanel()
    panel.refresh_targets(("awg_0", "awg_1"), (1, 3))
    legacy = stability.default_stability_settings(("awg_0", "awg_1"))
    legacy.pop("qcs_integration_duration_s")
    legacy["trace_samples_per_point"] = 80
    normalized = stability.normalize_stability_settings(
        legacy,
        output_names=("awg_0", "awg_1"),
    )

    assert normalized["qcs_integration_duration_s"] == pytest.approx(
        80 / stability.QCS_M5200_SAMPLE_RATE_HZ
    )
    panel.load_settings(normalized)
    panel.set_hardware_backend("qcs")

    assert panel.trace_samples.value() == 80
    assert panel.qcs_integration_duration_us.value() == pytest.approx(0.020)
    assert panel.config(full_scale_mv=2500.0).trace_samples_per_point == 96
    assert "80" not in panel.qcs_integration_note.text()
    assert "96 M5200 samples" in panel.qcs_integration_note.text()
    app.processEvents()
    panel.close()


def test_stability_qcs_bias_t_exposes_only_fixed_time_dc_compensation():
    app = _application()
    panel = stability.StabilityDiagramPanel()
    panel.refresh_targets(("awg_0", "awg_1"), (1, 3))
    panel.bias_t_group.setChecked(True)
    panel.bias_t_type.setCurrentIndex(panel.bias_t_type.findData("filter"))
    panel.bias_t_mode.setCurrentIndex(
        panel.bias_t_mode.findData("fixed_voltage")
    )
    panel.bias_t_duration_us.setValue(4.25)
    panel.show()

    panel.set_hardware_backend("qcs")
    app.processEvents()

    assert panel.bias_t_group.isVisible() is True
    assert panel.bias_t_duration_us.isVisible() is True
    assert panel.bias_t_duration_us.isEnabled() is True
    for unsupported_control in (
        panel.bias_t_type,
        panel.bias_t_mode,
        panel.bias_t_compensation_mv,
        panel.bias_t_filter_tau_us,
    ):
        assert unsupported_control.isVisible() is False
        assert unsupported_control.isEnabled() is False

    config = panel.config(full_scale_mv=2500.0)
    assert config.bias_t_compensation_enabled is True
    assert config.bias_t_compensation_type == "dc"
    assert config.bias_t_compensation_mode == "fixed_time"
    assert config.bias_t_compensation_duration_us == pytest.approx(4.25)
    assert panel.settings_dict()["bias_t_compensation"] == {
        "enabled": True,
        "type": "filter",
        "mode": "fixed_voltage",
        "voltage_mv": panel.bias_t_compensation_mv.value(),
        "duration_us": 4.25,
        "filter_tau_us": panel.bias_t_filter_tau_us.value(),
    }

    panel.set_hardware_backend("qick")
    app.processEvents()
    assert panel.bias_t_type.isVisible() is True
    assert panel.bias_t_mode.isVisible() is True
    assert panel.bias_t_type.currentData() == "filter"
    assert panel.bias_t_mode.currentData() == "fixed_voltage"
    panel.close()


def test_stability_qcs_rf_output_selection_round_trips():
    app = _application()
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_x", "dc_y"),
        {0: "rf_primary", 1: "rf_secondary"},
        "digitizer",
    )
    panel = stability.StabilityDiagramPanel()
    panel.refresh_targets(("awg_0", "awg_1"), (0, 1))
    panel.set_qcs_front_panel_configuration(configuration)
    panel.set_hardware_backend("qcs")
    selector = panel.path_diagram.qcs_output_mapping_selector
    selector.setCurrentIndex(selector.findData(1))
    settings = panel.settings_dict()

    assert settings["qcs_rf_output_logical_index"] == 1

    restored = stability.StabilityDiagramPanel()
    restored.refresh_targets(("awg_0", "awg_1"), (0, 1))
    restored.set_qcs_front_panel_configuration(configuration)
    restored.set_hardware_backend("qcs")
    restored.load_settings(settings)
    app.processEvents()

    assert (
        restored.path_diagram.qcs_output_mapping_selector.currentData() == 1
    )
    assert restored.qcs_rf_acquisition_front_panel_selections() == (
        ("rf", 1),
        ("acquisition", 0),
    )
    restored.close()
    panel.close()


def test_stability_panel_controls_and_settings_round_trip(tmp_path):
    app = _application()
    panel = stability.StabilityDiagramPanel()
    panel.refresh_targets(
        ("awg_0", "awg_1", "awg_2"),
        (1, 3, 5),
        ("set_0", "set_1"),
    )
    panel.x_axis.output.setCurrentIndex(panel.x_axis.output.findData("awg_0"))
    panel.y_axis.output.setCurrentIndex(panel.y_axis.output.findData("awg_2"))
    panel.x_axis.start_mv.setValue(-250.0)
    panel.x_axis.stop_mv.setValue(125.0)
    panel.x_axis.points.setValue(11)
    panel.y_axis.points.setValue(7)
    panel.repetitions.setValue(4)
    panel.trace_samples.setValue(321)
    panel.settle_time_us.setValue(75.5)
    panel.modulation_frequency_mhz.setValue(12.5)
    panel.modulation_gain.setValue(12345)
    panel.plot.magnitude_range_control.set_manual_levels(10.0, 100.0)
    panel.plot.phase_range_control.set_manual_levels(-90.0, 45.0)
    panel.plot.load_visible_data(["i", "q", "magnitude"])
    panel.bias_t_group.setChecked(True)
    panel.bias_t_mode.setCurrentIndex(
        panel.bias_t_mode.findData("fixed_time")
    )
    panel.bias_t_duration_us.setValue(2.5)
    database_path = tmp_path / "stability_single_shot.db"
    saved_plot_path = tmp_path / "saved_stability.db"
    panel.database_path.setText(str(database_path))
    panel.saved_database_path.setText(str(saved_plot_path))
    panel._preferred_saved_run_id = 29
    panel.saved_plot_data.setCurrentIndex(
        panel.saved_plot_data.findData("q")
    )
    app.processEvents()

    config = panel.config(full_scale_mv=2500.0)
    assert config.x_axis.output_name == "awg_0"
    assert config.x_axis.segment_name == stability.STABILITY_HOLD_SEGMENT
    assert config.y_axis.output_name == "awg_2"
    assert config.point_count == 77
    assert config.trace_samples_per_point == 321
    assert config.settle_time_us == 75.5
    assert config.modulation_frequency_mhz == 12.5
    assert config.modulation_gain == 12345
    assert config.bias_t_compensation_enabled is True
    assert config.bias_t_compensation_type == "dc"
    assert config.bias_t_compensation_mode == "fixed_time"
    assert config.bias_t_compensation_duration_us == 2.5
    assert panel.bias_t_compensation_mv.isEnabled() is False
    assert panel.bias_t_duration_us.isEnabled() is True
    assert panel.point_count.text() == "77"
    assert not hasattr(panel.x_axis, "segment")
    assert "segment_name" not in panel.x_axis.settings_dict()
    assert not hasattr(panel, "rf_editor_tabs")
    assert panel.database_path_value() == str(database_path)
    assert panel.saved_database_path_value() == str(saved_plot_path)
    assert panel.layout().indexOf(panel.controls_scroll) >= 0

    saved = panel.settings_dict()
    restored = stability.StabilityDiagramPanel()
    restored.refresh_targets(
        ("awg_0", "awg_1", "awg_2"),
        (1, 3, 5),
        ("set_0", "set_1"),
    )
    restored.load_settings(saved)
    assert restored.settings_dict() == saved
    assert restored.bias_t_group.isChecked() is True
    assert restored.bias_t_mode.currentData() == "fixed_time"
    assert restored.bias_t_duration_us.value() == 2.5
    assert restored.settle_time_us.value() == 75.5
    assert restored.plot.color_range_settings() == {
        "i": {
            "auto": True,
            "minimum": -1.0,
            "maximum": 1.0,
        },
        "q": {
            "auto": True,
            "minimum": -1.0,
            "maximum": 1.0,
        },
        "magnitude": {
            "auto": False,
            "minimum": 10.0,
            "maximum": 100.0,
        },
        "phase": {
            "auto": False,
            "minimum": -90.0,
            "maximum": 45.0,
        },
    }
    assert restored.plot.visible_data() == ("i", "q", "magnitude")
    assert restored.saved_database_path.text() == str(saved_plot_path)
    assert restored._preferred_saved_run_id == 29
    assert restored.saved_plot_data.currentData() == "q"

    dc_changes = []
    calibration_changes = []
    panel.dc_measure_changed.connect(
        lambda enabled, gain: dc_changes.append((enabled, gain))
    )
    panel.dc_calibration_changed.connect(
        lambda enabled, path, run_id: calibration_changes.append(
            (enabled, path, run_id)
        )
    )
    panel.set_dc_measure_context(
        "DC_In",
        True,
        2.0e6,
        True,
        "dc_calibration.db",
        4,
    )
    assert panel.dc_measure_mode.isChecked() is True
    assert panel.dc_measure_mode.isEnabled() is True
    assert panel.dc_measure_gain_v_per_a.isEnabled() is True
    assert panel.dc_calibration_group.isChecked() is True
    assert panel.dc_calibration_group.isEnabled() is True
    assert panel.dc_calibration_path.text() == "dc_calibration.db"
    assert panel.dc_calibration_run_id.value() == 4
    panel.dc_measure_gain_v_per_a.setValue(3.0e6)
    panel.dc_calibration_run_id.setValue(5)
    app.processEvents()
    assert dc_changes[-1] == (True, 3.0e6)
    assert calibration_changes[-1] == (True, "dc_calibration.db", 5)
    calibrated_settings = panel.settings_dict()
    restored.load_settings(calibrated_settings)
    assert restored.settings_dict() == calibrated_settings
    panel.set_dc_measure_context("RF_In", True, 4.0e6)
    assert panel.dc_measure_mode.isChecked() is False
    assert panel.dc_measure_mode.isEnabled() is False
    assert panel.dc_measure_gain_v_per_a.isEnabled() is False
    assert panel.dc_calibration_group.isChecked() is False
    assert panel.dc_calibration_group.isEnabled() is False

    panel.set_running(True, "running")
    assert panel.start_button.isEnabled() is False
    assert panel.stop_button.isEnabled() is True
    assert panel.single_shot_button.isEnabled() is False
    assert panel.trace_samples.isEnabled() is False
    assert panel.qcs_integration_duration_us.isEnabled() is False
    assert panel.settle_time_us.isEnabled() is False
    assert panel.modulation_frequency_mhz.isEnabled() is False
    assert panel.bias_t_group.isEnabled() is False
    panel.set_stopping()
    assert panel.stop_button.isEnabled() is False
    panel.refresh_targets(("awg_0", "awg_1"), (1, 3))
    assert panel.start_button.isEnabled() is False
    assert panel.single_shot_button.isEnabled() is False
    panel.set_running(False, "ready")
    assert panel.start_button.isEnabled() is True
    assert panel.trace_samples.isEnabled() is True
    assert panel.qcs_integration_duration_us.isEnabled() is True
    assert panel.settle_time_us.isEnabled() is True
    assert panel.modulation_frequency_mhz.isEnabled() is True
    assert panel.bias_t_group.isEnabled() is True
    panel.set_saved_run_loading(True, run_id=29)
    panel.refresh_targets(("awg_0", "awg_1"), (1, 3))
    assert panel.start_button.isEnabled() is False
    assert panel.single_shot_button.isEnabled() is False
    panel.set_saved_run_loading(False)
    assert panel.start_button.isEnabled() is True
    panel.close()
    restored.close()


def _stored_stability_metadata():
    return {
        "created_at_utc": "2026-07-27T12:34:56+00:00",
        "gui_settings": {
            "qick": {
                "full_scale_mv": 800.0,
                "fir_rate_profile": "50_ksps",
                "fir_stability_capture_mode": (
                    "immediate_continuous_fir_output"
                ),
            },
            "stability_diagram": {
                "x_axis": {"output_name": "awg_0"},
                "y_axis": {"output_name": "awg_1"},
            },
        },
        "measurement_layout": {
            "iq_shape": [4, 2, 3, 2],
            "iq_unit": "ADC units",
            "measurement_mode": "raw_iq",
            "sample_rate_hz": 50_000.0,
            "sample_period_us": 20.0,
            "sweep_axes": [
                {
                    "parameter": "awg_0_set_0_voltage_mv",
                    "output_name": "awg_0",
                    "segment_name": stability.STABILITY_HOLD_SEGMENT,
                    "axis_kind": "amplitude",
                    "unit": "mV",
                    "count": 2,
                },
                {
                    "parameter": "awg_1_set_0_voltage_mv",
                    "output_name": "awg_1",
                    "segment_name": stability.STABILITY_HOLD_SEGMENT,
                    "axis_kind": "amplitude",
                    "unit": "mV",
                    "count": 2,
                },
            ],
        },
    }


def test_saved_stability_run_listing_filters_other_qick_runs(tmp_path):
    database_path = tmp_path / "stability.db"
    metadata = _stored_stability_metadata()
    programmable_delay = json.loads(json.dumps(metadata))
    programmable_delay["gui_settings"]["qick"][
        "fir_stability_capture_mode"
    ] = "programmable_fpga_delay"
    non_stability = json.loads(json.dumps(metadata))
    non_stability["gui_settings"]["qick"].pop("fir_stability_capture_mode")
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE runs ("
            "run_id INTEGER PRIMARY KEY, "
            "qick_experiment_json TEXT)"
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?)",
            (10, json.dumps(non_stability)),
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?)",
            (11, json.dumps(metadata)),
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?)",
            (12, json.dumps(programmable_delay)),
        )

    summaries = stability.list_stability_runs(database_path)

    assert [summary.run_id for summary in summaries] == [12, 11]
    assert summaries[0].x_axis_label == "awg_0"
    assert summaries[0].y_axis_label == "awg_1"
    assert summaries[0].sample_rate_hz == 50_000.0
    assert "Run 12" in summaries[0].display_label
    assert "50 kSPS" in summaries[0].display_label


def test_saved_stability_iq_arrays_restore_cartesian_grid(tmp_path):
    metadata = _stored_stability_metadata()
    metadata["gui_settings"]["qick"][
        "fir_stability_capture_mode"
    ] = "programmable_fpga_delay"
    iq = np.empty((4, 2, 3, 2), dtype=np.int32)
    for point_index in range(4):
        iq[point_index, :, :, 0] = point_index + 1
        iq[point_index, :, :, 1] = -(point_index + 1)
    arrays = {
        "metadata": metadata,
        "iq": iq,
        "iq_unit": "ADC units",
        "measurement_mode": "raw_iq",
        "sweep_coordinates": {
            "awg_0_set_0_voltage_mv": np.asarray(
                [
                    [-100.0, -100.0],
                    [-100.0, -100.0],
                    [100.0, 100.0],
                    [100.0, 100.0],
                ]
            ),
            "awg_1_set_0_voltage_mv": np.asarray(
                [
                    [-50.0, -50.0],
                    [50.0, 50.0],
                    [-50.0, -50.0],
                    [50.0, 50.0],
                ]
            ),
        },
    }

    result = stability.stability_result_from_stored_arrays(
        arrays,
        database_path=tmp_path / "stability.db",
        run_id=23,
    )

    np.testing.assert_allclose(result.x_voltage_mv, [-100.0, 100.0])
    np.testing.assert_allclose(result.y_voltage_mv, [-50.0, 50.0])
    np.testing.assert_allclose(result.i_mean, [[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_allclose(result.q_mean, [[-1.0, -3.0], [-2.0, -4.0]])
    assert result.repetition_count == 2
    assert result.samples_per_trace == 3
    assert result.sample_rate_hz == 50_000.0
    assert result.fir_rate_profile == "50_ksps"
    assert result.source_label == "QCoDeS Run 23"
    assert result.run_id == 23


def test_stability_overlay_selector_lists_run_and_emits_selection(tmp_path):
    app = _application()
    database_path = tmp_path / "stability.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE runs ("
            "run_id INTEGER PRIMARY KEY, "
            "qick_experiment_json TEXT)"
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?)",
            (31, json.dumps(_stored_stability_metadata())),
        )

    selector = stability.StabilityOverlaySelector()
    selector.database_path.setText(str(database_path))
    selector.refresh_runs()
    emitted = []
    selector.load_requested.connect(
        lambda path, run_id, quantity: emitted.append(
            (path, run_id, quantity)
        )
    )
    selector.quantity_combo.setCurrentIndex(
        selector.quantity_combo.findData("phase")
    )
    selector.load_button.click()
    app.processEvents()

    assert selector.run_combo.count() == 1
    assert selector.run_combo.currentData() == 31
    assert emitted == [(str(database_path), 31, "phase")]
    selector.close()


def test_stability_panel_selects_and_requests_saved_diagram(tmp_path):
    app = _application()
    database_path = tmp_path / "stability.db"
    save_database_path = tmp_path / "new_stability.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE runs ("
            "run_id INTEGER PRIMARY KEY, "
            "qick_experiment_json TEXT)"
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?)",
            (41, json.dumps(_stored_stability_metadata())),
        )

    panel = stability.StabilityDiagramPanel()
    panel.database_path.setText(str(save_database_path))
    panel.saved_database_path.setText(str(database_path))
    requested = []
    panel.saved_run_requested.connect(
        lambda path, run_id: requested.append((path, run_id))
    )
    panel.refresh_saved_runs()
    panel.saved_plot_data.setCurrentIndex(
        panel.saved_plot_data.findData("phase")
    )
    panel.load_saved_run_button.click()
    app.processEvents()

    assert panel.saved_run_combo.count() == 1
    assert panel.saved_run_combo.currentData() == 41
    assert requested == [(str(database_path), 41)]
    assert panel.database_path_value() == str(save_database_path)
    assert panel.plot.visible_data() == ("phase",)

    panel.set_saved_run_loading(True, run_id=41)
    assert panel.database_path.isEnabled() is True
    assert panel.saved_database_path.isEnabled() is False
    assert panel.refresh_saved_runs_button.isEnabled() is False
    assert panel.load_saved_run_button.isEnabled() is False
    panel.set_saved_run_loading(False)
    assert panel.database_path.isEnabled() is True
    assert panel.saved_database_path.isEnabled() is True
    assert panel.refresh_saved_runs_button.isEnabled() is True
    assert panel.load_saved_run_button.isEnabled() is True
    panel.close()


def test_saved_stability_run_loads_from_real_qcodes_database(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "stability_qcodes.db"
    monkeypatch.setenv(QCODES_STAGING_ENV, str(tmp_path / "staging"))
    axes = (
        AmplitudeSweep(
            stability.STABILITY_HOLD_SEGMENT,
            "awg_0",
            -0.5,
            0.5,
            2,
        ),
        AmplitudeSweep(
            stability.STABILITY_HOLD_SEGMENT,
            "awg_1",
            -0.25,
            0.25,
            2,
        ),
    )
    iq = np.empty((4, 1, 2, 2), dtype=np.int32)
    for point_index in range(4):
        iq[point_index, :, :, 0] = 10 + point_index
        iq[point_index, :, :, 1] = -20 - point_index
    ddr_result = FineTuneDdrResult(
        sweep_points=np.asarray(
            [
                [-0.5, -0.25],
                [-0.5, 0.25],
                [0.5, -0.25],
                [0.5, 0.25],
            ]
        ),
        iq=iq,
        sweep_axes=axes,
        sweep_shape=(2, 2),
        cross_capacitance=np.eye(2),
    )
    gui_settings = {
        "qick": {
            "fabric_mhz": 300.0,
            "tproc_mhz": 300.0,
            "full_scale_mv": 800.0,
            "fir_rate_profile": "50_ksps",
            "fir_stability_capture_mode": (
                "programmable_fpga_delay"
            ),
        },
        "stability_diagram": {
            "x_axis": {"output_name": "awg_0"},
            "y_axis": {"output_name": "awg_1"},
        },
        "awg": {"cross_capacitance": np.eye(2).tolist()},
    }
    dataset, _row_count = store_qick_result(
        ddr_result,
        run_config=QcodesRunConfig(
            database_path=str(database_path),
            experiment_name="Stability loader test",
            sample_name="simulated device",
            sample_rate_hz=50_000.0,
        ),
        connection_config=QickConnectionConfig(
            "192.0.2.10",
            8888,
            "myqick",
        ),
        program_summary={},
        gui_settings=gui_settings,
        rf_settings={},
    )

    summaries = stability.list_stability_runs(database_path)
    result = stability.load_stability_diagram_run(
        database_path,
        dataset.run_id,
    )

    assert [summary.run_id for summary in summaries] == [dataset.run_id]
    np.testing.assert_allclose(result.x_voltage_mv, [-400.0, 400.0])
    np.testing.assert_allclose(result.y_voltage_mv, [-200.0, 200.0])
    np.testing.assert_allclose(result.i_mean, [[10.0, 12.0], [11.0, 13.0]])
    np.testing.assert_allclose(result.q_mean, [[-20.0, -22.0], [-21.0, -23.0]])
    assert result.source_label == f"QCoDeS Run {dataset.run_id}"
    assert result.sample_rate_hz == 50_000.0


def test_saved_qcs_hardware_stability_run_is_listed_and_loaded(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "qcs_stability_qcodes.db"
    monkeypatch.setenv(QCODES_STAGING_ENV, str(tmp_path / "staging"))
    axes = (
        AmplitudeSweep(
            stability.STABILITY_HOLD_SEGMENT,
            "awg_0",
            -0.5,
            0.5,
            2,
        ),
        AmplitudeSweep(
            stability.STABILITY_HOLD_SEGMENT,
            "awg_1",
            -0.25,
            0.25,
            2,
        ),
    )
    iq = np.empty((4, 2, 1, 2), dtype=float)
    for point_index in range(4):
        iq[point_index, :, 0, 0] = point_index + np.arange(2)
        iq[point_index, :, 0, 1] = -point_index
    ddr_result = FineTuneDdrResult(
        sweep_points=np.asarray([
            [-0.5, -0.25],
            [-0.5, 0.25],
            [0.5, -0.25],
            [0.5, 0.25],
        ]),
        iq=iq,
        sweep_axes=axes,
        sweep_shape=(2, 2),
        cross_capacitance=np.eye(2),
        sample_rate_hz=1_000_000.0,
        fir_rate_profile="qcs_hardware_demod",
    )
    gui_settings = {
        "qick": {
            "fabric_mhz": 300.0,
            "full_scale_mv": 800.0,
        },
        "stability_diagram": {
            "capture_mode": "qcs_hardware_sweep",
            "x_axis": {"output_name": "awg_0"},
            "y_axis": {"output_name": "awg_1"},
        },
        "awg": {"cross_capacitance": np.eye(2).tolist()},
    }
    dataset, _row_count = store_qick_result(
        ddr_result,
        run_config=QcodesRunConfig(
            database_path=str(database_path),
            experiment_name="QCS Stability loader test",
            sample_name="simulated device",
            sample_rate_hz=1_000_000.0,
        ),
        connection_config=QcsConnectionConfig(
            mapper_path="qcs_mapper.qcs",
            dc_channel_names=("dc_x", "dc_y"),
            acquisition_channel_name="digitizer",
        ),
        program_summary={
            "backend": "qcs",
            "hardware_sweep": True,
            "program_count": 1,
        },
        gui_settings=gui_settings,
        rf_settings={"readout_details": {"hw_demod": True}},
        backend_name="qcs",
    )

    summaries = stability.list_stability_runs(database_path)
    result = stability.load_stability_diagram_run(
        database_path,
        dataset.run_id,
    )

    assert [summary.run_id for summary in summaries] == [dataset.run_id]
    np.testing.assert_allclose(result.x_voltage_mv, [-400.0, 400.0])
    np.testing.assert_allclose(result.y_voltage_mv, [-200.0, 200.0])
    np.testing.assert_allclose(
        result.i_mean,
        [[0.5, 2.5], [1.5, 3.5]],
    )
    assert result.repetition_count == 2
    assert result.samples_per_trace == 1
    assert result.fir_rate_profile == "qcs_hardware_demod"
