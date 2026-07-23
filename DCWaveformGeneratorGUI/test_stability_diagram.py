"""Headless tests for two-electrode stability-diagram acquisition.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtWidgets
import pytest

from qick_qcodes_experiment import QcodesRunConfig, QickConnectionConfig
import stability_diagram as stability


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _fir_profile(*, is_50_ksps: bool = False):
    sample_rate_hz = 50_000.0 if is_50_ksps else 1_000_000.0
    trigger_delay_samples = 50 if is_50_ksps else 0
    sample_period_us = 1_000_000.0 / sample_rate_hz
    return SimpleNamespace(
        name="50_ksps" if is_50_ksps else "1_msps",
        sample_rate_hz=sample_rate_hz,
        sample_period_us=sample_period_us,
        trigger_delay_samples=trigger_delay_samples,
        trigger_delay_us=trigger_delay_samples * sample_period_us,
        software_warmup_compensation=not is_50_ksps,
        timing_label=(
            "50 kSPS (20 us/sample, FPGA delay 50 samples (1000 us))"
            if is_50_ksps
            else "1 MSPS (1 us/sample, tProcessor FIR warm-up compensation)"
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
        "readout_spec": object(),
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


def test_stability_50ksps_hold_covers_trace_and_fpga_delay():
    config = _config()
    sequence = stability.build_stability_hold_sequence(
        config,
        output_names=("awg_0", "awg_1"),
        fabric_mhz=300.0,
        full_scale_mv=100.0,
        sample_period_us=20.0,
        fpga_trigger_delay_us=1000.0,
    )

    expected_hold_us = (
        config.settle_time_us
        + 1000.0
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
        + profile.trigger_delay_us
        + _config().trace_samples_per_point * profile.sample_period_us
        + stability.DEFAULT_STABILITY_POINT_GUARD_US
    )
    assert sequence.segments[0].duration_cycles == int(
        np.ceil(expected_hold_us * 300.0)
    )
    assert captured["rf_specs"][0].duration_us == (
        profile.trigger_delay_us
        + _config().trace_samples_per_point * profile.sample_period_us
    )


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
    panel.bias_t_group.setChecked(True)
    panel.bias_t_mode.setCurrentIndex(
        panel.bias_t_mode.findData("fixed_time")
    )
    panel.bias_t_duration_us.setValue(2.5)
    database_path = tmp_path / "stability_single_shot.db"
    panel.database_path.setText(str(database_path))
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
    assert panel.settle_time_us.isEnabled() is False
    assert panel.modulation_frequency_mhz.isEnabled() is False
    assert panel.bias_t_group.isEnabled() is False
    panel.set_stopping()
    assert panel.stop_button.isEnabled() is False
    panel.set_running(False, "ready")
    assert panel.start_button.isEnabled() is True
    assert panel.trace_samples.isEnabled() is True
    assert panel.settle_time_us.isEnabled() is True
    assert panel.modulation_frequency_mhz.isEnabled() is True
    assert panel.bias_t_group.isEnabled() is True
    panel.close()
    restored.close()
