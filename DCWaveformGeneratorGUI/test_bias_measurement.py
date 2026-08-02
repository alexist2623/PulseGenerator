"""Tests for Bias sweep generation, current readers, and QCoDeS storage.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtWidgets
import pytest

from bias_measurement import (
    CurrentReading,
    default_bias_measurement_settings,
    make_gate_sweep,
    normalize_bias_measurement_settings,
    ramp_bias_channel,
    ramp_bias_channels,
    run_bias_measurement,
    validate_bias_sweep_voltage_limit,
)
from bias_measurement_gui import BiasMeasurementTabs
from qick_qcodes_experiment import QickConnectionConfig


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeSoc:
    def __init__(self):
        self.values = {channel: 0.01 * channel for channel in range(8)}

    def rfb_get_bias(self, channel):
        return self.values[int(channel)]

    def rfb_set_bias(self, channel, voltage):
        self.values[int(channel)] = float(voltage)
        return float(voltage)


class FakeAdcReader:
    def __init__(self, soc, _soccfg, _connection, _settings, **_kwargs):
        self.soc = soc
        self.closed = False

    def read(self):
        value = float(self.soc.values[2])
        return CurrentReading(value, -value, abs(value) * np.sqrt(2.0), -45.0)

    def close(self, **_kwargs):
        self.closed = True


class FakeParameter:
    def __init__(self, value):
        self.value = value

    def __call__(self, *args):
        if args:
            self.value = args[0]
        return self.value


class FakeSr860:
    last = None

    def __init__(self, _name, address):
        FakeSr860.last = self
        self.address = address
        self.reference_source = FakeParameter("EXT")
        self.frequency = FakeParameter(1.0)
        self.phase = FakeParameter(0.0)
        self.signal_input = FakeParameter("voltage")
        self.input_gain = FakeParameter(1e6)
        self.sensitivity = FakeParameter(1e-9)
        self.filter_slope = FakeParameter(6)
        self.time_constant = FakeParameter(1e-3)
        self.amplitude = FakeParameter(0.0)
        self.closed = False

    def get_values(self, *_names):
        current = float(self.amplitude()) * 2.0
        return current, 0.0, abs(current), 0.0

    def close(self):
        self.closed = True


def test_nested_gate_sweep_and_ramp_are_bounded():
    values = make_gate_sweep(
        0.0,
        -0.6,
        4,
        2,
        return_leg=True,
        largest_loop_first=False,
    )
    assert values.tolist() == pytest.approx(
        [0.0, -0.1, -0.2, -0.3, -0.2, -0.1, 0.0,
         0.0, -0.2, -0.4, -0.6, -0.4, -0.2, 0.0]
    )

    soc = FakeSoc()
    writes = []
    original = soc.rfb_set_bias

    def record(channel, voltage):
        writes.append(float(voltage))
        return original(channel, voltage)

    soc.rfb_set_bias = record
    ramp_bias_channel(
        soc,
        0,
        0.025,
        max_step_v=0.01,
        pause_s=0.0,
        voltage_limit_v=0.1,
        sleeper=lambda _seconds: None,
    )
    assert writes == pytest.approx([0.0083333333, 0.0166666667, 0.025])
    assert max(np.diff([0.0, *writes])) <= 0.01


def test_vector_ramp_moves_channels_on_one_shared_fraction():
    soc = FakeSoc()
    soc.values[0] = 0.0
    soc.values[1] = 0.0
    writes = []
    pauses = []
    original = soc.rfb_set_bias

    def record(channel, voltage):
        writes.append((int(channel), float(voltage)))
        return original(channel, voltage)

    soc.rfb_set_bias = record
    result = ramp_bias_channels(
        soc,
        {0: 0.02, 1: -0.03},
        max_step_v=0.01,
        pause_s=0.001,
        voltage_limit_v=0.1,
        sleeper=pauses.append,
    )

    np.testing.assert_allclose(
        np.asarray(writes, dtype=float),
        np.asarray([
            (0, 0.02 / 3), (1, -0.01),
            (0, 0.04 / 3), (1, -0.02),
            (0, 0.02), (1, -0.03),
        ]),
    )
    assert pauses == pytest.approx([0.001, 0.001, 0.001])
    assert result == pytest.approx({0: 0.02, 1: -0.03})


def test_ramp_never_writes_when_target_or_current_voltage_exceeds_limit():
    soc = FakeSoc()
    writes = []
    original = soc.rfb_set_bias

    def record(channel, voltage):
        writes.append((int(channel), float(voltage)))
        return original(channel, voltage)

    soc.rfb_set_bias = record
    with pytest.raises(ValueError, match="target.*exceeds"):
        ramp_bias_channels(
            soc,
            {0: 0.11},
            max_step_v=0.01,
            pause_s=0.0,
            voltage_limit_v=0.1,
            sleeper=lambda _seconds: None,
        )
    assert writes == []

    soc.values[0] = 0.11
    with pytest.raises(ValueError, match="current voltage.*outside"):
        ramp_bias_channels(
            soc,
            {0: 0.0},
            max_step_v=0.01,
            pause_s=0.0,
            voltage_limit_v=0.1,
            sleeper=lambda _seconds: None,
        )
    assert writes == []


@pytest.mark.parametrize("kind", ["gate", "wall_wall", "nested"])
def test_every_bias_sweep_type_preflights_all_endpoints(kind):
    settings = default_bias_measurement_settings()[kind]
    if kind == "gate":
        settings["gate_stop_v"] = 0.51
    elif kind == "wall_wall":
        settings["fast_stop_v"] = -0.51
    else:
        settings["axes"][1]["stop_v"][1] = 0.51
    normalized = normalize_bias_measurement_settings({kind: settings})[kind]
    with pytest.raises(ValueError, match="exceeds the configured"):
        validate_bias_sweep_voltage_limit(kind, normalized, 0.5)


def test_out_of_limit_nested_sweep_fails_before_qick_connection(tmp_path):
    settings = default_bias_measurement_settings()["nested"]
    settings["database_path"] = str(tmp_path / "must_not_exist.db")
    settings["axes"][1]["stop_v"][0] = 0.2
    connected = []

    def connector(**_kwargs):
        connected.append(True)
        return FakeSoc(), {}

    with pytest.raises(ValueError, match="exceeds the configured"):
        run_bias_measurement(
            connection_config=QickConnectionConfig(
                "127.0.0.1", 8888, "myqick"
            ),
            kind="nested",
            settings=settings,
            channel_names=("",) * 8,
            voltage_limit_v=0.1,
            connector=connector,
            adc_reader_factory=FakeAdcReader,
            sleeper=lambda _seconds: None,
        )
    assert connected == []
    assert not (tmp_path / "must_not_exist.db").exists()


def test_bias_measurement_settings_round_trip_and_legacy_defaults():
    defaults = default_bias_measurement_settings()
    normalized = normalize_bias_measurement_settings({})
    assert normalized == defaults
    assert normalized["two_point"]["sr860"]["filter_slope_db_oct"] == 24
    assert normalized["wall_wall"]["fast_channel"] == 1
    assert normalized["nested"]["axes"][1]["channels"] == [0, 1]

    app = _application()
    tabs = BiasMeasurementTabs()
    tabs.set_channel_names(["P", "BL", "BR", "S0", "AccL", "AccR", "", ""])
    tabs.load_settings(defaults)
    app.processEvents()
    persisted = tabs.settings_dict()
    assert persisted == defaults
    assert tabs.pages["gate"].gate_channel.itemText(1) == "BIAS1 (BL)"
    assert "BIAS0 (P)" in (
        tabs.pages["nested"].nested_axis_editors[1].channel_summary.text()
    )
    tabs.close()


def test_nested_settings_reject_reused_channels_and_bad_vectors():
    settings = default_bias_measurement_settings()["nested"]
    settings = json.loads(json.dumps(settings))
    settings["axes"][1]["channels"] = [2]
    settings["axes"][1]["start_v"] = [0.0]
    settings["axes"][1]["stop_v"] = [0.1]
    with pytest.raises(ValueError, match="more than one nested axis"):
        normalize_bias_measurement_settings({"nested": settings})

    settings = default_bias_measurement_settings()["nested"]
    settings = json.loads(json.dumps(settings))
    settings["axes"][1]["stop_v"] = [0.1]
    with pytest.raises(ValueError, match="must contain 2 voltage values"):
        normalize_bias_measurement_settings({"nested": settings})


def test_nested_axis_gui_add_remove_and_reorder():
    app = _application()
    tabs = BiasMeasurementTabs()
    page = tabs.pages["nested"]
    page._add_nested_axis({
        "name": "third",
        "channels": [3],
        "start_v": [-0.2],
        "stop_v": [0.2],
        "points": 5,
    })
    app.processEvents()
    assert [axis["name"] for axis in page.settings_dict()["axes"]] == [
        "outer", "inner_vector", "third",
    ]

    third = page.nested_axis_editors[2]
    page._move_nested_axis(third, -1)
    assert [axis["name"] for axis in page.settings_dict()["axes"]] == [
        "outer", "third", "inner_vector",
    ]
    page._remove_nested_axis(third)
    app.processEvents()
    assert [axis["name"] for axis in page.settings_dict()["axes"]] == [
        "outer", "inner_vector",
    ]
    tabs.close()


def test_gate_qick_adc_measurement_saves_bias_metadata(tmp_path):
    database = tmp_path / "bias_measurement.db"
    settings = default_bias_measurement_settings()["gate"]
    settings.update({
        "database_path": str(database),
        "sample_name": "GateTest",
        "current_mode": "qick_adc",
        "gate_channel": 2,
        "gate_start_v": -0.1,
        "gate_stop_v": 0.1,
        "points_per_leg": 3,
        "loops": 1,
        "return_leg": False,
        "repetitions_per_point": 2,
        "settle_s": 0.0,
        "ramp_max_step_v": 1.0,
        "ramp_pause_s": 0.0,
    })
    soc = FakeSoc()
    initial = dict(soc.values)

    result = run_bias_measurement(
        connection_config=QickConnectionConfig("127.0.0.1", 8888, "myqick"),
        kind="gate",
        settings=settings,
        channel_names=("P", "BL", "BR", "S0", "AccL", "AccR", "", ""),
        voltage_limit_v=1.0,
        connector=lambda **_kwargs: (soc, {}),
        adc_reader_factory=FakeAdcReader,
        sleeper=lambda _seconds: None,
    )

    assert result.run_id >= 1
    assert result.x_values.tolist() == pytest.approx([-0.1, 0.0, 0.1])
    assert result.magnitude_a.shape == (3,)
    assert soc.values == pytest.approx(initial)

    from qcodes import initialise_or_create_database_at, load_by_id

    initialise_or_create_database_at(str(database))
    dataset = load_by_id(result.run_id)
    initial_metadata = json.loads(
        dataset.get_metadata("bias_channel_voltages_initial_json")
    )
    final_metadata = json.loads(
        dataset.get_metadata("bias_channel_voltages_final_json")
    )
    assert len(initial_metadata["channels"]) == 8
    assert initial_metadata["channels"][2]["name"] == "BR"
    assert final_metadata["channels"][2]["voltage_v"] == pytest.approx(0.1)
    adc_metadata = json.loads(dataset.get_metadata("qick_adc_settings_json"))
    assert adc_metadata["readout_ch"] == settings["qick_adc"]["readout_ch"]
    assert adc_metadata["fir_samples"] == settings["qick_adc"]["fir_samples"]
    # QCoDeS counts the six dependent values written for each of six reads.
    assert dataset.number_of_results == 36


def test_wall_wall_result_shape_with_qick_adc(tmp_path):
    settings = default_bias_measurement_settings()["wall_wall"]
    settings.update({
        "database_path": str(tmp_path / "wall.db"),
        "sample_name": "WallTest",
        "current_mode": "qick_adc",
        "slow_channel": 1,
        "fast_channel": 2,
        "slow_start_v": 0.0,
        "slow_stop_v": 0.1,
        "slow_points": 2,
        "fast_start_v": -0.1,
        "fast_stop_v": 0.1,
        "fast_points": 3,
        "repetitions_per_point": 1,
        "settle_s": 0.0,
        "ramp_max_step_v": 1.0,
        "ramp_pause_s": 0.0,
    })
    soc = FakeSoc()
    result = run_bias_measurement(
        connection_config=QickConnectionConfig("127.0.0.1", 8888, "myqick"),
        kind="wall_wall",
        settings=settings,
        channel_names=("",) * 8,
        voltage_limit_v=1.0,
        connector=lambda **_kwargs: (soc, {}),
        adc_reader_factory=FakeAdcReader,
        sleeper=lambda _seconds: None,
    )
    assert result.magnitude_a.shape == (2, 3)
    assert result.x_values.tolist() == pytest.approx([-0.1, 0.0, 0.1])
    assert result.y_values.tolist() == pytest.approx([0.0, 0.1])


def test_general_nested_vector_sweep_saves_all_axis_setpoints(tmp_path):
    database = tmp_path / "nested.db"
    settings = default_bias_measurement_settings()["nested"]
    settings.update({
        "database_path": str(database),
        "sample_name": "NestedTest",
        "current_mode": "qick_adc",
        "axes": [
            {
                "name": "outer_gate",
                "channels": [2],
                "start_v": [0.0],
                "stop_v": [0.2],
                "points": 2,
            },
            {
                "name": "paired_inner",
                "channels": [0, 1],
                "start_v": [-0.1, 0.1],
                "stop_v": [0.1, -0.1],
                "points": 3,
            },
        ],
        "repetitions_per_point": 2,
        "settle_s": 0.0,
        "ramp_max_step_v": 1.0,
        "ramp_pause_s": 0.0,
    })
    soc = FakeSoc()
    initial = dict(soc.values)
    result = run_bias_measurement(
        connection_config=QickConnectionConfig("127.0.0.1", 8888, "myqick"),
        kind="nested",
        settings=settings,
        channel_names=("P", "BL", "BR", "S0", "", "", "", ""),
        voltage_limit_v=1.0,
        connector=lambda **_kwargs: (soc, {}),
        adc_reader_factory=FakeAdcReader,
        sleeper=lambda _seconds: None,
    )

    assert result.magnitude_a.shape == (2, 3)
    assert result.x_values.tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert result.y_values.tolist() == pytest.approx([0.0, 0.2])
    assert soc.values == pytest.approx(initial)

    from qcodes import initialise_or_create_database_at, load_by_id

    initialise_or_create_database_at(str(database))
    dataset = load_by_id(result.run_id)
    assert dataset.number_of_results == 72
    parameter_data = dataset.get_parameter_data("i_r_a")["i_r_a"]
    assert set(parameter_data) >= {
        "nested_axis_0_index",
        "nested_axis_0_bias2_voltage_v",
        "nested_axis_1_index",
        "nested_axis_1_bias0_voltage_v",
        "nested_axis_1_bias1_voltage_v",
        "repetition_index",
        "i_r_a",
    }
    assert np.unique(
        parameter_data["nested_axis_1_bias0_voltage_v"]
    ).tolist() == pytest.approx([-0.1, 0.0, 0.1])
    metadata = json.loads(dataset.get_metadata("bias_measurement_json"))
    assert metadata["configuration"]["axes"][1]["channels"] == [0, 1]


def test_two_point_sr860_sweeps_bias_and_saves_settings(tmp_path):
    settings = default_bias_measurement_settings()["two_point"]
    settings.update({
        "database_path": str(tmp_path / "two_point.db"),
        "sample_name": "TwoPointTest",
        "bias_start_v": 0.0,
        "bias_stop_v": 100e-6,
        "points": 3,
        "repetitions_per_point": 1,
    })
    settings["sr860"] = dict(settings["sr860"])
    settings["sr860"]["settle_time_constants"] = 0.0
    soc = FakeSoc()
    result = run_bias_measurement(
        connection_config=QickConnectionConfig("127.0.0.1", 8888, "myqick"),
        kind="two_point",
        settings=settings,
        channel_names=("",) * 8,
        voltage_limit_v=1.0,
        connector=lambda **_kwargs: (soc, {}),
        sr860_factory=FakeSr860,
        sleeper=lambda _seconds: None,
    )
    assert result.x_values.tolist() == pytest.approx([0.0, 50e-6, 100e-6])
    assert result.magnitude_a.tolist() == pytest.approx([0.0, 100e-6, 200e-6])
    assert FakeSr860.last.frequency() == pytest.approx(43.5371)
    assert FakeSr860.last.filter_slope() == 24
    assert FakeSr860.last.amplitude() == pytest.approx(0.0)
    assert FakeSr860.last.closed is True

    from qcodes import initialise_or_create_database_at, load_by_id

    initialise_or_create_database_at(settings["database_path"])
    dataset = load_by_id(result.run_id)
    sr_metadata = json.loads(dataset.get_metadata("sr860_settings_json"))
    assert sr_metadata["time_constant_s"] == pytest.approx(0.1)
    assert sr_metadata["filter_slope_db_oct"] == 24
    assert sr_metadata["sensitivity_a"] == pytest.approx(100e-9)
