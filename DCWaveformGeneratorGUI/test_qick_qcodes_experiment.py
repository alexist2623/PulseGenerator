"""QCoDeS persistence and injected QICK execution tests.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np

from dc_waveform_core import QickDdrReadoutSpec, QickRfPulseSpec
from qick_fine_tune_sweep import (
    AmplitudeSweep,
    FineTuneDdrResult,
    FineTuneSequence,
)
from qick_qcodes_experiment import (
    IQ_TRACE_PARAMETER,
    QCODES_STAGING_ENV,
    QcodesRunConfig,
    QickConnectionConfig,
    _sweep_parameter_names,
    build_awg_vertex_metadata,
    build_runtime_ddr_readout,
    build_runtime_rf_pulses,
    connect_qick,
    load_qick_iq_arrays,
    run_qick_qcodes_experiment,
    store_qick_result,
)


def test_sweep_parameter_names_identify_output_segment_and_voltage_unit():
    axes = (
        AmplitudeSweep("set_1", "awg_0", -0.5, 0.5, 3),
        AmplitudeSweep("gate hold", "awg-3", -0.25, 0.25, 5),
    )

    assert _sweep_parameter_names(axes) == (
        "awg_0_set_1_voltage_mv",
        "awg_3_gate_hold_voltage_mv",
    )


def _ddr_result():
    iq = np.asarray(
        [
            [
                [[1, -1], [2, -2], [3, -3]],
                [[4, -4], [5, -5], [6, -6]],
            ],
            [
                [[10, 20], [11, 21], [12, 22]],
                [[13, 23], [14, 24], [15, 25]],
            ],
        ],
        dtype=np.int32,
    )
    return FineTuneDdrResult(
        sweep_points=np.asarray([-0.5, 0.5]),
        iq=iq,
        sweep_axes=(AmplitudeSweep("set_1", "awg_0", -0.5, 0.5, 2),),
        sweep_shape=(2,),
        cross_capacitance=np.eye(1),
    )


def _gui_metadata():
    return {
        "waveforms": {
            "time_ns": [0.0, 1000.0],
            "virtual_mv": [[100.0, 100.0]],
            "physical_mv": [[100.0, 100.0]],
        },
        "awg": {"cross_capacitance": [[1.0]]},
        "qick": {"fabric_mhz": 300.0, "full_scale_mv": 100.0},
        "rf_outputs": [{"frequency_mhz": 50.0, "gain": 12000}],
        "rf_readout": {"readout_frequency_mhz": 25.0},
    }


def test_store_qick_result_writes_iq_and_awg_vertices_as_data(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "qick_trace.db"
    staging_root = tmp_path / "staging"
    monkeypatch.setenv(QCODES_STAGING_ENV, str(staging_root))
    progress_updates = []
    sequence = FineTuneSequence(("awg_0", "awg_1"))
    sequence.add_set("start", (0.0, 0.25), 10)
    sequence.set_amplitude_sweep("start", "awg_0", -0.5, 0.5, 2)
    gui_metadata = _gui_metadata()
    gui_metadata["awg_waveform_vertices"] = build_awg_vertex_metadata(
        sequence,
        fabric_mhz=300.0,
        full_scale_mv=100.0,
    )
    dataset, row_count = store_qick_result(
        _ddr_result(),
        run_config=QcodesRunConfig(
            str(database_path),
            experiment_name="QICK test",
            sample_name="simulated device",
            notes="headless persistence test",
        ),
        connection_config=QickConnectionConfig("192.0.2.10", 8888, "myqick"),
        program_summary={"program_instructions": 42},
        gui_settings=gui_metadata,
        rf_settings={"outputs": [[10.0, 12.0]], "readout": 20.0},
        progress_callback=lambda percent, message: progress_updates.append(
            (percent, message)
        ),
        batch_rows=5,
    )

    assert database_path.exists()
    assert row_count == 12
    # Four IQ traces plus virtual/physical vertex arrays for two outputs at
    # two sweep points. Each dependent array occupies one QCoDeS result row.
    assert dataset.number_of_results == 12
    sweep_parameter = "awg_0_set_1_voltage_mv"
    dataset_parameters = {
        parameter.strip() for parameter in dataset.parameters.split(",")
    }
    assert IQ_TRACE_PARAMETER in dataset_parameters
    assert sweep_parameter in dataset_parameters
    for removed_parameter in (
        "point_index",
        "sample_index",
        "time_us",
        "i",
        "q",
        "magnitude",
        "phase_deg",
        "awg_output_index",
        "awg_vertex_index",
        "awg_vertex_time_us",
        "awg_virtual_vertex_mv",
        "awg_physical_vertex_mv",
    ):
        assert removed_parameter not in dataset_parameters

    trace_data = dataset.get_parameter_data(IQ_TRACE_PARAMETER)[
        IQ_TRACE_PARAMETER
    ]
    expected_flat_iq = _ddr_result().iq.reshape(4, 3, 2)
    assert trace_data[IQ_TRACE_PARAMETER].shape == (4, 3, 2)
    np.testing.assert_array_equal(
        trace_data[IQ_TRACE_PARAMETER],
        expected_flat_iq,
    )
    np.testing.assert_allclose(
        trace_data[sweep_parameter][:, 0, 0],
        [-50.0, -50.0, 50.0, 50.0],
    )
    np.testing.assert_array_equal(
        trace_data["repetition_index"][:, 0, 0],
        [0, 1, 0, 1],
    )

    loaded = load_qick_iq_arrays(dataset)
    np.testing.assert_array_equal(loaded["iq"], _ddr_result().iq)
    np.testing.assert_array_equal(loaded["i"], _ddr_result().iq[..., 0])
    np.testing.assert_array_equal(loaded["q"], _ddr_result().iq[..., 1])
    np.testing.assert_array_equal(loaded["sample_index"], [0, 1, 2])
    np.testing.assert_allclose(loaded["time_us"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        loaded["magnitude"],
        np.hypot(_ddr_result().iq[..., 0], _ddr_result().iq[..., 1]),
    )

    awg_0_virtual = dataset.get_parameter_data(
        "awg_0_virtual_vertices_mv"
    )["awg_0_virtual_vertices_mv"]
    awg_0_physical = dataset.get_parameter_data(
        "awg_0_physical_vertices_mv"
    )["awg_0_physical_vertices_mv"]
    awg_1_virtual = dataset.get_parameter_data(
        "awg_1_virtual_vertices_mv"
    )["awg_1_virtual_vertices_mv"]
    awg_1_physical = dataset.get_parameter_data(
        "awg_1_physical_vertices_mv"
    )["awg_1_physical_vertices_mv"]
    np.testing.assert_allclose(
        awg_0_virtual["awg_0_virtual_vertices_mv"],
        [[-50.0, -50.0], [50.0, 50.0]],
    )
    np.testing.assert_allclose(
        awg_0_physical["awg_0_physical_vertices_mv"],
        [[-50.0, -50.0], [50.0, 50.0]],
    )
    np.testing.assert_allclose(
        awg_1_virtual["awg_1_virtual_vertices_mv"],
        [[25.0, 25.0], [25.0, 25.0]],
    )
    np.testing.assert_allclose(
        awg_1_physical["awg_1_physical_vertices_mv"],
        [[25.0, 25.0], [25.0, 25.0]],
    )
    np.testing.assert_allclose(
        awg_0_virtual[sweep_parameter][:, 0],
        [-50.0, 50.0],
    )
    metadata = json.loads(dataset.get_metadata("qick_experiment_json"))
    assert metadata["qick_connection"]["host"] == "192.0.2.10"
    assert metadata["measurement_layout"]["iq_shape"] == [2, 2, 3, 2]
    assert metadata["measurement_layout"]["awg_vertex_shape"] == [2, 2, 2]
    assert "awg_waveform_vertices" not in metadata["gui_settings"]
    layout = metadata["measurement_layout"]
    assert layout["sample_period_us"] == 1.0
    assert layout["storage_format"] == "qcodes_array_per_trace_v1"
    assert layout["sql_rows_per_trace"] == 1
    assert layout["time_reconstruction"].startswith("sample_index = arange")
    assert layout["sweep_axes"][0]["parameter"] == sweep_parameter
    assert layout["sweep_axes"][0]["voltage_start_mv"] == -50.0
    assert layout["sweep_axes"][0]["voltage_stop_mv"] == 50.0
    assert "point_index" not in layout["setpoint_meanings"]
    assert "time_us" not in layout["setpoint_meanings"]
    assert layout["awg_vertex_channels"] == {
        "awg_0": {
            "virtual_parameter": "awg_0_virtual_vertices_mv",
            "physical_parameter": "awg_0_physical_vertices_mv",
            "time_us": [0.0, 10 / 300],
            "vertex_count": 2,
        },
        "awg_1": {
            "virtual_parameter": "awg_1_virtual_vertices_mv",
            "physical_parameter": "awg_1_physical_vertices_mv",
            "time_us": [0.0, 10 / 300],
            "vertex_count": 2,
        },
    }
    assert json.loads(dataset.get_metadata("cross_capacitance_json")) == [[1.0]]
    assert progress_updates[0][0] == 65
    assert progress_updates[-1][0] == 99
    assert [item[0] for item in progress_updates] == sorted(
        item[0] for item in progress_updates
    )
    assert not list(staging_root.glob("qick_qcodes_*"))


def test_store_qick_result_keeps_cartesian_sweeps_and_repetitions_grouped(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(QCODES_STAGING_ENV, str(tmp_path / "staging"))
    axis_0 = AmplitudeSweep("set_1", "awg_0", 0.1, 0.3, 5)
    axis_1 = AmplitudeSweep("set_1", "awg_1", -0.1, -0.2, 5)
    coordinates = np.asarray([
        (voltage_0, voltage_1)
        for voltage_0 in np.linspace(axis_0.start, axis_0.stop, axis_0.count)
        for voltage_1 in np.linspace(axis_1.start, axis_1.stop, axis_1.count)
    ])
    iq = np.arange(25 * 2 * 128 * 2, dtype=np.int32).reshape(25, 2, 128, 2)
    result = FineTuneDdrResult(
        sweep_points=coordinates,
        iq=iq,
        sweep_axes=(axis_0, axis_1),
        sweep_shape=(5, 5),
        cross_capacitance=np.eye(2),
    )
    database_path = tmp_path / "cartesian_traces.db"
    dataset, row_count = store_qick_result(
        result,
        run_config=QcodesRunConfig(
            str(database_path),
            experiment_name="Cartesian trace test",
            sample_name="two AWG outputs",
        ),
        connection_config=QickConnectionConfig("192.0.2.10", 8888, "myqick"),
        program_summary={"program_instructions": 100},
        gui_settings={
            "qick": {"fabric_mhz": 300.0, "full_scale_mv": 800.0},
        },
        rf_settings={},
        batch_rows=512,
    )

    assert row_count == 25 * 2 * 128
    assert dataset.number_of_results == 50
    with sqlite3.connect(database_path) as connection:
        sql_row_count = connection.execute(
            f'SELECT COUNT(*) FROM "{dataset.table_name}"'
        ).fetchone()[0]
    assert sql_row_count == 50

    trace_data = dataset.get_parameter_data(IQ_TRACE_PARAMETER)[
        IQ_TRACE_PARAMETER
    ]
    assert trace_data[IQ_TRACE_PARAMETER].shape == (50, 128, 2)
    np.testing.assert_array_equal(
        trace_data[IQ_TRACE_PARAMETER],
        iq.reshape(50, 128, 2),
    )
    np.testing.assert_allclose(
        trace_data["awg_0_set_1_voltage_mv"][:, 0, 0],
        np.repeat(coordinates[:, 0] * 800.0, 2),
    )
    np.testing.assert_allclose(
        trace_data["awg_1_set_1_voltage_mv"][:, 0, 0],
        np.repeat(coordinates[:, 1] * 800.0, 2),
    )
    np.testing.assert_array_equal(
        trace_data["repetition_index"][:, 0, 0],
        np.tile([0, 1], 25),
    )
    loaded = load_qick_iq_arrays(dataset)
    np.testing.assert_array_equal(loaded["iq"], iq)
    assert loaded["iq"].shape == (25, 2, 128, 2)
    dataset_parameters = {
        parameter.strip() for parameter in dataset.parameters.split(",")
    }
    for unstored_parameter in (
        "sample_index",
        "time_us",
        "i",
        "q",
        "magnitude",
        "phase_deg",
        "awg_vertex_time_us",
    ):
        assert unstored_parameter not in dataset_parameters


def test_local_staging_preserves_existing_database_runs(tmp_path, monkeypatch):
    staging_root = tmp_path / "staging"
    monkeypatch.setenv(QCODES_STAGING_ENV, str(staging_root))
    database_path = tmp_path / "cumulative.db"
    run_config = QcodesRunConfig(
        str(database_path),
        experiment_name="Cumulative trace test",
        sample_name="same database",
    )
    common_arguments = {
        "run_config": run_config,
        "connection_config": QickConnectionConfig(
            "192.0.2.10", 8888, "myqick"
        ),
        "program_summary": {"program_instructions": 42},
        "gui_settings": _gui_metadata(),
        "rf_settings": {},
    }

    first_dataset, _ = store_qick_result(_ddr_result(), **common_arguments)
    first_guid = first_dataset.guid
    first_dataset.conn.close()
    second_dataset, _ = store_qick_result(_ddr_result(), **common_arguments)

    with sqlite3.connect(database_path) as connection:
        runs = connection.execute(
            "SELECT guid FROM runs ORDER BY run_id"
        ).fetchall()
    assert len(runs) == 2
    assert runs[0][0] == first_guid
    assert runs[1][0] == second_dataset.guid
    assert not list(staging_root.glob("qick_qcodes_*"))


def test_awg_vertex_metadata_stores_all_sweep_points_in_both_spaces():
    sequence = FineTuneSequence(("awg_0", "awg_1"))
    sequence.set_cross_capacitance(((1.0, 0.0), (0.5, 1.0)))
    sequence.add_set("start", (0.0, 0.0), 10)
    sequence.add_ramp("to_gate", 5)
    sequence.add_set("gate", (0.2, -0.1), 15)
    sequence.set_amplitude_sweep("gate", "awg_0", -0.2, 0.2, 3)

    metadata = build_awg_vertex_metadata(
        sequence,
        fabric_mhz=300.0,
        full_scale_mv=100.0,
    )
    virtual = metadata["virtual"]
    physical = metadata["physical"]
    assert virtual["time_cycles"] == [0.0, 10.0, 15.0, 30.0]
    assert virtual["time_us"] == [0.0, 10 / 300, 15 / 300, 30 / 300]
    assert virtual["value_shape"] == [3, 2, 4]
    assert virtual["values_mv"][0][0] == [0.0, 0.0, -20.0, -20.0]
    assert physical["values_mv"][0][1] == [0.0, 0.0, -20.0, -20.0]


def test_runtime_configs_use_actual_qick_clocks():
    soccfg = {
        "tprocs": [{"f_time": 300.0}],
        "gens": [{"f_fabric": 250.0}],
    }
    rf = QickRfPulseSpec(
        0, "set_0", 0.2, 0.4, 43.5, 10000, 6.0, 7.0
    )
    ddr = QickDdrReadoutSpec(0, "set_0", 0.25, 16, 12.5)
    runtime_rf = build_runtime_rf_pulses(soccfg, (rf,))[0]
    runtime_ddr = build_runtime_ddr_readout(soccfg, ddr)

    assert runtime_rf.length_cycles == 100
    assert runtime_rf.delay_tproc_cycles == 60
    assert runtime_ddr.trigger_delay_tproc_cycles == 75
    assert runtime_ddr.samples_per_trigger == 16


def test_connect_and_run_support_injected_qick_server(tmp_path, monkeypatch):
    monkeypatch.setenv(QCODES_STAGING_ENV, str(tmp_path / "staging"))
    ddr_result = _ddr_result()
    calls = []
    progress_updates = []

    class FakeSoc:
        def rfb_set_gen_rf(self, gen_ch, att1, att2):
            calls.append(("gen", gen_ch, att1, att2))
            return att1, att2

        def rfb_set_ro_rf(self, ro_ch, attenuation):
            calls.append(("ro", ro_ch, attenuation))
            return attenuation

        def rfb_set_ro_filter(self, ro_ch, **kwargs):
            calls.append(("filter", ro_ch, kwargs))

    class FakeProgram:
        def acquire_fir_ddr(self, soc, progress=False, counter_progress=None):
            calls.append(("acquire", soc, progress))
            if counter_progress is not None:
                counter_progress(0, 4)
                counter_progress(2, 4)
                counter_progress(4, 4)
            return ddr_result

        def summary(self):
            return {"program_instructions": 7}

    class FakeSequence:
        def make_program(self, soccfg, **kwargs):
            calls.append(("program", soccfg, kwargs))
            return FakeProgram()

    soc = FakeSoc()
    soccfg = {
        "tprocs": [{"f_time": 300.0}],
        "gens": [{"f_fabric": 300.0}],
    }

    def connector(**kwargs):
        calls.append(("connect", kwargs))
        return soc, soccfg

    connection = QickConnectionConfig("198.51.100.9", 9999, "testqick")
    assert connect_qick(connection, connector=connector) == (soc, soccfg)
    result = run_qick_qcodes_experiment(
        connection_config=connection,
        run_config=QcodesRunConfig(
            str(tmp_path / "end_to_end.db"),
            experiment_name="Injected QICK",
            sample_name="fake soc",
        ),
        sequence=FakeSequence(),
        awg_channels=(0,),
        repetitions_per_sweep=2,
        rf_specs=(QickRfPulseSpec(
            0, "set_0", 0.0, 0.4, 50.0, 10000, 5.0, 6.0
        ),),
        readout_spec=QickDdrReadoutSpec(0, "set_0", 0.0, 3),
        gui_settings=_gui_metadata(),
        connector=connector,
        progress_callback=lambda percent, message: progress_updates.append(
            (percent, message)
        ),
    )

    assert result.row_count == 12
    assert result.database_path.exists()
    assert calls.count(("gen", 0, 5.0, 6.0)) == 1
    assert any(call[0] == "acquire" for call in calls)
    assert progress_updates[0][0] == 0
    assert progress_updates[-1] == (100, "Experiment saved")
    assert any(percent == 32 for percent, _ in progress_updates)
    assert any(percent == 55 for percent, _ in progress_updates)
    assert any(percent == 60 for percent, _ in progress_updates)
