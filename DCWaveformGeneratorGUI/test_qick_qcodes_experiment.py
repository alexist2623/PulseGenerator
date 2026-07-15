"""QCoDeS persistence and injected QICK execution tests.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import json

import numpy as np

from dc_waveform_core import QickDdrReadoutSpec, QickRfPulseSpec
from qick_fine_tune_sweep import (
    AmplitudeSweep,
    FineTuneDdrResult,
    FineTuneSequence,
)
from qick_qcodes_experiment import (
    QcodesRunConfig,
    QickConnectionConfig,
    _sweep_parameter_names,
    build_awg_vertex_metadata,
    build_runtime_ddr_readout,
    build_runtime_rf_pulses,
    connect_qick,
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


def test_store_qick_result_writes_iq_and_awg_vertices_as_data(tmp_path):
    database_path = tmp_path / "qick_trace.db"
    progress_updates = []
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("start", (0.0,), 10)
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
    # QCoDeS counts one result per dependent parameter. row_count remains the
    # number of acquired time samples written to each I/Q quantity.
    vertex_row_count = 2 * 1 * 2
    assert dataset.number_of_results == 4 * row_count + 2 * vertex_row_count
    data = dataset.get_parameter_data("i")["i"]
    assert data["i"].tolist() == [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    ]
    sweep_parameter = "awg_0_set_1_voltage_mv"
    assert data[sweep_parameter].tolist() == [-50.0] * 6 + [50.0] * 6
    assert data["sample_index"].tolist() == [0, 1, 2] * 4
    dataset_parameters = {
        parameter.strip() for parameter in dataset.parameters.split(",")
    }
    assert "point_index" not in dataset_parameters
    assert "time_us" not in dataset_parameters
    assert sweep_parameter in dataset_parameters
    virtual_vertices = dataset.get_parameter_data("awg_virtual_vertex_mv")[
        "awg_virtual_vertex_mv"
    ]
    physical_vertices = dataset.get_parameter_data("awg_physical_vertex_mv")[
        "awg_physical_vertex_mv"
    ]
    assert virtual_vertices["awg_virtual_vertex_mv"].tolist() == [
        -50.0, -50.0, 50.0, 50.0,
    ]
    assert physical_vertices["awg_physical_vertex_mv"].tolist() == [
        -50.0, -50.0, 50.0, 50.0,
    ]
    assert virtual_vertices["awg_vertex_time_us"].tolist() == [
        0.0, 10 / 300, 0.0, 10 / 300,
    ]
    assert virtual_vertices["awg_output_index"].tolist() == [0, 0, 0, 0]
    assert virtual_vertices["awg_vertex_index"].tolist() == [0, 1, 0, 1]
    assert virtual_vertices[sweep_parameter].tolist() == [
        -50.0, -50.0, 50.0, 50.0,
    ]
    metadata = json.loads(dataset.get_metadata("qick_experiment_json"))
    assert metadata["qick_connection"]["host"] == "192.0.2.10"
    assert metadata["measurement_layout"]["iq_shape"] == [2, 2, 3, 2]
    assert metadata["measurement_layout"]["awg_vertex_shape"] == [2, 1, 2]
    assert "awg_waveform_vertices" not in metadata["gui_settings"]
    layout = metadata["measurement_layout"]
    assert layout["sample_period_us"] == 1.0
    assert layout["time_reconstruction"].startswith("time_us = sample_index")
    assert layout["sweep_axes"][0]["parameter"] == sweep_parameter
    assert layout["sweep_axes"][0]["voltage_start_mv"] == -50.0
    assert layout["sweep_axes"][0]["voltage_stop_mv"] == 50.0
    assert "point_index" not in layout["setpoint_meanings"]
    assert "time_us" not in layout["setpoint_meanings"]
    assert json.loads(dataset.get_metadata("cross_capacitance_json")) == [[1.0]]
    assert progress_updates[0][0] == 65
    assert progress_updates[-1][0] == 99
    assert [item[0] for item in progress_updates] == sorted(
        item[0] for item in progress_updates
    )


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


def test_connect_and_run_support_injected_qick_server(tmp_path):
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
