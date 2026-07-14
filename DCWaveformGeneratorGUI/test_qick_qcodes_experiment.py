"""QCoDeS persistence and injected QICK execution tests.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import json

import numpy as np

from dc_waveform_core import QickDdrReadoutSpec, QickRfPulseSpec
from qick_fine_tune_sweep import AmplitudeSweep, FineTuneDdrResult
from qick_qcodes_experiment import (
    QcodesRunConfig,
    QickConnectionConfig,
    build_runtime_ddr_readout,
    build_runtime_rf_pulses,
    connect_qick,
    run_qick_qcodes_experiment,
    store_qick_result,
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
        "rf_outputs": [{"frequency_mhz": 50.0, "gain": 12000}],
        "rf_readout": {"readout_frequency_mhz": 25.0},
    }


def test_store_qick_result_writes_trace_rows_and_metadata(tmp_path):
    database_path = tmp_path / "qick_trace.db"
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
        gui_settings=_gui_metadata(),
        rf_settings={"outputs": [[10.0, 12.0]], "readout": 20.0},
    )

    assert database_path.exists()
    assert row_count == 12
    # QCoDeS counts one result per dependent parameter, while row_count is
    # the number of acquired time samples written to each I/Q quantity.
    assert dataset.number_of_results == 4 * row_count
    data = dataset.get_parameter_data("i")["i"]
    assert data["i"].tolist() == [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    ]
    assert data["time_us"].tolist() == [0.0, 1.0, 2.0] * 4
    metadata = json.loads(dataset.get_metadata("qick_experiment_json"))
    assert metadata["qick_connection"]["host"] == "192.0.2.10"
    assert metadata["measurement_layout"]["iq_shape"] == [2, 2, 3, 2]
    assert json.loads(dataset.get_metadata("cross_capacitance_json")) == [[1.0]]


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
        def acquire_fir_ddr(self, soc, progress=False):
            calls.append(("acquire", soc, progress))
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
    )

    assert result.row_count == 12
    assert result.database_path.exists()
    assert calls.count(("gen", 0, 5.0, 6.0)) == 1
    assert any(call[0] == "acquire" for call in calls)
