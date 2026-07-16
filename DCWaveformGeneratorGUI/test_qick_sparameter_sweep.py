"""RF-only hardware S-parameter sweep, storage, and GUI tests.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import json
import os
import sqlite3

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5 import QtWidgets

# Desktop imports need the PYNQ stubs installed before regular QICK modules.
from qick.sim import QickSim  # noqa: F401
from qick.qick_asm import QickConfig

import DCWaveform_Generator as gui
import qick_sparameter_sweep as sparameter_module
from qick_qcodes_experiment import (
    QCODES_STAGING_ENV,
    QcodesRunConfig,
    QickConnectionConfig,
)
from qick_sparameter_sweep import (
    CALIBRATED_GAIN_PARAMETER,
    MAGNITUDE_DB_PARAMETER,
    MAX_RF_OUTPUT_GAIN,
    POWER_GAIN_PARAMETER,
    SParameterPowerSweepResult,
    SParameterSweepConfig,
    SParameterSweepProgram,
    SParameterSweepResult,
    configure_sparameter_rf_board,
    load_sparameter_run,
    run_sparameter_sweep,
    store_sparameter_result,
)


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _mock_soccfg(*, generator_type="axis_signal_gen_v6"):
    awg = generator_type == "axis_awg_tuning_v1"
    return QickConfig({
        "sw_version": "0.2.357",
        "refclk_freq": 300.0,
        "tprocs": [{
            "type": "axis_tproc64x32_x8",
            "f_time": 300.0,
            "pmem_size": 65536,
            "dmem_size": 4096,
            "output_pins": [],
        }],
        "gens": [{
            "type": generator_type,
            "gen_type": "awg_tuning" if awg else "signal_generator",
            "tproc_ch": 0,
            "tmux_ch": 0,
            "f_fabric": 300.0,
            "f_dds": 300.0,
            "fs_mult": 1,
            "fs_div": 1,
            "fdds_div": 1,
            "samps_per_clk": 16,
            "maxlen": 16384,
            "maxv": 32764 if awg else 32766,
            "maxv_scale": 1.0,
            "complex_env": not awg,
            "has_mixer": False,
            "has_dds": not awg,
            "b_dds": 32,
            "b_phase": 32,
            "dac": "00",
            "interpolation": 1,
            **({
                "n_pts": 16,
                "frac": 16,
                "cmd_width": 160,
                "step_width": 24,
                "duration_width": 23,
                "fixed_width": 48,
            } if awg else {}),
        }],
        "readouts": [{
            "type": "axis_dyn_readout_v1",
            "ro_type": "axis_dyn_readout_v1",
            "tproc_ctrl": 1,
            "tmux_ch": 0,
            "f_fabric": 300.0,
            "f_output": 300.0,
            "f_dds": 300.0,
            "fs_mult": 1,
            "fs_div": 1,
            "fdds_div": 1,
            "b_dds": 32,
            "b_phase": 32,
            "adc": "00",
            "buf_maxlen": 16384,
            "has_weights": False,
            "has_edge_counter": False,
            "trigger_type": "dport",
            "trigger_port": 0,
            "trigger_bit": 0,
        }],
        "ddr4_buf": {
            "sample_capture": True,
            "fir_enabled": True,
            "fir_output_fs_mhz": 1.0,
            "fir_decimation": 300,
            "fir_group_delay_input_samples": 8677,
            "fir_input_fs_mhz": 300.0,
            "trigger_type": "dport",
            "trigger_port": 0,
            "trigger_bit": 1,
        },
    })


def _config(**updates):
    values = {
        "frequency_start_mhz": 10.0,
        "frequency_end_mhz": 20.0,
        "frequency_points": 11,
        "scan_time_us": 4.0,
        "settle_seconds": 0.0,
    }
    values.update(updates)
    return SParameterSweepConfig(**values)


def _result():
    frequencies = np.asarray([10.0, 11.0, 12.0])
    iq = np.asarray([
        [[3, 4], [3, 4], [3, 4], [3, 4]],
        [[-3, 1], [-3, 1], [-3, 1], [-3, 1]],
        [[-3, -1], [-3, -1], [-3, -1], [-3, -1]],
    ], dtype=np.int32)
    return SParameterSweepResult.from_iq(frequencies, frequencies, iq)


def test_result_uses_mean_iq_db_magnitude_and_unwrapped_phase():
    result = _result()

    expected_complex = result.mean_i + 1j * result.mean_q
    np.testing.assert_allclose(
        result.magnitude_db,
        20.0 * np.log10(np.abs(expected_complex)),
    )
    np.testing.assert_allclose(
        result.phase_unwrapped_deg,
        np.degrees(np.unwrap(np.angle(expected_complex))),
    )
    assert result.mean_i[0] == 3.0
    assert result.mean_q[0] == 4.0


def test_gain_has_hard_limit():
    assert SParameterSweepConfig(gain=MAX_RF_OUTPUT_GAIN).gain == 32766
    with pytest.raises(ValueError, match="must not exceed"):
        SParameterSweepConfig(gain=MAX_RF_OUTPUT_GAIN + 1)


def test_power_gain_grid_supports_linear_and_log_spacing():
    linear = SParameterSweepConfig(
        power_sweep_enabled=True,
        power_start_gain=1000,
        power_end_gain=20000,
        power_points=5,
        power_scale="linear",
    )
    logarithmic = SParameterSweepConfig(
        power_sweep_enabled=True,
        power_start_gain=100,
        power_end_gain=10000,
        power_points=5,
        power_scale="log",
    )

    np.testing.assert_array_equal(
        linear.power_gains,
        [1000, 5750, 10500, 15250, 20000],
    )
    np.testing.assert_array_equal(
        logarithmic.power_gains,
        [100, 316, 1000, 3162, 10000],
    )
    with pytest.raises(ValueError, match="greater than zero"):
        SParameterSweepConfig(
            power_sweep_enabled=True,
            power_start_gain=0,
            power_end_gain=100,
            power_points=3,
            power_scale="log",
        )
    with pytest.raises(ValueError, match="duplicate integer gain"):
        SParameterSweepConfig(
            power_sweep_enabled=True,
            power_start_gain=1,
            power_end_gain=2,
            power_points=3,
            power_scale="linear",
        )


def test_program_is_fixed_size_hardware_sweep_and_updates_both_dds_registers():
    short = SParameterSweepProgram(_mock_soccfg(), _config(frequency_points=3))
    long = SParameterSweepProgram(_mock_soccfg(), _config(frequency_points=1001))
    short.compile()
    long.compile()

    assert len(short.prog_list) == len(long.prog_list)
    assert sum(inst["name"] == "math" for inst in long.prog_list) == 2
    assert sum(inst["name"] == "loopnz" for inst in long.prog_list) == 2
    assert long.summary()["sweep_execution"] == "tproc_hardware_register_add"
    assert long.summary()["awg_tuning_used"] is False
    assert long.scan_samples == 4
    assert long.loop_dims == [1001, 1]


def test_calibrated_program_reads_gain_table_from_dmem_and_compresses_addresses():
    gain_table = tuple(1000 + index for index in range(4000))
    config = _config(
        frequency_points=5001,
        frequency_start_mhz=10.0,
        frequency_end_mhz=20.0,
        power_calibration_enabled=True,
        calibration_database_path="calibration.db",
        output_power_dbm=-20.0,
        calibrated_gain_table=gain_table,
        calibrated_frequency_point_count=5001,
        calibrated_output_power_dbm=-20.0,
        calibrated_nominal_gain_code=5000,
        calibrated_reference_response_dbm=10.0,
        calibrated_correction_min_db=-1.0,
        calibrated_correction_max_db=0.0,
        calibration_output_run_id=31,
    )
    program = SParameterSweepProgram(_mock_soccfg(), config)
    program.compile()

    assert program.summary()["gain_source"] == "tproc_dmem_frequency_table"
    assert program.summary()["gain_dmem_entry_count"] == 4000
    assert program.summary()["gain_table_compressed"] is True
    assert sum(inst["name"] == "memr" for inst in program.prog_list) == 1
    assert any(
        inst["name"] == "condj"
        and "reuse adjacent calibrated gain" in inst.get("comment", "")
        for inst in program.prog_list
    )
    np.testing.assert_array_equal(
        program._expanded_gain_codes()[:4],
        [1000, 1000, 1001, 1002],
    )

    calls = []

    class FakeSoc:
        def reload_mem(self):
            calls.append(("reload",))

        def load_mem(self, data, mem_sel, addr):
            calls.append(("load", np.asarray(data).copy(), mem_sel, addr))

    program._load_runtime_dmem(FakeSoc())
    assert calls[0] == ("reload",)
    assert calls[1][2:] == ("dmem", 16)
    np.testing.assert_array_equal(calls[1][1], gain_table)


def test_program_quantizes_from_channel_metadata_when_refclk_is_missing():
    soccfg = _mock_soccfg()
    del soccfg._cfg["refclk_freq"]
    program = SParameterSweepProgram(soccfg, _config())
    program.compile()

    assert program.frequency_quantum_mhz > 0.0
    assert sum(inst["name"] == "math" for inst in program.prog_list) == 2


def test_long_scan_uses_periodic_start_and_timed_zero_stop():
    program = SParameterSweepProgram(
        _mock_soccfg(),
        _config(scan_time_us=1000.0),
    )
    program.compile()

    summary = program.summary()
    assert summary["rf_output_mode"] == "periodic_start_timed_zero_stop"
    assert summary["rf_periodic_word_fabric_cycles"] == 3
    assert summary["rf_stop_word_fabric_cycles"] == 3
    assert summary["readout_period_fabric_cycles"] == 3
    assert summary["rf_stop_command_tproc_cycle"] > 65535

    gain_writes = [
        inst["args"][2]
        for inst in program.prog_list
        if inst.get("comment", "").startswith("gain =")
    ]
    assert gain_writes == [program.sweep.gain, 0]

    generator_modes = [
        inst["args"][2]
        for inst in program.prog_list
        if inst.get("comment", "").startswith("phrst|")
    ]
    assert len(generator_modes) == 2
    start_mode, stop_mode = generator_modes
    assert start_mode & 0xFFFF == 3
    assert stop_mode & 0xFFFF == 3
    assert (start_mode >> 16) & 0b00100
    assert not ((stop_mode >> 16) & 0b00100)

    initial_frequency_writes = [
        inst
        for inst in program.prog_list
        if inst.get("comment", "").startswith("freq =")
    ]
    assert len(initial_frequency_writes) == 2


def test_program_rejects_awg_tuning_as_rf_sweep_output():
    with pytest.raises(ValueError, match="not axis_awg_tuning_v1"):
        SParameterSweepProgram(
            _mock_soccfg(generator_type="axis_awg_tuning_v1"),
            _config(),
        )


def test_fir_ddr_acquisition_keeps_one_trace_per_frequency(monkeypatch):
    program = SParameterSweepProgram(
        _mock_soccfg(), _config(frequency_points=3, scan_time_us=4.0)
    )
    raw = np.arange(3 * 4 * 2, dtype=np.int32).reshape(12, 2)

    class FakeSoc:
        def arm_ddr4_fir_samples(self, **kwargs):
            self.arm_kwargs = kwargs
            return 24

        def get_ddr4_fir_samples(self, **kwargs):
            self.read_kwargs = kwargs
            return raw

    soc = FakeSoc()
    monkeypatch.setattr(program, "run_rounds", lambda *_args, **_kwargs: None)
    result = program.acquire_fir_ddr(soc)

    assert soc.arm_kwargs["n_triggers"] == 3
    assert soc.arm_kwargs["n_samples"] == 4
    assert result.iq_traces.shape == (3, 4, 2)
    np.testing.assert_array_equal(result.iq_traces.reshape(12, 2), raw)
    np.testing.assert_allclose(result.mean_i, raw.reshape(3, 4, 2)[:, :, 0].mean(1))


def test_rf_board_output_and_readout_controls_are_applied():
    calls = []

    class FakeSoc:
        def rfb_set_gen_rf(self, *args):
            calls.append(("gen_rf", args))
            return args[1], args[2]

        def rfb_set_gen_filter(self, *args, **kwargs):
            calls.append(("gen_filter", args, kwargs))

        def rfb_set_ro_rf(self, *args):
            calls.append(("ro_rf", args))
            return args[1]

        def rfb_set_ro_filter(self, *args, **kwargs):
            calls.append(("ro_filter", args, kwargs))

    actual = configure_sparameter_rf_board(FakeSoc(), _config())

    assert [call[0] for call in calls] == [
        "gen_rf", "gen_filter", "ro_rf", "ro_filter"
    ]
    assert actual["output"]["commanded_att1_db"] == 10.0
    assert actual["readout"]["commanded_attenuation_db"] == 20.0


def test_qcodes_round_trip_stores_split_traces_and_derived_response(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "sparameter.db"
    monkeypatch.setenv(QCODES_STAGING_ENV, str(tmp_path / "staging"))
    connection = QickConnectionConfig(host="127.0.0.1")
    run = QcodesRunConfig(
        database_path=str(database_path),
        experiment_name="RF tank response",
        sample_name="unit-test",
    )
    result = _result()
    dataset, row_count = store_sparameter_result(
        result,
        config=_config(frequency_points=3),
        connection_config=connection,
        run_config=run,
        program_summary={"sweep_execution": "tproc_hardware_register_add"},
        rf_settings={"output": {}, "readout": {}},
    )

    assert database_path.exists()
    assert row_count == 12
    loaded = load_sparameter_run(database_path, dataset.run_id)
    np.testing.assert_array_equal(loaded.result.iq_traces, result.iq_traces)
    np.testing.assert_allclose(loaded.result.magnitude_db, result.magnitude_db)
    np.testing.assert_allclose(
        loaded.result.phase_unwrapped_deg, result.phase_unwrapped_deg
    )

    from qcodes import (
        Measurement,
        Parameter,
        initialise_or_create_database_at,
        load_or_create_experiment,
    )

    initialise_or_create_database_at(str(database_path))
    unrelated_measurement = Measurement(
        exp=load_or_create_experiment("unrelated", "unit-test")
    )
    unrelated = Parameter("unrelated_scalar")
    unrelated_measurement.register_parameter(unrelated)
    with unrelated_measurement.run() as datasaver:
        datasaver.add_result((unrelated, 1.0))

    latest_sparameter = load_sparameter_run(database_path, 0)
    assert latest_sparameter.run_id == dataset.run_id


def test_calibrated_single_power_round_trip_stores_applied_gain_per_frequency(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "calibrated_sparameter.db"
    monkeypatch.setenv(QCODES_STAGING_ENV, str(tmp_path / "staging-calibrated"))
    connection = QickConnectionConfig(host="127.0.0.1")
    run = QcodesRunConfig(
        database_path=str(database_path),
        experiment_name="Calibrated RF response",
        sample_name="unit-test",
    )
    base = _result()
    result = SParameterSweepResult.from_iq(
        base.requested_frequencies_mhz,
        base.frequencies_mhz,
        base.iq_traces,
        output_power_dbm=-20.0,
        nominal_gain_code=1200,
        frequency_gain_codes=[1000, 1100, 1200],
        actual_output_powers_dbm=[-30.0, -29.0, -28.0],
        input_powers_dbm=[-50.0, -48.0, -46.0],
    )
    config = _config(
        frequency_points=3,
        power_calibration_enabled=True,
        calibration_database_path=str(tmp_path / "gain_pwr_calb.db"),
        output_power_dbm=-20.0,
    )

    dataset, _row_count = store_sparameter_result(
        result,
        config=config,
        connection_config=connection,
        run_config=run,
        program_summary={"gain_source": "tproc_dmem_frequency_table"},
        rf_settings={"power_calibration": {"output_run": {"run_id": 31}}},
    )
    loaded = load_sparameter_run(database_path, dataset.run_id)

    assert loaded.result.output_power_dbm == -20.0
    assert loaded.result.nominal_gain_code == 1200
    np.testing.assert_array_equal(
        loaded.result.frequency_gain_codes,
        [1000, 1100, 1200],
    )
    assert loaded.result.physical_power_calibrated
    np.testing.assert_allclose(
        loaded.result.actual_output_powers_dbm,
        [-30.0, -29.0, -28.0],
    )
    np.testing.assert_allclose(
        loaded.result.input_powers_dbm,
        [-50.0, -48.0, -46.0],
    )
    np.testing.assert_allclose(loaded.result.magnitude_db, [-20.0, -19.0, -18.0])
    gain_data = dataset.get_parameter_data(CALIBRATED_GAIN_PARAMETER)[
        CALIBRATED_GAIN_PARAMETER
    ]
    np.testing.assert_array_equal(
        gain_data[CALIBRATED_GAIN_PARAMETER],
        [1000, 1100, 1200],
    )


def test_software_power_sweep_publishes_one_live_db_run_per_power(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "power_sparameter.db"
    monkeypatch.setenv(QCODES_STAGING_ENV, str(tmp_path / "staging"))
    connection = QickConnectionConfig(host="127.0.0.1")
    run = QcodesRunConfig(
        database_path=str(database_path),
        experiment_name="RF power dependence",
        sample_name="unit-test",
    )
    config = _config(
        frequency_points=3,
        scan_time_us=4.0,
        power_sweep_enabled=True,
        power_start_gain=100,
        power_end_gain=10000,
        power_points=3,
        power_scale="log",
        settle_seconds=0.0,
    )

    class FakeProgram:
        def __init__(self, point_config):
            self.sweep = point_config

        def acquire_fir_ddr(self, _soc, counter_progress=None):
            if counter_progress is not None:
                counter_progress(0, self.sweep.frequency_points)
                counter_progress(
                    self.sweep.frequency_points,
                    self.sweep.frequency_points,
                )
            frequencies = self.sweep.requested_frequencies_mhz
            iq = np.zeros((frequencies.size, 4, 2), dtype=np.int32)
            iq[:, :, 0] = self.sweep.gain
            iq[:, :, 1] = np.arange(frequencies.size)[:, np.newaxis]
            return SParameterSweepResult.from_iq(
                frequencies,
                frequencies,
                iq,
            )

        def summary(self):
            return {
                "gain": self.sweep.gain,
                "sweep_execution": "tproc_hardware_register_add",
            }

    monkeypatch.setattr(
        sparameter_module,
        "configure_sparameter_rf_board",
        lambda _soc, _config: {"output": {}, "readout": {}},
    )
    monkeypatch.setattr(
        sparameter_module,
        "build_sparameter_program",
        lambda _soccfg, point_config, tproc_mhz=None: FakeProgram(
            point_config
        ),
    )
    live_updates = []
    progress_updates = []

    def capture_partial(stored):
        with sqlite3.connect(database_path) as connection_db:
            payload_text = connection_db.execute(
                "SELECT sparameter_result_json FROM runs WHERE run_id = ?",
                (stored.run_id,),
            ).fetchone()[0]
        payload = json.loads(payload_text)
        live_updates.append((
            stored.run_id,
            stored.result.power_count,
            payload["completed_power_points"],
            tuple(payload["power_gains"]),
        ))

    stored = run_sparameter_sweep(
        connection_config=connection,
        run_config=run,
        sweep_config=config,
        connector=lambda **_kwargs: (object(), object()),
        partial_callback=capture_partial,
        progress_callback=lambda percent, _message: progress_updates.append(
            percent
        ),
    )

    assert [update[1] for update in live_updates] == [1, 2, 3]
    assert [update[2] for update in live_updates] == [1, 2, 3]
    assert len({update[0] for update in live_updates}) == 1
    assert live_updates[-1][3] == (100, 1000, 10000)
    assert progress_updates == sorted(progress_updates)
    assert progress_updates[-1] == 100
    assert isinstance(stored.result, SParameterPowerSweepResult)
    assert stored.result.iq_traces.shape == (3, 3, 4, 2)
    assert stored.row_count == 3 * 3 * 4
    loaded = load_sparameter_run(database_path, stored.run_id)
    assert isinstance(loaded.result, SParameterPowerSweepResult)
    np.testing.assert_array_equal(loaded.result.power_gains, [100, 1000, 10000])
    np.testing.assert_array_equal(
        loaded.result.iq_traces,
        stored.result.iq_traces,
    )
    magnitude_data = loaded.dataset.get_parameter_data(
        MAGNITUDE_DB_PARAMETER
    )[MAGNITUDE_DB_PARAMETER]
    assert POWER_GAIN_PARAMETER in magnitude_data
    np.testing.assert_array_equal(
        magnitude_data[POWER_GAIN_PARAMETER],
        np.repeat([100, 1000, 10000], 3),
    )


def test_compensated_power_sweep_builds_one_frequency_gain_table_per_power(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "compensated_power_sparameter.db"
    monkeypatch.setenv(QCODES_STAGING_ENV, str(tmp_path / "staging-compensated"))
    connection = QickConnectionConfig(host="127.0.0.1")
    run = QcodesRunConfig(
        database_path=str(database_path),
        experiment_name="Compensated RF power dependence",
        sample_name="unit-test",
    )
    calibration_path = tmp_path / "gain_pwr_calb.db"
    calibration_path.touch()
    config = _config(
        frequency_points=3,
        scan_time_us=4.0,
        power_calibration_enabled=True,
        calibration_database_path=str(calibration_path),
        output_power_dbm=-30.0,
        power_sweep_enabled=True,
        power_start_dbm=-30.0,
        power_end_dbm=-20.0,
        power_points=2,
        power_scale="log",
        settle_seconds=0.0,
    )

    class Summary:
        run_id = 31
        sample_name = "RF_Out_400MHz"

        def as_dict(self):
            return {"run_id": self.run_id, "sample_name": self.sample_name}

    class FakeOutputCalibration:
        def output_power_dbm(self, frequencies, gains, **_kwargs):
            return -20.0 + np.zeros_like(frequencies, dtype=float)

    class FakeInputCalibration:
        summary = Summary()

        def input_power_dbm(self, frequencies, adc_magnitude_db, **_kwargs):
            del frequencies
            return np.asarray(adc_magnitude_db, dtype=float) - 40.0

    built_tables = []

    class FakeProgram:
        def __init__(self, target_power_dbm, nominal_gain, gains):
            self.target_power_dbm = target_power_dbm
            self.nominal_gain = nominal_gain
            self.gains = np.asarray(gains, dtype=np.int64)
            self.frequencies_mhz = config.requested_frequencies_mhz

        def acquire_fir_ddr(self, _soc, counter_progress=None):
            if counter_progress is not None:
                counter_progress(config.frequency_points, config.frequency_points)
            iq = np.zeros((config.frequency_points, 4, 2), dtype=np.int32)
            iq[:, :, 0] = self.gains[:, np.newaxis]
            return SParameterSweepResult.from_iq(
                self.frequencies_mhz,
                self.frequencies_mhz,
                iq,
                output_power_dbm=self.target_power_dbm,
                nominal_gain_code=self.nominal_gain,
                frequency_gain_codes=self.gains,
            )

        def summary(self):
            return {
                "gain_source": "tproc_dmem_frequency_table",
                "nominal_gain": self.nominal_gain,
            }

    def build_for_power(
        _soccfg,
        _config_value,
        *,
        target_power_dbm,
        **_kwargs,
    ):
        nominal_gain = int(round((target_power_dbm + 40.0) * 100.0))
        gains = np.asarray(
            [nominal_gain, nominal_gain - 10, nominal_gain - 20],
            dtype=np.int64,
        )
        built_tables.append((float(target_power_dbm), gains.copy()))
        return FakeProgram(target_power_dbm, nominal_gain, gains), object()

    monkeypatch.setattr(
        sparameter_module,
        "configure_sparameter_rf_board",
        lambda _soc, _config: {"output": {}, "readout": {}},
    )
    monkeypatch.setattr(
        sparameter_module,
        "_prepare_power_calibration",
        lambda *_args, **_kwargs: (
            object(),
            FakeOutputCalibration(),
            Summary(),
            FakeInputCalibration(),
        ),
    )
    monkeypatch.setattr(
        sparameter_module,
        "_build_calibrated_program",
        build_for_power,
    )

    stored = run_sparameter_sweep(
        connection_config=connection,
        run_config=run,
        sweep_config=config,
        connector=lambda **_kwargs: (object(), object()),
    )

    assert [entry[0] for entry in built_tables] == [-30.0, -20.0]
    np.testing.assert_array_equal(built_tables[0][1], [1000, 990, 980])
    np.testing.assert_array_equal(built_tables[1][1], [2000, 1990, 1980])
    np.testing.assert_array_equal(stored.result.power_gains, [1000, 2000])
    np.testing.assert_array_equal(stored.result.output_powers_dbm, [-30.0, -20.0])
    np.testing.assert_array_equal(
        stored.result.frequency_gain_codes,
        [[1000, 990, 980], [2000, 1990, 1980]],
    )


def test_gui_has_independent_sparameter_tab_gain_limit_and_settings_round_trip(
    tmp_path,
):
    app = _application()
    window = gui.MainWindow()
    panel = window._sparameter_panel
    calibration_panel = window._calibration_panel

    assert window._control_tabs.indexOf(panel) >= 0
    assert window._control_tabs.indexOf(calibration_panel) >= 0
    assert panel.gain.maximum() == MAX_RF_OUTPUT_GAIN
    panel.frequency_start_mhz.setValue(42.0)
    panel.frequency_end_mhz.setValue(84.0)
    panel.frequency_points.setValue(33)
    panel.power_sweep_enabled.setChecked(True)
    panel.power_start_gain.setValue(100)
    panel.power_end_gain.setValue(10000)
    panel.power_points.setValue(3)
    panel.power_scale.setCurrentIndex(panel.power_scale.findData("log"))
    experiment_database = tmp_path / "experiment.db"
    sparameter_database = tmp_path / "sparameter.db"
    window._experiment_panel.database_path.setText(str(experiment_database))
    panel.database_path.setText(str(sparameter_database))
    panel.output_filter_type.setCurrentText("lowpass")
    panel.readout_filter_type.setCurrentText("bandpass")
    calibration_database = tmp_path / "gain_pwr_calb.db"
    calibration_panel.database_path.setText(str(calibration_database))
    calibration_panel.scope_resource.setText("USB::MOCK")
    calibration_panel.input_board.setCurrentText("RF_In")
    calibration_panel.input_path_loss.setValue(3.5)
    settings = window._settings_to_dict()
    decoded = window._decode_settings(settings)
    assert decoded["s_parameter"]["frequency_start_mhz"] == 42.0
    assert decoded["s_parameter"]["frequency_points"] == 33
    assert decoded["s_parameter"]["power_sweep_enabled"] is True
    assert decoded["s_parameter"]["power_start_gain"] == 100
    assert decoded["s_parameter"]["power_end_gain"] == 10000
    assert decoded["s_parameter"]["power_points"] == 3
    assert decoded["s_parameter"]["power_scale"] == "log"
    assert decoded["s_parameter"]["database_path"] == str(
        sparameter_database
    )
    assert decoded["run_config"].database_path == str(experiment_database)
    assert decoded["s_parameter"]["output_filter_type"] == "lowpass"
    assert decoded["s_parameter"]["readout_filter_type"] == "bandpass"
    assert decoded["calibration"]["database_path"] == str(calibration_database)
    assert (
        decoded["calibration"]["output"]["oscilloscope"]["visa_resource"]
        == "USB::MOCK"
    )
    assert decoded["calibration"]["input"]["input_board_type"] == "RF_In"
    assert decoded["calibration"]["input"]["path_loss_db"] == 3.5

    legacy_settings = dict(settings)
    legacy_settings["version"] = 8
    legacy_settings.pop("calibration")
    legacy_decoded = window._decode_settings(legacy_settings)
    assert legacy_decoded["calibration"]["output"]["output_board_type"] == "RF_Out"
    assert legacy_decoded["calibration"]["input"]["input_board_type"] == "RF_In"

    run_arguments = window._sparameter_run_arguments()
    assert run_arguments["run_config"].database_path == str(
        sparameter_database
    )
    window._experiment_panel.database_path.setText(str(tmp_path / "changed.db"))
    assert window._sparameter_run_arguments()["run_config"].database_path == str(
        sparameter_database
    )
    window._experiment_panel.database_path.clear()
    assert window._sparameter_run_arguments()["run_config"].database_path == str(
        sparameter_database
    )

    window._sparameter_plot.set_result(_result())
    window._sparameter_plot.set_result(
        SParameterPowerSweepResult.from_sweeps(
            [100, 1000],
            [_result(), _result()],
        )
    )
    app.processEvents()
    window.close()


def test_gui_round_trips_board_types_calibration_database_and_dbm_power(tmp_path):
    app = _application()
    window = gui.MainWindow()
    panel = window._sparameter_panel
    calibration_path = tmp_path / "gain_pwr_calb.db"

    panel.power_calibration_enabled.setChecked(True)
    panel.calibration_database_path.setText(str(calibration_path))
    panel.output_board_type.setCurrentText("DC_Out")
    panel.input_board_type.setCurrentText("RF_In")
    panel.output_power_dbm.setValue(-23.5)
    panel.power_sweep_enabled.setChecked(True)
    panel.power_start_dbm.setValue(-40.0)
    panel.power_end_dbm.setValue(-10.0)
    panel.power_points.setValue(7)
    panel.power_scale.setCurrentIndex(panel.power_scale.findData("log"))

    settings = window._settings_to_dict()
    decoded = window._decode_settings(settings)["s_parameter"]

    assert decoded["power_calibration_enabled"] is True
    assert decoded["calibration_database_path"] == str(calibration_path)
    assert decoded["output_board_type"] == "DC_Out"
    assert decoded["input_board_type"] == "RF_In"
    assert decoded["output_power_dbm"] == -23.5
    assert decoded["power_start_dbm"] == -40.0
    assert decoded["power_end_dbm"] == -10.0
    assert panel.gain.isEnabled() is False
    assert panel.power_start_gain.isEnabled() is False
    assert panel.power_start_dbm.isEnabled() is True
    app.processEvents()
    window.close()
