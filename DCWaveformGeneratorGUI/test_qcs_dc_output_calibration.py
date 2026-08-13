"""Focused tests for M5301A-to-1-Mohm-scope DC output calibration."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qcs_dc_output_calibration import (
    QCS_M5301_DC_OUTPUT_CALIBRATION_SCHEMA,
    KeysightDcVoltageMeter,
    QcsDcOscilloscopeConfig,
    QcsM5301DcCalibrationConfig,
    QcsM5301DcOutputCalibration,
    StoredQcsM5301DcCalibrationRun,
    load_qcs_m5301_dc_output_calibration,
    run_qcs_m5301_dc_output_calibration,
    store_qcs_m5301_dc_output_calibration,
)


def _config(database_path: Path, **updates):
    values = {
        "database_path": str(database_path),
        "chassis": 1,
        "slot": 7,
        "channel": 2,
        "voltage_start_v": -1.0,
        "voltage_stop_v": 1.0,
        "voltage_points": 3,
        "nominal_full_scale_v": 5.0,
        "oscilloscope": QcsDcOscilloscopeConfig(
            visa_resource="USB::SCOPE",
            channel=2,
            average_count=1,
            settle_seconds=0.0,
            sample_interval_seconds=0.0,
        ),
    }
    values.update(updates)
    return QcsM5301DcCalibrationConfig(**values)


def test_config_requires_bipolar_points_and_one_megohm(tmp_path):
    with pytest.raises(ValueError, match="1 Mohm"):
        QcsDcOscilloscopeConfig(input_impedance_ohm=50.0)

    with pytest.raises(ValueError, match="bipolar"):
        _config(
            tmp_path / "cal.db",
            voltage_start_v=0.0,
            voltage_stop_v=1.0,
        )

    with pytest.raises(ValueError, match="explicit zero"):
        _config(
            tmp_path / "cal.db",
            voltage_start_v=-1.0,
            voltage_stop_v=1.0,
            voltage_points=4,
        )

    with pytest.raises(ValueError, match="symmetric about zero"):
        _config(
            tmp_path / "cal.db",
            voltage_start_v=-1.0,
            voltage_stop_v=2.0,
            voltage_points=4,
        )


def test_origin_constrained_fit_and_compensated_maximum():
    commanded = np.asarray([-5.0, -2.5, 0.0, 2.5, 5.0])
    # The 10 mV offset remains in the residual. It is not fitted/compensated.
    measured = 0.98 * commanded + 0.01
    calibration = QcsM5301DcOutputCalibration.fit(
        commanded,
        measured,
        chassis=1,
        slot=7,
        channel=1,
        scope_input_impedance_ohm=1.0e6,
        nominal_full_scale_v=5.0,
    )

    assert calibration.gain_a == pytest.approx(0.98)
    assert calibration.measured_zero_v == pytest.approx(0.01)
    assert calibration.rmse_v == pytest.approx(0.01)
    assert calibration.max_abs_residual_v == pytest.approx(0.01)
    assert calibration.corrected_maximum_abs_voltage_v == pytest.approx(4.9)
    assert calibration.command_voltage_for_target(4.8) == pytest.approx(
        4.8 / 0.98
    )
    assert calibration.relative_amplitude_for_target(2.45) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="reachable"):
        calibration.command_voltage_for_target(4.91)

    with pytest.raises(ValueError, match="must be positive"):
        QcsM5301DcOutputCalibration.fit(
            commanded,
            -measured,
            chassis=1,
            slot=7,
            channel=1,
        )


class _Channel:
    def __init__(self, name):
        self.name = name


class _Mapper:
    def __init__(self, name):
        self.channels = [_Channel(name)]


class _Envelope:
    pass


class _Waveform:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Program:
    def __init__(self, name=None):
        self.name = name
        self.waveforms = []
        self.shots = None

    def add_waveform(self, waveform, channel, **kwargs):
        self.waveforms.append((waveform, channel, kwargs))

    def n_shots(self, shots):
        self.shots = shots


class _FakeQcs:
    Program = _Program
    DCWaveform = _Waveform
    ConstantEnvelope = _Envelope


class _Executor:
    def __init__(self):
        self.programs = []

    def execute(self, program):
        self.programs.append(program)
        return None


class _Scope:
    input_impedance_ohm = 1.0e6
    idn = "KEYSIGHT,MOCK"

    def __init__(self, values, *, fail_at=None):
        self.values = list(values)
        self.index = 0
        self.fail_at = fail_at
        self.ensure_calls = 0

    def ensure_one_megohm(self):
        self.ensure_calls += 1
        return self.input_impedance_ohm

    def measure_voltage(self):
        if self.fail_at is not None and self.index == self.fail_at:
            raise RuntimeError("scope failed")
        value = self.values[self.index]
        self.index += 1
        return value


def test_runner_executes_relative_levels_and_always_resets(tmp_path):
    config = _config(tmp_path / "cal.db")
    executor = _Executor()
    scope = _Scope([-0.97, 0.0, 0.97])
    callback_values = []

    def callback(**values):
        callback_values.append(
            (
                values["commanded_voltage_v"],
                values["relative_amplitude"],
                values["is_reset"],
            )
        )

    def store(config, commands, measured, calibration, **_kwargs):
        return StoredQcsM5301DcCalibrationRun(
            run_id=19,
            guid="test",
            database_path=Path(config.database_path),
            row_count=len(commands),
            calibration=calibration,
            result={"measured": list(measured)},
        )

    stored = run_qcs_m5301_dc_output_calibration(
        calibration_config=config,
        mapper=_Mapper(config.virtual_channel_name),
        executor=executor,
        scope=scope,
        qcs_module=_FakeQcs,
        program_callback=callback,
        storage_callback=store,
    )

    assert stored.run_id == 19
    assert stored.calibration.gain_a == pytest.approx(0.97)
    assert len(executor.programs) == 4
    amplitudes = [
        program.waveforms[0][0].kwargs["amplitude"]
        for program in executor.programs
    ]
    assert amplitudes == pytest.approx([-0.2, 0.0, 0.2, 0.0])
    assert callback_values[-1] == (0.0, 0.0, True)
    assert scope.ensure_calls == 3


def test_runner_rejects_nonzero_mapped_physical_offset(tmp_path):
    config = _config(tmp_path / "cal.db")

    class MapperWithOffset(_Mapper):
        def get_physical_channels(self, _channel):
            return (
                types.SimpleNamespace(
                    settings=types.SimpleNamespace(
                        offset=types.SimpleNamespace(value=0.125)
                    )
                ),
            )

    executor = _Executor()
    with pytest.raises(ValueError, match="physical-channel offset.*zero"):
        run_qcs_m5301_dc_output_calibration(
            calibration_config=config,
            mapper=MapperWithOffset(config.virtual_channel_name),
            executor=executor,
            scope=_Scope([-1.0, 0.0, 1.0]),
            qcs_module=_FakeQcs,
            storage_callback=lambda *_args, **_kwargs: None,
        )

    # Validation occurs before any output program is submitted.
    assert executor.programs == []


def test_runner_resets_after_scope_failure(tmp_path):
    config = _config(tmp_path / "cal.db")
    executor = _Executor()
    scope = _Scope([-0.97, 0.0, 0.97], fail_at=1)

    with pytest.raises(RuntimeError, match="scope failed"):
        run_qcs_m5301_dc_output_calibration(
            calibration_config=config,
            mapper=_Mapper(config.virtual_channel_name),
            executor=executor,
            scope=scope,
            qcs_module=_FakeQcs,
            storage_callback=lambda *_args, **_kwargs: None,
        )

    # Two requested levels were programmed before failure, then zero reset.
    assert len(executor.programs) == 3
    assert executor.programs[-1].waveforms[0][0].kwargs["amplitude"] == 0.0


def test_runner_reports_both_calibration_and_reset_failures_as_unsafe(tmp_path):
    config = _config(tmp_path / "cal.db")
    scope = _Scope([-0.97, 0.0, 0.97], fail_at=1)

    class ResetFailingExecutor(_Executor):
        def execute(self, program):
            self.programs.append(program)
            if len(self.programs) == 3:
                raise RuntimeError("reset transport failed")
            return None

    executor = ResetFailingExecutor()
    with pytest.raises(RuntimeError, match="output state is unsafe") as caught:
        run_qcs_m5301_dc_output_calibration(
            calibration_config=config,
            mapper=_Mapper(config.virtual_channel_name),
            executor=executor,
            scope=scope,
            qcs_module=_FakeQcs,
            storage_callback=lambda *_args, **_kwargs: None,
        )

    assert "scope failed" in str(caught.value)
    assert "reset transport failed" in str(caught.value)
    assert str(caught.value.calibration_error) == "scope failed"
    assert str(caught.value.reset_error) == "reset transport failed"
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_keysight_scope_adapter_sets_and_verifies_one_megohm(monkeypatch):
    class Instrument:
        def __init__(self):
            self.writes = []
            self.timeout = None
            self.closed = False

        def write(self, command):
            self.writes.append(command)

        def query(self, command):
            if command == "*IDN?":
                return "KEYSIGHT,MOCK"
            if command.endswith(":IMPedance?"):
                return "ONEM"
            if command.startswith(":MEASure:VAVerage?"):
                return "1.25"
            raise AssertionError(command)

        def close(self):
            self.closed = True

    instrument = Instrument()

    class ResourceManager:
        def open_resource(self, resource):
            assert resource == "USB::SCOPE"
            return instrument

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "pyvisa",
        types.SimpleNamespace(ResourceManager=lambda: ResourceManager()),
    )
    config = QcsDcOscilloscopeConfig(
        visa_resource="USB::SCOPE",
        channel=3,
        average_count=2,
        settle_seconds=0.0,
        sample_interval_seconds=0.0,
    )
    with KeysightDcVoltageMeter(config) as meter:
        assert meter.measure_voltage() == pytest.approx(1.25)

    assert ":CHANnel3:IMPedance ONEMeg" in instrument.writes
    assert ":CHANnel3:COUPling DC" in instrument.writes
    assert instrument.closed is True


def test_qcodes_round_trip_is_strict_by_address_and_one_megohm(tmp_path):
    database_path = tmp_path / "m5301_calibration.db"
    config = _config(database_path, module_serial="MY123")
    commands = config.commanded_voltages_v
    measured = 0.96 * commands
    calibration = QcsM5301DcOutputCalibration.fit(
        commands,
        measured,
        database_path=database_path,
        chassis=config.chassis,
        slot=config.slot,
        channel=config.channel,
        module_serial=config.module_serial,
        scope_resource=config.oscilloscope.visa_resource,
        scope_channel=config.oscilloscope.channel,
        nominal_full_scale_v=config.nominal_full_scale_v,
    )
    stored = store_qcs_m5301_dc_output_calibration(
        config,
        commands,
        measured,
        calibration,
        scope_identity="KEYSIGHT,MOCK",
    )
    try:
        loaded = load_qcs_m5301_dc_output_calibration(
            database_path,
            chassis=1,
            slot=7,
            channel=2,
            module_serial="MY123",
        )
        assert loaded.run_id == stored.run_id
        assert loaded.gain_a == pytest.approx(0.96)
        assert loaded.scope_input_impedance_ohm == 1.0e6

        with pytest.raises(LookupError, match="compatible run"):
            load_qcs_m5301_dc_output_calibration(
                database_path,
                chassis=1,
                slot=7,
                channel=3,
            )
        with pytest.raises(ValueError, match=r"not \(1, 7, 3\)"):
            load_qcs_m5301_dc_output_calibration(
                database_path,
                chassis=1,
                slot=7,
                channel=3,
                run_id=stored.run_id,
            )
    finally:
        if stored.dataset is not None:
            stored.dataset.conn.close()


def test_loader_rejects_qick_dc_loopback_schema(tmp_path):
    path = tmp_path / "qick_loopback.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE experiments (exp_id INTEGER PRIMARY KEY, sample_name TEXT)"
        )
        connection.execute(
            "CREATE TABLE runs (run_id INTEGER PRIMARY KEY, exp_id INTEGER, "
            "result_table_name TEXT, DC_Voltage_Calibration_Config TEXT)"
        )
        connection.execute("INSERT INTO experiments VALUES (1, 'DC_In')")
        connection.execute(
            "CREATE TABLE qick_results (dc_voltage_mv REAL, mean_adc REAL)"
        )
        connection.execute("INSERT INTO qick_results VALUES (-100, -5)")
        connection.execute("INSERT INTO qick_results VALUES (100, 5)")
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?)",
            (
                1,
                1,
                "qick_results",
                json.dumps(
                    {"schema": "qstl-qick-dc-voltage-calibration-v1"}
                ),
            ),
        )

    with pytest.raises(ValueError, match="QICK DC_Out-to-DC_In loopback"):
        load_qcs_m5301_dc_output_calibration(
            path,
            chassis=1,
            slot=7,
            channel=2,
            run_id=1,
        )


def test_stored_metadata_declares_distinct_qcs_schema(tmp_path):
    database_path = tmp_path / "metadata.db"
    config = _config(database_path)
    commands = config.commanded_voltages_v
    measured = commands.copy()
    calibration = QcsM5301DcOutputCalibration.fit(
        commands,
        measured,
        chassis=1,
        slot=7,
        channel=2,
        nominal_full_scale_v=5.0,
    )
    stored = store_qcs_m5301_dc_output_calibration(
        config, commands, measured, calibration
    )
    try:
        with sqlite3.connect(database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (stored.run_id,)
            ).fetchone()
            metadata = json.loads(
                row["QCS_M5301_DC_Output_Calibration_Config"]
            )
            assert metadata["schema"] == QCS_M5301_DC_OUTPUT_CALIBRATION_SCHEMA
            assert metadata["instrument_model"] == "M5301A"
            assert metadata["scope"]["input_impedance_ohm"] == 1.0e6
            assert metadata["fit_intercept_fixed_v"] == 0.0
    finally:
        if stored.dataset is not None:
            stored.dataset.conn.close()
