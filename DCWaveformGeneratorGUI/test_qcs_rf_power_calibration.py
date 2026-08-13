"""Focused M5300A/M5200A RF power-calibration tests."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

try:
    from .qcs_rf_power_calibration import (
        M5200PowerReference,
        M5300PowerCalibrationConfig,
        NOMINAL_M5200_50OHM,
        QCS_RF_CALIBRATION_POINT_TABLE,
        QCS_RF_CALIBRATION_RUN_TABLE,
        QCS_RF_POWER_CALIBRATION_SCHEMA,
        QcsPhysicalChannelIdentity,
        REFERENCE_CALIBRATED,
        load_m5300_power_calibration,
        run_m5300_power_calibration,
        store_m5300_power_calibration,
    )
except ImportError:
    from qcs_rf_power_calibration import (
        M5200PowerReference,
        M5300PowerCalibrationConfig,
        NOMINAL_M5200_50OHM,
        QCS_RF_CALIBRATION_POINT_TABLE,
        QCS_RF_CALIBRATION_RUN_TABLE,
        QCS_RF_POWER_CALIBRATION_SCHEMA,
        QcsPhysicalChannelIdentity,
        REFERENCE_CALIBRATED,
        load_m5300_power_calibration,
        run_m5300_power_calibration,
        store_m5300_power_calibration,
    )


OUTPUT = QcsPhysicalChannelIdentity(1, 1, 4, 1, "M5300A")
INPUT = QcsPhysicalChannelIdentity(1, 1, 18, 1, "M5200A")
DIGEST = "a" * 64


def _reference():
    # This synthetic reference maps magnitude 10**(P/20) directly to P dBm.
    return M5200PowerReference.calibrated(
        slope=1.0,
        intercept_dbm=0.0,
        source="traceable synthetic M5200 reference",
        uncertainty_db=0.2,
    )


def _power_grid(frequencies, amplitudes):
    full_scale = np.interp(frequencies, [1.0e9, 2.0e9], [-10.0, -8.0])
    return full_scale[:, None] + 20.0 * np.log10(amplitudes)[None, :]


def _store(path, *, output=OUTPUT, input_channel=INPUT):
    frequencies = np.asarray([1.0e9, 1.5e9, 2.0e9])
    amplitudes = np.asarray([0.25, 0.5, 1.0])
    powers = _power_grid(frequencies, amplitudes)
    magnitudes = 10.0 ** (powers / 20.0)
    return store_m5300_power_calibration(
        path,
        frequencies_hz=frequencies,
        relative_amplitudes=amplitudes,
        mean_i=magnitudes,
        mean_q=np.zeros_like(magnitudes),
        output_identity=output,
        input_identity=input_channel,
        mapper_sha256=DIGEST,
        lo_frequency_hz=1.2e9,
        integration_duration_s=1.0e-6,
        repetitions=20,
        input_reference=_reference(),
        notes="unit-test calibration",
    )


def test_nominal_m5200_conversion_requires_explicit_acknowledgement():
    with pytest.raises(ValueError, match="explicit"):
        M5200PowerReference.nominal_50ohm(
            volts_per_iq_unit=1.0,
        )

    reference = M5200PowerReference.nominal_50ohm(
        volts_per_iq_unit=2.0,
        path_loss_db=3.0,
        acknowledge_nominal_scaling=True,
        source="nominal HCL scaling",
    )
    # |IQ|=0.5 and 2 V/unit gives 1 V peak.  Into 50 ohm that is
    # 10 mW = 10 dBm, before adding 3 dB path loss.
    assert reference.mode == NOMINAL_M5200_50OHM
    assert reference.output_power_dbm([0.5])[0] == pytest.approx(13.0)


def test_store_load_fit_and_dbm_to_relative_amplitude(tmp_path):
    stored = _store(tmp_path / "qcs_power.db")
    assert stored.point_count == 9
    calibration = load_m5300_power_calibration(
        stored.database_path,
        run_id=stored.run_id,
        expected_output=OUTPUT,
        expected_input=INPUT,
        expected_mapper_sha256=DIGEST,
        expected_lo_frequency_hz=1.2e9,
        required_frequencies_hz=[1.1e9, 1.9e9],
    )

    assert calibration.calibration_quality == REFERENCE_CALIBRATED
    assert calibration.full_scale_power_dbm([1.0e9, 1.5e9, 2.0e9]) == (
        pytest.approx([-10.0, -9.0, -8.0])
    )
    target = np.asarray([-15.0, -14.0, -13.0])
    expected = np.full(3, 10.0 ** (-5.0 / 20.0))
    assert calibration.relative_amplitudes_for_power(
        [1.0e9, 1.5e9, 2.0e9], target
    ) == pytest.approx(expected)
    assert calibration.provenance["schema"] == QCS_RF_POWER_CALIBRATION_SCHEMA
    assert calibration.provenance["output"]["slot"] == 4
    assert calibration.notes == "unit-test calibration"

    with sqlite3.connect(stored.database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert QCS_RF_CALIBRATION_RUN_TABLE in tables
    assert QCS_RF_CALIBRATION_POINT_TABLE in tables


def test_mapping_rejects_frequency_and_power_extrapolation(tmp_path):
    calibration = _store(tmp_path / "qcs_power.db").calibration
    with pytest.raises(ValueError, match="frequency extrapolation"):
        calibration.relative_amplitudes_for_power([0.9e9], [-15.0])
    with pytest.raises(ValueError, match="power extrapolation"):
        calibration.relative_amplitudes_for_power([1.0e9], [-40.0])
    with pytest.raises(ValueError, match="full-scale"):
        calibration.relative_amplitudes_for_power(
            [1.0e9],
            [-5.0],
            allow_power_extrapolation=True,
        )
    assert calibration.relative_amplitudes_for_power(
        [1.0e9],
        [-25.0],
        allow_power_extrapolation=True,
    )[0] == pytest.approx(10.0 ** (-15.0 / 20.0))


def test_loader_rejects_qick_database_and_wrong_hardware(tmp_path):
    legacy = tmp_path / "legacy_gain_power.db"
    with sqlite3.connect(legacy) as connection:
        connection.execute("CREATE TABLE runs (run_id INTEGER PRIMARY KEY)")
    with pytest.raises(ValueError, match="legacy QICK"):
        load_m5300_power_calibration(legacy)

    database = tmp_path / "qcs_power.db"
    _store(database)
    wrong_output = QcsPhysicalChannelIdentity(1, 1, 4, 2, "M5300A")
    with pytest.raises(ValueError, match="selected connector"):
        load_m5300_power_calibration(
            database,
            run_id=1,
            expected_output=wrong_output,
        )
    with pytest.raises(ValueError, match="LO frequency"):
        load_m5300_power_calibration(
            database,
            run_id=1,
            expected_lo_frequency_hz=1.3e9,
        )
    with pytest.raises(ValueError, match="active QCS mapper"):
        load_m5300_power_calibration(
            database,
            run_id=1,
            expected_mapper_sha256="b" * 64,
        )


def test_auto_loader_skips_newer_run_for_other_connector(tmp_path):
    database = tmp_path / "qcs_power.db"
    first = _store(database)
    other_output = QcsPhysicalChannelIdentity(1, 1, 4, 2, "M5300A")
    second = _store(database, output=other_output)
    assert second.run_id > first.run_id
    selected = load_m5300_power_calibration(
        database,
        expected_output=OUTPUT,
    )
    assert selected.run_id == first.run_id


class _Address:
    def __init__(self, host_controller, chassis, slot, channel):
        self.host_controller = host_controller
        self.chassis = chassis
        self.slot = slot
        self.channel = channel


class _Mapper:
    def __init__(self):
        self.output = SimpleNamespace(name="rf_probe")
        self.input = SimpleNamespace(name="digitizer")
        self.channels = (self.output, self.input)
        self._physical = {
            "rf_probe": SimpleNamespace(
                address=_Address(1, 1, 4, 1),
                instrument="InstrumentEnum.M5300AWG",
                settings=SimpleNamespace(
                    lo_frequency=SimpleNamespace(value=1.2e9)
                ),
            ),
            "digitizer": SimpleNamespace(
                address=_Address(1, 1, 18, 1),
                instrument="InstrumentEnum.M5200Digitizer",
                settings=SimpleNamespace(),
            ),
        }

    def get_physical_channels(self, channel):
        return (self._physical[channel.name],)


def test_injected_runner_uses_flat_cartesian_grid_and_stores_result(tmp_path):
    calls = []

    def measurement_runner(**kwargs):
        calls.append(kwargs)
        frequencies = kwargs["frequencies_hz"]
        amplitudes = kwargs["relative_amplitudes"]
        full_scale = np.interp(
            frequencies,
            [1.0e9, 2.0e9],
            [-10.0, -8.0],
        )
        powers = full_scale + 20.0 * np.log10(amplitudes)
        return 10.0 ** (powers / 20.0), np.zeros(powers.size)

    config = M5300PowerCalibrationConfig(
        database_path=str(tmp_path / "injected.db"),
        mapper_path=str(tmp_path / "injected_mapper.qcs"),
        rf_channel_name="rf_probe",
        acquisition_channel_name="digitizer",
        frequencies_hz=(1.0e9, 2.0e9),
        relative_amplitudes=(0.5, 1.0),
        input_reference=_reference(),
        repetitions=7,
        expected_lo_frequency_hz=1.2e9,
    )
    progress = []
    stored = run_m5300_power_calibration(
        config,
        mapper=_Mapper(),
        measurement_runner=measurement_runner,
        progress_callback=lambda value, message: progress.append(
            (value, message)
        ),
    )
    assert len(calls) == 1
    assert calls[0]["frequencies_hz"] == pytest.approx(
        [1.0e9, 1.0e9, 2.0e9, 2.0e9]
    )
    assert calls[0]["relative_amplitudes"] == pytest.approx(
        [0.5, 1.0, 0.5, 1.0]
    )
    assert stored.point_count == 4
    assert stored.calibration.output_identity == OUTPUT
    assert stored.calibration.input_identity == INPUT
    assert progress[-1][0] == 100


def test_models_are_normalized_but_wrong_modules_are_rejected():
    assert QcsPhysicalChannelIdentity(
        1, 1, 4, 1, "InstrumentEnum.M5300AWG"
    ).module_model == "M5300A"
    with pytest.raises(ValueError, match="output_identity"):
        store_m5300_power_calibration(
            "unused.db",
            frequencies_hz=[1.0e9, 2.0e9],
            relative_amplitudes=[0.5, 1.0],
            mean_i=np.ones((2, 2)),
            mean_q=np.zeros((2, 2)),
            output_identity=QcsPhysicalChannelIdentity(
                1, 1, 7, 1, "M5301A"
            ),
            input_identity=INPUT,
            mapper_sha256=DIGEST,
            lo_frequency_hz=1.2e9,
            integration_duration_s=1.0e-6,
            repetitions=1,
            input_reference=_reference(),
        )
