"""Focused M5300A/M5200A RF power-calibration tests."""

from __future__ import annotations

import os
import sqlite3
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5 import QtWidgets

try:
    from . import qcs_rf_power_calibration as calibration_backend
    from .calibration_gui import (
        CalibrationPanel,
        default_calibration_settings,
    )
    from .qcs_rf_power_calibration import (
        M5200PowerReference,
        M5300PowerCalibrationConfig,
        NOMINAL_M5200_50OHM,
        QCS_M5200_VOLTAGE_50OHM,
        QCS_RF_CALIBRATION_POINT_TABLE,
        QCS_RF_CALIBRATION_RUN_TABLE,
        QCS_RF_CALIBRATION_MAX_TOTAL_INTEGRATION_DURATION_S,
        QCS_RF_POWER_CALIBRATION_SCHEMA,
        QcsPhysicalChannelIdentity,
        load_m5300_power_calibration,
        resolve_m5300_m5200_identities,
        run_m5300_power_calibration,
        store_m5300_power_calibration,
    )
except ImportError:
    import qcs_rf_power_calibration as calibration_backend
    from calibration_gui import (
        CalibrationPanel,
        default_calibration_settings,
    )
    from qcs_rf_power_calibration import (
        M5200PowerReference,
        M5300PowerCalibrationConfig,
        NOMINAL_M5200_50OHM,
        QCS_M5200_VOLTAGE_50OHM,
        QCS_RF_CALIBRATION_POINT_TABLE,
        QCS_RF_CALIBRATION_RUN_TABLE,
        QCS_RF_CALIBRATION_MAX_TOTAL_INTEGRATION_DURATION_S,
        QCS_RF_POWER_CALIBRATION_SCHEMA,
        QcsPhysicalChannelIdentity,
        load_m5300_power_calibration,
        resolve_m5300_m5200_identities,
        run_m5300_power_calibration,
        store_m5300_power_calibration,
    )


OUTPUT = QcsPhysicalChannelIdentity(1, 1, 4, 1, "M5300A")
INPUT = QcsPhysicalChannelIdentity(1, 1, 18, 1, "M5200A")
DIGEST = "a" * 64


def _reference():
    return M5200PowerReference.qcs_voltage_50ohm()


def _power_grid(frequencies, amplitudes):
    full_scale = np.interp(frequencies, [1.0e9, 2.0e9], [-10.0, -8.0])
    return full_scale[:, None] + 20.0 * np.log10(amplitudes)[None, :]


def _store(path, *, output=OUTPUT, input_channel=INPUT):
    frequencies = np.asarray([1.0e9, 1.5e9, 2.0e9])
    amplitudes = np.asarray([0.25, 0.5, 1.0])
    powers = _power_grid(frequencies, amplitudes)
    # A coherent M5200 IntegrationFilter returns peak voltage. Convert the
    # synthetic connector powers back to Vpk so the production automatic
    # voltage-to-50-ohm path is covered by every storage/fit test.
    magnitudes = 10.0 ** ((powers - 10.0) / 20.0)
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


def test_qcs_m5200_voltage_reference_is_automatic_peak_voltage_conversion():
    reference = M5200PowerReference.qcs_voltage_50ohm()

    assert reference.mode == QCS_M5200_VOLTAGE_50OHM
    # One volt peak into 50 ohms is 10 mW = 10 dBm.
    assert reference.output_power_dbm([1.0])[0] == pytest.approx(10.0)
    assert reference.output_power_dbm([0.123])[0] == pytest.approx(
        20.0 * np.log10(0.123) + 10.0,
    )


def test_legacy_nominal_m5200_conversion_remains_readable():
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

    assert calibration.calibration_quality == QCS_M5200_VOLTAGE_50OHM
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
    def __init__(self, *, lo_frequency_hz=1.2e9):
        self.output = SimpleNamespace(name="rf_probe")
        self.input = SimpleNamespace(name="digitizer")
        self.channels = (self.output, self.input)
        self._physical = {
            "rf_probe": SimpleNamespace(
                address=_Address(1, 1, 4, 1),
                instrument="InstrumentEnum.M5300AWG",
                settings=SimpleNamespace(
                    lo_frequency=SimpleNamespace(value=lo_frequency_hz)
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


def test_zero_hz_m5300_lo_is_valid_calibration_metadata(tmp_path):
    config = M5300PowerCalibrationConfig(
        database_path=str(tmp_path / "zero_lo.db"),
        mapper_path=str(tmp_path / "zero_lo.qcs"),
        rf_channel_name="rf_probe",
        acquisition_channel_name="digitizer",
        frequencies_hz=(10.0e6, 20.0e6),
        relative_amplitudes=(0.5, 1.0),
        input_reference=_reference(),
        expected_lo_frequency_hz=0.0,
    )
    assert config.expected_lo_frequency_hz == 0.0

    output, input_channel, lo_frequency_hz = (
        resolve_m5300_m5200_identities(
            _Mapper(lo_frequency_hz=0.0),
            "rf_probe",
            "digitizer",
        )
    )
    assert output == OUTPUT
    assert input_channel == INPUT
    assert lo_frequency_hz == 0.0

    stored = store_m5300_power_calibration(
        tmp_path / "zero_lo.db",
        frequencies_hz=[10.0e6, 20.0e6],
        relative_amplitudes=[0.5, 1.0],
        mean_i=[[0.01, 0.02], [0.015, 0.03]],
        mean_q=np.zeros((2, 2)),
        output_identity=OUTPUT,
        input_identity=INPUT,
        mapper_sha256=DIGEST,
        lo_frequency_hz=0.0,
        integration_duration_s=1.0e-6,
        repetitions=1,
        input_reference=_reference(),
    )
    loaded = load_m5300_power_calibration(
        stored.database_path,
        run_id=stored.run_id,
        expected_lo_frequency_hz=0.0,
    )
    assert loaded.lo_frequency_hz == 0.0

    with pytest.raises(ValueError, match="nonnegative"):
        M5300PowerCalibrationConfig(
            database_path="unused.db",
            mapper_path="unused.qcs",
            rf_channel_name="rf_probe",
            acquisition_channel_name="digitizer",
            frequencies_hz=(10.0e6, 20.0e6),
            relative_amplitudes=(0.5, 1.0),
            input_reference=_reference(),
            expected_lo_frequency_hz=-1.0,
        )


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
        return 10.0 ** ((powers - 10.0) / 20.0), np.zeros(powers.size)

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


def test_100ms_calibration_plan_uses_reusable_100us_passes():
    plans = calibration_backend._integration_pass_plan(0.1)

    assert len(plans) == 1_000
    assert {plan.segment_sample_counts for plan in plans} == {
        (32_000,) * 15
    }
    assert sum(plan.sample_count for plan in plans) == 480_000_000
    assert QCS_RF_CALIBRATION_MAX_TOTAL_INTEGRATION_DURATION_S == pytest.approx(
        0.1
    )

    with pytest.raises(ValueError, match="must not exceed 100 ms"):
        M5300PowerCalibrationConfig(
            database_path="unused.db",
            mapper_path="unused.qcs",
            rf_channel_name="rf_probe",
            acquisition_channel_name="digitizer",
            frequencies_hz=(1.0e9, 2.0e9),
            relative_amplitudes=(0.5, 1.0),
            input_reference=_reference(),
            integration_duration_s=0.100001,
        )


def test_square_qcs_calibration_result_uses_point_first_repeat_axis():
    point_first = np.asarray(
        [[1.0 + 10.0j, 3.0 + 30.0j], [20.0 + 2.0j, 40.0 + 4.0j]]
    )

    class Results:
        @staticmethod
        def get_iq(_channels, *, avg=False, acq_index=-1):
            assert avg is False
            assert acq_index == 0
            return point_first

    decoded = calibration_backend._point_repetition_iq_from_result(
        SimpleNamespace(results=Results()),
        object(),
        2,
        2,
        acquisition_index=0,
    )

    np.testing.assert_array_equal(decoded, point_first)


class _FakeScalar:
    def __init__(self, name, value, dtype):
        self.name = name
        self.value = value
        self.dtype = dtype


class _FakeArray(_FakeScalar):
    pass


class _FakeProgram:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name
        self.waveforms = []
        self.acquisitions = []
        self.repetitions = None
        self.sweep_values = None
        owner.programs.append(self)

    def add_waveform(self, waveform, channels, **options):
        self.waveforms.append((waveform, channels, options))

    def add_acquisition(self, integration_filter, channels, **options):
        self.acquisitions.append((integration_filter, channels, options))

    def n_shots(self, repetitions):
        self.repetitions = repetitions

    def sweep(self, values, variables):
        self.sweep_values = (values, variables)


class _FakePassResults:
    def __init__(self, pass_index, point_count, repetitions):
        self.pass_index = pass_index
        self.point_count = point_count
        self.repetitions = repetitions

    def get_iq(self, _channels, *, avg=False, acq_index=-1):
        assert avg is False
        if self.pass_index == 0:
            # The first 100 us pass has 15 equal-size filters. Its weighted
            # result is therefore the mean of 1..15 = 8.
            value = float(acq_index + 1)
        else:
            # The 16-sample tail must contribute only 16/480016 of the total.
            value = 100.0
        return np.full(
            (self.point_count, self.repetitions),
            value + 0.5j * value,
            dtype=np.complex128,
        )


class _FakeExecutor:
    def __init__(self, owner):
        self.owner = owner

    def execute(self, program):
        pass_index = len(self.owner.executed_programs)
        self.owner.executed_programs.append(program)
        values, _variables = program.sweep_values
        point_count = len(values[0].value)
        return SimpleNamespace(
            results=_FakePassResults(
                pass_index,
                point_count,
                program.repetitions,
            )
        )


class _FakeQcs:
    def __init__(self):
        self.programs = []
        self.backends = []
        self.executors = []
        self.executed_programs = []

    @staticmethod
    def Scalar(name, value, dtype):
        return _FakeScalar(name, value, dtype)

    @staticmethod
    def Array(name, value, dtype):
        return _FakeArray(name, np.asarray(value), dtype)

    @staticmethod
    def ConstantEnvelope():
        return SimpleNamespace(kind="constant")

    @staticmethod
    def RFWaveform(**kwargs):
        return SimpleNamespace(**kwargs)

    @staticmethod
    def IntegrationFilter(waveform):
        return SimpleNamespace(waveform=waveform)

    def Program(self, name):
        return _FakeProgram(self, name)

    def HclBackend(self, **kwargs):
        backend = SimpleNamespace(**kwargs)
        self.backends.append(backend)
        return backend

    def Executor(self, backend):
        assert backend is self.backends[0]
        executor = _FakeExecutor(self)
        self.executors.append(executor)
        return executor


def test_long_calibration_reuses_backend_executor_and_sample_weights_passes(
    tmp_path,
):
    qcs = _FakeQcs()
    requested_samples = 480_000 + 16
    config = M5300PowerCalibrationConfig(
        database_path=str(tmp_path / "segmented.db"),
        mapper_path=str(tmp_path / "injected_mapper.qcs"),
        rf_channel_name="rf_probe",
        acquisition_channel_name="digitizer",
        frequencies_hz=(1.0e9, 2.0e9),
        relative_amplitudes=(0.5, 1.0),
        input_reference=_reference(),
        integration_duration_s=(
            requested_samples / calibration_backend.M5200_SAMPLE_RATE_HZ
        ),
        repetitions=2,
        expected_lo_frequency_hz=1.2e9,
    )
    progress = []

    stored = run_m5300_power_calibration(
        config,
        mapper=_Mapper(),
        qcs_module=qcs,
        progress_callback=lambda value, message: progress.append(
            (value, message)
        ),
    )

    assert len(qcs.backends) == 1
    assert len(qcs.executors) == 1
    assert len(qcs.programs) == 2
    assert len(qcs.executed_programs) == 2
    assert len(qcs.programs[0].acquisitions) == 15
    assert len(qcs.programs[1].acquisitions) == 1
    assert all(
        acquisition[0].waveform.duration
        <= calibration_backend.M5200_MAX_INTEGRATION_SAMPLES
        / calibration_backend.M5200_SAMPLE_RATE_HZ
        for program in qcs.programs
        for acquisition in program.acquisitions
    )
    expected = (8.0 * 480_000 + 100.0 * 16) / requested_samples
    np.testing.assert_allclose(stored.calibration.mean_i, expected)
    np.testing.assert_allclose(stored.calibration.mean_q, 0.5 * expected)
    assert stored.calibration.integration_duration_s == pytest.approx(
        requested_samples / calibration_backend.M5200_SAMPLE_RATE_HZ
    )
    assert any("pass 1/2" in message for _value, message in progress)
    assert any("pass 2/2" in message for _value, message in progress)
    assert progress[-1][0] == 100


def test_equal_100us_calibration_passes_reuse_the_same_program(tmp_path):
    qcs = _FakeQcs()
    config = M5300PowerCalibrationConfig(
        database_path=str(tmp_path / "reused_program.db"),
        mapper_path=str(tmp_path / "injected_mapper.qcs"),
        rf_channel_name="rf_probe",
        acquisition_channel_name="digitizer",
        frequencies_hz=(1.0e9, 2.0e9),
        relative_amplitudes=(0.5, 1.0),
        input_reference=_reference(),
        integration_duration_s=200.0e-6,
        repetitions=2,
        expected_lo_frequency_hz=1.2e9,
    )

    stored = run_m5300_power_calibration(
        config,
        mapper=_Mapper(),
        qcs_module=qcs,
    )

    assert len(qcs.programs) == 1
    assert qcs.executed_programs == [qcs.programs[0], qcs.programs[0]]
    np.testing.assert_allclose(stored.calibration.mean_i, (8.0 + 100.0) / 2)
    np.testing.assert_allclose(
        stored.calibration.mean_q,
        (4.0 + 50.0) / 2,
    )


def test_qcs_rf_calibration_gui_exposes_100ms_total_average():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = CalibrationPanel()

    assert panel.qcs_rf_integration_us.maximum() == pytest.approx(100_000.0)
    labels = [
        item.text()
        for item in panel.findChildren(QtWidgets.QLabel)
        if "integrated I/Q" in item.text()
    ]
    assert "Total integrated I/Q averaging time:" in labels
    assert "separate QCS executions" in panel.qcs_rf_integration_us.toolTip()

    assert not hasattr(panel, "qcs_rf_reference_mode")
    assert not hasattr(panel, "qcs_rf_reference_slope")
    assert not hasattr(panel, "qcs_rf_reference_intercept_dbm")
    assert not hasattr(panel, "qcs_rf_nominal_volts_per_iq")
    assert not hasattr(panel, "qcs_rf_reference_description")
    assert not hasattr(panel, "qcs_rf_path_loss_db")
    assert not any(
        "M5200A 50 Ohm Power Reference" in group.title()
        for group in panel.findChildren(QtWidgets.QGroupBox)
    )

    saved_reference = panel.settings_dict()["qcs_rf_output"]
    assert "reference_mode" not in saved_reference
    assert "reference_slope" not in saved_reference
    assert "reference_intercept_dbm" not in saved_reference
    assert "nominal_volts_per_iq_unit" not in saved_reference
    assert "path_loss_db" not in saved_reference

    # Older settings remain loadable, but their manual conversion fields are
    # intentionally discarded. Only the useful cable/path correction remains.
    legacy_settings = dict(default_calibration_settings())
    legacy_settings["qcs_rf_output"] = {
        **legacy_settings["qcs_rf_output"],
        "reference_mode": "reference_calibrated",
        "reference_slope": 3.0,
        "reference_intercept_dbm": -17.0,
        "nominal_volts_per_iq_unit": 0.9,
        "acknowledge_nominal_scaling": True,
        "path_loss_db": 2.5,
    }
    panel.load_settings(legacy_settings)
    normalized_reference = panel.settings_dict()["qcs_rf_output"]
    assert "path_loss_db" not in normalized_reference

    panel.close()
    panel.deleteLater()
    app.processEvents()
