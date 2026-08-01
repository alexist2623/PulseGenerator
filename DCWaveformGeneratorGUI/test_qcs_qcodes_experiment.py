"""Tests for the Keysight QCS Experiment backend."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dc_waveform_core import PulseSequence, generate_qcs_program_code
from qick_fine_tune_sweep import FineTuneSequence
import qcs_qcodes_experiment as backend
from qcs_qcodes_experiment import (
    QcsAcquisitionConfig,
    QcsConnectionConfig,
    QcsRfPulseConfig,
    QcsUnsupportedFeatureError,
    compile_qcs_point,
    execute_qcs_sequence,
    normalize_qcs_hardware_sweep_iq,
    normalize_qcs_iq,
)
from qick_qcodes_experiment import QcodesRunConfig


class _Channel:
    def __init__(self, name):
        self.name = name


class _Mapper:
    def __init__(self, *names):
        self.channels = [_Channel(name) for name in names]


class _Envelope:
    def __init__(self, *args):
        self.args = args


class _Waveform:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _Expression:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return _Expression(("add", self, other))

    __radd__ = __add__

    def __mul__(self, other):
        return _Expression(("mul", self, other))

    __rmul__ = __mul__


class _Scalar(_Expression):
    def __init__(self, name, value=None, dtype=None):
        super().__init__(value)
        self.name = name
        self.dtype = dtype


class _Array:
    def __init__(self, name, value=None, dtype=None):
        self.name = name
        self.value = np.asarray(value)
        self.dtype = dtype


class _Program:
    def __init__(self, name=None):
        self.name = name
        self.waveforms = []
        self.acquisitions = []
        self.sweeps = []
        self.shots = None

    def add_waveform(self, pulse, channels, **kwargs):
        self.waveforms.append((pulse, channels, kwargs))

    def add_acquisition(self, **kwargs):
        self.acquisitions.append(kwargs)

    def n_shots(self, value):
        self.shots = value
        return self

    def sweep(self, values, target):
        self.sweeps.append((values, target))
        return self


class _FakeQcs:
    Program = _Program
    DCWaveform = _Waveform
    RFWaveform = _Waveform
    ConstantEnvelope = _Envelope
    GaussianEnvelope = _Envelope
    ArbitraryEnvelope = _Envelope
    Scalar = _Scalar
    Array = _Array


def _sequence():
    return FineTuneSequence(("gate",)).add_set("read", [0.5], 300)


def _connection(**changes):
    values = {
        "mapper_path": "unused.json",
        "dc_channel_names": ("dc_gate",),
        "acquisition_channel_name": "digitizer",
    }
    values.update(changes)
    return QcsConnectionConfig(**values)


def _acquisition(**changes):
    values = {
        "at_segment": "read",
        "duration_s": 100e-9,
        "sample_rate_hz": backend.QCS_M5200_SAMPLE_RATE_HZ,
        "frequency_hz": 50e6,
    }
    values.update(changes)
    return QcsAcquisitionConfig(**values)


def _stability_sequence():
    return (
        FineTuneSequence(("x_gate", "y_gate"))
        .add_set("set_0", [0.0, 0.0], 30_000)
        .add_amplitude_sweep("set_0", "x_gate", -0.25, 0.25, 3)
        .add_amplitude_sweep("set_0", "y_gate", -0.5, 0.5, 2)
        .set_cross_capacitance([[1.0, 0.1], [-0.2, 1.0]])
    )


def test_mapper_digest_is_rechecked_immediately_before_load(tmp_path):
    mapper_path = tmp_path / "mapper.qcs"
    mapper_path.write_bytes(b"expected mapper")
    expected_digest = hashlib.sha256(b"expected mapper").hexdigest()

    class Loader:
        ChannelMapper = _Mapper

        @staticmethod
        def load(_path):
            return _Mapper("dc_gate", "digitizer")

    connection = _connection(
        mapper_path=str(mapper_path),
        mapper_sha256=expected_digest,
    )
    assert isinstance(
        backend.load_qcs_channel_mapper(connection, qcs_module=Loader),
        _Mapper,
    )

    mapper_path.write_bytes(b"replaced mapper")
    with pytest.raises(ValueError, match="does not match"):
        backend.load_qcs_channel_mapper(connection, qcs_module=Loader)


def test_connection_config_is_qcodes_metadata_serializable():
    payload = asdict(
        _connection(rf_channel_names={3: "readout_rf"})
    )
    assert payload["dc_channel_names"] == ("dc_gate",)
    assert payload["dc_full_scale_v"] == 2.5
    assert payload["rf_channel_names"] == {3: "readout_rf"}
    assert payload["init_time_s"] == pytest.approx(100e-6)


def test_connection_config_prevents_hcl_init_time_nanosecond_truncation():
    # This is the exact float produced by the GUI's 100 us -> seconds path.
    requested = 100.0 * 1e-6
    assert int(requested * 1e9) == 99_999

    connection = _connection(init_time_s=requested)

    assert connection.init_time_s == pytest.approx(100e-6)
    assert int(connection.init_time_s * 1e9) == 100_000
    assert int(connection.init_time_s * 1e9) % 10 == 0

    rounded_up = _connection(init_time_s=0.125e-6)
    assert rounded_up.init_time_s == pytest.approx(0.130e-6)
    assert int(rounded_up.init_time_s * 1e9) == 130


def test_stability_executor_resets_phase_for_hardware_iq():
    captured = {}

    class HclBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class Executor:
        def __init__(self, hcl_backend):
            self.backend = hcl_backend

    Qcs = type(
        "Qcs",
        (),
        {"HclBackend": HclBackend, "Executor": Executor},
    )

    mapper = object()
    executor = backend.build_qcs_executor(
        _connection(),
        mapper,
        qcs_module=Qcs,
    )

    assert isinstance(executor, Executor)
    assert captured["channel_mapper"] is mapper
    assert captured["hw_demod"] is True
    assert captured["reset_phase_every_shot"] is True
    assert captured["init_time"] == pytest.approx(100e-6)


def test_compile_converts_gui_millivolts_to_qcs_relative_amplitude():
    compiled = compile_qcs_point(
        _sequence(),
        0,
        connection_config=_connection(),
        mapper=_Mapper("dc_gate", "digitizer"),
        repetitions_per_sweep=2,
        acquisition=_acquisition(),
        qcs_module=_FakeQcs,
    )

    dc_waveform = compiled.program.waveforms[0][0]
    # 0.5 of the GUI's 800 mV scale is 400 mV, or 0.16 of 2.5 V.
    assert dc_waveform.kwargs["amplitude"] == pytest.approx(0.16)
    assert compiled.program.waveforms[0][2]["new_layer"] is True
    assert compiled.program.shots == 2
    assert compiled.program.acquisitions[0]["pre_delay"] == 0.0
    assert compiled.program.acquisitions[0]["new_layer"] is False
    assert isinstance(
        compiled.program.acquisitions[0]["integration_filter"],
        _Waveform,
    )


def test_compile_raw_trace_uses_duration_instead_of_integration_filter():
    acquisition = _acquisition()
    compiled = compile_qcs_point(
        _sequence(),
        0,
        connection_config=_connection(hw_demod=False),
        mapper=_Mapper("dc_gate", "digitizer"),
        repetitions_per_sweep=1,
        acquisition=acquisition,
        qcs_module=_FakeQcs,
    )

    request = compiled.program.acquisitions[0]
    assert request["integration_filter"] == pytest.approx(
        acquisition.duration_s
    )
    assert request["new_layer"] is False


def test_compile_rejects_dc_voltage_above_configured_qcs_range():
    with pytest.raises(ValueError, match=r"reaches 0\.4 V.*\+/-0\.1 V"):
        compile_qcs_point(
            _sequence(),
            0,
            connection_config=_connection(dc_full_scale_v=0.1),
            mapper=_Mapper("dc_gate", "digitizer"),
            repetitions_per_sweep=1,
            acquisition=_acquisition(),
            qcs_module=_FakeQcs,
        )


def test_execute_software_points_normalizes_complex_iq_in_c_order():
    sequence = _sequence().add_amplitude_sweep(
        "read", "gate", 0.25, 0.5, 2
    )
    returned = [
        np.asarray([1 + 2j, 3 + 4j]),
        np.asarray([5 + 6j, 7 + 8j]),
    ]

    class Executor:
        def execute(self, _program):
            return returned.pop(0)

    result = execute_qcs_sequence(
        connection_config=_connection(),
        sequence=sequence,
        repetitions_per_sweep=2,
        acquisition=_acquisition(),
        qcs_module=_FakeQcs,
        mapper=_Mapper("dc_gate", "digitizer"),
        executor=Executor(),
    )

    assert result.ddr_result.iq.shape == (2, 2, 1, 2)
    np.testing.assert_allclose(
        result.ddr_result.iq[:, :, 0],
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
    )
    np.testing.assert_allclose(
        result.ddr_result.sweep_points, [0.25, 0.5]
    )


def test_qcs_stability_hardware_sweep_executes_one_program_in_c_order():
    sequence = _stability_sequence()
    connection = _connection(
        dc_channel_names=("dc_x", "dc_y"),
        dc_full_scale_v=1.0,
    )
    semantic_raw = np.empty((4, 3, 2), dtype=complex)
    for repetition in range(4):
        for x_index in range(3):
            for y_index in range(2):
                semantic_raw[repetition, x_index, y_index] = (
                    100 * x_index + 10 * y_index + repetition
                ) + 1j * repetition
    raw = semantic_raw.reshape(4, 6)

    calls = {"execute": 0}

    class Executor:
        def execute(self, program):
            calls["execute"] += 1
            assert program.shots == 4
            assert len(program.sweeps) == 1
            arrays, variables = program.sweeps[0]
            assert [variable.name for variable in variables] == [
                "stability_dc_0_amplitude",
                "stability_dc_1_amplitude",
            ]
            np.testing.assert_allclose(
                arrays[0].value,
                [-0.24, -0.16, -0.04, 0.04, 0.16, 0.24],
            )
            np.testing.assert_allclose(
                arrays[1].value,
                [-0.36, 0.44, -0.4, 0.4, -0.44, 0.36],
            )
            dc_amplitudes = [
                waveform[0].kwargs["amplitude"]
                for waveform in program.waveforms[:2]
            ]
            assert all(
                isinstance(amplitude, _Scalar)
                for amplitude in dc_amplitudes
            )
            return raw

    result = backend.execute_qcs_stability_hardware_sweep(
        connection_config=connection,
        sequence=sequence,
        repetitions_per_point=4,
        fabric_mhz=300.0,
        source_full_scale_mv=800.0,
        acquisition=_acquisition(
            at_segment="set_0",
            duration_s=20e-6,
            pre_delay_s=10e-6,
                sample_count=32,
        ),
        qcs_module=_FakeQcs,
        mapper=_Mapper("dc_x", "dc_y", "digitizer"),
        executor=Executor(),
    )

    assert calls["execute"] == 1
    assert len(result.programs) == 1
    assert len(result.raw_results) == 1
    assert result.program_summary["hardware_sweep"] is True
    assert result.program_summary["hardware_sweep_dimensions"] == 1
    assert result.program_summary["hardware_sweep_shape"] == [6]
    assert result.program_summary["stability_grid_shape"] == [3, 2]
    assert result.program_summary["reset_phase_every_shot"] is True
    assert result.program_summary["acquisition_result_type"] == "integrated_iq"
    assert (
        result.rf_settings["readout_details"]["reset_phase_every_shot"]
        is True
    )
    assert (
        result.rf_settings["readout_details"]["acquisition_result_type"]
        == "integrated_iq"
    )
    assert result.program_summary["program_count"] == 1
    assert result.ddr_result.iq.shape == (6, 4, 1, 2)
    np.testing.assert_array_equal(
        result.ddr_result.iq[:, 0, 0, 0],
        [0, 10, 100, 110, 200, 210],
    )
    np.testing.assert_array_equal(
        result.ddr_result.iq[0, :, 0, 1],
        [0, 1, 2, 3],
    )
    assert result.ddr_result.sweep_shape == (3, 2)
    np.testing.assert_allclose(
        result.ddr_result.sweep_points,
        [
            [-0.25, -0.5],
            [-0.25, 0.5],
            [0.0, -0.5],
            [0.0, 0.5],
            [0.25, -0.5],
            [0.25, 0.5],
        ],
    )


def test_qcs_stability_hardware_sweep_requires_hardware_demodulation():
    with pytest.raises(
        QcsUnsupportedFeatureError,
        match="hardware demodulation",
    ):
        backend.execute_qcs_stability_hardware_sweep(
            connection_config=_connection(
                dc_channel_names=("dc_x", "dc_y"),
                hw_demod=False,
            ),
            sequence=_stability_sequence(),
            repetitions_per_point=1,
            acquisition=_acquisition(at_segment="set_0"),
            qcs_module=_FakeQcs,
            mapper=_Mapper("dc_x", "dc_y", "digitizer"),
            executor=object(),
        )


def test_qcs_stability_flat_result_uses_native_shot_x_y_order():
    native = np.arange(12).reshape(2, 3, 2).astype(complex)
    normalized = normalize_qcs_hardware_sweep_iq(
        native.reshape(-1),
        repetitions_per_point=2,
        sweep_shape=(3, 2),
    )

    assert normalized.shape == (6, 2, 1, 2)
    np.testing.assert_array_equal(
        normalized[:, :, 0, 0],
        np.moveaxis(native, 0, -1).reshape(6, 2),
    )

    flattened = normalize_qcs_hardware_sweep_iq(
        native.reshape(2, 6),
        repetitions_per_point=2,
        sweep_shape=(3, 2),
    )
    flattened_shot_last = normalize_qcs_hardware_sweep_iq(
        native.reshape(2, 6).T,
        repetitions_per_point=2,
        sweep_shape=(3, 2),
    )
    np.testing.assert_array_equal(flattened, normalized)
    np.testing.assert_array_equal(flattened_shot_last, normalized)


def test_qcs_stability_flattened_hardware_array_budget():
    sequence = (
        FineTuneSequence(("x_gate", "y_gate"))
        .add_set("set_0", [0.0, 0.0], 30_000)
        .add_amplitude_sweep("set_0", "x_gate", -0.25, 0.25, 983)
        .add_amplitude_sweep("set_0", "y_gate", -0.5, 0.5, 25)
        .set_cross_capacitance(np.eye(2))
    )
    connection = _connection(
        dc_channel_names=("dc_x", "dc_y"),
        dc_full_scale_v=1.0,
    )
    compiled = backend.compile_qcs_stability_hardware_sweep(
        sequence,
        connection_config=connection,
        mapper=_Mapper("dc_x", "dc_y", "digitizer"),
        repetitions_per_point=1,
        acquisition=_acquisition(at_segment="set_0"),
        qcs_module=_FakeQcs,
    )

    assert compiled.sweep_shape == (983, 25)
    arrays, variables = compiled.program.sweeps[0]
    assert len(arrays) == len(variables) == 2
    assert all(array.value.size == 24_575 for array in arrays)

    oversized = (
        FineTuneSequence(("x_gate", "y_gate"))
        .add_set("set_0", [0.0, 0.0], 30_000)
        .add_amplitude_sweep("set_0", "x_gate", -0.25, 0.25, 1_024)
        .add_amplitude_sweep("set_0", "y_gate", -0.5, 0.5, 24)
        .set_cross_capacitance(np.eye(2))
    )
    with pytest.raises(
        QcsUnsupportedFeatureError,
        match="24,575",
    ):
        backend.compile_qcs_stability_hardware_sweep(
            oversized,
            connection_config=connection,
            mapper=_Mapper("dc_x", "dc_y", "digitizer"),
            repetitions_per_point=1,
            acquisition=_acquisition(at_segment="set_0"),
            qcs_module=_FakeQcs,
        )


@pytest.mark.parametrize(
    ("x_count", "y_count", "repetitions", "message"),
    (
        (257, 96, 1, "Cartesian points"),
        (200, 100, 101, "hardware-demodulated IQ values"),
    ),
)
def test_qcs_stability_rejects_unsafe_scan_budget(
    x_count,
    y_count,
    repetitions,
    message,
):
    sequence = (
        FineTuneSequence(("x_gate", "y_gate"))
        .add_set("set_0", [0.0, 0.0], 30_000)
        .add_amplitude_sweep(
            "set_0", "x_gate", -0.25, 0.25, x_count
        )
        .add_amplitude_sweep(
            "set_0", "y_gate", -0.5, 0.5, y_count
        )
        .set_cross_capacitance(np.eye(2))
    )
    with pytest.raises(QcsUnsupportedFeatureError, match=message):
        backend.compile_qcs_stability_hardware_sweep(
            sequence,
            connection_config=_connection(
                dc_channel_names=("dc_x", "dc_y"),
                dc_full_scale_v=1.0,
            ),
            mapper=_Mapper("dc_x", "dc_y", "digitizer"),
            repetitions_per_point=repetitions,
            acquisition=_acquisition(at_segment="set_0"),
            qcs_module=_FakeQcs,
        )


def test_qcs_stability_preflights_dc_instrument_and_phase():
    class Physical:
        def __init__(self, instrument):
            self.instrument = instrument

    class PhysicalMapper(_Mapper):
        def __init__(self, *, dc_instrument, dc_absolute_phase):
            super().__init__("dc_x", "dc_y", "digitizer")
            for channel in self.channels:
                channel.labels = (0,)
                channel.absolute_phase = (
                    dc_absolute_phase
                    if channel.name.startswith("dc_")
                    else True
                )
            self._physical = {
                "dc_x": Physical(dc_instrument),
                "dc_y": Physical("M5301AWG"),
                "digitizer": Physical("M5200Digitizer"),
            }

        def get_physical_channels(self, channel):
            return (self._physical[channel.name],)

    connection = _connection(
        dc_channel_names=("dc_x", "dc_y"),
        dc_full_scale_v=1.0,
    )
    with pytest.raises(ValueError, match="must map to M5301AWG"):
        backend.compile_qcs_stability_hardware_sweep(
            _stability_sequence(),
            connection_config=connection,
            mapper=PhysicalMapper(
                dc_instrument="M5300AWG",
                dc_absolute_phase=False,
            ),
            repetitions_per_point=1,
            acquisition=_acquisition(at_segment="set_0"),
            qcs_module=_FakeQcs,
        )
    with pytest.raises(ValueError, match="absolute_phase=False"):
        backend.compile_qcs_stability_hardware_sweep(
            _stability_sequence(),
            connection_config=connection,
            mapper=PhysicalMapper(
                dc_instrument="M5301AWG",
                dc_absolute_phase=True,
            ),
            repetitions_per_point=1,
            acquisition=_acquisition(at_segment="set_0"),
            qcs_module=_FakeQcs,
        )


def test_execute_raw_trace_uses_reported_hardware_rate_and_shot_first_axis():
    trace = np.asarray(
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    )

    class Results:
        def get_trace(self, channels, avg=False):
            assert avg is False
            return {channels: trace}

    class ExecutedProgram:
        results = Results()

        def get_sample_rates(self, channels):
            return {channels: 4.8e9}

    class Executor:
        def execute(self, _program):
            return ExecutedProgram()

    result = execute_qcs_sequence(
        connection_config=_connection(hw_demod=False),
        sequence=_sequence(),
        repetitions_per_sweep=2,
        acquisition=_acquisition(),
        qcs_module=_FakeQcs,
        mapper=_Mapper("dc_gate", "digitizer"),
        executor=Executor(),
    )

    assert result.ddr_result.sample_rate_hz == pytest.approx(4.8e9)
    assert result.ddr_result.iq.shape == (1, 2, 3, 2)
    np.testing.assert_array_equal(
        result.ddr_result.iq[0, ..., 0], trace
    )


def test_hardware_demod_uses_mapped_m5200_rate_and_sample_blocks():
    class Physical:
        sample_rate = 4.8e9

    class Mapper(_Mapper):
        @staticmethod
        def get_physical_channels(_channel):
            return (Physical(),)

    mapper = Mapper("dc_gate", "digitizer")
    compiled = compile_qcs_point(
        _sequence(),
        0,
        connection_config=_connection(hw_demod=True),
        mapper=mapper,
        repetitions_per_sweep=1,
        acquisition=_acquisition(
            duration_s=64e-6,
            sample_rate_hz=1e6,
            sample_count=64,
        ),
        qcs_module=_FakeQcs,
    )

    assert compiled.acquisition_sample_rate_hz == pytest.approx(4.8e9)
    assert compiled.acquisition_duration_s == pytest.approx(64 / 4.8e9)
    integration_filter = compiled.program.acquisitions[0][
        "integration_filter"
    ]
    assert integration_filter.kwargs["duration"] == pytest.approx(
        64 / 4.8e9
    )

    with pytest.raises(ValueError, match="multiple of 16 samples"):
        compile_qcs_point(
            _sequence(),
            0,
            connection_config=_connection(hw_demod=True),
            mapper=mapper,
            repetitions_per_sweep=1,
            acquisition=_acquisition(sample_count=65),
            qcs_module=_FakeQcs,
        )


def test_normalize_accepts_final_iq_axis_and_rejects_bad_count():
    values = np.arange(24).reshape(2, 6, 2)
    normalized = normalize_qcs_iq(
        values, repetitions_per_sweep=4
    )
    assert normalized.shape == (4, 3, 2)
    raw_trace = normalize_qcs_iq(
        np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        repetitions_per_sweep=2,
        real_is_i_trace=True,
    )
    assert raw_trace.shape == (2, 3, 2)
    np.testing.assert_allclose(
        raw_trace[..., 0],
        [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
    )
    np.testing.assert_array_equal(raw_trace[..., 1], 0.0)
    qcs_iq = normalize_qcs_iq(
        np.asarray(
            [
                [1.0 + 10.0j, 2.0 + 20.0j],
                [3.0 + 30.0j, 4.0 + 40.0j],
            ]
        ),
        repetitions_per_sweep=2,
    )
    np.testing.assert_allclose(
        qcs_iq[..., 0],
        [[1.0, 2.0], [3.0, 4.0]],
    )
    np.testing.assert_allclose(
        qcs_iq[..., 1],
        [[10.0, 20.0], [30.0, 40.0]],
    )
    square_raw = normalize_qcs_iq(
        np.arange(9.0).reshape(3, 3),
        repetitions_per_sweep=3,
        real_is_i_trace=True,
    )
    np.testing.assert_array_equal(
        square_raw[..., 0],
        np.arange(9.0).reshape(3, 3),
    )
    with pytest.raises(ValueError, match="not divisible"):
        normalize_qcs_iq(
            np.asarray([1 + 1j, 2 + 2j, 3 + 3j]),
            repetitions_per_sweep=2,
        )


def test_calibrated_rf_power_sweep_fails_explicitly():
    sequence = _sequence().add_rf_power_sweep(
        "read", 3, -30.0, -20.0, 3
    )
    with pytest.raises(
        QcsUnsupportedFeatureError, match="connector-power"
    ):
        backend.validate_qcs_capabilities(
            connection_config=_connection(),
            sequence=sequence,
            acquisition=_acquisition(),
        )


def test_nonblocking_execution_fails_before_data_readback():
    with pytest.raises(
        QcsUnsupportedFeatureError, match="blocking=True"
    ):
        backend.validate_qcs_capabilities(
            connection_config=_connection(blocking=False),
            sequence=_sequence(),
            acquisition=_acquisition(),
        )


def test_run_qcs_qcodes_experiment_uses_qcs_storage(monkeypatch, tmp_path):
    execution = backend.QcsExecutionResult(
        ddr_result=backend.FineTuneDdrResult(
            sweep_points=np.asarray([0.0]),
            iq=np.zeros((1, 1, 1, 2)),
        ),
        programs=("program",),
        raw_results=("raw",),
        program_summary={"backend": "qcs"},
        rf_settings={"backend": "qcs"},
    )
    monkeypatch.setattr(
        backend, "execute_qcs_sequence", lambda **_kwargs: execution
    )
    captured = {}

    class Dataset:
        run_id = 17
        guid = "qcs-guid"

    def fake_store(result, **kwargs):
        captured["result"] = result
        captured.update(kwargs)
        return Dataset(), 2

    monkeypatch.setattr(backend, "store_experiment_result", fake_store)
    run_config = QcodesRunConfig(str(tmp_path / "qcs.db"))
    stored = backend.run_qcs_qcodes_experiment(
        connection_config=_connection(),
        run_config=run_config,
        sequence=_sequence(),
        repetitions_per_sweep=1,
        source_full_scale_mv=1000.0,
        acquisition=_acquisition(),
    )

    assert stored.run_id == 17
    assert stored.programs == ("program",)
    assert captured["backend_name"] == "qcs"
    assert captured["connection_config"] == _connection()
    assert captured["gui_settings"]["qick"]["full_scale_mv"] == 1000.0


def test_full_qcs_adapter_writes_real_qcodes_database(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "QSTL_QCODES_STAGING_DIR", str(tmp_path / "staging")
    )

    class Executor:
        def execute(self, _program):
            return np.asarray([1 + 2j, 3 + 4j])

    stored = backend.run_qcs_qcodes_experiment(
        connection_config=_connection(),
        run_config=QcodesRunConfig(
            str(tmp_path / "qcs_full.db"),
            experiment_name="QCS adapter integration",
            sample_name="unit test",
        ),
        sequence=_sequence(),
        repetitions_per_sweep=2,
        acquisition=_acquisition(),
        qcs_module=_FakeQcs,
        mapper=_Mapper("dc_gate", "digitizer"),
        executor=Executor(),
        gui_settings={
            "qick": {
                "fabric_mhz": 300.0,
                "full_scale_mv": 800.0,
            }
        },
    )

    assert stored.row_count == 2
    assert stored.database_path.is_file()
    metadata = json.loads(
        stored.dataset.get_metadata("qcs_experiment_json")
    )
    assert metadata["qcs_connection"]["mapper_path"] == "unused.json"
    assert metadata["program_summary"]["backend"] == "qcs"
    assert metadata["measurement_layout"]["iq_shape"] == [1, 2, 1, 2]


def test_real_qcs_255_builds_program_offline():
    qcs = pytest.importorskip("keysight.qcs")
    sequence = _sequence()
    dc = qcs.Channels(0, "dc_gate")
    dc_second = qcs.Channels(0, "dc_second")
    digitizer = qcs.Channels(0, "digitizer", absolute_phase=True)
    mapper = qcs.ChannelMapper()
    mapper.add_channel_mapping(
        dc, [(1, 2, 1)], qcs.InstrumentEnum.M5301AWG
    )
    mapper.add_channel_mapping(
        dc_second, [(1, 2, 2)], qcs.InstrumentEnum.M5301AWG
    )
    mapper.add_channel_mapping(
        digitizer,
        [(1, 4, 1)],
        qcs.InstrumentEnum.M5200Digitizer,
    )

    compiled = compile_qcs_point(
        sequence,
        0,
        connection_config=_connection(),
        mapper=mapper,
        repetitions_per_sweep=2,
        acquisition=_acquisition(),
        qcs_module=qcs,
    )
    assert isinstance(compiled.program, qcs.Program)
    assert compiled.acquisition_channels.name == "digitizer"
    assert len(compiled.program.layers) == 1
    dc_waveform = compiled.program.layers[0].operations[dc][0]
    assert dc_waveform.amplitudes[0].value == pytest.approx(0.16)

    raw_compiled = compile_qcs_point(
        sequence,
        0,
        connection_config=_connection(hw_demod=False),
        mapper=mapper,
        repetitions_per_sweep=2,
        acquisition=_acquisition(
            duration_s=1e-6,
            sample_count=480,
        ),
        qcs_module=qcs,
    )
    assert len(raw_compiled.program.layers) == 1
    acquisition_operation = next(
        operation
        for operation in raw_compiled.program.layers[0].operations[digitizer]
        if isinstance(operation, qcs.Acquisition)
    )
    assert acquisition_operation.integration_filter is None
    assert acquisition_operation.duration.value == pytest.approx(100e-9)
    assert raw_compiled.acquisition_sample_rate_hz == pytest.approx(4.8e9)

    two_output_sequence = FineTuneSequence(
        ("gate", "second")
    ).add_set("read", [0.5, -0.25], 300)
    two_output_compiled = compile_qcs_point(
        two_output_sequence,
        0,
        connection_config=_connection(
            dc_channel_names=("dc_gate", "dc_second")
        ),
        mapper=mapper,
        repetitions_per_sweep=2,
        acquisition=_acquisition(),
        qcs_module=qcs,
    )
    assert len(two_output_compiled.program.layers) == 1
    layer_operations = two_output_compiled.program.layers[0].operations
    assert dc in layer_operations
    assert dc_second in layer_operations
    assert digitizer in layer_operations


def test_real_qcs_255_builds_native_two_axis_stability_hardware_sweep():
    qcs = pytest.importorskip("keysight.qcs")
    dc_x = qcs.Channels(0, "dc_x", absolute_phase=False)
    dc_y = qcs.Channels(0, "dc_y", absolute_phase=False)
    rf_out = qcs.Channels(0, "rf_out", absolute_phase=True)
    digitizer = qcs.Channels(0, "digitizer", absolute_phase=True)
    mapper = qcs.ChannelMapper()
    mapper.add_channel_mapping(
        dc_x, [(1, 7, 1)], qcs.InstrumentEnum.M5301AWG
    )
    mapper.add_channel_mapping(
        dc_y, [(1, 7, 2)], qcs.InstrumentEnum.M5301AWG
    )
    mapper.add_channel_mapping(
        rf_out, [(1, 4, 1)], qcs.InstrumentEnum.M5300AWG
    )
    mapper.set_lo_frequencies(qcs.Address(1, 4, 1), 6.0e9)
    mapper.add_channel_mapping(
        digitizer,
        [(1, 18, 1)],
        qcs.InstrumentEnum.M5200Digitizer,
    )
    compiled = backend.compile_qcs_stability_hardware_sweep(
        _stability_sequence(),
        connection_config=_connection(
            dc_channel_names=("dc_x", "dc_y"),
            dc_full_scale_v=1.0,
            rf_channel_names={0: "rf_out"},
        ),
        mapper=mapper,
        repetitions_per_point=4,
        fabric_mhz=300.0,
        source_full_scale_mv=800.0,
        rf_pulses=(
            QcsRfPulseConfig(
                gen_ch=0,
                at_segment="set_0",
                duration_s=20.001e-6,
                amplitude=0.2,
                frequency_hz=50e6,
                phase_rad=0.0,
                delay_s=10.001e-6,
                envelope="constant",
                require_within_segment=False,
            ),
        ),
        acquisition=_acquisition(
            at_segment="set_0",
            duration_s=20e-6,
            pre_delay_s=10.001e-6,
            sample_count=32,
        ),
        qcs_module=qcs,
    )

    repetitions = compiled.program.repetitions
    assert [type(item).__name__ for item in repetitions.items] == [
        "Repeat",
        "Sweep",
    ]
    assert repetitions.shape == (4, 6)
    assert repetitions.averaged_shape == (6,)
    assert repetitions.num_hw_items == 2
    sweep = repetitions.items[1]
    association_by_name = {
        variable.name: array.value
        for variable, array in sweep.associations.items()
    }
    assert set(association_by_name) == {
        "stability_dc_0_amplitude",
        "stability_dc_1_amplitude",
    }
    np.testing.assert_allclose(
        association_by_name["stability_dc_0_amplitude"],
        [-0.24, -0.16, -0.04, 0.04, 0.16, 0.24],
    )
    np.testing.assert_allclose(
        association_by_name["stability_dc_1_amplitude"],
        [-0.36, 0.44, -0.4, 0.4, -0.44, 0.36],
    )
    assert compiled.sweep_shape == (3, 2)
    assert len(compiled.program.layers) == 1
    assert len(compiled.program.layers[0].operations) == 4
    for channel in (dc_x, dc_y):
        waveform = compiled.program.layers[0].operations[channel][0]
        assert len(waveform.amplitudes) == 1
        assert isinstance(waveform.amplitudes[0], qcs.Scalar)

    def assert_300_mhz_aligned(seconds):
        ticks = seconds * 300e6
        assert ticks == pytest.approx(round(ticks), abs=1e-7)

    generated, _ = qcs.SequenceBuilder(channel_map=mapper).build(
        compiled.program
    )
    for operations in generated.layers[0].operations.values():
        for operation in operations:
            assert_300_mhz_aligned(operation.duration.value)
    assert compiled.program.render(mapper=mapper) is not None


def test_generated_qcs_code_aligns_outputs_in_parallel_layers():
    qcs = pytest.importorskip("keysight.qcs")
    gate_a = PulseSequence(0.0, 10.0)
    gate_a.add_flat_ramp(5.0, 10.0, 100.0)
    gate_b = PulseSequence(50.0, 15.0)
    gate_b.add_flat_ramp(7.0, 8.0, -50.0)

    code = generate_qcs_program_code(
        (gate_a, gate_b),
        channel_names=("gate_a", "gate_b"),
    )
    namespace = {}
    exec(code, namespace)
    program = namespace["generate_dc_waveforms"](
        qcs.Program(),
        qcs.Channels(0, "gate_a"),
        qcs.Channels(0, "gate_b"),
    )

    interval_count = len(np.unique(np.concatenate((gate_a.t, gate_b.t)))) - 1
    assert isinstance(program, qcs.Program)
    assert code.count("new_layer=True") == interval_count
    assert code.count("new_layer=False") == interval_count
