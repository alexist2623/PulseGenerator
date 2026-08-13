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


class _PhysicalSettings:
    def __init__(self, name):
        self.offset = _Scalar(f"{name}_offset", value=0.0, dtype=float)


class _PhysicalChannel:
    def __init__(self, name, instrument, sample_rate=None):
        self.name = name
        self.instrument = instrument
        self.sample_rate = sample_rate
        self.settings = _PhysicalSettings(name)


class _PhysicalMapper(_Mapper):
    """Fake native mapper exposing M5301 offset and M5200 timing."""

    def __init__(self, *names):
        super().__init__(*names)
        self._physical = {}
        for channel in self.channels:
            is_acquisition = "digitizer" in channel.name
            self._physical[channel] = _PhysicalChannel(
                channel.name,
                "M5200Digitizer" if is_acquisition else "M5301AWG",
                (
                    backend.QCS_M5200_SAMPLE_RATE_HZ
                    if is_acquisition
                    else None
                ),
            )

    def get_physical_channels(self, channel):
        return (self._physical[channel],)

    def offset_scalar(self, name):
        channel = next(
            channel for channel in self.channels if channel.name == name
        )
        return self._physical[channel].settings.offset


class _Envelope:
    def __init__(self, *args):
        self.args = args


class _Waveform:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __add__(self, other):
        return _CombinedWaveform(self, other)


class _CombinedWaveform(_Waveform):
    def __init__(self, *components):
        super().__init__(*components)
        self.components = components


class _Hold(_Waveform):
    pass


class _Delay(_Waveform):
    pass


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
        self.repetition_calls = []

    def add_waveform(self, pulse, channels, **kwargs):
        self.waveforms.append((pulse, channels, kwargs))

    def add_acquisition(self, **kwargs):
        self.acquisitions.append(kwargs)

    def n_shots(self, value):
        self.shots = value
        self.repetition_calls.append("shots")
        return self

    def sweep(self, values, target):
        self.sweeps.append((values, target))
        self.repetition_calls.append("sweep")
        return self


class _FakeQcs:
    Program = _Program
    DCWaveform = _Waveform
    Hold = _Hold
    Delay = _Delay
    RFWaveform = _Waveform
    ConstantEnvelope = _Envelope
    GaussianEnvelope = _Envelope
    ArbitraryEnvelope = _Envelope
    Scalar = _Scalar
    Array = _Array


def _sequence():
    return FineTuneSequence(("gate",)).add_set("read", [0.5], 300)


def _fixed_baseline_offset_sequence():
    return (
        FineTuneSequence(("gate",))
        .add_set("baseline_before", [0.10], 32)
        .add_ramp("ramp_up", 128)
        .add_set("swept_level", [0.20], 32)
        .add_ramp("ramp_down", 128)
        .add_set("baseline_after", [0.10], 32)
        .add_amplitude_sweep(
            "swept_level",
            "gate",
            0.20,
            0.40,
            3,
        )
    )


def _swept_negative_plateau_sequence(
    *,
    plateau_cycles=30_000,
    ramp_cycles=3_000,
):
    """Pulse used by the GUI's swept final-negative-plateau workflow."""

    return (
        FineTuneSequence(("gate",))
        # The GUI's 800 mV scale maps +/-0.125 to +/-100 mV.
        .add_set("initial_negative", [-0.125], 3_000)
        .add_ramp("ramp_to_positive", ramp_cycles)
        .add_set("positive", [0.125], 120)
        .add_ramp("ramp_to_swept_negative", ramp_cycles)
        .add_set("swept_negative", [-0.125], plateau_cycles)
        .add_amplitude_sweep(
            "swept_negative",
            "gate",
            -0.125,
            -0.3125,
            5,
        )
    )


def _two_output_swept_negative_plateau_sequence(*, levels=101):
    return (
        FineTuneSequence(("gate_a", "gate_b"))
        .add_set("initial_negative", [-0.125, -0.125], 3_000)
        .add_ramp("ramp_to_positive", 3_000)
        .add_set("positive", [0.125, 0.125], 120)
        .add_ramp("ramp_to_swept_negative", 3_000)
        .add_set("swept_negative", [-0.125, -0.125], 30_000)
        .add_amplitude_sweep(
            "swept_negative", "gate_a", -0.125, -0.3125, levels
        )
        .add_amplitude_sweep(
            "swept_negative", "gate_b", -0.125, -0.375, levels
        )
    )


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


def _program_waveform_operations(entry):
    pulse = entry[0]
    return pulse if isinstance(pulse, list) else [pulse]


def test_qcs_cancellation_aborts_only_owned_pending_programs():
    aborted = []
    pending = [
        {
            "AccessionId": 101,
            "State": "Running",
            "Description": "PulseGenerator point 4 [qcs-gui-owned]",
        },
        {
            "AccessionId": 102,
            "State": "Compiling",
            "Description": (
                "PulseGenerator synchronized AWG tuning sweep "
                "[qcs-gui-owned]"
            ),
        },
        {
            "AccessionId": 103,
            "State": "Running",
            "Description": "PulseGenerator point 8 [qcs-gui-other]",
        },
        {
            "AccessionId": 104,
            "State": "Running",
            "Description": "PulseGenerator emergency DC reset",
        },
    ]

    class AbortBackend:
        def __init__(self, channel_mapper):
            assert channel_mapper is mapper

        def get_pending_programs_info(self):
            return pending

        def abort_program(self, accession_id):
            aborted.append(accession_id)
            return object()

    class AbortQcs:
        HclBackend = AbortBackend

    mapper = object()
    controller = backend.QcsCancellationController(
        program_name_tag="owned"
    )
    controller.bind(AbortQcs, mapper)

    assert controller.request_stop() is True
    assert controller.request_stop() is False
    assert controller.wait_for_abort(1.0) is True
    assert aborted == [101, 102]


def test_qcs_cancellation_retries_rejected_abort_until_accepted():
    abort_results = iter((False, False, True))
    abort_calls = []
    status = []

    class AbortBackend:
        def __init__(self, channel_mapper):
            assert channel_mapper is mapper

        def get_pending_programs_info(self):
            return [
                {
                    "AccessionId": 201,
                    "State": "Running",
                    "Description": "PulseGenerator [qcs-gui-retry]",
                }
            ]

        def abort_program(self, accession_id):
            abort_calls.append(accession_id)
            return next(abort_results)

    class AbortQcs:
        HclBackend = AbortBackend

    mapper = object()
    controller = backend.QcsCancellationController(
        program_name_tag="retry",
        status_callback=status.append,
    )
    controller.bind(AbortQcs, mapper)

    controller.request_stop()
    assert controller.wait_for_abort(1.0) is True
    assert abort_calls == [201, 201, 201]
    assert any("accepted the abort" in message for message in status)


def test_qcs_cancellation_waits_past_five_queries_for_registration():
    query_count = 0
    aborted = []

    class AbortBackend:
        def __init__(self, channel_mapper):
            assert channel_mapper is mapper

        def get_pending_programs_info(self):
            nonlocal query_count
            query_count += 1
            if query_count <= 6:
                return []
            return [
                {
                    "AccessionId": 301,
                    "State": "Compiling",
                    "Description": "PulseGenerator [qcs-gui-register]",
                }
            ]

        def abort_program(self, accession_id):
            aborted.append(accession_id)
            return True

    class AbortQcs:
        HclBackend = AbortBackend

    mapper = object()
    controller = backend.QcsCancellationController(
        program_name_tag="register"
    )
    controller.bind(AbortQcs, mapper)

    controller.request_stop()
    assert controller.wait_for_abort(2.0) is True
    assert query_count == 7
    assert aborted == [301]


def test_qcs_cancellation_rejects_request_after_stop_window_closes():
    status = []
    controller = backend.QcsCancellationController(
        program_name_tag="completed",
        status_callback=status.append,
    )

    assert controller.close_stop_window() is True
    assert controller.request_stop() is False
    assert controller.is_stop_requested() is False
    assert "already completed" in status[-1]


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


def test_compile_lowers_awg_set_ramp_set_to_real_dc_waveforms():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("initial_hold", [0.0], 30_000)
        .add_ramp("linear_ramp", 3_000)
        .add_set("final_hold", [0.4], 3_000)
    )

    compiled = compile_qcs_point(
        sequence,
        0,
        connection_config=_connection(),
        mapper=_Mapper("dc_gate", "digitizer"),
        repetitions_per_sweep=1,
        qcs_module=_FakeQcs,
    )

    operations = _program_waveform_operations(
        compiled.program.waveforms[0]
    )
    assert [type(operation) for operation in operations] == [
        _Delay,
        _Waveform,
        _Waveform,
    ]
    durations = [operation.kwargs["duration"] for operation in operations]
    np.testing.assert_allclose(
        durations,
        [100e-6, 10e-6, 10e-6],
        rtol=0.0,
        atol=1e-15,
    )
    assert sum(durations) == pytest.approx(120e-6)
    assert compiled.duration_s == pytest.approx(120e-6)

    # 0.0 -> 0.4 of the GUI's 800 mV scale becomes 0.0 -> 0.128
    # relative to the configured 2.5 V QCS full scale. The following SET uses
    # a real constant DCWaveform; HCL Hold does not preserve M5301 voltage.
    ramp_vertices = operations[1].kwargs["envelope"].args[1]
    assert ramp_vertices[0] == pytest.approx(0.0)
    assert ramp_vertices[-1] == pytest.approx(0.128)
    assert operations[2].kwargs["amplitude"] == pytest.approx(0.128)


@pytest.mark.parametrize(
    ("start", "end", "expected_first", "expected_second"),
    [
        (0.048, -0.024, 86, 42),
        (-0.024, 0.048, 42, 86),
    ],
)
def test_numeric_bipolar_ramp_splits_at_zero_without_waveform_addition(
    start,
    end,
    expected_first,
    expected_second,
):
    operations = backend._qcs_ramp_dc_interval(
        _FakeQcs,
        duration_cycles=128,
        start_amplitude=start,
        end_amplitude=end,
        name="bipolar",
        fabric_hz=300e6,
    )

    assert isinstance(operations, list)
    assert [type(operation) for operation in operations] == [
        _Waveform,
        _Waveform,
    ]
    assert not any(isinstance(operation, _CombinedWaveform) for operation in operations)
    np.testing.assert_allclose(
        [operation.kwargs["duration"] * 300e6 for operation in operations],
        [expected_first, expected_second],
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        operations[0].kwargs["envelope"].args[1],
        [start, 0.0],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        operations[1].kwargs["envelope"].args[1],
        [0.0, end],
        rtol=0.0,
        atol=1e-15,
    )


def test_numeric_bipolar_ramp_rejects_duration_too_short_to_split():
    with pytest.raises(QcsUnsupportedFeatureError, match="at least 8"):
        backend._qcs_ramp_dc_interval(
            _FakeQcs,
            duration_cycles=6,
            start_amplitude=0.1,
            end_amplitude=-0.1,
            name="short_bipolar",
            fabric_hz=300e6,
        )


def test_compile_preserves_instantaneous_set_before_next_hold():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("first", [0.1], 3_000)
        .add_set("second", [0.2], 3_000)
    )

    compiled = compile_qcs_point(
        sequence,
        0,
        connection_config=_connection(),
        mapper=_Mapper("dc_gate", "digitizer"),
        repetitions_per_sweep=1,
        qcs_module=_FakeQcs,
    )

    operations = _program_waveform_operations(
        compiled.program.waveforms[0]
    )
    assert [type(operation) for operation in operations] == [
        _Waveform,
        _Waveform,
    ]
    assert operations[0].kwargs["amplitude"] == pytest.approx(0.032)
    assert operations[1].kwargs["amplitude"] == pytest.approx(0.064)
    assert sum(
        operation.kwargs["duration"] for operation in operations
    ) == pytest.approx(20e-6)


def test_compile_emits_explicit_terminal_dc_reset_for_compensation():
    times_cycles = np.asarray([0.0, 300.0, 300.0])
    amplitudes = np.asarray([0.1, 0.1, 0.0])

    operations = backend._qcs_dc_waveform_operations(
        _FakeQcs,
        times_cycles=times_cycles,
        amplitudes=amplitudes,
        name="compensated_gate",
        fabric_hz=300e6,
        append_terminal_value=True,
    )

    assert [type(operation) for operation in operations] == [
        _Waveform,
        _Delay,
    ]
    assert operations[0].kwargs["amplitude"] == pytest.approx(0.1)
    assert operations[1].kwargs["duration"] == pytest.approx(4 / 300e6)


def test_m5301_ramp_rejects_unrepresentable_odd_fabric_cycle_duration():
    with pytest.raises(ValueError, match="16-sample waveform granularity"):
        backend._qcs_ramp_dc_interval(
            _FakeQcs,
            duration_cycles=301,
            start_amplitude=0.0,
            end_amplitude=0.1,
            name="odd_ramp",
            fabric_hz=300e6,
        )


def test_awg_ramp_preflight_matches_measured_98304_sample_hcl_budget():
    exact_limit = (
        FineTuneSequence(("gate",))
        .add_set("initial", [0.0], 30_000)
        .add_ramp("ramp", 6_000)
        .add_set("final", [0.001], 6_288)
    )
    compiled = compile_qcs_point(
        exact_limit,
        0,
        connection_config=_connection(),
        mapper=_Mapper("dc_gate", "digitizer"),
        repetitions_per_sweep=1,
        qcs_module=_FakeQcs,
    )
    operations = _program_waveform_operations(
        compiled.program.waveforms[0]
    )
    rendered_cycles = sum(
        round(operation.kwargs["duration"] * 300e6)
        for operation in operations
        if type(operation) is _Waveform
    )
    assert rendered_cycles == 12_288
    exact_report = backend.qcs_m5301_waveform_capacity_report(exact_limit)
    assert exact_report.worst_channel.rendered_samples == 98_304
    assert exact_report.usage_fraction == pytest.approx(1.0)
    assert exact_report.exceeds_capacity is False

    over_limit = (
        FineTuneSequence(("gate",))
        .add_set("initial", [0.0], 30_000)
        .add_ramp("ramp", 6_000)
        .add_set("final", [0.001], 6_290)
    )
    with pytest.raises(
        QcsUnsupportedFeatureError,
        match=r"98,320 / 98,304 samples.*40\.966667 / 40\.960000 us",
    ):
        compile_qcs_point(
            over_limit,
            0,
            connection_config=_connection(),
            mapper=_Mapper("dc_gate", "digitizer"),
            repetitions_per_sweep=1,
            qcs_module=_FakeQcs,
        )
    over_report = backend.qcs_m5301_waveform_capacity_report(over_limit)
    assert over_report.worst_channel.rendered_samples == 98_320
    assert over_report.exceeds_capacity is True
    with pytest.raises(
        QcsUnsupportedFeatureError,
        match=r"waveform capacity exceeded.*sweep point 1",
    ):
        backend.validate_qcs_m5301_waveform_capacity(over_limit)


def test_m5301_capacity_is_per_output_and_zero_delays_use_no_samples():
    two_outputs = (
        FineTuneSequence(("gate_x", "gate_y"))
        .add_set("initial", [0.0, 0.0], 30_000)
        .add_ramp("ramp", 6_000)
        .add_set("final", [0.001, 0.002], 6_288)
    )
    report = backend.qcs_m5301_waveform_capacity_report(two_outputs)

    assert [channel.rendered_samples for channel in report.channels] == [
        98_304,
        98_304,
    ]
    assert report.usage_fraction == pytest.approx(1.0)
    assert report.exceeds_capacity is False

    long_zero_delay = FineTuneSequence(("gate",)).add_set(
        "zero_wait",
        [0.0],
        100_001,
    )
    zero_report = backend.qcs_m5301_waveform_capacity_report(long_zero_delay)
    assert zero_report.worst_channel.rendered_samples == 0


def test_m5301_capacity_preview_accounts_for_automatic_fixed_offset():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("baseline_before", [0.10], 20_000)
        .add_ramp("ramp_up", 1_000)
        .add_set("swept_level", [0.20], 1_000)
        .add_ramp("ramp_down", 1_000)
        .add_set("baseline_after", [0.10], 20_000)
        .add_amplitude_sweep("swept_level", "gate", 0.20, 0.30, 3)
    )
    conservative = backend.qcs_m5301_waveform_capacity_report(
        sequence,
        amplitude_scale=0.8 / 2.5,
    )
    assert conservative.exceeds_capacity is True

    optimized = backend.validate_qcs_m5301_waveform_capacity(
        sequence,
        amplitude_scale=0.8 / 2.5,
        auto_fixed_dc_offsets=True,
        source_full_scale_mv=800.0,
        dc_full_scale_v=2.5,
    )
    assert optimized.exceeds_capacity is False
    # Both constant plateaus directly follow ramps and are retained with Hold;
    # only the two 1,000-cycle ramps consume rendered waveform memory.
    assert optimized.worst_channel.rendered_fabric_cycles == 2_000
    assert optimized.worst_channel.rendered_samples == 16_000


def test_m5301_capacity_counts_only_nonzero_terminal_waveform():
    class _TerminalSequence:
        output_names = ("active", "zero")
        sweep_point_count = 1
        sweep_axes = ()

        @staticmethod
        def compensated_waveform_vertices(_point_index):
            return (
                np.asarray([0.0, 12_284.0, 12_284.0]),
                {
                    "active": np.asarray([0.1, 0.1, 0.2]),
                    "zero": np.asarray([0.0, 0.0, 0.0]),
                },
                {},
            )

        @staticmethod
        def sweep_coordinate(_point_index):
            return ()

    report = backend.qcs_m5301_waveform_capacity_report(_TerminalSequence())
    assert report.channels[0].rendered_samples == 98_304
    assert report.channels[1].rendered_samples == 0


def test_m5301_capacity_requires_300_mhz_and_precedes_mapper_loading(
    monkeypatch,
):
    over_limit = (
        FineTuneSequence(("gate",))
        .add_set("initial", [0.0], 30_000)
        .add_ramp("ramp", 12_290)
        .add_set("final", [0.001], 100)
    )
    with pytest.raises(ValueError, match="fixed 300 MHz"):
        backend.qcs_m5301_waveform_capacity_report(
            over_limit,
            fabric_mhz=250.0,
        )

    mapper_loaded = False

    def _unexpected_mapper_load(*_args, **_kwargs):
        nonlocal mapper_loaded
        mapper_loaded = True
        raise AssertionError("mapper must not load before capacity validation")

    monkeypatch.setattr(backend, "load_qcs_channel_mapper", _unexpected_mapper_load)
    with pytest.raises(QcsUnsupportedFeatureError, match="capacity exceeded"):
        execute_qcs_sequence(
            connection_config=_connection(),
            sequence=over_limit,
            repetitions_per_sweep=1,
            acquisition=_acquisition(at_segment="final"),
            qcs_module=_FakeQcs,
        )
    assert mapper_loaded is False


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


def test_raw_trace_can_cross_segment_while_single_iq_remains_scoped():
    acquisition = _acquisition(
        duration_s=2e-6,
        pre_delay_s=0.5e-6,
        sample_count=9_600,
    )
    raw_connection = _connection(hw_demod=False)
    raw_point = compile_qcs_point(
        _sequence(),
        0,
        connection_config=raw_connection,
        mapper=_Mapper("dc_gate", "digitizer"),
        repetitions_per_sweep=1,
        acquisition=acquisition,
        qcs_module=_FakeQcs,
    )

    request = raw_point.program.acquisitions[0]
    assert request["integration_filter"] == pytest.approx(2e-6)
    assert request["pre_delay"] == pytest.approx(0.5e-6)
    assert raw_point.duration_s == pytest.approx(2.5e-6)

    swept = _sequence().add_amplitude_sweep(
        "read", "gate", 0.25, 0.5, 2
    )
    raw_sweep = backend.compile_qcs_synchronized_sweep(
        swept,
        connection_config=raw_connection,
        mapper=_Mapper("dc_gate", "digitizer"),
        repetitions_per_sweep=1,
        acquisition=acquisition,
        qcs_module=_FakeQcs,
    )
    assert raw_sweep.hardware_sweep is False
    assert raw_sweep.duration_s == pytest.approx(2.5e-6)
    assert raw_sweep.program.acquisitions[0][
        "integration_filter"
    ] == pytest.approx(2e-6)

    with pytest.raises(ValueError, match="acquisition exceeds segment"):
        compile_qcs_point(
            _sequence(),
            0,
            connection_config=_connection(hw_demod=True),
            mapper=_Mapper("dc_gate", "digitizer"),
            repetitions_per_sweep=1,
            acquisition=acquisition,
            qcs_module=_FakeQcs,
        )
    with pytest.raises(ValueError, match="acquisition exceeds segment"):
        backend.compile_qcs_synchronized_sweep(
            swept,
            connection_config=_connection(hw_demod=True),
            mapper=_Mapper("dc_gate", "digitizer"),
            repetitions_per_sweep=1,
            acquisition=acquisition,
            qcs_module=_FakeQcs,
        )


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


def test_execute_native_awg_sweep_uses_one_program_in_c_order():
    sequence = _sequence().add_amplitude_sweep(
        "read", "gate", 0.25, 0.5, 2
    )
    returned = np.asarray(
        [
            [1 + 2j, 5 + 6j],
            [3 + 4j, 7 + 8j],
        ]
    )
    calls = {"execute": 0}

    class Executor:
        def execute(self, program):
            calls["execute"] += 1
            assert program.repetition_calls == ["sweep", "shots"]
            assert len(program.sweeps) == 1
            arrays, variables = program.sweeps[0]
            assert len(arrays) == len(variables) == 1
            np.testing.assert_allclose(arrays[0].value, [0.08, 0.16])
            return returned

    result = execute_qcs_sequence(
        connection_config=_connection(),
        sequence=sequence,
        repetitions_per_sweep=2,
        acquisition=_acquisition(),
        qcs_module=_FakeQcs,
        mapper=_Mapper("dc_gate", "digitizer"),
        executor=Executor(),
    )

    assert calls["execute"] == 1
    assert len(result.programs) == 1
    assert len(result.raw_results) == 1
    assert result.program_summary["hardware_sweep"] is True
    assert result.program_summary["sweep_execution_mode"] == "hardware_flattened"
    assert result.program_summary["program_count"] == 1
    assert result.program_summary["executor_call_count"] == 1
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


def test_fixed_m5301_offset_enables_native_segment_amplitude_sweep():
    sequence = _fixed_baseline_offset_sequence()
    mapper = _PhysicalMapper("dc_gate", "digitizer")
    offset = mapper.offset_scalar("dc_gate")
    returned = np.asarray(
        [
            [1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j],
            [4.0 + 4.0j, 5.0 + 5.0j, 6.0 + 6.0j],
        ]
    )
    calls = []

    class Executor:
        def execute(self, program):
            calls.append(program)
            if program.name == "PulseGenerator emergency DC reset":
                assert offset.value == pytest.approx(0.0)
                return None

            # GUI amplitudes use the 800 mV source scale. The common 0.10
            # baseline therefore becomes an 80 mV physical-channel offset.
            assert offset.value == pytest.approx(0.080)
            assert program.repetition_calls == ["sweep", "shots"]
            assert len(program.sweeps) == 1
            arrays, variables = program.sweeps[0]
            assert len(arrays) == len(variables) == 2
            assert all(variable is not offset for variable in variables)
            for array in arrays:
                # Residual target levels are 80/160/240 mV, normalized by
                # the independent 2.5 V DCWaveform full scale.
                np.testing.assert_allclose(
                    array.value,
                    [0.032, 0.064, 0.096],
                )
            operations = _program_waveform_operations(program.waveforms[0])
            assert not any(
                isinstance(operation, _CombinedWaveform)
                for operation in operations
            )
            return returned

    result = execute_qcs_sequence(
        connection_config=_connection(),
        sequence=sequence,
        repetitions_per_sweep=2,
        acquisition=_acquisition(at_segment="swept_level"),
        qcs_module=_FakeQcs,
        mapper=mapper,
        executor=Executor(),
    )

    assert len(calls) == 2
    assert calls[-1].name == "PulseGenerator emergency DC reset"
    assert offset.value == pytest.approx(0.0)
    assert result.program_summary["hardware_sweep"] is True
    assert result.program_summary["sweep_execution_mode"] == "hardware_flattened"
    assert result.program_summary["program_count"] == 1
    assert result.program_summary["executor_call_count"] == 1
    assert result.program_summary["safety_reset_executor_call_count"] == 1
    assert result.program_summary["total_executor_call_count"] == 2
    assert result.program_summary["fixed_dc_offset_residualization"] is True
    np.testing.assert_allclose(
        result.program_summary["dc_channel_offsets_v"],
        [0.080],
    )
    assert result.ddr_result.iq.shape == (3, 2, 1, 2)
    np.testing.assert_allclose(
        result.ddr_result.iq[:, :, 0, 0],
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
    )


def test_swept_negative_plateau_uses_one_native_hardware_sweep_and_hold():
    sequence = _swept_negative_plateau_sequence()
    mapper = _PhysicalMapper("dc_gate", "digitizer")

    compiled = backend.compile_qcs_synchronized_sweep(
        sequence,
        connection_config=_connection(dc_full_scale_v=2.5),
        mapper=mapper,
        repetitions_per_sweep=1,
        source_full_scale_mv=800.0,
        acquisition=_acquisition(
            at_segment="swept_negative",
            duration_s=1e-6,
        ),
        qcs_module=_FakeQcs,
    )

    assert compiled.hardware_sweep is True
    assert compiled.software_sweep_reasons == ()
    assert compiled.program.repetition_calls == ["sweep", "shots"]
    np.testing.assert_allclose(compiled.dc_channel_offsets_v, [0.100])
    assert mapper.offset_scalar("dc_gate").value == pytest.approx(0.100)

    operations = _program_waveform_operations(
        compiled.program.waveforms[0]
    )
    operations_by_name = {
        operation.kwargs.get("name"): operation
        for operation in operations
    }
    ramp = operations_by_name["awg_dc_0_interval_3_rising"]
    assert ramp.kwargs["duration"] == pytest.approx(10e-6)
    plateau_operations = [
        operation
        for operation in operations
        if str(operation.kwargs.get("name", "")).startswith(
            "awg_dc_0_interval_4"
        )
    ]
    assert any(
        isinstance(operation, _Hold)
        for operation in plateau_operations
    )
    assert sum(
        operation.kwargs["duration"] for operation in plateau_operations
    ) == pytest.approx(100e-6)

    arrays, variables = compiled.program.sweeps[0]
    sweep_values_by_variable = {
        id(variable): np.asarray(array.value)
        for array, variable in zip(arrays, variables)
    }
    expected_residual = np.linspace(-0.100, -0.250, 5)
    expected_residual = (expected_residual - 0.100) / 2.5
    np.testing.assert_allclose(
        sweep_values_by_variable[id(ramp.kwargs["amplitude"])],
        expected_residual,
    )

    # A short swept SET may precede Hold, or the compiler may directly Hold
    # the endpoint established by the incoming ramp. In both cases the
    # plateau voltage follows the same hardware-swept endpoint values.
    plateau_seed = next(
        (
            operation
            for operation in plateau_operations
            if isinstance(operation, _Waveform)
            and not isinstance(operation, _Hold)
        ),
        None,
    )
    if plateau_seed is None:
        assert isinstance(plateau_operations[0], _Hold)
        assert operations.index(plateau_operations[0]) == (
            operations.index(ramp) + 1
        )
    else:
        np.testing.assert_allclose(
            sweep_values_by_variable[
                id(plateau_seed.kwargs["amplitude"])
            ],
            expected_residual,
        )


def test_swept_negative_plateau_capacity_preview_counts_hold_seed_only():
    report = backend.qcs_m5301_waveform_capacity_report(
        _swept_negative_plateau_sequence(),
        fabric_mhz=300.0,
        amplitude_scale=800.0 / 2_500.0,
        auto_fixed_dc_offsets=True,
        source_full_scale_mv=800.0,
        dc_full_scale_v=2.5,
    )

    assert report.exhaustive is True
    assert report.exceeds_capacity is False
    assert report.worst_channel.rendered_samples <= (
        backend.QCS_M5301_MAX_RENDERED_SAMPLES
    )


def test_duration_swept_plateau_capacity_preview_does_not_assume_hold():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("zero", [0.0], 300)
        .add_ramp("ramp", 3_000)
        .add_set("long_plateau", [-0.125], 30_000)
        .add_hold_duration_sweep(
            "long_plateau",
            100.0,
            110.0,
            2,
            sequence_fabric_mhz=300.0,
        )
    )
    report = backend.qcs_m5301_waveform_capacity_report(
        sequence,
        fabric_mhz=300.0,
        amplitude_scale=800.0 / 2_500.0,
        auto_fixed_dc_offsets=True,
        source_full_scale_mv=800.0,
        dc_full_scale_v=2.5,
    )

    assert report.exceeds_capacity is True
    assert report.worst_channel.rendered_fabric_cycles == 36_000


def test_swept_negative_plateau_preview_reports_planned_hardware_mode():
    preview = backend.qcs_sweep_execution_preview(
        _swept_negative_plateau_sequence(),
        hardware_demodulation=True,
        source_full_scale_mv=800.0,
        dc_full_scale_v=2.5,
    )

    assert preview.mode == "hardware"
    assert preview.exact is False
    assert preview.dc_channel_offsets_v == pytest.approx((0.100,))
    assert "confirmed" in preview.reasons[0]

    trace_preview = backend.qcs_sweep_execution_preview(
        _swept_negative_plateau_sequence(),
        hardware_demodulation=False,
        source_full_scale_mv=800.0,
        dc_full_scale_v=2.5,
    )
    assert trace_preview.mode == "software"
    assert any("Raw trace" in reason for reason in trace_preview.reasons)


def test_101_by_101_voltage_grid_uses_hardware_limit_not_software_limit():
    sequence = _two_output_swept_negative_plateau_sequence()
    connection = _connection(
        dc_channel_names=("dc_gate_a", "dc_gate_b"),
        dc_full_scale_v=2.5,
    )
    mapper = _PhysicalMapper("dc_gate_a", "dc_gate_b", "digitizer")

    preview = backend.qcs_sweep_execution_preview(
        sequence,
        hardware_demodulation=True,
        source_full_scale_mv=800.0,
        dc_full_scale_v=2.5,
        init_time_s=100e-6,
    )
    compiled = backend.compile_qcs_synchronized_sweep(
        sequence,
        connection_config=connection,
        mapper=mapper,
        repetitions_per_sweep=1,
        source_full_scale_mv=800.0,
        acquisition=_acquisition(
            at_segment="swept_negative",
            duration_s=1e-6,
        ),
        qcs_module=_FakeQcs,
    )

    assert sequence.sweep_point_count == 10_201
    assert preview.mode == "hardware"
    assert preview.exact is False
    assert "100 us inter-shot" in " ".join(preview.reasons)
    assert compiled.hardware_sweep is True
    assert compiled.sweep_shape == (101, 101)
    assert compiled.sweep_variable_count == 2
    assert compiled.sweep_array_value_count == 20_402
    assert compiled.program.repetition_calls == ["sweep", "shots"]


def test_101_by_101_trace_grid_is_rejected_as_oversized_software_sweep():
    sequence = _two_output_swept_negative_plateau_sequence()
    preview = backend.qcs_sweep_execution_preview(
        sequence,
        hardware_demodulation=False,
        source_full_scale_mv=800.0,
        dc_full_scale_v=2.5,
    )

    assert preview.mode == "invalid"
    assert "10,000-point limit" in " ".join(preview.reasons)

    with pytest.raises(
        QcsUnsupportedFeatureError,
        match="software sweeps are limited to 10,000",
    ):
        backend.compile_qcs_synchronized_sweep(
            sequence,
            connection_config=_connection(
                dc_channel_names=("dc_gate_a", "dc_gate_b"),
                dc_full_scale_v=2.5,
                hw_demod=False,
            ),
            mapper=_PhysicalMapper(
                "dc_gate_a", "dc_gate_b", "digitizer"
            ),
            repetitions_per_sweep=1,
            source_full_scale_mv=800.0,
            acquisition=_acquisition(
                at_segment="swept_negative",
                duration_s=1e-6,
            ),
            qcs_module=_FakeQcs,
        )


def test_unrelated_fixed_ramp_does_not_block_offset_hardware_sweep():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("initial", [-0.25], 300)
        .add_ramp("unrelated_fixed_ramp", 600)
        .add_set("fixed_level", [0.375], 120)
        .add_set("sweep_anchor", [0.125], 120)
        .add_ramp("swept_ramp", 600)
        .add_set("swept_plateau", [-0.125], 300)
        .add_amplitude_sweep(
            "swept_plateau", "gate", -0.125, -0.3125, 5
        )
    )
    mapper = _PhysicalMapper("dc_gate", "digitizer")

    compiled = backend.compile_qcs_synchronized_sweep(
        sequence,
        connection_config=_connection(dc_full_scale_v=2.5),
        mapper=mapper,
        repetitions_per_sweep=1,
        source_full_scale_mv=800.0,
        acquisition=_acquisition(
            at_segment="swept_plateau",
            duration_s=100e-9,
        ),
        qcs_module=_FakeQcs,
    )

    assert compiled.hardware_sweep is True
    assert compiled.sweep_variable_count == 1
    assert compiled.dc_channel_offsets_v == pytest.approx((0.100,))
    assert compiled.program.repetition_calls == ["sweep", "shots"]


def test_zero_offset_ramp_to_long_plateau_uses_hold_in_direct_compiler():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("zero", [0.0], 300)
        .add_ramp("swept_ramp", 3_000)
        .add_set("swept_plateau", [-0.125], 30_000)
        .add_amplitude_sweep(
            "swept_plateau", "gate", -0.125, -0.3125, 5
        )
    )
    compiled = backend.compile_qcs_synchronized_sweep(
        sequence,
        connection_config=_connection(dc_full_scale_v=2.5),
        mapper=_PhysicalMapper("dc_gate", "digitizer"),
        repetitions_per_sweep=1,
        source_full_scale_mv=800.0,
        acquisition=_acquisition(
            at_segment="swept_plateau",
            duration_s=100e-9,
        ),
        qcs_module=_FakeQcs,
    )

    assert compiled.hardware_sweep is True
    assert compiled.dc_channel_offsets_v == pytest.approx((0.0,))
    operations = _program_waveform_operations(compiled.program.waveforms[0])
    assert any(isinstance(operation, _Hold) for operation in operations)


def test_fixed_time_compensation_stays_in_hardware_when_within_budget():
    sequence = _swept_negative_plateau_sequence(
        plateau_cycles=300,
        ramp_cycles=600,
    ).set_bias_t_compensation(
        0.1,
        mode="fixed_time",
        fixed_duration_cycles=3_000,
        inter_output_gap_cycles=2,
    )
    mapper = _PhysicalMapper("dc_gate", "digitizer")
    calls = []

    class Executor:
        def execute(self, program):
            calls.append(program)
            if len(calls) == 1:
                assert mapper.offset_scalar("dc_gate").value == pytest.approx(
                    0.100
                )
                return np.full(sequence.sweep_point_count, 1.0 + 2.0j)
            assert mapper.offset_scalar("dc_gate").value == pytest.approx(0.0)
            return None

    result = execute_qcs_sequence(
        connection_config=_connection(dc_full_scale_v=2.5),
        sequence=sequence,
        repetitions_per_sweep=1,
        source_full_scale_mv=800.0,
        acquisition=_acquisition(
            at_segment="swept_negative",
            duration_s=100e-9,
        ),
        qcs_module=_FakeQcs,
        mapper=mapper,
        executor=Executor(),
    )

    assert len(calls) == 2
    assert result.program_summary["hardware_sweep"] is True
    assert result.program_summary["sweep_execution_mode"] == "hardware_flattened"
    assert result.program_summary["program_count"] == 1
    assert result.program_summary["executor_call_count"] == 1
    assert result.program_summary["safety_reset_executor_call_count"] == 1
    assert result.program_summary["dc_channel_offsets_v"] == pytest.approx(
        [0.100]
    )
    assert result.program_summary[
        "dc_offset_init_compensation_v"
    ] == pytest.approx([-0.800])

    arrays, variables = calls[0].sweeps[0]
    values_by_name = {
        variable.name: np.asarray(array.value)
        for array, variable in zip(arrays, variables)
    }
    compensation_values = values_by_name[
        "awg_dc_0_interval_6_amplitude"
    ]
    original_target = sequence.bias_t_compensation_preview(0)[
        0
    ].target_amplitude
    expected_first = (
        (original_target - 0.800 / 0.800) * (800.0 / 2_500.0)
        - 0.100 / 2.5
    )
    assert compensation_values[0] == pytest.approx(expected_first)

    preview = backend.qcs_sweep_execution_preview(
        sequence,
        hardware_demodulation=True,
        source_full_scale_mv=800.0,
        dc_full_scale_v=2.5,
        init_time_s=100e-6,
    )
    assert "includes that inter-shot offset area" in " ".join(
        preview.reasons
    )


def test_nonzero_offset_trace_sweep_uses_zero_offset_fixed_numeric_points():
    sequence = _swept_negative_plateau_sequence(
        plateau_cycles=300,
        ramp_cycles=600,
    )
    mapper = _PhysicalMapper("dc_gate", "digitizer")
    calls = []

    class Executor:
        def execute(self, program):
            calls.append(program)
            return np.asarray([[1.0, 2.0, 3.0]])

    result = execute_qcs_sequence(
        connection_config=_connection(
            dc_full_scale_v=2.5,
            hw_demod=False,
        ),
        sequence=sequence,
        repetitions_per_sweep=1,
        source_full_scale_mv=800.0,
        acquisition=_acquisition(
            at_segment="swept_negative",
            duration_s=100e-9,
        ),
        qcs_module=_FakeQcs,
        mapper=mapper,
        executor=Executor(),
    )

    assert len(calls) == sequence.sweep_point_count
    assert result.program_summary["hardware_sweep"] is False
    assert result.program_summary[
        "sweep_execution_mode"
    ] == "software_fixed_numeric_dc_ramp"
    assert result.program_summary["dc_channel_offsets_v"] == [0.0]
    assert mapper.offset_scalar("dc_gate").value == pytest.approx(0.0)
    assert "cannot remain active" in " ".join(
        result.program_summary["software_sweep_reasons"]
    )


def test_fixed_offset_is_not_used_without_a_common_idle_baseline():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("zero", [0.0], 32)
        .add_ramp("to_swept", 128)
        .add_set("swept", [0.05], 32)
        .add_ramp("to_fixed", 128)
        .add_set("fixed", [-0.05], 32)
        .add_amplitude_sweep("swept", "gate", 0.05, 0.10, 2)
    )
    mapper = _PhysicalMapper("dc_gate", "digitizer")
    calls = []

    class Executor:
        def execute(self, program):
            calls.append(program)
            assert mapper.offset_scalar("dc_gate").value == pytest.approx(0.0)
            return np.asarray([1.0 + 2.0j])

    result = execute_qcs_sequence(
        connection_config=_connection(),
        sequence=sequence,
        repetitions_per_sweep=1,
        acquisition=_acquisition(at_segment="swept"),
        qcs_module=_FakeQcs,
        mapper=mapper,
        executor=Executor(),
    )

    assert len(calls) == 2
    assert result.program_summary["hardware_sweep"] is False
    assert result.program_summary["sweep_execution_mode"] == (
        "software_fixed_numeric_dc_ramp"
    )


def test_fixed_offset_execution_failure_resets_physical_offset():
    mapper = _PhysicalMapper("dc_gate", "digitizer")
    offset = mapper.offset_scalar("dc_gate")
    calls = []

    class Executor:
        def execute(self, program):
            calls.append(program)
            if len(calls) == 1:
                assert offset.value == pytest.approx(0.080)
                raise RuntimeError("injected fixed-offset failure")
            assert program.name == "PulseGenerator emergency DC reset"
            assert offset.value == pytest.approx(0.0)
            return None

    with pytest.raises(RuntimeError, match="injected fixed-offset failure"):
        execute_qcs_sequence(
            connection_config=_connection(),
            sequence=_fixed_baseline_offset_sequence(),
            repetitions_per_sweep=1,
            acquisition=_acquisition(at_segment="swept_level"),
            qcs_module=_FakeQcs,
            mapper=mapper,
            executor=Executor(),
        )

    assert len(calls) == 2
    assert offset.value == pytest.approx(0.0)


def test_execute_bipolar_awg_sweep_uses_fixed_numeric_programs_and_one_executor():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("positive", [0.0125], 192)
        .add_ramp("bipolar", 128)
        .add_set("negative", [-0.075], 192)
        .add_amplitude_sweep("positive", "gate", 0.0125, 0.15, 3)
    )
    returned = [
        np.asarray([point + 1 + 1j, point + 2 + 2j])
        for point in range(3)
    ]
    calls = {"execute": 0}

    class Executor:
        def execute(self, program):
            point = calls["execute"]
            calls["execute"] += 1
            operations = _program_waveform_operations(program.waveforms[0])
            assert not any(
                isinstance(operation, _CombinedWaveform)
                for operation in operations
            )
            return returned[point]

    result = execute_qcs_sequence(
        connection_config=_connection(),
        sequence=sequence,
        repetitions_per_sweep=2,
        acquisition=_acquisition(at_segment="bipolar"),
        qcs_module=_FakeQcs,
        mapper=_Mapper("dc_gate", "digitizer"),
        executor=Executor(),
    )

    assert calls["execute"] == 3
    assert len(result.programs) == len(result.raw_results) == 3
    assert result.program_summary["hardware_sweep"] is False
    assert (
        result.program_summary["sweep_execution_mode"]
        == "software_fixed_numeric_dc_ramp"
    )
    assert result.program_summary["program_count"] == 3
    assert result.program_summary["executor_call_count"] == 3
    assert result.ddr_result.iq.shape == (3, 2, 1, 2)
    np.testing.assert_allclose(
        result.ddr_result.iq[:, :, 0, 0],
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
    )
    np.testing.assert_allclose(
        result.ddr_result.iq[:, :, 0, 1],
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
    )


def test_same_sign_nonzero_ramp_uses_fixed_numeric_programs():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("start", [0.05], 32)
        .add_ramp("same_sign", 128)
        .add_set("end", [0.10], 32)
        .add_amplitude_sweep("start", "gate", 0.05, 0.075, 2)
    )
    calls = []

    class Executor:
        def execute(self, program):
            calls.append(program)
            operations = _program_waveform_operations(program.waveforms[0])
            assert not any(
                isinstance(operation, _CombinedWaveform)
                for operation in operations
            )
            return np.asarray([1.0 + 2.0j])

    result = execute_qcs_sequence(
        connection_config=_connection(),
        sequence=sequence,
        repetitions_per_sweep=1,
        acquisition=_acquisition(at_segment="same_sign"),
        qcs_module=_FakeQcs,
        mapper=_Mapper("dc_gate", "digitizer"),
        executor=Executor(),
    )

    assert len(calls) == 2
    assert result.program_summary["sweep_execution_mode"] == (
        "software_fixed_numeric_dc_ramp"
    )


def test_public_synchronized_compiler_rejects_waveform_addition():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("start", [0.05], 32)
        .add_ramp("same_sign", 128)
        .add_set("end", [0.10], 32)
        .add_amplitude_sweep("start", "gate", 0.04, 0.06, 2)
        .add_amplitude_sweep("end", "gate", 0.09, 0.11, 2)
    )

    with pytest.raises(QcsUnsupportedFeatureError, match="waveform addition"):
        backend.compile_qcs_synchronized_sweep(
            sequence,
            connection_config=_connection(),
            mapper=_Mapper("dc_gate", "digitizer"),
            repetitions_per_sweep=1,
            acquisition=_acquisition(at_segment="same_sign"),
            qcs_module=_FakeQcs,
        )


def test_fixed_numeric_execution_failure_resets_dc_outputs_to_zero():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("positive", [0.05], 32)
        .add_ramp("bipolar", 128)
        .add_set("negative", [-0.05], 32)
        .add_amplitude_sweep("positive", "gate", 0.05, 0.10, 2)
    )
    calls = []

    class Executor:
        def execute(self, program):
            calls.append(program)
            if len(calls) == 2:
                raise RuntimeError("injected point failure")
            if program.name == "PulseGenerator emergency DC reset":
                return None
            return np.asarray([1.0 + 2.0j])

    with pytest.raises(RuntimeError, match="injected point failure"):
        execute_qcs_sequence(
            connection_config=_connection(),
            sequence=sequence,
            repetitions_per_sweep=1,
            acquisition=_acquisition(at_segment="bipolar"),
            qcs_module=_FakeQcs,
            mapper=_Mapper("dc_gate", "digitizer"),
            executor=Executor(),
        )

    assert len(calls) == 3
    reset_program = calls[-1]
    assert reset_program.name == "PulseGenerator emergency DC reset"
    assert reset_program.shots == 1
    reset_operations = _program_waveform_operations(
        reset_program.waveforms[0]
    )
    assert len(reset_operations) == 1
    assert reset_operations[0].kwargs["amplitude"] == 0.0


def test_fixed_numeric_user_stop_prevents_next_point_and_resets_dc():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("positive", [0.05], 32)
        .add_ramp("bipolar", 128)
        .add_set("negative", [-0.05], 32)
        .add_amplitude_sweep("positive", "gate", 0.05, 0.10, 3)
    )
    controller = backend.QcsCancellationController(
        program_name_tag="fixed-stop"
    )
    calls = []

    class Executor:
        def execute(self, program):
            calls.append(program)
            if program.name == "PulseGenerator emergency DC reset":
                return None
            controller.request_stop()
            return np.asarray([1.0 + 2.0j])

    with pytest.raises(backend.QcsExperimentCancelled, match="stopped by user"):
        execute_qcs_sequence(
            connection_config=_connection(),
            sequence=sequence,
            repetitions_per_sweep=1,
            acquisition=_acquisition(at_segment="bipolar"),
            qcs_module=_FakeQcs,
            mapper=_Mapper("dc_gate", "digitizer"),
            executor=Executor(),
            cancellation=controller,
        )

    assert len(calls) == 2
    assert "[qcs-gui-fixed-stop]" in calls[0].name
    assert calls[1].name == "PulseGenerator emergency DC reset"
    assert controller.wait_for_abort(1.0) is True


def test_native_user_stop_always_resets_dc_even_without_fixed_offset():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("read", [0.0], 300)
        .add_amplitude_sweep("read", "gate", 0.0, 0.10, 3)
    )
    controller = backend.QcsCancellationController(
        program_name_tag="native-stop"
    )
    calls = []

    class Executor:
        def execute(self, program):
            calls.append(program)
            if program.name == "PulseGenerator emergency DC reset":
                return None
            controller.request_stop()
            return np.zeros((3, 1), dtype=complex)

    with pytest.raises(backend.QcsExperimentCancelled, match="stopped by user"):
        execute_qcs_sequence(
            connection_config=_connection(),
            sequence=sequence,
            repetitions_per_sweep=1,
            acquisition=_acquisition(at_segment="read"),
            qcs_module=_FakeQcs,
            mapper=_Mapper("dc_gate", "digitizer"),
            executor=Executor(),
            cancellation=controller,
        )

    assert len(calls) == 2
    assert "[qcs-gui-native-stop]" in calls[0].name
    assert calls[1].name == "PulseGenerator emergency DC reset"
    assert controller.wait_for_abort(1.0) is True


def test_fixed_numeric_stop_reports_emergency_dc_reset_failure():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("positive", [0.05], 32)
        .add_ramp("bipolar", 128)
        .add_set("negative", [-0.05], 32)
        .add_amplitude_sweep("positive", "gate", 0.05, 0.10, 2)
    )
    controller = backend.QcsCancellationController(
        program_name_tag="fixed-reset-failure"
    )

    class Executor:
        def execute(self, program):
            if program.name == "PulseGenerator emergency DC reset":
                raise RuntimeError("injected reset failure")
            controller.request_stop()
            return np.asarray([1.0 + 2.0j])

    with pytest.raises(RuntimeError, match="physical output state is unknown"):
        execute_qcs_sequence(
            connection_config=_connection(),
            sequence=sequence,
            repetitions_per_sweep=1,
            acquisition=_acquisition(at_segment="bipolar"),
            qcs_module=_FakeQcs,
            mapper=_Mapper("dc_gate", "digitizer"),
            executor=Executor(),
            cancellation=controller,
        )


def test_native_stop_reports_emergency_dc_reset_failure():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("read", [0.0], 300)
        .add_amplitude_sweep("read", "gate", 0.0, 0.10, 3)
    )
    controller = backend.QcsCancellationController(
        program_name_tag="native-reset-failure"
    )

    class Executor:
        def execute(self, program):
            if program.name == "PulseGenerator emergency DC reset":
                raise RuntimeError("injected reset failure")
            controller.request_stop()
            return np.zeros((3, 1), dtype=complex)

    with pytest.raises(RuntimeError, match="physical output state is unknown"):
        execute_qcs_sequence(
            connection_config=_connection(),
            sequence=sequence,
            repetitions_per_sweep=1,
            acquisition=_acquisition(at_segment="read"),
            qcs_module=_FakeQcs,
            mapper=_Mapper("dc_gate", "digitizer"),
            executor=Executor(),
            cancellation=controller,
        )


def test_stop_during_native_result_normalization_resets_dc(monkeypatch):
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("read", [0.0], 300)
        .add_amplitude_sweep("read", "gate", 0.0, 0.10, 3)
    )
    controller = backend.QcsCancellationController(
        program_name_tag="normalization-stop"
    )
    calls = []

    class Executor:
        def execute(self, program):
            calls.append(program)
            if program.name == "PulseGenerator emergency DC reset":
                return None
            return np.zeros((3, 1), dtype=complex)

    def request_stop_during_normalization(*args, **kwargs):
        controller.request_stop()
        # Deliberately malformed: accepted cancellation must reset and win
        # before the common result-shape validation can raise instead.
        return np.zeros((1, 1), dtype=complex)

    monkeypatch.setattr(
        backend,
        "normalize_qcs_hardware_sweep_iq",
        request_stop_during_normalization,
    )

    with pytest.raises(backend.QcsExperimentCancelled, match="after acquisition"):
        execute_qcs_sequence(
            connection_config=_connection(),
            sequence=sequence,
            repetitions_per_sweep=1,
            acquisition=_acquisition(at_segment="read"),
            qcs_module=_FakeQcs,
            mapper=_Mapper("dc_gate", "digitizer"),
            executor=Executor(),
            cancellation=controller,
        )

    assert len(calls) == 2
    assert calls[-1].name == "PulseGenerator emergency DC reset"


def test_duration_sweep_uses_one_qcs_managed_software_program():
    sequence = _sequence().add_hold_duration_sweep("read", 0.5, 0.6, 2)
    returned = np.asarray(
        [
            [1 + 2j, 3 + 4j],
            [5 + 6j, 7 + 8j],
        ]
    )
    calls = {"execute": 0}

    class Executor:
        def execute(self, program):
            calls["execute"] += 1
            assert program.repetition_calls == ["shots", "sweep"]
            return returned

    result = execute_qcs_sequence(
        connection_config=_connection(),
        sequence=sequence,
        repetitions_per_sweep=2,
        acquisition=_acquisition(),
        qcs_module=_FakeQcs,
        mapper=_Mapper("dc_gate", "digitizer"),
        executor=Executor(),
    )

    assert calls["execute"] == 1
    assert result.program_summary["hardware_sweep"] is False
    assert (
        result.program_summary["sweep_execution_mode"]
        == "software_single_program"
    )
    assert result.program_summary["software_sweep_points"] == 2
    assert result.program_summary["program_count"] == 1
    assert any(
        "duration" in reason
        for reason in result.program_summary["software_sweep_reasons"]
    )
    np.testing.assert_allclose(
        result.ddr_result.iq[:, :, 0],
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
    )


def test_fixed_time_bias_t_zero_area_point_keeps_one_hardware_program():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("read", [0.0], 300)
        .add_amplitude_sweep("read", "gate", -0.1, 0.1, 3)
        .set_bias_t_compensation(
            0.1,
            mode="fixed_time",
            fixed_duration_cycles=300,
        )
    )

    compiled = backend.compile_qcs_synchronized_sweep(
        sequence,
        connection_config=_connection(),
        mapper=_Mapper("dc_gate", "digitizer"),
        repetitions_per_sweep=2,
        acquisition=_acquisition(),
        qcs_module=_FakeQcs,
    )

    assert compiled.hardware_sweep is True
    assert compiled.program.repetition_calls == ["sweep", "shots"]
    operations = _program_waveform_operations(
        compiled.program.waveforms[0]
    )
    assert [type(operation) for operation in operations] == [
        _Waveform,
        _Delay,
        _Waveform,
        _Delay,
    ]
    assert operations[1].kwargs["duration"] == pytest.approx(33 / 300e6)
    assert operations[3].kwargs["duration"] == pytest.approx(4 / 300e6)
    arrays, variables = compiled.program.sweeps[0]
    by_name = {
        variable.name: array.value
        for variable, array in zip(variables, arrays)
    }
    expected_compensation = [
        sequence.bias_t_compensation_preview(point_index)[0].target_amplitude
        * (800.0 / 2500.0)
        for point_index in range(3)
    ]
    np.testing.assert_allclose(
        by_name["awg_dc_0_interval_2_amplitude"],
        expected_compensation,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("start_cycles", "stop_cycles", "expected_hardware"),
    [
        (1, 3, False),
        (4, 6, True),
    ],
)
def test_zero_delay_duration_sweep_uses_hardware_only_above_minimum(
    start_cycles,
    stop_cycles,
    expected_hardware,
):
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("wait", [0.0], start_cycles)
        .add_hold_duration_sweep(
            "wait",
            start_cycles / 300.0,
            stop_cycles / 300.0,
            3,
        )
    )
    acquisition = _acquisition(
        at_segment="wait",
        duration_s=1 / 300e6,
        sample_count=16,
    )

    compiled = backend.compile_qcs_synchronized_sweep(
        sequence,
        connection_config=_connection(),
        mapper=_Mapper("dc_gate", "digitizer"),
        repetitions_per_sweep=2,
        acquisition=acquisition,
        qcs_module=_FakeQcs,
    )

    assert compiled.hardware_sweep is expected_hardware
    assert compiled.program.repetition_calls == (
        ["sweep", "shots"]
        if expected_hardware
        else ["shots", "sweep"]
    )
    if expected_hardware:
        assert compiled.software_sweep_reasons == ()
    else:
        assert any(
            "interval 0 duration" in reason
            for reason in compiled.software_sweep_reasons
        )


def test_short_constant_zero_delay_compiles_but_short_waveform_does_not():
    zero = FineTuneSequence(("gate",)).add_set("wait", [0.0], 1)
    compiled = backend.compile_qcs_synchronized_sweep(
        zero,
        connection_config=_connection(),
        mapper=_Mapper("dc_gate", "digitizer"),
        repetitions_per_sweep=1,
        acquisition=_acquisition(
            at_segment="wait",
            duration_s=1 / 300e6,
            sample_count=16,
        ),
        qcs_module=_FakeQcs,
    )
    operation = _program_waveform_operations(
        compiled.program.waveforms[0]
    )[0]
    assert isinstance(operation, _Delay)
    assert operation.kwargs["duration"] == pytest.approx(1 / 300e6)

    nonzero = FineTuneSequence(("gate",)).add_set("short", [0.1], 1)
    with pytest.raises(ValueError, match="M5301 waveform"):
        backend.compile_qcs_synchronized_sweep(
            nonzero,
            connection_config=_connection(),
            mapper=_Mapper("dc_gate", "digitizer"),
            repetitions_per_sweep=1,
            acquisition=_acquisition(
                at_segment="short",
                duration_s=1 / 300e6,
                sample_count=16,
            ),
            qcs_module=_FakeQcs,
        )


def test_rf_frequency_sweep_uses_one_native_hardware_program():
    sequence = _sequence().add_rf_frequency_sweep(
        "read", 3, 25.0, 75.0, 3
    )
    returned = np.asarray(
        [
            [1 + 10j, 2 + 20j, 3 + 30j],
            [4 + 40j, 5 + 50j, 6 + 60j],
        ]
    )
    calls = {"execute": 0}

    class Executor:
        def execute(self, program):
            calls["execute"] += 1
            assert program.repetition_calls == ["sweep", "shots"]
            assert len(program.sweeps) == 1
            arrays, variables = program.sweeps[0]
            assert len(arrays) == len(variables) == 1
            assert variables[0].name == "awg_rf_0_frequency"
            np.testing.assert_allclose(
                arrays[0].value,
                [25.0e6, 50.0e6, 75.0e6],
            )
            return returned

    result = execute_qcs_sequence(
        connection_config=_connection(
            rf_channel_names={3: "rf_drive"},
        ),
        sequence=sequence,
        repetitions_per_sweep=2,
        rf_pulses=(
            QcsRfPulseConfig(
                gen_ch=3,
                at_segment="read",
                duration_s=100e-9,
                amplitude=0.2,
                frequency_hz=25e6,
            ),
        ),
        acquisition=_acquisition(),
        qcs_module=_FakeQcs,
        mapper=_Mapper("dc_gate", "rf_drive", "digitizer"),
        executor=Executor(),
    )

    assert calls["execute"] == 1
    assert result.program_summary["hardware_sweep"] is True
    assert result.program_summary["sweep_execution_mode"] == "hardware_flattened"
    assert result.program_summary["executor_call_count"] == 1
    np.testing.assert_allclose(
        result.ddr_result.iq[:, :, 0],
        [
            [[1.0, 10.0], [4.0, 40.0]],
            [[2.0, 20.0], [5.0, 50.0]],
            [[3.0, 30.0], [6.0, 60.0]],
        ],
    )


def test_rf_duration_sweep_uses_one_qcs_managed_software_program():
    sequence = _sequence().add_rf_duration_sweep(
        "read", 3, 0.1, 0.2, 3
    )
    returned = np.asarray(
        [
            [1 + 10j, 2 + 20j],
            [3 + 30j, 4 + 40j],
            [5 + 50j, 6 + 60j],
        ]
    )
    calls = {"execute": 0}

    class Executor:
        def execute(self, program):
            calls["execute"] += 1
            assert program.repetition_calls == ["shots", "sweep"]
            assert len(program.sweeps) == 1
            arrays, variables = program.sweeps[0]
            assert len(arrays) == len(variables) == 1
            assert variables[0].name == "awg_rf_0_duration"
            np.testing.assert_allclose(
                arrays[0].value,
                [100e-9, 150e-9, 200e-9],
            )
            return returned

    result = execute_qcs_sequence(
        connection_config=_connection(
            rf_channel_names={3: "rf_drive"},
        ),
        sequence=sequence,
        repetitions_per_sweep=2,
        rf_pulses=(
            QcsRfPulseConfig(
                gen_ch=3,
                at_segment="read",
                duration_s=100e-9,
                amplitude=0.2,
                frequency_hz=50e6,
            ),
        ),
        acquisition=_acquisition(),
        qcs_module=_FakeQcs,
        mapper=_Mapper("dc_gate", "rf_drive", "digitizer"),
        executor=Executor(),
    )

    assert calls["execute"] == 1
    assert result.program_summary["hardware_sweep"] is False
    assert (
        result.program_summary["sweep_execution_mode"]
        == "software_single_program"
    )
    assert result.program_summary["executor_call_count"] == 1
    assert any(
        reason == "RF gen_ch 3 duration"
        for reason in result.program_summary["software_sweep_reasons"]
    )
    np.testing.assert_allclose(
        result.ddr_result.iq[:, :, 0],
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
            [[5.0, 50.0], [6.0, 60.0]],
        ],
    )


def test_same_rf_channel_uses_relative_gap_after_previous_pulse():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("prep", [0.5], 300)
        .add_set("read", [0.5], 600)
    )
    compiled = backend.compile_qcs_synchronized_sweep(
        sequence,
        connection_config=_connection(
            rf_channel_names={0: "rf_drive", 1: "rf_drive"},
        ),
        mapper=_Mapper("dc_gate", "rf_drive", "digitizer"),
        repetitions_per_sweep=1,
        rf_pulses=(
            QcsRfPulseConfig(
                gen_ch=0,
                at_segment="read",
                delay_s=200e-9,
                duration_s=100e-9,
                amplitude=0.2,
                frequency_hz=50e6,
            ),
            QcsRfPulseConfig(
                gen_ch=1,
                at_segment="prep",
                delay_s=100e-9,
                duration_s=100e-9,
                amplitude=0.2,
                frequency_hz=50e6,
            ),
        ),
        acquisition=_acquisition(at_segment="read"),
        qcs_module=_FakeQcs,
    )

    rf_entries = [
        entry
        for entry in compiled.program.waveforms
        if entry[1].name == "rf_drive"
    ]
    assert [entry[0].kwargs["name"] for entry in rf_entries] == [
        "awg_rf_1",
        "awg_rf_0",
    ]
    assert [entry[2]["new_layer"] for entry in rf_entries] == [False, False]
    assert rf_entries[0][2]["pre_delay"] == pytest.approx(100e-9)
    # The late pulse starts at 1.2 us. The first pulse ends at 0.2 us,
    # therefore QCS needs a 1.0 us relative gap, not another 1.2 us delay.
    assert rf_entries[1][2]["pre_delay"] == pytest.approx(1.0e-6)


def test_different_rf_channels_keep_independent_absolute_pre_delays():
    sequence = (
        FineTuneSequence(("gate",))
        .add_set("prep", [0.5], 300)
        .add_set("read", [0.5], 600)
    )
    compiled = backend.compile_qcs_synchronized_sweep(
        sequence,
        connection_config=_connection(
            rf_channel_names={0: "rf_late", 1: "rf_early"},
        ),
        mapper=_Mapper("dc_gate", "rf_late", "rf_early", "digitizer"),
        repetitions_per_sweep=1,
        rf_pulses=(
            QcsRfPulseConfig(
                gen_ch=0,
                at_segment="read",
                delay_s=200e-9,
                duration_s=100e-9,
                amplitude=0.2,
                frequency_hz=50e6,
            ),
            QcsRfPulseConfig(
                gen_ch=1,
                at_segment="prep",
                delay_s=100e-9,
                duration_s=100e-9,
                amplitude=0.2,
                frequency_hz=50e6,
            ),
        ),
        acquisition=_acquisition(at_segment="read"),
        qcs_module=_FakeQcs,
    )

    delays_by_channel = {
        entry[1].name: entry[2]["pre_delay"]
        for entry in compiled.program.waveforms
        if entry[1].name in {"rf_late", "rf_early"}
    }
    assert delays_by_channel["rf_late"] == pytest.approx(1.2e-6)
    assert delays_by_channel["rf_early"] == pytest.approx(100e-9)


def test_overlapping_pulses_on_one_rf_channel_are_rejected():
    sequence = FineTuneSequence(("gate",)).add_set("prep", [0.5], 300)
    with pytest.raises(ValueError, match=r"overlap.*rf_drive.*point 1"):
        backend.compile_qcs_synchronized_sweep(
            sequence,
            connection_config=_connection(
                rf_channel_names={0: "rf_drive", 1: "rf_drive"},
            ),
            mapper=_Mapper("dc_gate", "rf_drive", "digitizer"),
            repetitions_per_sweep=1,
            rf_pulses=(
                QcsRfPulseConfig(
                    gen_ch=0,
                    at_segment="prep",
                    delay_s=200e-9,
                    duration_s=200e-9,
                    amplitude=0.2,
                    frequency_hz=50e6,
                ),
                QcsRfPulseConfig(
                    gen_ch=1,
                    at_segment="prep",
                    delay_s=100e-9,
                    duration_s=200e-9,
                    amplitude=0.2,
                    frequency_hz=50e6,
                ),
            ),
            acquisition=_acquisition(at_segment="prep"),
            qcs_module=_FakeQcs,
        )


def test_hardware_array_count_budget_falls_back_to_one_software_program():
    sequence = FineTuneSequence(("gate",))
    for index in range(9):
        sequence.add_set(f"set_{index}", [0.0], 30)
    for index in range(9):
        sequence.add_amplitude_sweep(
            f"set_{index}", "gate", -0.01, 0.01, 2
        )
    point_count = sequence.sweep_point_count
    calls = {"execute": 0}

    class Executor:
        def execute(self, program):
            calls["execute"] += 1
            assert program.repetition_calls == ["shots", "sweep"]
            assert len(program.sweeps) == 1
            arrays, variables = program.sweeps[0]
            assert len(arrays) == len(variables) == 9
            assert all(array.value.shape == (point_count,) for array in arrays)
            return np.zeros((point_count, 2), dtype=complex)

    result = execute_qcs_sequence(
        connection_config=_connection(),
        sequence=sequence,
        repetitions_per_sweep=2,
        acquisition=_acquisition(at_segment="set_8"),
        qcs_module=_FakeQcs,
        mapper=_Mapper("dc_gate", "digitizer"),
        executor=Executor(),
    )

    assert calls["execute"] == 1
    assert point_count == 512
    assert result.program_summary["hardware_sweep"] is False
    assert result.program_summary["program_count"] == 1
    assert result.program_summary["executor_call_count"] == 1
    assert any(
        "uses 9 swept arrays" in reason
        and "supports at most 8" in reason
        for reason in result.program_summary["software_sweep_reasons"]
    )
    assert result.ddr_result.iq.shape == (point_count, 2, 1, 2)


def test_raw_trace_sweep_uses_one_program_and_preserves_samples():
    sequence = _sequence().add_amplitude_sweep(
        "read", "gate", 0.25, 0.5, 2
    )
    returned = np.arange(12, dtype=float).reshape(2, 2, 3)
    calls = {"execute": 0}

    class Executor:
        def execute(self, program):
            calls["execute"] += 1
            assert program.repetition_calls == ["shots", "sweep"]
            return returned

    result = execute_qcs_sequence(
        connection_config=_connection(hw_demod=False),
        sequence=sequence,
        repetitions_per_sweep=2,
        acquisition=_acquisition(),
        qcs_module=_FakeQcs,
        mapper=_Mapper("dc_gate", "digitizer"),
        executor=Executor(),
    )

    assert calls["execute"] == 1
    assert result.program_summary["hardware_sweep"] is False
    assert result.program_summary["program_count"] == 1
    assert result.ddr_result.iq.shape == (2, 2, 3, 2)
    np.testing.assert_array_equal(result.ddr_result.iq[..., 0], returned)
    np.testing.assert_array_equal(result.ddr_result.iq[..., 1], 0.0)


def test_multi_axis_cross_capacitance_bipolar_ramp_uses_safe_numeric_points():
    sequence = (
        FineTuneSequence(("x_gate", "y_gate"))
        .add_set("initial", [0.0, 0.0], 30)
        .add_ramp("ramp", 8)
        .add_set("read", [0.0, 0.0], 300)
        .add_amplitude_sweep("initial", "x_gate", -0.25, 0.25, 3)
        .add_amplitude_sweep("read", "y_gate", -0.5, 0.5, 2)
        .set_cross_capacitance([[1.0, 0.1], [-0.2, 1.0]])
    )
    raw = np.arange(12).reshape(6, 2).astype(complex)
    calls = {"execute": 0}

    class Executor:
        def execute(self, program):
            point = calls["execute"]
            calls["execute"] += 1
            assert program.repetition_calls == ["shots"]
            operations = [
                operation
                for entry in program.waveforms[:2]
                for operation in _program_waveform_operations(entry)
            ]
            assert not any(
                isinstance(operation, _CombinedWaveform)
                for operation in operations
            )
            return raw[point]

    result = execute_qcs_sequence(
        connection_config=_connection(
            dc_channel_names=("dc_x", "dc_y"),
            dc_full_scale_v=1.0,
        ),
        sequence=sequence,
        repetitions_per_sweep=2,
        acquisition=_acquisition(at_segment="read"),
        qcs_module=_FakeQcs,
        mapper=_Mapper("dc_x", "dc_y", "digitizer"),
        executor=Executor(),
    )

    assert calls["execute"] == 6
    assert result.program_summary["hardware_sweep"] is False
    assert (
        result.program_summary["sweep_execution_mode"]
        == "software_fixed_numeric_dc_ramp"
    )
    assert result.program_summary["program_count"] == 6
    assert result.program_summary["executor_call_count"] == 6
    assert result.ddr_result.sweep_shape == (3, 2)
    np.testing.assert_array_equal(
        result.ddr_result.iq[:, :, 0, 0],
        raw.real,
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
                _program_waveform_operations(waveform)[0].kwargs[
                    "amplitude"
                ]
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


def test_qcs_stability_fixed_time_compensation_is_in_same_hardware_sweep():
    sequence = _stability_sequence().set_bias_t_compensation(
        0.1,
        mode="fixed_time",
        fixed_duration_cycles=60_000,
    )
    compiled = backend.compile_qcs_stability_hardware_sweep(
        sequence,
        connection_config=_connection(
            dc_channel_names=("dc_x", "dc_y"),
            dc_full_scale_v=1.0,
        ),
        mapper=_Mapper("dc_x", "dc_y", "digitizer"),
        repetitions_per_point=4,
        fabric_mhz=300.0,
        source_full_scale_mv=800.0,
        acquisition=_acquisition(at_segment="set_0"),
        qcs_module=_FakeQcs,
    )

    program = compiled.program
    assert program.shots == 4
    assert len(program.sweeps) == 1
    assert len(program.waveforms) == 4

    measurement_x, measurement_y, compensation_x, compensation_y = (
        program.waveforms
    )
    assert measurement_x[1].name == compensation_x[1].name == "dc_x"
    assert measurement_y[1].name == compensation_y[1].name == "dc_y"
    assert compensation_x[2]["new_layer"] is True
    assert compensation_y[2]["new_layer"] is False

    measurement_x_operations = _program_waveform_operations(measurement_x)
    measurement_y_operations = _program_waveform_operations(measurement_y)
    compensation_x_operations = _program_waveform_operations(compensation_x)
    compensation_y_operations = _program_waveform_operations(compensation_y)
    for operations in (
        measurement_x_operations,
        measurement_y_operations,
        compensation_x_operations,
        compensation_y_operations,
    ):
        assert [type(operation) for operation in operations] == [
            _Waveform,
            _Hold,
        ]
        assert operations[0].kwargs["duration"] == pytest.approx(1e-6)

    compensation_duration_s = sum(
        operation.kwargs["duration"]
        for operation in compensation_x_operations
    )
    assert sum(
        operation.kwargs["duration"]
        for operation in compensation_y_operations
    ) == pytest.approx(compensation_duration_s)
    assert compensation_duration_s == pytest.approx(60_000 / 300.0e6)
    assert compensation_duration_s * 300.0e6 == pytest.approx(60_000)

    arrays, variables = program.sweeps[0]
    assert len(arrays) == len(variables) == 4
    values_by_variable = {
        id(variable): np.asarray(array.value)
        for array, variable in zip(arrays, variables)
    }
    hold_x_values = values_by_variable[
        id(measurement_x_operations[0].kwargs["amplitude"])
    ]
    hold_y_values = values_by_variable[
        id(measurement_y_operations[0].kwargs["amplitude"])
    ]
    compensation_x_values = values_by_variable[
        id(compensation_x_operations[0].kwargs["amplitude"])
    ]
    compensation_y_values = values_by_variable[
        id(compensation_y_operations[0].kwargs["amplitude"])
    ]

    np.testing.assert_allclose(
        hold_x_values,
        [-0.24, -0.16, -0.04, 0.04, 0.16, 0.24],
    )
    np.testing.assert_allclose(
        hold_y_values,
        [-0.36, 0.44, -0.4, 0.4, -0.44, 0.36],
    )
    np.testing.assert_allclose(
        compensation_x_values,
        [0.12, 0.08, 0.02, -0.02, -0.08, -0.12],
    )
    np.testing.assert_allclose(
        compensation_y_values,
        [0.18, -0.22, 0.2, -0.2, 0.22, -0.18],
    )
    assert not np.array_equal(compensation_x_values, compensation_y_values)


def test_qcs_stability_fixed_time_compensation_rejects_overrange_voltage():
    sequence = _stability_sequence().set_bias_t_compensation(
        0.1,
        mode="fixed_time",
        fixed_duration_cycles=30,
    )

    with pytest.raises(
        ValueError,
        match=r"(?i)(bias-t|compensation).*(too short|full scale|exceed)",
    ):
        backend.compile_qcs_stability_hardware_sweep(
            sequence,
            connection_config=_connection(
                dc_channel_names=("dc_x", "dc_y"),
                dc_full_scale_v=1.0,
            ),
            mapper=_Mapper("dc_x", "dc_y", "digitizer"),
            repetitions_per_point=1,
            fabric_mhz=300.0,
            source_full_scale_mv=800.0,
            acquisition=_acquisition(at_segment="set_0"),
            qcs_module=_FakeQcs,
        )


def test_qcs_stability_without_compensation_keeps_original_single_layer_sweep():
    compiled = backend.compile_qcs_stability_hardware_sweep(
        _stability_sequence(),
        connection_config=_connection(
            dc_channel_names=("dc_x", "dc_y"),
            dc_full_scale_v=1.0,
        ),
        mapper=_Mapper("dc_x", "dc_y", "digitizer"),
        repetitions_per_point=1,
        fabric_mhz=300.0,
        source_full_scale_mv=800.0,
        acquisition=_acquisition(at_segment="set_0"),
        qcs_module=_FakeQcs,
    )

    assert len(compiled.program.waveforms) == 2
    assert [entry[2]["new_layer"] for entry in compiled.program.waveforms] == [
        True,
        False,
    ]
    for entry in compiled.program.waveforms:
        operations = _program_waveform_operations(entry)
        assert [type(operation) for operation in operations] == [
            _Waveform,
            _Hold,
        ]
        assert sum(
            operation.kwargs["duration"] for operation in operations
        ) == pytest.approx(100e-6)
    arrays, variables = compiled.program.sweeps[0]
    assert len(arrays) == len(variables) == 2


def test_qcs_stability_hold_avoids_the_measured_direct_waveform_limit():
    # The connected QCS 2.5.5 sandbox accepts 98,304 M5301 samples but
    # rejects the next valid 16-sample increment (98,320 samples). Verify
    # that an interval beyond that boundary renders only a short seed.
    duration_s = 98_320 / 2.4e9
    amplitude = _Scalar("swept_amplitude", value=0.0, dtype=float)

    operations = backend._qcs_constant_dc_interval(
        _FakeQcs,
        duration_s=duration_s,
        amplitude=amplitude,
        name="limit_probe",
        fabric_hz=300e6,
    )

    assert [type(operation) for operation in operations] == [
        _Waveform,
        _Hold,
    ]
    assert operations[0].kwargs["amplitude"] is amplitude
    assert operations[0].kwargs["duration"] == pytest.approx(1e-6)
    assert sum(
        operation.kwargs["duration"] for operation in operations
    ) == pytest.approx(duration_s)
    assert all(
        operation.kwargs["duration"] * 300e6
        == pytest.approx(
            round(operation.kwargs["duration"] * 300e6)
        )
        for operation in operations
    )


def test_qcs_constant_dc_interval_preserves_short_timing_boundaries():
    amplitude = _Scalar("amplitude", value=0.0, dtype=float)

    direct = backend._qcs_constant_dc_interval(
        _FakeQcs,
        duration_s=300 / 300e6,
        amplitude=amplitude,
        name="direct",
        fabric_hz=300e6,
    )
    held = backend._qcs_constant_dc_interval(
        _FakeQcs,
        duration_s=301 / 300e6,
        amplitude=amplitude,
        name="held",
        fabric_hz=300e6,
    )

    assert type(direct) is _Waveform
    assert direct.kwargs["duration"] == pytest.approx(1e-6)
    assert [type(operation) for operation in held] == [_Waveform, _Hold]
    assert held[0].kwargs["duration"] == pytest.approx(1e-6)
    assert held[1].kwargs["duration"] == pytest.approx(1 / 300e6)
    assert sum(
        operation.kwargs["duration"] for operation in held
    ) == pytest.approx(301 / 300e6)

    with pytest.raises(ValueError, match="at least 4 QCS fabric cycles"):
        backend._qcs_constant_dc_interval(
            _FakeQcs,
            duration_s=3 / 300e6,
            amplitude=amplitude,
            name="too_short",
            fabric_hz=300e6,
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


def test_qcs_stability_hardware_sweep_requires_300_mhz_fabric():
    with pytest.raises(ValueError, match="300 MHz synchronization fabric"):
        backend.compile_qcs_stability_hardware_sweep(
            _stability_sequence(),
            connection_config=_connection(
                dc_channel_names=("dc_x", "dc_y"),
            ),
            mapper=_Mapper("dc_x", "dc_y", "digitizer"),
            repetitions_per_point=1,
            fabric_mhz=250.0,
            acquisition=_acquisition(at_segment="set_0"),
            qcs_module=_FakeQcs,
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


def test_flat_software_sweep_iq_is_point_major_after_shape_is_stripped():
    point_shot = np.asarray(
        [
            [1 + 10j, 2 + 20j],
            [3 + 30j, 4 + 40j],
            [5 + 50j, 6 + 60j],
        ]
    )

    normalized = normalize_qcs_hardware_sweep_iq(
        point_shot.reshape(-1),
        repetitions_per_point=2,
        sweep_shape=(3,),
        hardware_sweep=False,
    )

    assert normalized.shape == (3, 2, 1, 2)
    np.testing.assert_array_equal(normalized[:, :, 0, 0], point_shot.real)
    np.testing.assert_array_equal(normalized[:, :, 0, 1], point_shot.imag)


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
            sample_count=9_600,
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
    assert acquisition_operation.duration.value == pytest.approx(2e-6)
    assert raw_compiled.duration_s == pytest.approx(2e-6)
    assert raw_compiled.program.layers[0].duration().value == pytest.approx(
        2e-6
    )
    assert raw_compiled.acquisition_sample_rate_hz == pytest.approx(4.8e9)
    rendered_program, _layer_map = qcs.SequenceBuilder(
        channel_map=mapper
    ).build(raw_compiled.program)
    assert len(rendered_program.layers) == 1
    assert rendered_program.layers[0].duration().value == pytest.approx(2e-6)
    rendered_dc_operations = rendered_program.layers[0].operations[dc]
    assert isinstance(rendered_dc_operations[-1], qcs.Delay)
    assert rendered_dc_operations[-1].duration.value == pytest.approx(1e-6)

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
        operations = compiled.program.layers[0].operations[channel]
        assert len(operations) == 2
        waveform, hold = operations
        assert isinstance(waveform, qcs.DCWaveform)
        assert isinstance(hold, qcs.Hold)
        assert len(waveform.amplitudes) == 1
        assert isinstance(waveform.amplitudes[0], qcs.Scalar)
        assert waveform.duration.value == pytest.approx(1e-6)
        assert waveform.duration.value + hold.duration.value == pytest.approx(
            100e-6
        )

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


def test_real_qcs_255_builds_fixed_time_stability_compensation_layer():
    qcs = pytest.importorskip("keysight.qcs")
    dc_x = qcs.Channels(0, "dc_x", absolute_phase=False)
    dc_y = qcs.Channels(0, "dc_y", absolute_phase=False)
    digitizer = qcs.Channels(0, "digitizer", absolute_phase=True)
    mapper = qcs.ChannelMapper()
    mapper.add_channel_mapping(
        dc_x, [(1, 7, 1)], qcs.InstrumentEnum.M5301AWG
    )
    mapper.add_channel_mapping(
        dc_y, [(1, 7, 2)], qcs.InstrumentEnum.M5301AWG
    )
    mapper.add_channel_mapping(
        digitizer,
        [(1, 18, 1)],
        qcs.InstrumentEnum.M5200Digitizer,
    )
    sequence = _stability_sequence().set_bias_t_compensation(
        0.1,
        mode="fixed_time",
        fixed_duration_cycles=60_000,
    )

    compiled = backend.compile_qcs_stability_hardware_sweep(
        sequence,
        connection_config=_connection(
            dc_channel_names=("dc_x", "dc_y"),
            dc_full_scale_v=1.0,
        ),
        mapper=mapper,
        repetitions_per_point=2,
        fabric_mhz=300.0,
        source_full_scale_mv=800.0,
        acquisition=_acquisition(at_segment="set_0"),
        qcs_module=qcs,
    )

    assert len(compiled.program.layers) == 2
    compensation_layer = compiled.program.layers[1]
    assert set(compensation_layer.operations) == {dc_x, dc_y}
    compensation_operations = [
        compensation_layer.operations[channel]
        for channel in (dc_x, dc_y)
    ]
    for operations in compensation_operations:
        assert len(operations) == 2
        waveform, hold = operations
        assert isinstance(waveform, qcs.DCWaveform)
        assert isinstance(hold, qcs.Hold)
        assert isinstance(waveform.amplitudes[0], qcs.Scalar)
        assert waveform.duration.value == pytest.approx(1e-6)
        total_duration_s = waveform.duration.value + hold.duration.value
        assert total_duration_s == pytest.approx(60_000 / 300.0e6)
        assert total_duration_s * 300.0e6 == pytest.approx(60_000)

    sweep = compiled.program.repetitions.items[1]
    association_by_name = {
        variable.name: array.value
        for variable, array in sweep.associations.items()
    }
    compensation_values = [
        association_by_name[operations[0].amplitudes[0].name]
        for operations in compensation_operations
    ]
    np.testing.assert_allclose(
        compensation_values[0],
        [0.12, 0.08, 0.02, -0.02, -0.08, -0.12],
    )
    np.testing.assert_allclose(
        compensation_values[1],
        [0.18, -0.22, 0.2, -0.2, 0.22, -0.18],
    )
    assert len(sweep.associations) == 4

    generated, _ = qcs.SequenceBuilder(channel_map=mapper).build(
        compiled.program
    )
    assert len(generated.layers) == 1


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


def test_generated_qcs_code_uses_delay_for_long_zero_interval():
    qcs = pytest.importorskip("keysight.qcs")
    pulse = PulseSequence(0.0, 50_000.0)

    code = generate_qcs_program_code(
        (pulse,),
        channel_names=("gate",),
    )
    assert "qcs.Delay(" in code
    assert "qcs.Hold(" not in code
    namespace = {}
    exec(code, namespace)
    channel = qcs.Channels(0, "gate")
    program = namespace["generate_dc_waveforms"](
        qcs.Program(),
        channel,
    )
    operations = program.layers[0].operations[channel]
    assert [type(operation) for operation in operations] == [qcs.Delay]
    assert sum(
        operation.duration.value for operation in operations
    ) == pytest.approx(50e-6)


def test_generated_qcs_code_rejects_long_nonzero_hold():
    pulse = PulseSequence(25.0, 50_000.0)

    with pytest.raises(
        ValueError,
        match=r"120,000 rendered M5301 samples.*Hold.*did not preserve",
    ):
        generate_qcs_program_code(
            (pulse,),
            channel_names=("gate",),
        )


def test_generated_qcs_code_rejects_ramp_beyond_aggregate_hcl_buffer():
    pulse = PulseSequence(0.0, 1_000.0)
    pulse.add_flat_ramp(41_000.0, 1_000.0, 100.0)

    with pytest.raises(
        ValueError,
        match=r"rendered M5301 samples.*98,304-sample.*concatenating",
    ):
        generate_qcs_program_code(
            (pulse,),
            channel_names=("gate",),
        )
