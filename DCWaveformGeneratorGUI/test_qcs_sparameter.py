"""Focused tests for the QCS-resolved RF S-parameter sweep."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5 import QtWidgets

try:
    from .qcs_qcodes_experiment import (
        QCS_M5200_SAMPLE_RATE_HZ,
        QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES,
        QCS_SPARAMETER_INTER_SEGMENT_DELAY_S,
        QCS_SPARAMETER_MAX_INTEGRATION_DURATION_S,
        QcsConnectionConfig,
        QcodesRunConfig,
        compile_qcs_sparameter_sweep,
        execute_qcs_sparameter_sweep,
        run_qcs_sparameter_sweep,
    )
except ImportError:
    from qcs_qcodes_experiment import (
        QCS_M5200_SAMPLE_RATE_HZ,
        QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES,
        QCS_SPARAMETER_INTER_SEGMENT_DELAY_S,
        QCS_SPARAMETER_MAX_INTEGRATION_DURATION_S,
        QcsConnectionConfig,
        QcodesRunConfig,
        compile_qcs_sparameter_sweep,
        execute_qcs_sparameter_sweep,
        run_qcs_sparameter_sweep,
    )

try:
    from .qick_sparameter_sweep import (
        SAMPLE_INDEX_PARAMETER,
        SParameterSweepConfig,
        load_sparameter_run,
    )
except ImportError:
    from qick_sparameter_sweep import (
        SAMPLE_INDEX_PARAMETER,
        SParameterSweepConfig,
        load_sparameter_run,
    )

try:
    from .sparameter_gui import SParameterSweepPanel
except ImportError:
    from sparameter_gui import SParameterSweepPanel

try:
    from .qcs_rf_power_calibration import (
        M5200PowerReference,
        QcsPhysicalChannelIdentity,
        REFERENCE_CALIBRATED,
        store_m5300_power_calibration,
    )
except ImportError:
    from qcs_rf_power_calibration import (
        M5200PowerReference,
        QcsPhysicalChannelIdentity,
        REFERENCE_CALIBRATED,
        store_m5300_power_calibration,
    )


def _real_qcs_mapper(qcs):
    mapper = qcs.ChannelMapper()
    dc = qcs.Channels(0, "dc_gate")
    rf = qcs.Channels(0, "rf_probe", absolute_phase=True)
    digitizer = qcs.Channels(0, "digitizer", absolute_phase=True)
    mapper.add_channel_mapping(
        dc, [(1, 7, 1)], qcs.InstrumentEnum.M5301AWG
    )
    mapper.add_channel_mapping(
        rf, [(1, 4, 1)], qcs.InstrumentEnum.M5300AWG
    )
    mapper.set_lo_frequencies(qcs.Address(1, 4, 1), 1.2e9)
    mapper.add_channel_mapping(
        digitizer,
        [(1, 18, 1)],
        qcs.InstrumentEnum.M5200Digitizer,
    )
    return mapper, rf, digitizer


def _connection() -> QcsConnectionConfig:
    return QcsConnectionConfig(
        mapper_path="unused.qcs",
        dc_channel_names=("dc_gate",),
        rf_channel_names={0: "rf_probe"},
        acquisition_channel_name="digitizer",
        hw_demod=True,
        init_time_s=1e-6,
        blocking=True,
    )


def test_real_qcs_builds_one_shared_frequency_qcs_resolved_sweep():
    qcs = pytest.importorskip("keysight.qcs")
    mapper, rf, digitizer = _real_qcs_mapper(qcs)
    frequencies = np.asarray([1.24e9, 1.25e9, 1.26e9])

    compiled = compile_qcs_sparameter_sweep(
        connection_config=_connection(),
        mapper=mapper,
        frequencies_hz=frequencies,
        rf_gen_ch=0,
        rf_amplitude=0.005,
        integration_duration_s=1.0001e-6,
        repetitions_per_point=2,
        qcs_module=qcs,
    )

    assert compiled.integration_sample_count == 4_816
    assert compiled.integration_duration_s == pytest.approx(
        4_816 / QCS_M5200_SAMPLE_RATE_HZ
    )
    assert compiled.program.repetitions.shape == (3, 2)
    assert [
        type(item).__name__ for item in compiled.program.repetitions.items
    ] == ["Sweep", "Repeat"]
    sweep = compiled.program.repetitions.items[0]
    assert list(sweep.associations)[0].name == "sparameter_frequency_hz"
    np.testing.assert_allclose(
        next(iter(sweep.associations.values())).value,
        frequencies,
    )

    output = compiled.program.layers[0].operations[rf][0]
    acquisition = compiled.program.layers[0].operations[digitizer][0]
    integration_filter = acquisition.integration_filter
    assert isinstance(output, qcs.RFWaveform)
    assert isinstance(integration_filter, qcs.IntegrationFilter)
    assert output.rf_frequency.name == "sparameter_frequency_hz"
    assert (
        integration_filter.waveforms[0].rf_frequency.name
        == "sparameter_frequency_hz"
    )
    assert output.amplitudes[0].value == pytest.approx(0.005)
    assert integration_filter.waveforms[0].amplitudes[0].value == pytest.approx(
        1.0
    )

    generated, _layer_map = qcs.SequenceBuilder(channel_map=mapper).build(
        compiled.program
    )
    assert len(generated.layers) == 1


def test_qcs_sparameter_pairs_frequency_with_calibrated_amplitude():
    qcs = pytest.importorskip("keysight.qcs")
    mapper, rf, _digitizer = _real_qcs_mapper(qcs)
    frequencies = np.asarray([1.24e9, 1.25e9, 1.26e9])
    amplitudes = np.asarray([0.011, 0.012, 0.013])

    compiled = compile_qcs_sparameter_sweep(
        connection_config=_connection(),
        mapper=mapper,
        frequencies_hz=frequencies,
        rf_gen_ch=0,
        rf_amplitude=amplitudes,
        integration_duration_s=1e-6,
        repetitions_per_point=2,
        calibrated_output=True,
        qcs_module=qcs,
    )

    np.testing.assert_allclose(compiled.rf_amplitudes, amplitudes)
    assert compiled.amplitude_variable.name == "sparameter_relative_amplitude"
    sweep = compiled.program.repetitions.items[0]
    associations = {
        variable.name: values.value
        for variable, values in sweep.associations.items()
    }
    np.testing.assert_allclose(
        associations["sparameter_frequency_hz"], frequencies
    )
    np.testing.assert_allclose(
        associations["sparameter_relative_amplitude"], amplitudes
    )
    output = compiled.program.layers[0].operations[rf][0]
    assert output.amplitudes[0].name == "sparameter_relative_amplitude"
    generated, _layer_map = qcs.SequenceBuilder(channel_map=mapper).build(
        compiled.program
    )
    assert len(generated.layers) == 1


@pytest.mark.parametrize(
    "amplitudes, message",
    [
        ([0.1, 0.2], "one value for every frequency"),
        ([0.1, 0.0, 0.2], "must be in \\(0, 1\\]"),
        ([0.1, 1.01, 0.2], "must be in \\(0, 1\\]"),
    ],
)
def test_qcs_sparameter_rejects_invalid_calibrated_amplitude_schedule(
    amplitudes,
    message,
):
    qcs = pytest.importorskip("keysight.qcs")
    mapper, _rf, _digitizer = _real_qcs_mapper(qcs)
    with pytest.raises(ValueError, match=message):
        compile_qcs_sparameter_sweep(
            connection_config=_connection(),
            mapper=mapper,
            frequencies_hz=(1.24e9, 1.25e9, 1.26e9),
            rf_gen_ch=0,
            rf_amplitude=amplitudes,
            integration_duration_s=1e-6,
            calibrated_output=True,
            qcs_module=qcs,
        )


def test_qcs_sparameter_executes_once_and_normalizes_frequency_major_iq():
    qcs = pytest.importorskip("keysight.qcs")
    mapper, _rf, _digitizer = _real_qcs_mapper(qcs)
    compiled = compile_qcs_sparameter_sweep(
        connection_config=_connection(),
        mapper=mapper,
        frequencies_hz=(1.24e9, 1.25e9, 1.26e9),
        rf_gen_ch=0,
        rf_amplitude=0.005,
        integration_duration_s=1e-6,
        repetitions_per_point=2,
        qcs_module=qcs,
    )

    class Executor:
        def __init__(self):
            self.calls = []

        def execute(self, program):
            self.calls.append(program)
            # QCS software-resolved order is (frequency, shot).
            values = np.asarray(
                [
                    [1 + 10j, 4 + 40j],
                    [2 + 20j, 5 + 50j],
                    [3 + 30j, 6 + 60j],
                ]
            )

            class RawResult:
                def __getitem__(self, _channels):
                    return values

            return RawResult()

    executor = Executor()
    result = execute_qcs_sparameter_sweep(
        connection_config=_connection(),
        frequencies_hz=compiled.frequencies_hz,
        rf_gen_ch=0,
        rf_amplitude=0.005,
        integration_duration_s=1e-6,
        repetitions_per_point=2,
        qcs_module=qcs,
        mapper=mapper,
        executor=executor,
        compiled=compiled,
    )

    assert executor.calls == [compiled.program]
    assert result.iq.shape == (3, 2, 1, 2)
    np.testing.assert_allclose(result.iq[0, :, 0, 0], [1.0, 4.0])
    np.testing.assert_allclose(result.iq[2, :, 0, 1], [30.0, 60.0])
    assert result.program_summary["hardware_sweep"] is False
    assert result.program_summary["qcs_software_resolved_sweep"] is True
    assert result.program_summary["executor_call_count"] == 1
    np.testing.assert_allclose(
        result.program_summary["rf_relative_amplitudes"],
        [0.005, 0.005, 0.005],
    )
    np.testing.assert_allclose(
        result.rf_settings["output"]["relative_amplitudes"],
        [0.005, 0.005, 0.005],
    )
    assert (
        result.program_summary[
            "frequency_scalar_shared_with_integration_filter"
        ]
        is True
    )


def test_qcs_sparameter_decodes_each_segment_and_returns_one_weighted_iq():
    qcs = pytest.importorskip("keysight.qcs")
    mapper, _rf, _digitizer = _real_qcs_mapper(qcs)
    compiled = compile_qcs_sparameter_sweep(
        connection_config=_connection(),
        mapper=mapper,
        frequencies_hz=(1.24e9, 1.25e9, 1.26e9),
        rf_gen_ch=0,
        rf_amplitude=0.005,
        integration_duration_s=100e-6,
        repetitions_per_point=2,
        qcs_module=qcs,
    )

    class Results:
        def __init__(self):
            self.indices = []

        def get_iq(self, channels, *, avg, acq_index):
            assert avg is False
            self.indices.append(acq_index)
            frequency_shot = np.asarray(
                [
                    [1 + 10j, 4 + 40j],
                    [2 + 20j, 5 + 50j],
                    [3 + 30j, 6 + 60j],
                ]
            )
            return {channels: frequency_shot + acq_index}

    class Executor:
        def __init__(self):
            self.calls = []
            self.results = Results()

        def execute(self, program):
            self.calls.append(program)
            return SimpleNamespace(results=self.results)

    executor = Executor()
    result = execute_qcs_sparameter_sweep(
        connection_config=_connection(),
        frequencies_hz=compiled.frequencies_hz,
        rf_gen_ch=0,
        rf_amplitude=0.005,
        integration_duration_s=100e-6,
        repetitions_per_point=2,
        qcs_module=qcs,
        mapper=mapper,
        executor=executor,
        compiled=compiled,
    )

    assert executor.calls == [compiled.program]
    assert executor.results.indices == list(range(15))
    assert result.iq.shape == (3, 2, 1, 2)
    # Equal 32,000-sample filters make the segment mean 0..14 = 7.
    np.testing.assert_allclose(result.iq[0, :, 0, 0], [8.0, 11.0])
    np.testing.assert_allclose(result.iq[2, :, 0, 1], [30.0, 60.0])
    assert result.program_summary["integration_segment_count"] == 15
    assert result.program_summary["total_inter_segment_dead_time_s"] == (
        pytest.approx(14 * QCS_SPARAMETER_INTER_SEGMENT_DELAY_S)
    )
    assert result.program_summary["executor_call_count"] == 1


def test_qcs_sparameter_rejects_non_demodulated_connection():
    qcs = pytest.importorskip("keysight.qcs")
    mapper, _rf, _digitizer = _real_qcs_mapper(qcs)
    connection = QcsConnectionConfig(
        mapper_path=str(Path("unused.qcs")),
        dc_channel_names=("dc_gate",),
        rf_channel_names={0: "rf_probe"},
        acquisition_channel_name="digitizer",
        hw_demod=False,
    )
    with pytest.raises(ValueError, match="requires hardware demodulation"):
        compile_qcs_sparameter_sweep(
            connection_config=connection,
            mapper=mapper,
            frequencies_hz=(1.24e9, 1.25e9),
            rf_gen_ch=0,
            rf_amplitude=0.005,
            integration_duration_s=1e-6,
            qcs_module=qcs,
        )


def test_qcs_sparameter_segments_above_single_filter_limit():
    qcs = pytest.importorskip("keysight.qcs")
    mapper, rf, digitizer = _real_qcs_mapper(qcs)
    compiled = compile_qcs_sparameter_sweep(
        connection_config=_connection(),
        mapper=mapper,
        frequencies_hz=(1.24e9, 1.25e9),
        rf_gen_ch=0,
        rf_amplitude=0.005,
        integration_duration_s=100e-6,
        qcs_module=qcs,
    )

    assert compiled.quantized_requested_integration_sample_count == 480_000
    assert compiled.integration_sample_count == 480_000
    assert compiled.integration_segment_sample_counts == (32_000,) * 15
    assert len(compiled.program.layers) == 1
    assert float(compiled.program.duration().value) == pytest.approx(
        100e-6 + 14 * QCS_SPARAMETER_INTER_SEGMENT_DELAY_S
    )
    filter_names = set()
    layer = compiled.program.layers[0]
    outputs = list(layer.operations[rf])
    acquisitions = [
        operation
        for operation in layer.operations[digitizer]
        if isinstance(operation, qcs.Acquisition)
    ]
    delays = [
        operation
        for operation in layer.operations[digitizer]
        if isinstance(operation, qcs.Delay)
    ]
    assert len(outputs) == 15
    assert len(acquisitions) == 15
    assert len(delays) == 14
    for segment_index, (output, acquisition) in enumerate(
        zip(outputs, acquisitions)
    ):
        filter_waveform = acquisition.integration_filter.waveforms[0]
        filter_names.add(filter_waveform.name)
        expected_output_s = 32_000 / QCS_M5200_SAMPLE_RATE_HZ
        if segment_index < 14:
            expected_output_s += QCS_SPARAMETER_INTER_SEGMENT_DELAY_S
        assert float(output.duration.value) == pytest.approx(expected_output_s)
        assert float(filter_waveform.duration.value) == pytest.approx(
            32_000 / QCS_M5200_SAMPLE_RATE_HZ
        )
    assert len(filter_names) == 1


def test_qcs_sparameter_balances_remainder_into_identical_filters():
    qcs = pytest.importorskip("keysight.qcs")
    mapper, _rf, _digitizer = _real_qcs_mapper(qcs)
    compiled = compile_qcs_sparameter_sweep(
        connection_config=_connection(),
        mapper=mapper,
        frequencies_hz=(1.24e9, 1.25e9),
        rf_gen_ch=0,
        rf_amplitude=0.005,
        integration_duration_s=7.0001e-6,
        qcs_module=qcs,
    )

    assert compiled.quantized_requested_integration_sample_count == 33_616
    assert compiled.integration_segment_sample_counts == (16_816, 16_816)
    assert compiled.integration_sample_count == 33_632
    assert compiled.integration_duration_s == pytest.approx(
        33_632 / QCS_M5200_SAMPLE_RATE_HZ
    )


@pytest.mark.parametrize("absolute_phase", [False, True])
def test_qcs_sparameter_segmented_accepts_matching_phase_modes(absolute_phase):
    qcs = pytest.importorskip("keysight.qcs")
    mapper = qcs.ChannelMapper()
    dc = qcs.Channels(0, "dc_gate")
    rf = qcs.Channels(0, "rf_probe", absolute_phase=absolute_phase)
    digitizer = qcs.Channels(
        0,
        "digitizer",
        absolute_phase=absolute_phase,
    )
    mapper.add_channel_mapping(dc, [(1, 7, 1)], qcs.InstrumentEnum.M5301AWG)
    mapper.add_channel_mapping(rf, [(1, 4, 1)], qcs.InstrumentEnum.M5300AWG)
    mapper.add_channel_mapping(
        digitizer,
        [(1, 18, 2)],
        qcs.InstrumentEnum.M5200Digitizer,
    )

    compiled = compile_qcs_sparameter_sweep(
        connection_config=_connection(),
        mapper=mapper,
        frequencies_hz=(1.24e9, 1.25e9),
        rf_gen_ch=0,
        rf_amplitude=0.005,
        integration_duration_s=100e-6,
        qcs_module=qcs,
    )
    assert len(compiled.integration_segment_sample_counts) == 15


@pytest.mark.parametrize("rf_phase, acquisition_phase", [(True, False), (False, True)])
def test_qcs_sparameter_segmented_rejects_mismatched_phase_modes(
    rf_phase,
    acquisition_phase,
):
    qcs = pytest.importorskip("keysight.qcs")
    mapper = qcs.ChannelMapper()
    dc = qcs.Channels(0, "dc_gate")
    rf = qcs.Channels(0, "rf_probe", absolute_phase=rf_phase)
    digitizer = qcs.Channels(
        0,
        "digitizer",
        absolute_phase=acquisition_phase,
    )
    mapper.add_channel_mapping(dc, [(1, 7, 1)], qcs.InstrumentEnum.M5301AWG)
    mapper.add_channel_mapping(rf, [(1, 4, 1)], qcs.InstrumentEnum.M5300AWG)
    mapper.add_channel_mapping(
        digitizer,
        [(1, 18, 2)],
        qcs.InstrumentEnum.M5200Digitizer,
    )

    with pytest.raises(ValueError, match="matching absolute_phase settings"):
        compile_qcs_sparameter_sweep(
            connection_config=_connection(),
            mapper=mapper,
            frequencies_hz=(1.24e9, 1.25e9),
            rf_gen_ch=0,
            rf_amplitude=0.005,
            integration_duration_s=100e-6,
            qcs_module=qcs,
        )


def test_qcs_sparameter_rejects_above_hardware_tested_aggregate_limit():
    qcs = pytest.importorskip("keysight.qcs")
    mapper, _rf, _digitizer = _real_qcs_mapper(qcs)
    with pytest.raises(ValueError, match="480,000 integrated samples"):
        compile_qcs_sparameter_sweep(
            connection_config=_connection(),
            mapper=mapper,
            frequencies_hz=(1.24e9, 1.25e9),
            rf_gen_ch=0,
            rf_amplitude=0.005,
            integration_duration_s=100.1e-6,
            qcs_module=qcs,
        )


def test_qcs_panel_uses_segmented_integration_default_and_maximum():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = SParameterSweepPanel()
    panel.set_hardware_backend("qcs")

    assert panel.scan_time_us.value() == pytest.approx(1.0)
    assert panel.scan_time_us.maximum() == pytest.approx(100_000.0)
    assert panel.scan_time_label.text() == "Total I/Q averaging time:"
    assert "bounded QCS passes" in panel.scan_time_us.toolTip()
    assert panel.stop_button.isHidden() is False
    panel.show()
    application.processEvents()
    assert panel.stop_button.isVisible() is True
    assert panel.stop_button.isEnabled() is False
    stopped = []
    panel.stop_requested.connect(lambda: stopped.append(True))
    panel.set_running(True, "QCS sweep running")
    assert panel.stop_button.isEnabled() is True
    panel.stop_button.click()
    assert stopped == [True]
    panel.set_stopping()
    assert panel.stop_button.isEnabled() is False
    panel.set_running(False, "Ready")

    panel.scan_time_us.setValue(100_001.0)
    assert panel.scan_time_us.value() == pytest.approx(100_000.0)
    panel.deleteLater()
    application.processEvents()


def test_qcs_result_status_describes_integrated_shots_not_fir_samples():
    class RunId:
        def __init__(self):
            self.value = None

        def setValue(self, value):
            self.value = int(value)

    class PanelAdapter:
        def __init__(self):
            self.run_id = RunId()
            self.running_update = None

        def set_running(self, running, message):
            self.running_update = (bool(running), str(message))

    panel = PanelAdapter()
    stored = SimpleNamespace(
        run_id=17,
        database_path=Path("qcs_sparameter.db"),
        result=SimpleNamespace(
            frequencies_mhz=np.asarray([1_240.0, 1_250.0, 1_260.0]),
            sample_count=4,
        ),
        rf_settings={
            "backend": "qcs",
            "readout": {"integration_duration_s": 10e-6},
        },
    )

    SParameterSweepPanel.show_result(panel, stored)

    assert panel.run_id.value == 17
    assert panel.running_update is not None
    _running, message = panel.running_update
    assert "4 integrated I/Q shot(s) per point" in message
    assert "10 us total I/Q averaging" in message
    assert "FIR" not in message


def test_qcs_sparameter_database_round_trip_preserves_backend_and_shot_axis(
    tmp_path,
):
    qcs = pytest.importorskip("keysight.qcs")
    mapper, _rf, _digitizer = _real_qcs_mapper(qcs)

    class Executor:
        def execute(self, _program):
            values = np.asarray(
                [
                    [1 + 10j, 4 + 40j],
                    [2 + 20j, 5 + 50j],
                    [3 + 30j, 6 + 60j],
                ]
            )

            class RawResult:
                def __getitem__(self, _channels):
                    return values

            return RawResult()

    database_path = tmp_path / "qcs_sparameter.db"
    stored = run_qcs_sparameter_sweep(
        connection_config=_connection(),
        run_config=QcodesRunConfig(
            database_path=database_path,
            experiment_name="QCS S-parameter test",
            sample_name="loopback",
        ),
        sweep_config=SParameterSweepConfig(
            frequency_start_mhz=1240.0,
            frequency_end_mhz=1260.0,
            frequency_points=3,
            scan_time_us=1.0,
        ),
        rf_gen_ch=0,
        rf_amplitude=0.005,
        repetitions_per_point=2,
        qcs_module=qcs,
        mapper=mapper,
        executor=Executor(),
    )

    assert stored.result.iq_traces.shape == (3, 2, 2)
    assert stored.rf_settings["backend"] == "qcs"
    metadata = json.loads(
        stored.dataset.get_metadata("sparameter_experiment_json")
    )
    assert metadata["sample_axis"] == "hardware-demodulated repetition"
    paramspec = stored.dataset.paramspecs[SAMPLE_INDEX_PARAMETER]
    assert paramspec.label == "Integrated I/Q shot index"

    loaded = load_sparameter_run(database_path, stored.run_id)
    assert loaded.rf_settings["backend"] == "qcs"
    np.testing.assert_allclose(loaded.result.iq_traces, stored.result.iq_traces)


def test_qcs_sparameter_applies_distinct_m5300_power_calibration(tmp_path):
    qcs = pytest.importorskip("keysight.qcs")
    mapper, _rf, _digitizer = _real_qcs_mapper(qcs)
    calibration_path = tmp_path / "qcs_m5300_power.db"
    frequencies = np.asarray([1.24e9, 1.25e9, 1.26e9])
    amplitudes = np.asarray([0.01, 0.1])
    powers = np.asarray(
        [
            [-40.0, -20.0],
            [-42.0, -22.0],
            [-44.0, -24.0],
        ]
    )
    store_m5300_power_calibration(
        calibration_path,
        frequencies_hz=frequencies,
        relative_amplitudes=amplitudes,
        mean_i=np.ones((3, 2)),
        mean_q=np.zeros((3, 2)),
        power_dbm=powers,
        output_identity=QcsPhysicalChannelIdentity(1, 1, 4, 1, "M5300A"),
        input_identity=QcsPhysicalChannelIdentity(1, 1, 18, 1, "M5200A"),
        mapper_sha256="0" * 64,
        lo_frequency_hz=1.2e9,
        integration_duration_s=1e-6,
        repetitions=2,
        input_reference=M5200PowerReference(
            mode=REFERENCE_CALIBRATED,
            intercept_dbm=0.0,
        ),
    )

    class Executor:
        def __init__(self):
            self.program = None

        def execute(self, program):
            self.program = program
            values = np.ones((3, 2), dtype=complex)

            class RawResult:
                def __getitem__(self, _channels):
                    return values

            return RawResult()

    executor = Executor()
    stored = run_qcs_sparameter_sweep(
        connection_config=_connection(),
        run_config=QcodesRunConfig(
            database_path=tmp_path / "measurement.db",
            experiment_name="calibrated QCS S-parameter test",
            sample_name="loopback",
        ),
        sweep_config=SParameterSweepConfig(
            frequency_start_mhz=1240.0,
            frequency_end_mhz=1260.0,
            frequency_points=3,
            scan_time_us=1.0,
            power_calibration_enabled=True,
            calibration_database_path=str(calibration_path),
            output_power_dbm=-30.0,
        ),
        rf_gen_ch=0,
        rf_amplitude=0.005,
        repetitions_per_point=2,
        qcs_module=qcs,
        mapper=mapper,
        executor=executor,
    )

    calibration = stored.rf_settings["output"]["power_calibration"]
    assert calibration["schema"].endswith("power-calibration-v1")
    assert calibration["termination_ohm"] == pytest.approx(50.0)
    assert calibration["target_power_dbm"] == pytest.approx(-30.0)
    np.testing.assert_allclose(
        stored.rf_settings["output"]["relative_amplitudes"],
        10.0 ** ((-30.0 - np.asarray([0.0, -2.0, -4.0])) / 20.0),
    )
    sweep = executor.program.repetitions.items[0]
    assert {
        variable.name for variable in sweep.associations
    } == {
        "sparameter_frequency_hz",
        "sparameter_relative_amplitude",
    }
