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
    assert (
        result.program_summary[
            "frequency_scalar_shared_with_integration_filter"
        ]
        is True
    )


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


def test_qcs_sparameter_rejects_next_block_above_live_filter_limit():
    qcs = pytest.importorskip("keysight.qcs")
    mapper, _rf, _digitizer = _real_qcs_mapper(qcs)
    with pytest.raises(ValueError, match="at most 32,768 samples"):
        compile_qcs_sparameter_sweep(
            connection_config=_connection(),
            mapper=mapper,
            frequencies_hz=(1.24e9, 1.25e9),
            rf_gen_ch=0,
            rf_amplitude=0.005,
            integration_duration_s=(
                QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES + 16
            )
            / QCS_M5200_SAMPLE_RATE_HZ,
            qcs_module=qcs,
        )


def test_qcs_panel_uses_live_safe_integration_default_and_maximum():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = SParameterSweepPanel()
    panel.set_hardware_backend("qcs")

    assert panel.scan_time_us.value() == pytest.approx(1.0)
    assert panel.scan_time_us.maximum() == pytest.approx(6.826666)

    panel.scan_time_us.setValue(10.0)
    assert panel.scan_time_us.value() == pytest.approx(6.826666)
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
    assert "10 us integration" in message
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
