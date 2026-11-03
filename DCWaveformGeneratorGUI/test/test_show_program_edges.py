"""Show program describes emitted words, including combined/split fine edges."""
import pytest
import qick.sim  # noqa: F401 - desktop driver bootstrap

from dc_waveform_core import build_qick_sequence
from qick_dc_area import executed_commands
from qick_fine_ramp import scalar_runs
from qick_program_display import compiled_awg_commands, format_awg_commands
from test_awg_v7_fine import configuration, pulse
from test_execution_coordinates import compiled_bytes, execute
from test_sample_hold_sweep import compile_hold


@pytest.mark.parametrize('spec,end', [
    (dict(rise_start=2, rise_end=5, fall_start=14, fall_end=20, turn=100), 20),
    (dict(rise_start=15, rise_end=18, fall_start=31, fall_end=49, turn=-100), 20),
    (dict(rise_start=15, rise_end=16), 100),
    (dict(rise_start=3, rise_end=7), 0),
])
def test_edges_decode_issued_packet_and_first_target_sample(spec, end):
    seq = build_qick_sequence((pulse(spec, duration=20, end=end),))
    seq.set_voltage_scales(800., (800.,))
    program = seq.make_program(configuration(), awg_channels=(1,), repetitions_per_sweep=2)
    before = compiled_bytes(program)
    records = compiled_awg_commands(program, 0)
    commands = list(executed_commands(program, 0))
    model = execute(program)
    actual_packets = [event.word for event in model.output_events if event.word >> 151 & 1]
    assert len(actual_packets) == 2
    ramp = next(record for record in records if record['kind'] == 'ramp')
    assert int(ramp['packet_hex'], 16) == actual_packets[0] == actual_packets[1]
    command = next(command for command in commands if command.kind == 'ramp')
    samples = []
    for count, slope, intercept, low, high in scalar_runs(command, ramp['start_code_16']):
        samples.extend(max(low, min(high, (slope*i + intercept) // (1 << 20))) * 4
                       for i in range(count))
    for edge in ramp['edges']:
        start, stop = edge['start_sample'], edge['boundary_sample']
        expected = next(i for i in range(start, stop+1) if samples[i] == edge['target_code_16'])
        assert edge['target_sample'] == expected
        assert samples[start] == edge['start_code_16']
    assert ramp['terminal_code_16'] == samples[ramp['duration_samples']-1]
    assert ramp['second_edge_enabled'] == ('fall_start' in spec)
    text = '\n'.join(format_awg_commands(program, 0))
    assert 'None' not in text and 'ns' in text and 'mV' in text
    assert compiled_bytes(program) == before


def test_combined_to_separate_fall_branch_uses_actual_commands():
    # 48-sample rise + HOLD = 16383 (combined) or 16384 (two packets).
    program = compile_hold(16335, 16336, 2)
    before = compiled_bytes(program)
    first, second = (compiled_awg_commands(program, point) for point in (0, 1))
    assert [record['segment_name'] for record in first] == ['set_0', 'ramp_0_to_1']
    assert [record['segment_name'] for record in second] == ['set_0', 'ramp_0_to_1', 'ramp_1_to_2']
    assert len(first[1]['edges']) == 2
    assert len(second[1]['edges']) == len(second[2]['edges']) == 1
    assert second[2]['edges'][0]['direction'] == 'Falling'
    assert second[2]['edges'][0]['start_code_16'] == second[1]['terminal_code_16']
    assert second[2]['terminal_code_16'] == 0
    assert second[2]['command_cycle'] - second[1]['command_cycle'] == 1024
    assert 'ramp_1_to_2' not in '\n'.join(format_awg_commands(program, 0))
    assert 'ramp_1_to_2' in '\n'.join(format_awg_commands(program, 1))
    assert compiled_bytes(program) == before


def test_dialog_combined_and_split_points_do_not_change_assembly():
    from PyQt5 import QtWidgets
    import DCWaveform_Generator as gui

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    program = compile_hold(16335, 16336, 2, rf=True)
    before = compiled_bytes(program)
    dialog = gui.QickAssemblyDialog({'assembly': program.asm(), 'program': program})
    try:
        assert not dialog.isVisible()
        assert dialog.assembly_text.toPlainText() == program.asm()
        first = dialog.coordinate_text.toPlainText()
        assert 'Edge 1 (Rising)' in first and 'Edge 2 (Falling)' in first
        assert 'HOLD between edges:' in first and 'None' not in first
        assert 'ramp_1_to_2' not in first
        dialog.coordinate_point.setValue(1)
        app.processEvents()
        second = dialog.coordinate_text.toPlainText()
        assert 'ramp_1_to_2' in second and 'Edge 1 (Falling)' in second
        assert 'Second edge: disabled (11111)' in second
        assert 'None' not in second
        assert compiled_bytes(program) == before
    finally:
        dialog.close()
        dialog.deleteLater()
        app.sendPostedEvents(None, 0)
        app.processEvents()


def test_display_without_voltage_scale_keeps_dac_codes():
    program = compile_hold(31, 33, 3)
    text = '\n'.join(format_awg_commands(program, 1))
    assert 'DAC code' in text and 'None' not in text
    assert 'mV' not in text  # Do not invent a physical-voltage calibration.


@pytest.mark.parametrize('rf_enabled', (False, True))
def test_show_worker_and_run_compile_identical_programs(tmp_path, monkeypatch, rf_enabled):
    """Use both real compile paths; replace only connection/device I/O boundaries."""
    import copy
    from PyQt5 import QtWidgets
    import DCWaveform_Generator as gui
    import qick_qcodes_experiment as experiment
    from dc_waveform_core import QickDdrReadoutSpec, QickRfPulseSpec
    from qick_fine_tune_sweep import FineTuneAmplitudeSweepProgram
    from test_qick_fine_tune_sweep import _fir_soccfg

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setenv('QSTL_QICK_COMPILE_CACHE_DIR', str(tmp_path / 'cache'))
    cfg = _fir_soccfg(fir_rate_profile='50_ksps', ddr_trigger_port=7)
    cfg._cfg['gens'] = configuration()['gens']
    cfg._cfg['tprocs'] = configuration()['tprocs']
    cfg._cfg['readouts'][0]['tproc_ctrl'] = 2
    sequence = compile_hold(16335, 16336, 2).sequence
    sequence.set_voltage_scales(800., (800.,))
    rf_specs = (QickRfPulseSpec(0, 'set_2', .1, 97/4800, 198., 1000, 0., 0.,
                               output_board_type='DC_Out'),) if rf_enabled else ()
    kwargs = dict(sequence=sequence, awg_channels=(1,), repetitions_per_sweep=2,
                  rf_specs=rf_specs, readout_spec=QickDdrReadoutSpec(0, 'set_2', 0., 2),
                  compile_validation_mode='full')
    hardware = object()  # Any unexpected device method fails immediately.
    monkeypatch.setattr(gui, 'connect_qick', lambda config: (hardware, cfg))
    results, failures = [], []
    worker = gui.QickProgramWorker(None, copy.deepcopy(kwargs))
    worker.finished.connect(results.append)
    worker.failed.connect(failures.append)
    worker.run()
    assert not failures, failures
    assert len(results) == 1
    show = results[0]['program']
    before = compiled_bytes(show)
    if rf_enabled:
        assert any(before[2].values()), 'RF waveform-memory comparison must not be empty'
    phases, acquired = [], []
    monkeypatch.setattr(experiment, 'configure_rf_readout',
                        lambda *args: phases.append('readout setup boundary') or {})

    def acquire(program, soc, **options):
        assert soc is hardware
        assert compiled_bytes(program) == before
        acquired.append(program)
        return object()

    monkeypatch.setattr(FineTuneAmplitudeSweepProgram, 'acquire_fir_ddr', acquire)
    # Neither builder is mocked: runtime reaches the normal acquisition boundary.
    run, _, _ = experiment._execute_qick_sequence_once(hardware, cfg, **copy.deepcopy(kwargs))
    assert phases == ['readout setup boundary'] and acquired == [run]
    assert run.asm() == results[0]['assembly']
    assert compiled_bytes(run) == before
    for point in range(sequence.sweep_point_count):
        assert compiled_awg_commands(show, point) == compiled_awg_commands(run, point)
    app.processEvents()


def test_legacy_multi_output_sweeps_report_actual_quantized_targets():
    from test_execution_coordinates import awg_program
    program = awg_program()
    before = compiled_bytes(program)
    for point in range(program.sequence.sweep_point_count):
        commands = list(executed_commands(program, point))
        records = compiled_awg_commands(program, point)
        for command, record in zip(commands, records):
            if command.kind != 'ramp':
                continue
            edge, = record['edges']
            frac = int(program.soccfg['gens'][command.gen_ch]['frac'])
            count = command.duration_samples
            raw = [(record['start_code_16'] << frac) + int(command.step) * i
                   for i in range(count - 1)]
            samples = [max(-8192, min(8191, value // (1 << (frac + 2)))) * 4 for value in raw]
            samples.append(record['terminal_code_16'])
            first_target = samples.index(record['terminal_code_16'])
            assert edge['target_sample'] == first_target
    assert compiled_bytes(program) == before
