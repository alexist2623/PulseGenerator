"""Command occupancy, acquisition length and pre-loop initialization contracts."""
import pytest
import qick.sim  # desktop PYNQ stubs
from qick.awg_tuning import TProcV1BehaviorModel
from qick import QickConfig
from qick_fine_tune_sweep import FineTuneSequence, ReadoutConfig, DdrFirReadoutConfig
from qick_readout_timing import LEGACY_PENDING_COMMANDS, LEGACY_COMMAND_MAX_CYCLES
from test_qick_fine_tune_sweep import _fir_soccfg, _avg_soccfg


def events(p):
    m = TProcV1BehaviorModel(strict=True)
    if hasattr(p, 'load_runtime_dmem_into_model'):
        p.load_runtime_dmem_into_model(m)
    m.run(p, max_steps=100_000)
    assert m.ended and not m.timing_conflicts and not m.unsupported_instructions
    return m.output_events


def check(p, ch):
    ro_cfg = p.soccfg['readouts'][ch]
    ev = [e for e in events(p) if e.tproc_ch == ro_cfg['tproc_ctrl']]
    assert ev
    assert all((e.word >> 64) & 0xffff == 3 for e in ev)
    assert all((e.word >> 82) & 3 == 0 for e in ev)  # oneshot, no phase reset
    init = p.readout_initialization
    assert init['command_cycles'] == 3 and init['legacy_queue_bound'] == 8
    assert init['fifo_slot_wait_tproc_cycles'] >= (
        LEGACY_COMMAND_MAX_CYCLES * p.tproccfg['f_time'] / ro_cfg['f_fabric'])
    assert ev[0].cycle >= init['fifo_slot_wait_tproc_cycles'] + 128
    assert init['startup_wait_tproc_cycles'] >= (
        LEGACY_PENDING_COMMANDS * LEGACY_COMMAND_MAX_CYCLES
        * p.tproccfg['f_time'] / ro_cfg['f_fabric'])
    return ev


@pytest.mark.parametrize('kind', ['avg', 'ddr'])
def test_capture_lengths_and_repetitions_are_independent_of_three_cycle_command(kind):
    cfg = _fir_soccfg() if kind == 'ddr' else _avg_soccfg()
    seq = FineTuneSequence(('a',)).add_set('measure', (.01,), 300_000)
    kwargs = (dict(ddr_readout=DdrFirReadoutConfig(0, 512, 'measure', readout_period_cycles=65535))
              if kind == 'ddr' else dict(readout=ReadoutConfig(0, 512)))
    p = seq.make_program(cfg, awg_channels=(0,), repetitions_per_sweep=3, **kwargs)
    assert p.ro_chs[0]['length'] == 512
    assert len(check(p, 0)) == 4  # one initialization plus three real measurements
    if kind == 'ddr':
        assert p.ddr_readout_config.samples_per_trigger == 512
        assert p.ddr_readout_config.readout_period_cycles == 3


def test_initialization_preserves_every_measurement_word_and_relative_timestamp(monkeypatch):
    import qick_fine_tune_sweep as module
    cfg = _fir_soccfg(ddr_trigger_port=7)
    seq = FineTuneSequence(('a',)).add_set('measure', (.01,), 300_000)
    seq.add_amplitude_sweep('measure', 'a', .01, .03, 3)
    kwargs = dict(awg_channels=(0,), repetitions_per_sweep=2,
                  ddr_readout=DdrFirReadoutConfig(0, 20, 'measure'))
    actual = seq.make_program(cfg, **kwargs)
    actual_events = events(actual)
    monkeypatch.setattr(module, 'initialize_readout', lambda *a: None)
    reference = seq.make_program(cfg, **kwargs)
    reference_events = events(reference)
    assert actual_events[0].tproc_ch == cfg['readouts'][0]['tproc_ctrl']
    assert [e.word for e in actual_events[1:]] == [e.word for e in reference_events]
    offsets = {a.cycle-b.cycle for a,b in zip(actual_events[1:], reference_events)}
    assert len(offsets) == 1 and next(iter(offsets)) > 0


def test_noise_uses_minimum_command_without_shortening_capture():
    from noise_acquisition import NoiseAcquisitionConfig, build_noise_fir_program
    from test_noise_acquisition import _soccfg
    p = build_noise_fir_program(QickConfig(_soccfg()), NoiseAcquisitionConfig(fir_samples=100_000))
    assert len(check(p, 0)) == 2


@pytest.mark.parametrize('source', ['fir_ddr', 'avg_buffer'])
def test_sparameter_frequency_updates_use_minimum_commands(source):
    from qick_sparameter_sweep import SParameterSweepConfig, SParameterSweepProgram
    from test_qick_sparameter_sweep import _mock_soccfg
    config = SParameterSweepConfig(output_ch=0, readout_ch=0,
        frequency_start_mhz=10., frequency_end_mhz=12., frequency_points=3,
        scan_time_us=10., acquisition_source=source)
    cfg = _mock_soccfg()
    # Keep the mock trigger output distinct from the RF generator's port.
    cfg['readouts'][0]['trigger_port'] = 7
    p = SParameterSweepProgram(cfg, config)
    ro = check(p, 0)
    # Initialization uses the first point, then hardware updates advance DDS.
    frequencies = [e.word & 0xffffffff for e in ro]
    assert frequencies[0] == frequencies[1]
    assert len(set(frequencies)) == 3
