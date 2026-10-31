"""v7 holds the terminal target; duplicate SETs must not occupy the RF port."""
import pytest
import qick.sim
from qick.awg_tuning import TProcV1BehaviorModel
from qick_fine_tune_sweep import FineTuneSequence, RfPulseConfig, FineTuneAmplitudeSweepProgram
from qick_dc_area import executed_commands, waveform_ledger
from qick_execution_coordinates import build_execution_coordinates
from test_awg_v7_fine import configuration
from test_awg_v6_timing import configuration as v6_configuration
from test_sample_hold_sweep import compile_hold


def sequence():
    return (FineTuneSequence(('x',)).add_set('zero', (0.,), 30)
            .add_ramp('rise', 10).add_set('hold', (.2,), 100)
            .add_ramp('fall', 10).add_set('end', (0.,), 100))


@pytest.mark.parametrize('fine', (False, True))
def test_rf_starts_at_unchanged_post_ramp_boundary(fine):
    seq=sequence()
    if fine:seq.set_fine_ramp('rise', (dict(rise_start=0,rise_end=17),))
    p=seq.make_program(configuration(), awg_channels=(1,),
        rf_pulse=RfPulseConfig(0,'hold',10,1000,delay_tproc_cycles=0,freq_mhz=200.))
    p.compile()
    assert p.aux_timing['rf_0_start']+31==p.timing['segment_starts'][2]
    model=TProcV1BehaviorModel(strict=True);p.load_runtime_dmem_into_model(model);model.run(p)
    assert model.ended and not model.timing_conflicts
    awg=[e for e in model.output_events if e.word>>152&255==1]
    opcodes=[(e.word>>144)&3 for e in awg]
    first_ramp=opcodes.index(2)
    assert opcodes[first_ramp-1:]==[1,2,2]  # startup/reset commands precede the pulse
    assert [c.segment_name for c in executed_commands(p,0)]==['zero','rise','fall']
    exported=build_execution_coordinates(p)
    assert [x['segment_name'] for x in exported['awg_commands']]==['zero','rise','fall']


def test_voltage_sweeps_retain_exact_area_and_hold_time(monkeypatch):
    seq=sequence();seq.add_amplitude_sweep('hold','x',-.4,.4,11)
    p=seq.make_program(configuration(),awg_channels=(1,));p.compile()
    actual=[waveform_ledger(p,i,0) for i in range(11)]
    with monkeypatch.context() as m:
        m.setattr(FineTuneAmplitudeSweepProgram,'_command_is_emitted',lambda self,c:True)
        reference=seq.make_program(configuration(),awg_channels=(1,));reference.compile()
        expected=[waveform_ledger(reference,i,0) for i in range(11)]
    assert p.timing==reference.timing
    for a,b in zip(actual,expected):
        for field in ('area','prefix_min','prefix_max','scalar_samples','ideal_area'):
            assert a[field]==b[field]


def test_initial_and_standalone_sets_still_reserve_shared_port():
    seq=FineTuneSequence(('x',)).add_set('zero',(0.,),30).add_set('step',(.2,),100)
    with pytest.raises(ValueError,match='RF command conflicts'):
        seq.make_program(configuration(),awg_channels=(1,),rf_pulse=RfPulseConfig(0,'step',10,1000))
    p=seq.make_program(configuration(),awg_channels=(1,));p.compile()
    assert [c.kind for c in executed_commands(p,0)]==['set','set']


def test_older_firmware_retains_its_command_sequence():
    p=sequence().make_program(v6_configuration(),awg_channels=(1,));p.compile()
    assert [c.kind for c in executed_commands(p,0)]==['set','ramp','set','ramp','set']


def test_sample_hold_rf_zero_delay_does_not_reserve_omitted_set():
    original=compile_hold(1,33,33)
    p=original.sequence.make_program(configuration(),awg_channels=(1,),
        rf_pulse=RfPulseConfig(0,'set_2',7,1000,fine_duration=True,
                              duration_samples=97,delay_tproc_cycles=0))
    p.compile()
    model=TProcV1BehaviorModel(strict=True);p.load_runtime_dmem_into_model(model);model.run(p)
    assert model.ended and not model.timing_conflicts


def test_fixed_fine_rf_tracks_a_preceding_ramp_duration_sweep():
    seq = sequence()
    seq.add_ramp_duration_sweep('rise', .01, 1.01, 5)
    p = seq.make_program(configuration(), awg_channels=(1,),
        rf_pulse=RfPulseConfig(0, 'hold', 4, 1000, fine_duration=True,
                              duration_samples=64, delay_tproc_cycles=0))
    p.compile()
    model = TProcV1BehaviorModel(strict=True)
    p.load_runtime_dmem_into_model(model)
    model.run(p)
    assert model.ended and not model.timing_conflicts
    starts = [e.cycle for e in model.output_events
              if e.word >> 152 & 255 == 0 and e.word >> 96 & 65535 == 2000]
    ramps = [e.cycle for e in model.output_events
             if e.word >> 152 & 255 == 1 and e.word >> 144 & 3 == 2][::2]
    assert len(starts) == len(ramps) == 5
    assert [a-b for a,b in zip(starts, ramps)] == [3,78,153,228,303]
