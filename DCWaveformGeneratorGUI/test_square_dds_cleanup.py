"""Autonomous output is muted after every acquisition exit path."""
from types import SimpleNamespace
import pytest
import qick_qcodes_experiment as module
from qick_square_dds import SquarePulseConfig
from dc_waveform_core import QickDdrReadoutSpec


@pytest.mark.parametrize('fail',[False,True])
def test_acquisition_always_stops_square_output(monkeypatch,fail):
    calls=[]
    soc=SimpleNamespace(rfb_set_gen_dc=lambda ch:calls.append(('dc',ch)),
                        stop_square_pulse=lambda ch:calls.append(('stop',ch)))
    def acquire(*args,**kwargs):
        calls.append(('acquire',7))
        if fail: raise RuntimeError('capture failed')
        return 'result'
    program=SimpleNamespace(acquire_fir_ddr=acquire)
    monkeypatch.setattr(module,'configure_rf_readout',lambda *args:{})
    monkeypatch.setattr(module,'build_qick_program',lambda *args,**kwargs:program)
    sequence=SimpleNamespace(square_pulse_config=SquarePulseConfig(7),sweep_point_count=1)
    def run():
        return module._execute_qick_sequence_once(soc,{},sequence,awg_channels=(1,),
            repetitions_per_sweep=1,rf_specs=(),readout_spec=QickDdrReadoutSpec(0,'measure',0,2))
    if fail:
        with pytest.raises(RuntimeError,match='capture failed'): run()
    else:
        assert run()[1]=='result'
    assert calls==[('dc',7),('acquire',7),('stop',7)]
