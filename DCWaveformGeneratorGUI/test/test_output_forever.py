"""Output-only loop, cooperative digital stop and offscreen UI checks."""
import copy
import time
from types import SimpleNamespace
import pytest
from test_v8_controls import application, v8_config, pulse
from qick.awg_tuning import TProcV1BehaviorModel
from qick.output_forever import OutputForeverController
from dc_waveform_core import build_qick_sequence, generate_qick_program_code
from qick_fine_tune_sweep import FineTuneSequence, ReadoutConfig
from qick_output_forever import execute_output_forever, run_output_forever


class StopModel(TProcV1BehaviorModel):
    def __init__(self, shots):
        super().__init__(strict=True)
        self.shots = shots
        self.boundaries = []

    def _exec_memri(self, page, reg, addr):
        if addr == 2:
            self.boundaries.append(len(self.output_events))
            if len(self.boundaries) == self.shots:
                self.dmem[2] = 1
        super()._exec_memri(page, reg, addr)


@pytest.mark.parametrize('stop_shot', [1, 2, 6, 7, 12, 13, 25])
def test_complete_grid_repeats_and_stop_finishes_zero(stop_shot):
    seq = FineTuneSequence(('a',))
    seq.add_set('start', (0.0,), 8)
    seq.add_ramp('ramp', 12)
    seq.add_set('hold', (0.2,), 12)
    seq.add_amplitude_sweep('hold', 'a', 0.1, 0.3, 3)
    p = seq.make_program(v8_config(), (1,), repetitions_per_sweep=2,
                         execution_mode='output_forever')
    p.compile()
    model = StopModel(stop_shot)
    p.load_runtime_dmem_into_model(model)
    model.run(p, max_steps=1000000)
    assert model.ended and model.dmem[3] == 1
    assert len(model.boundaries) == stop_shot
    chunks=[]; start=0
    for end in model.boundaries:
        chunks.append([e.word for e in model.output_events[start:end]])
        start=end
    for shot in range(6, stop_shot):
        assert chunks[shot] == chunks[shot % 6]
    final = model.output_events[-1].word
    assert final & 0xffff == 0  # final AWG SET level
    assert p.ro_chs == {}
    assert p.summary()['execution_mode'] == 'output_forever'


@pytest.mark.parametrize('dc,rc', [(False,False),(True,False),(False,True),(True,True)])
def test_fine_edges_compensation_preserved_across_grid_and_stop(dc, rc):
    s=build_qick_sequence((pulse(),))
    s.set_bias_t_compensation(.1, enabled=dc, mode='fixed_time', fixed_duration_cycles=300)
    s.set_rc_compensation(200, enabled=rc)
    p=s.make_program(v8_config(),(1,),execution_mode='output_forever')
    p.compile()
    m=StopModel(3);p.load_runtime_dmem_into_model(m);m.run(p)
    assert m.dmem[3] == 1
    chunks=[]; start=0
    for end in m.boundaries:
        chunks.append([e.word for e in m.output_events[start:end]]);start=end
    assert chunks[0] == chunks[1] == chunks[2]


def test_readout_rejected_before_connect_or_configuration():
    calls=[]
    with pytest.raises(ValueError, match='RF Readout'):
        run_output_forever(connection_config=None,sequence=None,awg_channels=(),
            repetitions_per_sweep=1,readout_spec=object(), cancel_check=lambda:None,
            connector=lambda *a,**k:calls.append('connect'))
    assert calls == []
    s=build_qick_sequence((pulse(),))
    with pytest.raises(ValueError, match='Readout'):
        s.make_program(v8_config(),(1,), execution_mode='output_forever', readout=object())


def test_export_mode_and_cache_are_distinct():
    namespaces=[]
    for mode in ('finite','output_forever'):
        ns={'__name__':'offline'}
        exec(generate_qick_program_code((pulse(),),awg_channels=(1,),execution_mode=mode),ns)
        p=ns['build_program'](v8_config());p.compile();namespaces.append(p)
    assert namespaces[0].binprog != namespaces[1].binprog
    assert namespaces[1].execution_mode == 'output_forever'
    from qick_qcodes_experiment import build_qick_program
    # Actual cache keys include execution_mode through program_kwargs.
    sequence=build_qick_sequence((pulse(),))
    a=build_qick_program(v8_config(),sequence,awg_channels=(1,),repetitions_per_sweep=1)
    b=build_qick_program(v8_config(),sequence,awg_channels=(1,),repetitions_per_sweep=1,
                         execution_mode='output_forever')
    assert a.execution_mode == 'finite' and b.execution_mode == 'output_forever'


def board_fixture(bad_zero=False):
    calls=[]; memory={}
    def write(addr,data):
        calls.append(('write',addr,data));memory[addr]=data
        if addr==2 and data==1:memory[3]=1
    gen=SimpleNamespace(cfg={'output_clip_version':1},
        get_current=lambda:4 if bad_zero else 0,
        get_status=lambda decode:dict(idle=True,axi_override_pending=False),
        set_output_limits=lambda *a:calls.append(('clamp',a)))
    soc=SimpleNamespace(TPROC_VERSION=1,gens=[gen],
        tproc=SimpleNamespace(single_write=write,single_read=lambda addr:memory.get(addr,0)),
        start_src=lambda s:calls.append(('src',s)),stop_tproc=lambda:calls.append(('stop',)))
    return OutputForeverController(soc),calls


@pytest.mark.parametrize('lease_expired', [False, True])
def test_board_watchdog_stop_and_restart(lease_expired):
    c,calls=board_fixture();s=c.begin((0,),lease_seconds=.5)
    with pytest.raises(RuntimeError,match='active'):
        c.begin((0,))
    if lease_expired:
        deadline=time.monotonic()+2
        while c.status(s['token'])['state']=='running' and time.monotonic()<deadline:time.sleep(.02)
    result=c.stop(s['token'],timeout=2)
    assert result['state']=='stopped'
    assert result['reason']==('host_lease_expired' if lease_expired else 'user_stop')
    assert ('write',2,1) in calls and ('stop',) in calls
    second=c.begin((0,),lease_seconds=.5);assert second['token'] != s['token']
    c.stop(second['token'],timeout=2)


def test_cleanup_error_clamps_and_locks_next_start():
    c,calls=board_fixture(True);s=c.begin((0,),lease_seconds=.5)
    with pytest.raises(RuntimeError,match='cleanup incomplete'):
        c.stop(s['token'],timeout=2)
    assert ('clamp',(0,0,True)) in calls
    with pytest.raises(RuntimeError,match='cleanup is incomplete'):
        c.begin((0,))


def test_runtime_cancel_does_not_arm_or_save():
    calls=[]
    class Cancelled(Exception):pass
    def cancel():
        if 'start' in calls:raise Cancelled()
    p=SimpleNamespace(execution_mode='output_forever',ro_chs={},awg_channels=(1,),
        config_all=lambda *a,**k:calls.append('config'),
        _load_runtime_dmem=lambda *a,**k:calls.append('memory'))
    soc=SimpleNamespace(begin_output_forever=lambda c:dict(token='t'),
        check_output_forever_idle=lambda **kw:True,
        renew_output_forever=lambda t:dict(state='running'),
        stop_output_forever=lambda t:(calls.append('stop_ack') or dict(state='stopped')),
        start_src=lambda s:calls.append('internal'),start_tproc=lambda:calls.append('start'))
    with pytest.raises(Cancelled):execute_output_forever(soc,p,cancel_check=cancel)
    assert calls==['config','memory','internal','start','stop_ack']


@pytest.mark.parametrize('field', ['busy', 'armed', 'pending'])
def test_capture_busy_rejected_before_any_mailbox_write(field):
    c,calls=board_fixture()
    status=dict(busy=False,armed=False,pending=False);status[field]=True
    c.soc.ddr4_buf=SimpleNamespace(cfg=dict(supports_abort=True),abort_status=lambda:status)
    with pytest.raises(RuntimeError,match='DDR capture'):
        c.begin((0,))
    assert calls == []


def test_owned_session_cannot_be_reconfigured():
    c,calls=board_fixture();s=c.begin((0,))
    try:
        with pytest.raises(RuntimeError,match='active'):
            c.check_idle()
    finally:
        c.stop(s['token'],timeout=2)
    assert c.check_idle()


def test_square_and_external_marker_stop_even_when_finite_mute_is_disabled():
    from test_qick_square_dds import configuration
    from qick_square_dds import SquarePulseConfig, OutputTriggerConfig, attach_square_settings
    cfg=v8_config()
    cfg['gens']=[copy.deepcopy(cfg['gens'][1]),configuration()['gens'][1]]
    cfg['tprocs'][0]['output_pins']=[('output',7,6,'SPARE1_1V8')]
    seq=FineTuneSequence(('gate',)).add_set('hold',.1,300)
    attach_square_settings(seq,SquarePulseConfig(1,mute_on_finish=False),(),
                           OutputTriggerConfig(True,0,'loop','both',1))
    p=seq.make_program(cfg,(0,),execution_mode='output_forever');p.compile()
    m=StopModel(3);p.load_runtime_dmem_into_model(m);m.run(p)
    square=[e.word for e in m.output_events if e.tproc_ch==0 and (e.word>>152)&255==2]
    assert len(square)==4 and square[-1] & (1<<128)==0
    assert all(word & (1<<128) for word in square[:-1])
    marker=[e['word'] for e in m.output_pin_events if e['port']==7]
    assert marker and marker[-1]==0 and m.dmem[3]==1


def test_export_rejects_readout_even_when_generated_aux_is_disabled():
    ns={'__name__':'offline'}
    exec(generate_qick_program_code((pulse(),),awg_channels=(1,),execution_mode='output_forever'),ns)
    with pytest.raises(ValueError,match='forbids acquisition'):
        ns['build_program'](v8_config(),readout=object(),use_generated_aux=False)


def test_gui_start_stop_and_restart_uses_worker_without_measurement(application, monkeypatch):
    from PyQt5 import QtTest
    import DCWaveform_Generator as gui
    import qick_output_forever as runtime
    calls=[];errors=[]
    def fake_run(**kwargs):
        p=SimpleNamespace(execution_mode='output_forever',ro_chs={},awg_channels=(1,),
            config_all=lambda *a,**k:None,_load_runtime_dmem=lambda *a,**k:None)
        soc=SimpleNamespace(check_output_forever_idle=lambda **kw:True,
            begin_output_forever=lambda c:dict(token='t'),
            renew_output_forever=lambda t:dict(state='running'),
            stop_output_forever=lambda t:(calls.append('zero_ack') or dict(state='stopped')),
            start_src=lambda s:None,start_tproc=lambda:calls.append('start'))
        execute_output_forever(soc,p,cancel_check=kwargs['cancel_check'],
                               progress_callback=kwargs['progress_callback'])
    monkeypatch.setattr(runtime,'run_output_forever',fake_run)
    w=gui.MainWindow()
    monkeypatch.setattr(w,'_on_experiment_failed',lambda details:errors.append(details))
    monkeypatch.setattr(w._rf_readout_panel,'spec',lambda:None)
    monkeypatch.setattr(w,'_experiment_run_arguments',lambda **kw:dict(
        connection_config=None,sequence=None,awg_channels=(1,),repetitions_per_sweep=2,
        rf_specs=(),readout_spec=None,compile_validation_mode='full'))
    try:
        for run in (1,2):
            w._experiment_panel.forever_button.click()
            deadline=time.monotonic()+4
            while calls.count('start')<run and time.monotonic()<deadline:QtTest.QTest.qWait(20)
            assert calls.count('start')==run and not w.isVisible()
            w._experiment_panel.stop_button.click()
            while w._experiment_thread is not None and time.monotonic()<deadline:QtTest.QTest.qWait(20)
            assert w._experiment_thread is None and calls.count('zero_ack')==run
            assert 'digital zero acknowledged' in w._experiment_panel.run_status.text()
            assert w._experiment_panel.forever_button.isEnabled() and not errors
    finally:
        if w._experiment_worker is not None:
            w._experiment_worker.request_cancel()
            deadline=time.monotonic()+4
            while w._experiment_thread is not None and time.monotonic()<deadline:QtTest.QTest.qWait(20)
        w.close();w.deleteLater()


def test_rpc_failure_during_cleanup_is_not_reported_as_successful_cancel():
    from qick_qcodes_experiment import ExperimentCancelled
    started=[]
    def cancel():
        if started:raise ExperimentCancelled('stop')
    def failed_stop(token):raise ConnectionError('lost RPC')
    p=SimpleNamespace(execution_mode='output_forever',ro_chs={},awg_channels=(1,),
        config_all=lambda *a,**k:None,_load_runtime_dmem=lambda *a,**k:None)
    soc=SimpleNamespace(check_output_forever_idle=lambda **kw:True,
        begin_output_forever=lambda c:dict(token='t'),renew_output_forever=lambda t:None,
        stop_output_forever=failed_stop,start_src=lambda s:None,start_tproc=lambda:started.append(1))
    with pytest.raises(RuntimeError,match='Output-only cleanup incomplete: lost RPC'):
        execute_output_forever(soc,p,cancel_check=cancel)


def test_gui_button_no_visible_window_and_readout_rejection(application, monkeypatch):
    import DCWaveform_Generator as gui
    w=gui.MainWindow()
    try:
        panel=w._experiment_panel
        assert panel.forever_button.text()=='Run forever' and not w.isVisible()
        panel.set_running(True,'test',can_cancel=True,show_progress=False)
        assert not panel.forever_button.isEnabled() and panel.stop_button.isEnabled()
        panel.set_running(False,'test')
        messages=[]
        monkeypatch.setattr(w._rf_readout_panel,'spec',lambda:object())
        monkeypatch.setattr(gui.QtWidgets.QMessageBox,'warning',lambda *a:messages.append(a[-1]))
        w._run_output_forever()
        assert messages and 'Disable RF Readout' in messages[0]
        assert w._experiment_worker is None
    finally:
        w.close();w.deleteLater()
