"""Offscreen GUI -> compiler -> mocked board checks, including DDR cancellation."""
import copy
import json
import os
import socket
from types import SimpleNamespace
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt5 import QtCore, QtWidgets
from qick.sim import QickSim
from qick_output_limits_widget import OutputLimitsWidget
from qick_output_limits import (normalize_output_limits,
                                compile_output_limits, apply_output_limits)
from qick_capture_lifecycle import ddr_capture_guard
from test_awg_v7_fine import configuration, pulse
from dc_waveform_core import build_qick_sequence
from qick_fine_tune_sweep import FineTuneSequence, DdrFirReadoutConfig, FineTuneAmplitudeSweepProgram
from test_qick_fine_tune_sweep import _fir_soccfg
import stability_diagram as stability
import DCWaveform_Generator as gui


def v8_config():
    cfg = configuration()
    cfg['gens'][1].update(type='axis_awg_tuning_v8', output_clip_version=1,
                          output_clip_quantum=4)
    return cfg


@pytest.fixture
def application(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    def no_network(*args, **kwargs):
        pytest.fail('Unexpected network or device access')
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    yield app
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


def test_per_dac_voltage_scaling_inward_rounding_and_validation():
    cfg = v8_config()
    seq = build_qick_sequence((pulse(),), output_full_scales_mv=(1240.0,))
    seq.output_clip_limits = {'1': dict(enabled=True, min_mv=-100.001, max_mv=200.001)}
    bounds = compile_output_limits(cfg, seq, (1,))[0]
    assert bounds['min_code'] % 4 == bounds['max_code'] % 4 == 0
    assert bounds['effective_min_mv'] >= -100.001
    assert bounds['effective_max_mv'] <= 200.001
    assert bounds['max_code'] == 5284
    with pytest.raises(ValueError, match='v8'):
        compile_output_limits(configuration(), seq, (1,))
    seq.output_clip_limits['1']['min_mv'] = -1241
    with pytest.raises(ValueError, match='voltage range'):
        compile_output_limits(cfg, seq, (1,))
    for bad in [dict(enabled=True,min_mv=1,max_mv=2),
                dict(enabled=True,min_mv=-1,max_mv=float('nan')),
                dict(enabled='false',min_mv=-1,max_mv=2)]:
        with pytest.raises(ValueError): normalize_output_limits({'1':bad})


def test_fine_compile_cache_and_execution_use_same_limits(monkeypatch):
    seq = build_qick_sequence((pulse(),))
    seq.output_clip_limits = {'1':dict(enabled=True,min_mv=-80,max_mv=160)}
    p = seq.make_program(v8_config(), awg_channels=(1,), repetitions_per_sweep=3)
    assert p.output_clip_config[0]['min_code'] == -3276
    assert p.output_clip_config[0]['max_code'] == 6552
    calls = []
    class Board:
        def start_src(self, value): calls.append('internal')
        def stop_tproc(self): calls.append('stop')
        def set_awg_output_limits(self, **kwargs):
            calls.append(('limits', kwargs))
            return dict(kwargs,pending=False,configuration_error=False,clipped=False)
    from qick.qick_asm import AbsQickProgram
    parent = AbsQickProgram
    monkeypatch.setattr(parent, 'config_all', lambda *a,**k: calls.append('load'))
    p.config_all(Board())
    assert calls[:2] == ['internal','stop'] and calls[-1] == 'load'
    assert p.summary()['output_clip_readback'][0]['readback']['enabled']
    import qick_program_cache as cache
    before = cache._semantic(seq)
    seq.output_clip_limits['1']['max_mv'] = 200
    assert cache._semantic(seq) != before
    legacy = build_qick_sequence((pulse(),))
    assert compile_output_limits(configuration(), legacy, (1,)) == []
    disable = compile_output_limits(v8_config(), legacy, (1,))
    assert disable[0]['enabled'] is False
    assert apply_output_limits(Board(), disable)[0]['readback']['enabled'] is False


def test_export_preserves_limits_and_compilation_has_no_board_access(application):
    from dc_waveform_core import generate_qick_program_code
    limits = {'1': dict(enabled=True, min_mv=-40, max_mv=80)}
    namespace = {'__name__': 'offline_export'}
    exec(generate_qick_program_code((pulse(),), awg_channels=(1,),
                                    output_clip_limits=limits), namespace)
    seq = namespace['build_sequence']()
    assert seq.output_clip_limits == limits
    program = seq.make_program(v8_config(), awg_channels=(1,))
    program.compile()
    assert program.binprog
    assert program.summary()['output_clip_limits'][0]['requested'] == limits['1']
    assert program.summary()['output_clip_readback'] == []


def test_awg_and_stability_settings_roundtrip_without_window(application):
    w = gui.MainWindow()
    try:
        assert not w.isVisible()
        w._add_port()
        channels = tuple(w._qick_awg_channels)
        first = str(channels[0])
        limits = {first:dict(enabled=True,min_mv=-100,max_mv=200)}
        w._experiment_panel.output_limits.load_settings(limits)
        w._stability_panel.output_limits.load_settings(limits)
        saved = json.loads(json.dumps(w._settings_to_dict()))
        restored = w._decode_settings(saved)
        w._apply_decoded_settings(restored)
        assert w._experiment_panel.output_limits.settings()[first] == limits[first]
        assert w._stability_panel.output_limits.settings()[first] == limits[first]
        args = w._experiment_run_arguments(require_readout=False, require_run_config=False)
        assert args['sequence'].output_clip_limits[first] == limits[first]
        cfg = w._stability_panel.config(full_scale_mv=800)
        names, selected = w._stability_panel.run_output_mapping()
        sequence = stability.build_stability_hold_sequence(cfg, output_names=names, full_scale_mv=800,
                                                           fabric_mhz=300)
        assert sequence.output_clip_limits == cfg.output_clip_limits
        point = stability._stability_sequence_at_point(sequence, 0)
        assert point.output_clip_limits == sequence.output_clip_limits
        saved['qick'].pop('output_clip_limits')
        saved['stability_diagram'].pop('output_clip_limits')
        w._apply_decoded_settings(w._decode_settings(saved))
        assert not any(v['enabled'] for v in w._experiment_panel.output_limits.settings().values())
        assert not any(v['enabled'] for v in w._stability_panel.output_limits.settings().values())
        assert not w.isVisible()
    finally:
        w.close(); w.deleteLater()


def test_channel_reordering_keeps_limits_on_physical_generator(application):
    widget = OutputLimitsWidget()
    widget.set_channels((1,3))
    widget.load_settings({'1':dict(enabled=True,min_mv=-20,max_mv=40)})
    widget.set_channels((3,1))
    assert widget.settings()['1']['max_mv'] == 40
    assert not widget.settings()['3']['enabled']
    widget.deleteLater()


class Cancelled(RuntimeError):
    pass


@pytest.mark.parametrize('failure', ['none','timeout','stop','bad_ack','legacy'])
def test_guard_waits_for_abort_ack_and_preserves_cleanup_failures(failure):
    calls = []
    class Board:
        def stop_tproc(self):
            calls.append('stop')
            if failure == 'stop': raise RuntimeError('stop unavailable')
        def abort_ddr4_capture(self, timeout):
            calls.append('abort')
            if failure == 'timeout': raise TimeoutError('AXI response pending')
            return dict(acknowledged=failure != 'bad_ack', pending=False,busy=False,armed=False)
    board = Board()
    if failure == 'legacy': board.abort_ddr4_capture = None
    with pytest.raises(Cancelled if failure == 'none' else RuntimeError) as result:
        with ddr_capture_guard(board) as capture:
            capture.armed = True
            raise Cancelled('requested')
    assert calls[0] == 'stop'
    if failure not in ('none','legacy'):
        assert calls == ['stop','abort']
        assert isinstance(result.value.__cause__, Cancelled)
        assert 'do not re-arm' in str(result.value)


def test_cancel_immediately_after_arm_then_rerun(monkeypatch):
    seq = FineTuneSequence(('awg_0',))
    seq.add_set('capture', (0,), 300)
    p = seq.make_program(_fir_soccfg(fir_rate_profile='50_ksps'), awg_channels=(0,),
        ddr_readout=DdrFirReadoutConfig(ro_ch=0,samples_per_trigger=2,at_segment='capture',
                                       margin_input_samples=0,settle_seconds=0))
    calls=[]
    class Board:
        cancelled=False
        busy=False
        def arm_ddr4_fir_samples(self, **kwargs):
            assert not self.busy
            self.busy=True; calls.append('arm')
            return 8
        def stop_tproc(self): calls.append('stop')
        def abort_ddr4_capture(self, timeout):
            self.busy=False; calls.append('abort')
            return dict(acknowledged=True,pending=False,busy=False,armed=False)
    board=Board()
    def cancel():
        if board.busy: raise Cancelled('cancel after arm')
    for _ in range(2):
        with pytest.raises(Cancelled): p.acquire_fir_ddr(board,cancel_check=cancel,progress=False)
    assert calls == ['arm','stop','abort','arm','stop','abort']
    calls.clear()
    with pytest.raises(Cancelled):
        p.acquire_fir_ddr(board,cancel_check=lambda: (_ for _ in ()).throw(Cancelled()),progress=False)
    assert calls == []
