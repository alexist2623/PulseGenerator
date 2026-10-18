"""Compiled artifact reuse must preserve execution and never reuse acquired data."""
from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import qick.sim  # noqa: F401 -- desktop stubs
from qick.awg_tuning import TProcV1BehaviorModel
from qick_fine_tune_sweep import FineTuneSequence, RfPulseConfig
from test_qick_fine_tune_sweep import _independent_awg_soccfg, _shared_tmux_soccfg, _fir_soccfg
import qick_program_cache as cache
import qick_qcodes_experiment as experiment


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(cache.CACHE_DIR_ENV, str(tmp_path / 'cache'))
    monkeypatch.setenv(cache.CACHE_ENABLED_ENV, '1')
    cache.clear_memory_cache()
    yield
    cache.clear_memory_cache()


def sequence():
    seq = FineTuneSequence(('a', 'b')).add_set('hold', (.1, .2), 300)
    seq.add_amplitude_sweep('hold', 'a', .1, .2, 3)
    return seq


def build(seq=None, cfg=None, **kwargs):
    return experiment.build_qick_program(
        _independent_awg_soccfg() if cfg is None else cfg,
        sequence() if seq is None else seq,
        awg_channels=(0, 1), repetitions_per_sweep=1, **kwargs)


def execution(program):
    model = TProcV1BehaviorModel(strict=True)
    program.load_runtime_dmem_into_model(model)
    model.run(program, max_steps=1_000_000)
    assert not model.timing_conflicts
    return [(e.cycle, e.tproc_ch, e.word) for e in model.output_events]


def test_memory_disk_restore_execution_and_isolate_mutation():
    first = build()
    expected = execution(first)
    assert first.compile_cache_info['status'] == 'miss'
    second = build()
    assert second.compile_cache_info['source'] == 'memory'
    assert execution(second) == expected
    assert second.asm() == first.asm()
    assert cache._semantic(second.summary()) == cache._semantic(first.summary())
    assert second._gen_mgrs[0].prog is second
    second.prog_list.clear()
    second.sequence.cross_capacitance[0, 0] = 100
    second.acc_buf = np.ones(10)
    cache.clear_memory_cache()
    third = build()
    assert third.compile_cache_info['source'] == 'disk'
    assert third.acc_buf is None
    assert execution(third) == expected


def test_previously_evaluated_preview_has_same_semantic_key():
    seq = sequence()
    first = build(seq)
    assert seq.sweep_coordinates.shape[0] == 3
    assert build(seq).compile_cache_info['key'] == first.compile_cache_info['key']
    assert build(seq).compile_cache_info['status'] == 'hit'


def test_clear_removes_disk_and_memory_and_next_build_recompiles():
    first = build()
    assert build().compile_cache_info['status'] == 'hit'
    directory = cache.cache_directory()
    result = cache.clear_program_cache()
    assert result == {'directory': str(directory), 'files_removed': 1,
                      'memory_entries_removed': 1, 'errors': []}
    assert not cache._memory
    assert not list(directory.glob('*.json.gz'))
    rebuilt = build()
    assert rebuilt.compile_cache_info['status'] == 'miss'
    assert rebuilt.compile_cache_info['saved']
    assert rebuilt.asm() == first.asm()
    assert execution(rebuilt) == execution(first)


def test_clear_preserves_unrelated_files_and_does_not_recurse():
    directory = cache.cache_directory()
    directory.mkdir()
    preserved = [directory / name for name in
                 ('settings.json', 'measurements.db', 'waveform.json.gz', 'active.tmp')]
    nested = directory / ('a' * 64 + '.json.gz')
    nested.mkdir()
    preserved.append(nested / ('b' * 64 + '.json.gz'))
    for path in preserved:
        path.write_bytes(b'keep')
    # Even an unreadable/corrupt old artifact can be removed by its cache name.
    (directory / ('c' * 64 + '.json.gz')).write_bytes(b'corrupt')
    result = cache.clear_program_cache()
    assert result['files_removed'] == 1
    assert not result['errors']
    assert all(path.read_bytes() == b'keep' for path in preserved)


def test_clear_missing_directory_is_a_noop():
    assert not cache.cache_directory().exists()
    assert cache.clear_program_cache()['files_removed'] == 0
    assert not cache.cache_directory().exists()


def test_clear_reports_locked_entry_and_continues(monkeypatch):
    build()
    directory = cache.cache_directory()
    locked = next(directory.glob('*.json.gz'))
    removable = directory / ('f' * 64 + '.json.gz')
    removable.write_bytes(b'old')
    original = Path.unlink

    def unlink(path, *args, **kwargs):
        if path == locked:
            raise PermissionError('file locked')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', unlink)
    result = cache.clear_program_cache()
    assert result['files_removed'] == 1
    assert result['memory_entries_removed'] == 1
    assert len(result['errors']) == 1
    assert str(locked) in result['errors'][0]
    assert locked.exists() and not removable.exists()
    assert not cache._memory


def test_clear_directory_error_still_clears_memory(monkeypatch):
    build()

    def denied(_path):
        raise PermissionError('directory denied')

    monkeypatch.setattr(Path, 'iterdir', denied)
    result = cache.clear_program_cache()
    assert result['files_removed'] == 0
    assert result['memory_entries_removed'] == 1
    assert 'directory denied' in result['errors'][0]
    assert not cache._memory


@pytest.mark.parametrize('field', ['voltage', 'hold', 'matrix', 'groups', 'dc', 'rc',
                                   'scales', 'dac_current', 'firmware', 'clock',
                                   'repetitions', 'validation', 'rf', 'readout'])
def test_relevant_changes_invalidate(field):
    cfg, seq = _independent_awg_soccfg(), sequence()
    kw = dict(awg_channels=(0, 1), repetitions_per_sweep=1)
    old = cache._key(cfg, seq, kw)
    if field == 'voltage':
        seq.segments[0] = replace(seq.segments[0], amplitudes=(.15, .2))
    elif field == 'hold':
        seq.segments[0] = replace(seq.segments[0], duration_cycles=330)
    elif field == 'matrix':
        seq.set_cross_capacitance([[1, .3], [-.7, 1]])
    elif field == 'groups':
        seq._sweep_group_ids = {'amplitude:hold:a': 'group1'}
    elif field == 'dc':
        seq.set_bias_t_compensation(mode='fixed_time', fixed_duration_cycles=6000)
    elif field == 'rc':
        seq.set_rc_compensation(20.)
    elif field == 'scales':
        seq.set_voltage_scales(500, (400, 600))
    elif field == 'dac_current':
        seq.dac_current_settings[0] = 20000
    elif field == 'firmware':
        cfg['gens'][0]['output_sum_version'] = 3
    elif field == 'clock':
        cfg['tprocs'][0]['f_time'] = 400.
    elif field == 'repetitions':
        kw['repetitions_per_sweep'] = 2
    elif field == 'validation':
        kw['compile_validation_mode'] = 'boundary'
    elif field == 'rf':
        kw['rf_pulses'] = (RfPulseConfig(0, 'hold', 30, 1000),)
    elif field == 'readout':
        kw['recovery_tproc_cycles'] = 90
    assert cache._key(cfg, seq, kw) != old


def test_changed_code_in_open_process_bypasses_and_restart_key_changes(monkeypatch):
    first = build()
    monkeypatch.setattr(cache, '_source_fingerprint', lambda: 'different source bytes')
    assert build().compile_cache_info['status'] == 'bypass'
    monkeypatch.setattr(cache, '_loaded_source_fingerprint', 'different source bytes')
    revised = build()
    assert revised.compile_cache_info['status'] == 'miss'
    assert revised.compile_cache_info['key'] != first.compile_cache_info['key']


@pytest.mark.parametrize('damage', ['gzip', 'checksum', 'class'])
def test_corrupt_or_unsupported_cache_recompiles(damage):
    first = build()
    path = next(cache.cache_directory().glob('*.json.gz'))
    if damage == 'gzip':
        path.write_bytes(b'broken artifact')
    else:
        doc = json.loads(gzip.decompress(path.read_bytes()))
        if damage == 'checksum':
            doc['sha256'] = '0' * 64
        else:
            node = next(n for n in doc['program']['nodes'] if n[0] == 'object')
            node[1] = 'os.system'
            doc['sha256'] = hashlib.sha256(cache._json(doc['program'])).hexdigest()
        path.write_bytes(gzip.compress(cache._json(doc)))
    cache.clear_memory_cache()
    second = build()
    assert second.compile_cache_info['status'] == 'miss'
    assert second.binprog == first.binprog


def test_unwritable_cache_and_disable_still_compile(tmp_path, monkeypatch):
    blocked = tmp_path / 'file'; blocked.write_text('not a directory')
    monkeypatch.setenv(cache.CACHE_DIR_ENV, str(blocked))
    assert build().compile_cache_info['status'] == 'bypass'
    monkeypatch.setenv(cache.CACHE_ENABLED_ENV, '0')
    assert not hasattr(build(), 'compile_cache_info')


def test_cancel_callback_rebound_and_failed_build_not_cached(monkeypatch):
    calls = []
    first = build(cancel_check=lambda: calls.append('old'))
    calls.clear()
    current = lambda: calls.append('current')
    second = build(cancel_check=current)
    assert second.cancel_check is current
    assert calls and set(calls) == {'current'}
    class Cancelled(Exception):
        pass
    def cancel():
        raise Cancelled()
    with pytest.raises(Cancelled):
        build(cancel_check=cancel)
    before = set(cache.cache_directory().glob('*'))
    def fail(*args, **kwargs):
        raise ValueError('compile failed')
    monkeypatch.setattr(FineTuneSequence, 'make_program', fail)
    with pytest.raises(ValueError, match='compile failed'):
        build(compile_validation_mode='boundary')
    assert set(cache.cache_directory().glob('*')) == before
    assert first.binprog == second.binprog


def test_fine_rf_envelopes_dmem_and_hardware_loop_match():
    cfg = _shared_tmux_soccfg(); cfg['gens'][0]['f_dds'] = 4800.
    seq = FineTuneSequence(('awg_0',)).add_set('gate', .01, 400_000)
    seq.add_amplitude_sweep('gate', 'awg_0', .01, .02, 2)
    seq.add_rf_duration_sweep('gate', 0, 1/4800., 112/4800., 112,
                             segment_length_mode='fixed', sequence_fabric_mhz=300.)
    rf = RfPulseConfig(0, 'gate', 3, 2000, freq_mhz=192., phase_degrees=45.,
                      delay_tproc_cycles=30, fine_duration=True, duration_samples=1)
    kwargs = dict(awg_channels=(1,), rf_pulses=(rf,), repetitions_per_sweep=2,
                  compile_validation_mode='boundary')
    first = cache.build_cached_program(cfg, seq, kwargs)
    assert first.compile_cache_info['status'] == 'miss'
    cache.clear_memory_cache()
    second = cache.build_cached_program(cfg, seq, kwargs)
    assert second.compile_cache_info['status'] == 'hit'
    assert first.binprog == second.binprog
    assert first._runtime_dmem_words == second._runtime_dmem_words
    assert execution(first) == execution(second)
    assert len(second.envelopes[0]['envs']) == 16
    for name, env in first.envelopes[0]['envs'].items():
        np.testing.assert_array_equal(env['data'], second.envelopes[0]['envs'][name]['data'])


def test_show_hit_can_add_coordinate_snapshot_for_subsequent_runs(monkeypatch):
    first = build()
    shown = build()
    shown._execution_coordinate_snapshot = {'points': [1, 2, 3]}
    cache.save_execution_snapshot(shown)
    cache.clear_memory_cache()
    restored = build()
    assert restored._execution_coordinate_snapshot == shown._execution_coordinate_snapshot
    # It must return before attempting to recompute any coordinates.
    restored.soccfg = None
    experiment._cache_execution_coordinates(restored)
    restored.acquire_params = {'started': True}
    with pytest.raises(cache.Uncacheable, match='acquisition'):
        cache._write(first.compile_cache_info['key'], restored)


def test_disk_cache_survives_fresh_python_process():
    first = build()
    code = '''
import sys, json
sys.path[:0] = json.loads(sys.argv[1])
from test_qick_program_cache import build
p = build()
print(json.dumps({'cache': p.compile_cache_info, 'binary': p.binprog}))
'''
    result = subprocess.run([sys.executable, '-B', '-c', code,
                             json.dumps([str(Path(__file__).parent)] + sys.path)],
                            cwd='C:/GIT_Workspace/PulseGenerator', capture_output=True,
                            text=True, check=True, timeout=60)
    doc = json.loads(result.stdout.splitlines()[-1])
    assert doc['cache']['source'] == 'disk'
    assert doc['binary'] == first.binprog


def test_cache_hit_still_configures_and_acquires_new_data(monkeypatch):
    cfg = _fir_soccfg()
    seq = FineTuneSequence(('a',)).add_set('hold', .1, 30000)
    spec = experiment.QickDdrReadoutSpec(0, 'hold', 0., 3)
    calls = []
    monkeypatch.setattr(experiment, 'configure_rf_readout',
                        lambda soc, spec: calls.append('configure') or {})
    from qick_fine_tune_sweep import FineTuneAmplitudeSweepProgram
    def acquire(self, soc, **kwargs):
        calls.append('acquire')
        self.acquire_params = {'run': len(calls)}
        return {'new_measurement': len(calls)}
    monkeypatch.setattr(FineTuneAmplitudeSweepProgram, 'acquire_fir_ddr', acquire)
    args = dict(awg_channels=(0,), repetitions_per_sweep=1, rf_specs=(), readout_spec=spec)
    p1, data1, _ = experiment._execute_qick_sequence_once(object(), cfg, seq, **args)
    p2, data2, _ = experiment._execute_qick_sequence_once(object(), cfg, seq, **args)
    assert p1.compile_cache_info['status'] == 'miss'
    assert p2.compile_cache_info['status'] == 'hit'
    assert calls == ['configure', 'acquire', 'configure', 'acquire']
    assert data1 != data2


def test_cache_disk_entry_limit(monkeypatch):
    monkeypatch.setattr(cache, 'MAX_ENTRIES', 2)
    for n in range(4):
        seq = sequence(); seq.set_cross_capacitance([[1, n / 100.], [0, 1]])
        build(seq)
    assert len(list(cache.cache_directory().glob('*.json.gz'))) == 2


@pytest.mark.parametrize('shape', [(1, 1, 1), (1, 2, 1), (2, 1, 2)])
def test_v6_short_segments_cache_preserves_exact_timestamps(shape):
    from test_awg_v6_timing import configuration
    rise, hold, fall = shape
    seq = (FineTuneSequence(('x',)).add_set('zero', (0.,), 1)
           .add_ramp('rise', rise).add_set('hold', (.02,), hold)
           .add_ramp('fall', fall).add_set('end', (0.,), 1))
    cfg = configuration(); kwargs = dict(awg_channels=(1,))
    original = cache.build_cached_program(cfg, seq, kwargs)
    cache.clear_memory_cache()
    cached = cache.build_cached_program(cfg, seq, kwargs)
    assert cached.compile_cache_info['status'] == 'hit'
    assert cached.timing == original.timing
    assert execution(cached) == execution(original)


@pytest.mark.parametrize('mode', ['fixed_voltage', 'fixed_time'])
def test_grouped_matrix_dc_rc_cache_execution(mode):
    from test_grouped_sweep_compiler import grouped_sequence, config
    from test_rc_precompensation import upgraded
    seq = grouped_sequence(4)
    seq.set_cross_capacitance(((1., .2), (-.15, 1.)))
    seq.set_bias_t_compensation(.1, mode=mode,
                               fixed_duration_cycles=3000 if mode == 'fixed_time' else None)
    seq.set_rc_compensation(20.)
    cfg = upgraded(config())
    kwargs = dict(awg_channels=(0, 1), repetitions_per_sweep=2, compile_validation_mode='boundary')
    original = cache.build_cached_program(cfg, seq, kwargs)
    cached = cache.build_cached_program(cfg, seq, kwargs)
    assert cached.compile_cache_info['status'] == 'hit'
    assert execution(cached) == execution(original)
