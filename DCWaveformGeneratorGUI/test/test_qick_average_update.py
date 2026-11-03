"""The v7 file updater must verify the new Python API after a restart."""

import sys
from types import SimpleNamespace

import pytest

import qick_update as update
import qick_update_remote as remote


def test_incomplete_library_is_rejected_during_local_review(tmp_path):
    assert update.library_runtime_capabilities(tmp_path) == {}
    (tmp_path / 'trace_averaging.py').write_text('FIR_AVERAGE_API_VERSION = 1\n')
    (tmp_path / 'qick.py').write_text('class QickSoc:\n    pass\n')
    with pytest.raises(ValueError, match='Incomplete'):
        update.library_runtime_capabilities(tmp_path)
    (tmp_path / 'qick.py').write_text('class QickSoc:\n    def get_fir_average_capabilities(self): pass\n    def get_ddr4_fir_mean(self): pass\n')
    assert update.library_runtime_capabilities(tmp_path) == {'fir_average_api_version': 1}


@pytest.mark.parametrize('capabilities', [{}, {'fir_average_api_version': 0}, {'fir_average_api_version': 1}])
def test_v7_version_alone_is_not_enough_after_update(monkeypatch, capabilities):
    # Same v7 timestamp and package version can be present with an old server.
    runtime = dict(board='ZCU216', fw_timestamp='v7-timestamp', sw_version='0.2.357',
                   process_started=2., python_executable=sys.executable, generators=[],
                   runtime_capabilities=capabilities)
    monkeypatch.setattr(remote, 'live', lambda *args, **kwargs: runtime)
    manifest = dict(firmware_timestamp='v7-timestamp', library_version='0.2.357', generators=[],
                    required_runtime_capabilities={'fir_average_api_version': 1})
    if capabilities.get('fir_average_api_version') == 1:
        assert remote.verify_runtime({}, manifest, 1.) == runtime
    else:
        with pytest.raises(ValueError, match='restart'):
            remote.verify_runtime({}, manifest, 1.)
    runtime['process_started'] = .5
    with pytest.raises(ValueError, match='Restart'):
        remote.verify_runtime({}, manifest, 1.)


def test_runtime_verification_does_not_leave_short_rpc_timeout(monkeypatch):
    import Pyro4
    def fail(**kw):
        assert Pyro4.config.COMMTIMEOUT == 4
        raise RuntimeError('offline nameserver')
    monkeypatch.setattr(Pyro4, 'locateNS', fail)
    timeout = Pyro4.config.COMMTIMEOUT
    client = SimpleNamespace(connection=SimpleNamespace(host='192.0.2.1', ns_port=8888))
    with pytest.raises(RuntimeError, match='offline'):
        update.BoardUpdater.verify(client, 'test')
    assert Pyro4.config.COMMTIMEOUT == timeout
