"""Explicit session ownership for the preserved Keysight time-domain adapter.

The runner executes every method on the same background executor. This module
does not open a resource on import or construction. No instrument defaults to
an address or silently adopts an unknown identity.
"""
from pathlib import Path
import gzip
import importlib.util
import uuid
from .archive import atomic, digest, json_file, private_evidence_root


def _atomic_adapter_class():
    """Keep protocol code unchanged; inject atomic evidence writers per session."""
    path = Path(__file__).parent / '_scope' / 'compact_capture.py'
    spec = importlib.util.spec_from_file_location('_scope_session_' + uuid.uuid4().hex, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._json_exclusive = json_file

    def save_raw(path, raw):
        path = Path(path)
        packed = gzip.compress(raw, compresslevel=6, mtime=0)
        if gzip.decompress(packed) != raw:
            raise ValueError('Lossless RAW compression verification failed')
        atomic(path, packed)
        return dict(path=str(path), raw_bytes=len(raw), raw_sha256=digest(raw),
                    gzip_bytes=len(packed), gzip_sha256=digest(packed), archive_verified=True)
    module._gzip_exclusive = save_raw
    return module.CompactScopeCapture


class ScopeSession:
    def __init__(self, *, resource_factory, archive_directory, expected_identity,
                 configuration, max_sample_interval_s, max_origin_skew_samples=0.,
                 timeout_s=10.):
        self.factory = resource_factory
        self.directory = private_evidence_root(archive_directory)
        self.expected_identity = expected_identity
        self.configuration = dict(configuration)
        self.max_sample_interval_s = max_sample_interval_s
        self.max_origin_skew_samples = max_origin_skew_samples
        self.timeout_s = timeout_s
        self.resource = self.adapter = None
        self.restoration = None

    def arm(self):
        CompactScopeCapture = _atomic_adapter_class()
        if self.resource is not None:
            raise RuntimeError('A session can arm only once; use a fresh acquisition')
        self.directory.mkdir(parents=True, exist_ok=False)
        try:
            self.resource = self.factory()
            self.adapter = CompactScopeCapture(self.resource, expected_serial=None,
                raw_archive_directory=self.directory/'raw',
                max_sample_interval_s=self.max_sample_interval_s,
                max_origin_skew_samples=self.max_origin_skew_samples)
            snapshot = self.adapter.preflight()
            if snapshot['idn'].strip() != self.expected_identity.strip():
                raise RuntimeError(f'Scope identity mismatch: {snapshot["idn"]}')
            self.adapter.archive_backup(self.directory/'backup')
            self.adapter.configure(**self.configuration)
            self.adapter.arm(timeout_s=self.timeout_s)
            return dict(capture_id=self.adapter.capture_id, snapshot=snapshot,
                        configured=self.adapter.configured_readback)
        except Exception:
            self.close()
            raise

    def capture(self):
        from ._scope.compact_capture import save_capture
        if self.adapter is None:
            raise RuntimeError('Arm before capture')
        try:
            record = self.adapter.wait_and_read(timeout_s=self.timeout_s)
            return save_capture(record, self.directory/'raw')
        finally:
            # Any partial IEEE block/preamble remains in raw, even on errors.
            json_file(self.directory/'capture_events.json', dict(
                capture_id=self.adapter.capture_id, events=self.adapter.events,
                partial_preambles=self.adapter.partial_preambles,
                partial_archives=self.adapter.partial_archives))

    def close(self):
        if self.restoration is not None:
            return self.restoration
        report = dict(passed=True, scope_restore=None, resource_closed=False)
        try:
            if self.adapter is not None:
                report['scope_restore'] = self.adapter.restore()
                report['passed'] = bool(report['scope_restore'].get('passed'))
        except Exception as exc:
            report.update(passed=False, restore_error=repr(exc))
        finally:
            if self.resource is not None:
                try:
                    self.resource.close()
                    report['resource_closed'] = True
                except Exception as exc:
                    report.update(passed=False, close_error=repr(exc))
            self.restoration = report
            if self.directory.exists():
                json_file(self.directory/'restoration.json', report)
        return report


def visa_resource_factory(address, *, backend=None, timeout_ms=5000):
    """Return an unopened factory. PyVISA and the resource manager are lazy."""
    if not address:
        raise ValueError('A configured resource address is required')

    def connect():
        import pyvisa
        manager = pyvisa.ResourceManager(backend) if backend else pyvisa.ResourceManager()
        try:
            resource = manager.open_resource(address)
            resource.timeout = timeout_ms
            resource.write_termination = '\n'
            resource.read_termination = '\n'
        except Exception:
            manager.close()
            raise

        class OwnedResource:
            def __getattr__(self, key):
                return getattr(resource, key)

            def __setattr__(self, key, value):
                setattr(resource, key, value)

            def close(self):
                try:
                    resource.close()
                finally:
                    manager.close()
        return OwnedResource()
    return connect
