"""Incremental, immutable numerical evidence and expected/recorded accounting."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import html
import io
import json
import os
from pathlib import Path
import uuid

import numpy as np


DEFAULT_EVIDENCE_ROOT = Path('F:/QICK_Private_Analysis/gui_hardware_validation')


def private_evidence_root(root=None):
    """Evidence is private: refuse any path inside a Git checkout/worktree."""
    path = Path(root if root is not None else DEFAULT_EVIDENCE_ROOT).resolve()
    for parent in (path, *path.parents):
        if (parent / '.git').exists():
            raise ValueError('Hardware evidence must be outside every Git checkout')
    # Reject repository-shaped targets even when a caller bypasses a missing
    # checkout marker (for example a not-yet-created analysis_results folder).
    if 'analysis_results' in (part.lower() for part in path.parts):
        raise ValueError('Use a private external evidence directory, not analysis_results')
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        text = str(path)
        path = Path('\\\\?\\UNC\\' + text[2:] if text.startswith('\\\\') else '\\\\?\\' + text)
    return path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, data):
    """Publish a new file atomically, without replacing any previous evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + uuid.uuid4().hex + '.partial')
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, path)  # Atomic no-replace publication, same filesystem.
    finally:
        temp.unlink()


def json_file(path, value):
    atomic(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode('utf8'))


def npz_bytes(**arrays):
    if any(np.asarray(v).dtype.hasobject for v in arrays.values()):
        raise ValueError('Object arrays are not lossless numerical records')
    data = io.BytesIO()
    np.savez_compressed(data, **arrays)
    raw = data.getvalue()
    with np.load(io.BytesIO(raw), allow_pickle=False) as check:
        for key, values in arrays.items():
            assert np.array_equal(check[key], values, equal_nan=True)
    return raw


@dataclass(frozen=True)
class Point:
    index: int
    coordinates: dict
    repetitions: int = 1
    groups: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Reference:
    time_s: np.ndarray
    voltage_v: np.ndarray
    metadata: dict

    def arrays(self):
        t, v = np.asarray(self.time_s), np.asarray(self.voltage_v)
        if t.ndim != 1 or t.shape != v.shape or not len(t):
            raise ValueError('Reference time/value must be nonempty matching one-dimensional arrays')
        if not np.isfinite(t).all() or not np.isfinite(v).all() or np.any(np.diff(t) < 0):
            raise ValueError('Reference values must be finite and time ordered')
        required = {'units', 'time_origin', 'reference_plane', 'model_version', 'transformations'}
        if not required.issubset(self.metadata):
            raise ValueError(f'Reference metadata needs {sorted(required)}')
        return t, v


class RunArchive:
    """One attempt, never reused. Every point/repetition/channel has an identity.

    References may be deduplicated, measurements are acquired once and linked
    by native ROI indices. Full scope records remain intact; derived plots and
    residuals are separate. No missing/invalid measurement becomes a PASS.
    """
    def __init__(self, root=None, *, case, fixture, identity, points, measured_channels=(1, 2),
                 execution='hardware_loop', retry_of=None):
        if execution not in ('hardware_loop', 'individual_gui_log_grid', 'individual_gui'):
            raise ValueError('Specify actual execution semantics')
        fixture.validate()
        identity_data = identity.verify(case.get('required_capabilities', ()))
        self.case_id = digest(canonical(case))[:20]
        self.run_id = str(uuid.uuid4())
        self.root = private_evidence_root(root) / self.case_id / self.run_id
        self.fixture = fixture
        self.identity = identity
        self.channels = tuple(str(c) for c in measured_channels)
        if set(self.channels) != {'1', '2'}:
            raise ValueError('Scope archive requires A/B channels 1 and 2')
        points = tuple(points)
        if [p.index for p in points] != list(range(len(points))) or not points:
            raise ValueError('Points must be nonempty, contiguous, ordered, unique indices')
        if any(p.repetitions < 1 for p in points):
            raise ValueError('Each point needs at least one individual repetition')
        self.root.mkdir(parents=True, exist_ok=False)
        self.expected = {}
        for point in points:
            for rep in range(point.repetitions):
                for ch in self.channels:
                    key = self.key(point.index, rep, ch)
                    self.expected[key] = dict(point_id=f'{self.case_id}:p{point.index:08d}',
                        point_index=point.index, repetition_id=f'r{rep:06d}', repetition_index=rep,
                        channel_id=ch, coordinates=point.coordinates, groups=point.groups,
                        references={}, status='missing')
        self.acquisitions = {}
        self.events = []
        self.coverage = fixture.coverage()
        self.sealed = False
        self.finalized = False
        self.last_order = {ch: (-1, -1) for ch in self.channels}
        json_file(self.root/'run.json', dict(schema_version=1, run_id=self.run_id,
            case_id=self.case_id, case=case, retry_of=retry_of, execution=execution,
            fixture=asdict(fixture), identity=identity_data,
            created_utc=datetime.now(timezone.utc).isoformat(),
            numerical_format='lossless NPZ plus original IEEE WORD gzip/preamble JSON',
            physical_validation=False, note='Initialized only; physical execution is established by recorded evidence'))
        self.checkpoint()

    @staticmethod
    def key(point, repetition, channel):
        return f'p{int(point):08d}/r{int(repetition):06d}/ch{channel}'

    def event(self, name, **detail):
        if self.finalized:
            raise RuntimeError('Finalized evidence is immutable')
        row = dict(sequence=len(self.events), utc=datetime.now(timezone.utc).isoformat(),
                   name=name, detail=detail)
        json_file(self.root/'events'/f'{len(self.events):06d}.json', row)
        self.events.append(row)

    def checkpoint(self):
        folder = self.root/'checkpoints'
        revision = len(list(folder.glob('*.json'))) if folder.exists() else 0
        json_file(folder/f'{revision:06d}.json', dict(expected=self.expected,
            acquisitions=self.acquisitions, coverage=self.coverage, sealed=self.sealed))

    def record_coverage(self, category, status, *, reason, evidence):
        if category not in self.coverage or status not in ('passed', 'failed', 'inconclusive', 'unavailable'):
            raise ValueError('Unknown coverage category or status')
        if self.coverage[category]['status'] == 'unavailable' and status != 'unavailable':
            raise ValueError('Missing physical fixture cannot be replaced by other measurements')
        if not reason or not evidence:
            raise ValueError('Coverage needs a reason and explicit evidence references')
        self.coverage[category] = dict(status=status, reason=reason, evidence=evidence)
        self.checkpoint()

    def add_reference(self, point, channel, kind, reference):
        if self.sealed:
            raise RuntimeError('Intended references are immutable once execution is sealed')
        if kind not in ('requested', 'quantized', 'physical'):
            raise ValueError('Keep requested, quantized and physical expectations separate')
        t, v = reference.arrays()
        token = digest(canonical(reference.metadata) + canonical([t.dtype.str, v.dtype.str])
                       + t.tobytes() + v.tobytes())
        path = self.root/'references'/f'{token}.npz'
        if not path.exists():
            atomic(path, npz_bytes(time_s=t, voltage_v=v))
            json_file(path.with_suffix('.json'), reference.metadata)
        targets = [r for r in self.expected.values()
                   if r['point_index'] == point and r['channel_id'] == str(channel)]
        if not targets:
            raise KeyError((point, channel))
        for row in targets:
            if kind in row['references']:
                raise ValueError('Reference already assigned')
            row['references'][kind] = str(path.relative_to(self.root)).replace('\\', '/')

    def seal(self, *, gui_definition):
        if self.sealed:
            raise RuntimeError('Already sealed')
        missing = [k for k, v in self.expected.items()
                   if set(v['references']) != {'requested', 'quantized', 'physical'}]
        if missing:
            raise ValueError(f'Numerical references missing before execution: {missing[:5]}')
        self.identity.verify()
        json_file(self.root/'gui_requested.json', gui_definition)
        json_file(self.root/'intended_manifest.json', self.expected)
        self.sealed = True
        self.event('intended_sealed')
        self.checkpoint()

    def record_capture(self, metadata, directory):
        """Reload complete native channels and also retain calibrated arrays."""
        from ._scope.compact_capture import reload_channel, scaled_slice
        if not self.sealed:
            raise RuntimeError('Archive intended waveforms before acquisition')
        acquisition = str(metadata['capture_id'])
        if acquisition in self.acquisitions:
            raise ValueError('Duplicate or stale acquisition identity')
        if not metadata.get('same_acquisition') or not metadata.get('validation_passed'):
            raise ValueError('Invalid/incomplete acquisition; keep partial files, never mark recorded')
        directory = Path(directory).resolve()
        directory.relative_to(self.root)
        channels = {}
        for ch in self.channels:
            row = metadata['channels'][ch]
            info = row['archive']
            raw, pre = directory/Path(info['path']).name, directory/Path(info['preamble_path']).name
            if digest(raw.read_bytes()) != info['gzip_sha256'] or digest(pre.read_bytes()) != info['preamble_sha256']:
                raise ValueError('Original RAW/preamble hash mismatch')
            preamble_identity = json.loads(pre.read_text(encoding='utf8'))
            if str(preamble_identity.get('capture_id')) != acquisition or str(preamble_identity.get('channel')) != ch:
                raise ValueError('Stale or wrong-channel preamble identity')
            loaded = reload_channel(pre, raw)
            if loaded['raw_sha256'] != info['raw_sha256'] or loaded['payload_sha256'] != row['payload_sha256']:
                raise ValueError('RAW reload identity mismatch')
            t, v, invalid = scaled_slice(loaded)
            target = directory/f'ch{ch}.native.npz'
            atomic(target, npz_bytes(raw_codes=loaded['raw_codes'], time_s=t, voltage_v=v, invalid=invalid))
            channels[ch] = dict(path=str(target.relative_to(self.root)).replace('\\','/'),
                samples=len(t), invalid_samples=int(invalid.sum()), preamble=loaded['preamble'],
                raw_path=str(raw.relative_to(self.root)).replace('\\','/'),
                preamble_path=str(pre.relative_to(self.root)).replace('\\','/'))
        self.acquisitions[acquisition] = dict(channels=channels,
            identity_basis=metadata.get('same_acquisition_basis'), native_time_axes_preserved=True,
            measurement_kind=metadata.get('measurement_kind', 'unspecified'))
        self.event('capture_archived', acquisition_id=acquisition)
        self.checkpoint()
        return acquisition

    def link_measurement(self, point, repetition, channel, acquisition, start, stop, *, identity_evidence):
        """Use independently verified marker/sequence ROI identity, not guessed order."""
        key = self.key(point, repetition, channel)
        row = self.expected[key]
        if row['status'] != 'missing':
            raise ValueError('Duplicate point/repetition/channel measurement')
        if identity_evidence.get('verified') is not True or not identity_evidence.get('method'):
            raise ValueError('Point and repetition identity need verified evidence')
        order = (int(point), int(repetition))
        if order <= self.last_order[str(channel)]:
            raise ValueError('Out-of-order point or repetition mapping')
        capture = self.acquisitions[acquisition]['channels'][str(channel)]
        if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start < stop <= capture['samples']:
            raise ValueError('ROI outside native acquisition')
        for previous in self.expected.values():
            if previous.get('acquisition_id') == acquisition and previous['channel_id'] == str(channel):
                a, b = previous['native_roi']
                if start < b and a < stop:
                    raise ValueError('Overlapping point/repetition ROIs')
        row.update(status='recorded', acquisition_id=acquisition, native_roi=[start, stop],
                   identity_evidence=identity_evidence, comparison_status='inconclusive')
        self.last_order[str(channel)] = order

    def compare(self, point, repetition, channel, *, reference_kind='physical',
                time_offset_s=0., voltage_offset_v=0., absolute_tolerance_v=None):
        """Explicit derived interpolation/offset; original data remain untouched."""
        key = self.key(point, repetition, channel)
        row = self.expected[key]
        if row['status'] != 'recorded' or 'comparison' in row:
            raise ValueError('Comparison needs a fresh recorded measurement')
        native = self.acquisitions[row['acquisition_id']]['channels'][str(channel)]
        with np.load(self.root/native['path'], allow_pickle=False) as data:
            a, b = row['native_roi']
            t = data['time_s'][a:b].copy()
            measured = data['voltage_v'][a:b].copy()
            invalid = data['invalid'][a:b].copy()
        with np.load(self.root/row['references'][reference_kind], allow_pickle=False) as data:
            rt, rv = data['time_s'].copy(), data['voltage_v'].copy()
        aligned = t - float(time_offset_s)
        corrected = measured - float(voltage_offset_v)
        if not np.isfinite(time_offset_s) or not np.isfinite(voltage_offset_v):
            raise ValueError('Derived offsets must be finite and explicit')
        covered = bool(aligned.min() >= rt.min() and aligned.max() <= rt.max())
        valid = covered and not invalid.any() and np.isfinite(corrected).all()
        expected = np.interp(aligned, rt, rv) if covered else np.full(len(t), np.nan)
        residual = corrected - expected
        status = 'inconclusive'
        if absolute_tolerance_v is not None:
            if not np.isfinite(absolute_tolerance_v) or absolute_tolerance_v < 0:
                raise ValueError('Specify a finite nonnegative absolute tolerance')
            if valid:
                status = 'passed' if np.max(np.abs(residual)) <= absolute_tolerance_v else 'failed'
        derived = self.root/'comparisons'/key
        atomic(derived.with_suffix('.npz'), npz_bytes(original_time_s=t, original_voltage_v=measured,
            aligned_time_s=aligned, corrected_voltage_v=corrected, expected_voltage_v=expected, residual_v=residual))
        report = dict(status=status, numerical_comparison_valid=bool(valid), reference_kind=reference_kind,
            samples=len(t), covered=covered, invalid_samples=int(invalid.sum()),
            absolute_tolerance_v=absolute_tolerance_v, time_offset_s=time_offset_s,
            voltage_offset_v=voltage_offset_v, gain_fit=False, interpolation='numpy linear interpolation of supplied reference',
            max_abs_residual_v=float(np.max(np.abs(residual))) if valid else None,
            note='Interpolation and offsets are derived only; supplying a sampled reference does not certify analog bandwidth/timing.')
        json_file(derived.with_suffix('.json'), report)
        self._plot(derived.with_suffix('.png'), aligned, corrected, expected, residual, key, row)
        row.update(comparison_status=status, comparison=str(derived.relative_to(self.root)).replace('\\','/'))
        self.checkpoint()
        return report

    @staticmethod
    def _plot(path, time, voltage, expected, residual, key, row):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, constrained_layout=True)
        axes[0].plot(time, voltage, label='Physical measured (explicit correction)')
        axes[0].plot(time, expected, label='Supplied physical reference')
        axes[0].set(ylabel='V', title=f'{key} / {row["coordinates"]}')
        axes[0].legend()
        axes[1].plot(time, residual)
        axes[1].set(xlabel='Reference time (s)', ylabel='Residual (V)')
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def finalize(self, *, worker_status, cleanup, storage_identity=None, error=None):
        if self.finalized:
            raise RuntimeError('Attempt already finalized; retry requires a new run')
        missing = [key for key, row in self.expected.items() if row['status'] == 'missing']
        comparison_missing = [key for key, row in self.expected.items()
                              if row.get('comparison_status') not in ('passed', 'failed')]
        source_ok = True
        try:
            self.identity.verify()
        except Exception as exc:
            source_ok = False
            self.event('source_changed', error=str(exc))
        storage_ok = bool(storage_identity and all(storage_identity.get(key)
                          for key in ('run_id', 'guid', 'database_path')))
        complete = bool(self.sealed and not missing and not comparison_missing and worker_status == 'finished'
                        and cleanup.get('passed') is True and source_ok and storage_ok and error is None)
        verdict = ('failed' if any(r.get('comparison_status') == 'failed' for r in self.expected.values())
                   else 'passed') if complete else 'inconclusive'
        self.checkpoint()
        self._index()
        files = {str(p.relative_to(self.root)).replace('\\','/'):dict(bytes=p.stat().st_size, sha256=digest(p.read_bytes()))
                 for p in self.root.rglob('*') if p.is_file() and not p.name.endswith('.partial')}
        verified = _verify_files(self.root, files)
        complete = complete and verified['passed']
        if not complete:
            verdict = 'inconclusive'
        result = dict(schema_version=1, run_id=self.run_id, case_id=self.case_id, complete=complete,
            verdict=verdict, missing_measurements=missing, missing_comparisons=comparison_missing,
            worker_status=worker_status, cleanup=cleanup, storage_identity=storage_identity,
            source_hashes_match=source_ok, archive_verified=verified['passed'], error=error, expected=self.expected,
            acquisitions=self.acquisitions, coverage=self.coverage,
            execution_route=(json.loads((self.root/'execution_route.json').read_text(encoding='utf8'))
                             if (self.root/'execution_route.json').exists() else
                             dict(path='not_observed', scope_only_bypass=None)),
            full_suite_complete=all(r['status']=='passed' for r in self.coverage.values()), files=files)
        json_file(self.root/'manifest.json', result)
        self.finalized = True
        json_file(self.root/'reload_integrity.json', verified)
        return result

    def _index(self):
        rows = []
        for key, row in self.expected.items():
            comparison = row.get('comparison')
            link = f'<a href="{html.escape(comparison)}.png">overlay/residual</a>' if comparison else 'not available'
            native = self.acquisitions.get(row.get('acquisition_id'),{}).get('channels',{}).get(row['channel_id'])
            data = f'<a href="{native["path"]}">full native NPZ</a>' if native else 'missing'
            refs = ' '.join(f'<a href="{path}">{kind}</a>' for kind,path in row['references'].items())
            rows.append(f'<tr><td>{html.escape(key)}</td><td>{html.escape(str(row["coordinates"]))}</td>'
                        f'<td>{row.get("comparison_status",row["status"])}</td><td>{link}</td><td>{data}</td><td>{refs}</td></tr>')
        atomic(self.root/'index.html', ('<!doctype html><meta charset="utf-8"><title>GUI physical validation</title>'
            '<h1>GUI physical validation</h1><p>Original records are immutable; corrections are separate derived records. '
            'Missing third-output/marker fixtures remain unavailable.</p><table><tr><th>Identity</th><th>Coordinates</th>'
            '<th>Status</th><th>Plot</th><th>Measurement</th><th>References</th></tr>' + ''.join(rows) + '</table>').encode('utf8'))


def verify_archive(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    return _verify_files(folder, manifest['files'])


def _verify_files(folder, files):
    errors = []
    for rel, expected in files.items():
        path = (folder/rel).resolve()
        try:
            path.relative_to(folder)
            if path.stat().st_size != expected['bytes'] or digest(path.read_bytes()) != expected['sha256']:
                raise ValueError('content hash mismatch')
            if path.suffix == '.npz':
                with np.load(path, allow_pickle=False) as data:
                    for key in data.files:
                        data[key]  # Verify CRC and decompress every numerical array.
            if path.suffix == '.gz':
                gzip.decompress(path.read_bytes())
        except Exception as exc:
            errors.append(dict(path=rel, error=str(exc)))
    return dict(passed=not errors, files=len(files), errors=errors)
