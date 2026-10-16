"""Fixture capabilities and explicit coverage gaps for the v5 suite."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import numpy as np


REQUIRED_CATEGORIES = (
    'hold_duration', 'ramp_duration', 'rf_frequency', 'voltage',
    'grouped_voltage', 'grouped_lifecycle', 'rf_duration_within',
    'rf_duration_crossing', 'rf_duration_boundaries', 'combined_axes',
    'matrix_dc_rc', 'repetition_reset', 'failure_recovery',
    'third_output_loopback', 'marker_acquisition_timing',
)


def logarithmic_durations_us():
    """Ten individual GUI runs; not a native logarithmic FPGA sweep axis."""
    return tuple(float(v) for v in np.geomspace(.1, 100., 10))


@dataclass(frozen=True)
class Fixture:
    scope_resource: str
    scope_identity: str
    channels: dict
    third_output_loopback: dict | None = None
    marker: dict | None = None
    notes: str = ''

    def validate(self):
        if not self.scope_resource or not self.scope_identity:
            raise ValueError('Explicit scope resource and queried identity are required')
        if set(map(str, self.channels)) != {'1', '2'}:
            raise ValueError('This adapter requires exactly scope channels 1 and 2')
        rows = list(self.channels.values())
        for row in rows:
            for key in ('generator', 'physical_connector', 'cable', 'termination_ohm',
                        'probe_ratio', 'attenuation_db', 'coupling'):
                if key not in row:
                    raise ValueError(f'Fixture channel lacks {key}')
        if rows[0]['generator'] == rows[1]['generator']:
            raise ValueError('A and B must be distinct AWG outputs')
        if self.third_output_loopback:
            if self.third_output_loopback['generator'] in {r['generator'] for r in rows}:
                raise ValueError('C must be a separate third AWG, not a split of A or B')

    def coverage(self):
        self.validate()
        result = {key: {'status': 'inconclusive', 'reason': 'No physical case has been evaluated'}
                  for key in REQUIRED_CATEGORIES}
        if not self.third_output_loopback:
            result['third_output_loopback'] = {'status': 'unavailable',
                'reason': 'Separate third AWG-to-QICK input mapping/calibration is not provided'}
        if not self.marker:
            result['marker_acquisition_timing'] = {'status': 'unavailable',
                'reason': 'QICK marker-to-scope trigger wiring is not provided'}
        return result


@dataclass(frozen=True)
class FirmwareIdentity:
    """Explicit loaded-board identity plus local source content hashes.

    A local filename/branch is insufficient. The caller supplies the queried
    loaded-board configuration and a separately verified BIT/HWH relationship.
    """
    firmware_version: str
    loaded_board_config: dict
    bitstream_sha256: str
    hwh_sha256: str
    loaded_identity_evidence: dict
    source_files: dict
    capabilities: tuple[str, ...] = field(default_factory=tuple)

    def verify(self, required=()):
        if not self.firmware_version or not self.loaded_board_config:
            raise ValueError('Loaded firmware identity is required')
        for digest in (self.bitstream_sha256, self.hwh_sha256):
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest.lower()):
                raise ValueError('BIT and HWH require SHA256 identifiers')
        evidence = self.loaded_identity_evidence
        if evidence.get('verified') is not True or not evidence.get('method'):
            raise ValueError('Loaded firmware identity must be verified, not inferred from a path')
        if not self.source_files or not {'qick_driver', 'gui'}.issubset(self.source_files):
            raise ValueError('Archive both QICK driver and GUI source identities')
        for label, record in self.source_files.items():
            path = Path(record['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError(f'Source identity changed: {label}')
        missing = set(required) - set(self.capabilities)
        if missing:
            raise ValueError(f'Firmware capabilities unavailable: {sorted(missing)}')
        return asdict(self)

    def verify_observed(self, observed):
        """Compare a fresh read-only board probe to the declared loaded identity."""
        if observed.get('verified') is not True or not observed.get('method'):
            raise ValueError('Live identity requires a verified method and probe evidence')
        for key in ('firmware_version', 'bitstream_sha256', 'hwh_sha256', 'loaded_board_config'):
            wanted, found = getattr(self, key), observed.get(key)
            if json.dumps(wanted, sort_keys=True) != json.dumps(found, sort_keys=True):
                raise ValueError(f'Live loaded identity mismatch: {key}')
        if set(observed.get('capabilities', ())) != set(self.capabilities):
            raise ValueError('Live firmware capabilities differ from the intended case')
        return observed
