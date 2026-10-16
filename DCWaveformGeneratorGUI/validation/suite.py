"""Expected-case ledger. A partial fixture cannot silently complete the TODO."""
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import uuid

from .archive import atomic, canonical, digest, json_file, private_evidence_root, verify_archive
from .coverage import REQUIRED_CATEGORIES, logarithmic_durations_us


def duration_case_plan(segment):
    """Requested geometric grid, explicitly ten independent GUI executions."""
    if segment not in ('hold', 'ramp'):
        raise ValueError('Select hold or ramp')
    return [dict(name=f'{segment}_log_{i:02d}', category=f'{segment}_duration',
                 duration_us=value, execution='individual_gui_log_grid')
            for i, value in enumerate(logarithmic_durations_us())]


class SuiteManifest:
    """Incremental immutable attempts and explicit outstanding case/fixture gaps.

    This class never runs a GUI or device. The caller creates one configured
    GuiHardwareRunner at a time and records its finalized manifest here.
    """
    def __init__(self, root=None, *, name, fixture, planned_cases):
        self.root = private_evidence_root(root)/('suite_' + uuid.uuid4().hex)
        self.rows = {}
        for case in planned_cases:
            if case.get('category') not in REQUIRED_CATEGORIES:
                raise ValueError('Every planned case needs a known coverage category')
            key = digest(canonical(case))[:20]
            if key in self.rows:
                raise ValueError('Duplicate planned case')
            self.rows[key] = dict(case=case, status='pending', attempts=[])
        if not self.rows:
            raise ValueError('Declare planned cases before execution')
        self.fixture_coverage = fixture.coverage()
        self.root.mkdir(parents=True, exist_ok=False)
        self.revision = 0
        json_file(self.root/'plan.json', dict(name=name, cases=self.rows,
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  fixture_coverage=self.fixture_coverage))
        self.checkpoint()

    def record(self, archive_directory):
        folder = private_evidence_root(archive_directory)
        result = json.loads((folder/'manifest.json').read_text(encoding='utf8'))
        row = self.rows[result['case_id']]
        if any(a['run_id'] == result['run_id'] for r in self.rows.values() for a in r['attempts']):
            raise ValueError('Duplicate run attempt')
        checked = verify_archive(folder)
        status = result['verdict'] if result['complete'] and checked['passed'] else 'inconclusive'
        row['attempts'].append(dict(run_id=result['run_id'], directory=str(folder),
            status=status, manifest_sha256=digest((folder/'manifest.json').read_bytes()),
            reload_integrity=checked))
        row['status'] = status  # Previous failed attempts remain in the same ledger.
        self.checkpoint()

    def unavailable(self, case, reason):
        key = digest(canonical(case))[:20]
        if not reason or self.rows[key]['attempts']:
            raise ValueError('Unavailable needs a reason and no conflicting acquisition')
        self.rows[key].update(status='unavailable', reason=reason)
        self.checkpoint()

    def checkpoint(self):
        coverage = dict(self.fixture_coverage)
        for category in REQUIRED_CATEGORIES:
            selected = [r for r in self.rows.values() if r['case']['category'] == category]
            if coverage[category]['status'] != 'unavailable' and selected:
                states = {r['status'] for r in selected}
                status = 'passed' if states == {'passed'} else 'failed' if 'failed' in states else 'inconclusive'
                coverage[category] = dict(status=status, reason='Aggregate of explicitly planned cases')
        result = dict(cases=self.rows, coverage=coverage,
                      planned_cases_complete=all(r['status'] in ('passed', 'failed') for r in self.rows.values()),
                      full_suite_passed=all(r['status'] == 'passed' for r in coverage.values()))
        path = self.root/f'status_{self.revision:06d}.json'
        json_file(path, result)
        links = []
        for key, row in self.rows.items():
            attempts = ' '.join(f'<a href="{html.escape(os.path.relpath(Path(a["directory"])/"index.html", self.root).replace(chr(92), "/"))}">{html.escape(a["status"])}</a>'
                                for a in row['attempts'])
            links.append(f'<tr><td>{html.escape(row["case"]["name"])}</td><td>{row["status"]}</td><td>{attempts}</td></tr>')
        atomic(path.with_suffix('.html'), ('<!doctype html><meta charset="utf-8"><title>Validation suite</title>'
              '<p>Missing fixtures and unevaluated categories remain incomplete.</p><table>'+''.join(links)+'</table>').encode())
        self.revision += 1
        return result
