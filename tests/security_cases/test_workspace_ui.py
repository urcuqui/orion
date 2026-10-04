"""Presentation must reflect persisted security state without executing work."""
from orion.context.store import save_profile
from orion.experiments.lifecycle import ExperimentLifecycle, save, next_action
from orion.workspace_view import overview


def test_overview_is_read_only_and_uses_lifecycle_action(tmp_path):
    ws = ExperimentLifecycle(stages={'plan': 'APPROVED', 'attack': 'COMPLETE',
        'measure': 'COMPLETE', 'defend': 'APPLIED', 'retest': 'READY'},
        active_experiment_id='E1', defense_id='CONTROL-1', attack_run_id='RUN-1')
    save(ws, str(tmp_path))
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = overview(str(tmp_path))
    assert result['action']['label'] == next_action(ws)['label'] == 'RETEST SAME EXPERIMENT'
    assert result['current']['stages']['retest'] == 'READY'
    assert result['current']['retest_run_id'] is None
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_missing_linked_profile_does_not_substitute_unrelated_system(tmp_path):
    save_profile('ORN-SELF-OTHER', {'system_type': 'generative_ai'}, str(tmp_path))
    ws = ExperimentLifecycle(self_profile_id='ORN-SELF-MISSING')
    save(ws, str(tmp_path))
    system = overview(str(tmp_path))['profiles'][0]
    assert system['id'] == 'ORN-SELF-MISSING'
    assert not system['available']
    assert system['name'] is None


def test_overview_preserves_available_profiles_when_another_is_malformed(tmp_path):
    save_profile('ORN-TARGET-GOOD', {'name': 'support-agent'}, str(tmp_path))
    broken = tmp_path / 'ORN-SELF-BAD'
    broken.mkdir()
    (broken / 'self.json').write_text('{broken')
    result = overview(str(tmp_path))
    assert result['profiles'][1]['name'] == 'support-agent'
    assert result['errors']
    assert result['action']['href'] == '/context'


def test_security_states_have_distinct_accessible_labels():
    from app import app
    from flask import render_template_string
    with app.test_request_context('/'):
        html = render_template_string("{% from 'components/status.html' import status_badge %}{{ status_badge('HYPOTHESIS') }}{{ status_badge('UNKNOWN') }}{{ status_badge('NOT_APPLICABLE') }}{{ status_badge('APPLIED') }}")
    assert '◇' in html and ' UNKNOWN' in html
    assert ' UNKNOWN' in html and ' NOT_APPLICABLE' in html and ' APPLIED' in html
    assert 'CONFIRMED' not in html and 'MITIGATED' not in html


def test_observation_routes_remain_distinct_views_without_new_api():
    from app import app
    with app.test_client() as client:
        runs = client.get('/runs').data.decode()
        evidence = client.get('/runs?view=evidence').data.decode()
        assert 'Runs · execution history' in runs
        assert 'Evidence · records &amp; artifacts' in evidence
        assert 'data-view="runs"' in runs
        assert 'data-view="evidence"' in evidence
        assert 'aria-current="page"' in runs and 'aria-current="page"' in evidence


def test_retest_run_links_back_to_its_finding(monkeypatch):
    from app import app
    from orion.evidence import EvidenceStore, ExperimentRecord
    from orion.findings import FindingStore
    from orion.findings.model import Finding
    record = ExperimentRecord(trace_id='UI-RETEST', mode='hardened', status='ATTACK_BLOCKED')
    finding = Finding(id='UI-FINDING', title='Retested finding', retest_refs=['UI-RETEST'])
    monkeypatch.setattr(EvidenceStore, 'load', lambda self, trace: record)
    monkeypatch.setattr(FindingStore, 'find_by_evidence', lambda self, trace: None)
    monkeypatch.setattr(FindingStore, 'list', lambda self: [finding])
    with app.test_client() as client:
        response = client.get('/runs/UI-RETEST')
    assert response.status_code == 200
    assert 'href="/findings/UI-FINDING"' in response.data.decode()
