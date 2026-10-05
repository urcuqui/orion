"""Workstation refinements must preserve policy, finding and lifecycle meaning."""
from pathlib import Path
from flask import render_template_string
from app import app
from orion.experiments.lifecycle import ExperimentLifecycle, save, next_action
from orion.findings.model import Finding
from orion.workspace_view import finding_workspace


def test_shared_dragon_variants_are_decorative_and_cat_is_retired():
    with app.test_request_context('/'):
        body = render_template_string("{% from 'components/dragon.html' import dragon_sentinel %}{{ dragon_sentinel('compact') }}{{ dragon_sentinel('expanded') }}")
    assert 'dragon-compact' in body and 'dragon-expanded' in body
    assert body.count('aria-hidden="true"') == 2
    assert '◉' in body and 'WATCH / TRACE' not in body
    assert not Path('templates/components/cat.html').exists()


def test_shell_has_one_evidence_destination_and_preserves_console():
    with app.test_client() as client:
        body = client.get('/findings').data.decode()
        demo = client.get('/?demo=1').data.decode()
    nav = body.split('<nav class="orion-nav"')[1].split('</nav>')[0]
    assert nav.count('href="/runs"') == 1
    assert '>Runs<' not in nav
    assert 'data-orion-appearance="analyst"' in body
    assert 'data-orion-appearance="classic"' in demo
    assert 'ORION // AI SECURITY VALIDATION WORKSPACE' in body
    assert 'operator@orion:~/findings $' in body
    assert 'aria-controls="workflow-navigation"' in body


def test_policy_decision_is_not_a_security_outcome():
    with app.test_request_context('/'):
        html = render_template_string("{% from 'components/status.html' import policy_decision, status_badge %}{{ policy_decision('DENY') }}{{ status_badge('ATTACK_BLOCKED') }}{{ policy_decision(None) }}")
    assert 'policy-decision' in html and ' DENY' in html
    assert 'badge-attack_blocked' in html and ' UNKNOWN' in html
    css = Path('static/css/base.css').read_text()
    assert '.policy-decision { color: var(--status-neutral)' in css
    assert '.badge-confirmed { color: var(--status-success)' in css
    assert '.badge-observed { color: var(--status-info)' in css


def test_findings_have_seven_columns_and_keep_secondary_data(monkeypatch):
    import orion.findings as findings
    finding = Finding(id='QA-FINDING', title='Untrusted tool request', family='agentic_ai', affected_target='test-agent', applied_control='tool_allowlist')
    monkeypatch.setattr(findings, 'list_findings', lambda *args: [finding])
    with app.test_client() as client:
        html = client.get('/findings').data.decode()
    headings = html.split('<thead>')[1].split('</thead>')[0]
    assert headings.count('<th scope="col">') == 7
    for label in ['Finding', 'Severity', 'Status', 'Target', 'Evidence', 'Retest', 'Last observed']:
        assert label in headings
    assert '<summary>Details</summary>' in html and 'tool_allowlist' in html and 'QA-FINDING' in html
    assert 'data-filter="target"' in html and 'js/findings.js' in html


def test_finding_next_action_is_read_only_and_rejects_stale_link(tmp_path):
    ws = ExperimentLifecycle(finding_id='QA-FINDING', attack_run_id='QA-RUN', active_experiment_id='E1',
        stages={'plan': 'APPROVED', 'attack': 'COMPLETE', 'measure': 'COMPLETE', 'defend': 'APPLIED', 'retest': 'READY'})
    save(ws, str(tmp_path))
    finding = Finding(id='QA-FINDING', provenance={'experiment_workspace_id': ws.experiment_workspace_id})
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = finding_workspace(finding, str(tmp_path))
    assert result['next_action'] == next_action(ws)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    finding.id = 'DIFFERENT-FINDING'
    assert finding_workspace(finding, str(tmp_path)) is None


def test_applied_control_is_prominent_but_never_auto_mitigated(monkeypatch):
    import orion.findings as findings
    import orion.workspace_view as view
    finding = Finding(id='QA-FINDING', status='OBSERVED', evidence_refs=['QA-RUN'])
    ws = ExperimentLifecycle(finding_id=finding.id, attack_run_id='QA-RUN', defense_id='DEF-1', defense_control='tool_allowlist',
        active_experiment_id='E1', stages={'plan': 'APPROVED', 'attack': 'COMPLETE', 'measure': 'COMPLETE', 'defend': 'APPLIED', 'retest': 'READY'})
    monkeypatch.setattr(findings, 'load_finding', lambda *args: finding)
    monkeypatch.setattr(view, 'finding_workspace', lambda *args: ws.to_dict())
    with app.test_client() as client:
        html = client.get('/findings/QA-FINDING').data.decode()
    assert 'CONTROL APPLIED · MITIGATION NOT VERIFIED' in html
    assert ws.to_dict()['next_action']['label'] in html
    assert 'Open experiment and retest' in html and 'badge-mitigated' not in html
    assert html.index('What happened?') < html.index('Why does it matter?') < html.index('What proves it?') < html.index('What should I do?') < html.index('Provenance</div>')
    assert finding.status == 'OBSERVED'
