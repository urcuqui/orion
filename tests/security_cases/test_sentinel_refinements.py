"""Workstation refinements must preserve policy, finding and lifecycle meaning."""
from pathlib import Path
from flask import render_template_string
from app import app
from orion.experiments.lifecycle import ExperimentLifecycle, save, next_action
from orion.findings.model import Finding
from orion.workspace_view import finding_workspace


def test_shared_sentinel_variants_are_decorative_and_legacy_names_are_retired():
    with app.test_request_context('/'):
        body = render_template_string("{% from 'components/sentinel.html' import orion_sentinel %}{{ orion_sentinel('compact') }}{{ orion_sentinel('expanded') }}")
    assert 'orion-sentinel-compact' in body and 'orion-sentinel-expanded' in body
    assert body.count('aria-hidden="true"') == 2
    assert '◉' in body and 'WATCH / TRACE' not in body
    assert not Path('templates/components/cat.html').exists()
    assert not Path('templates/components/dragon.html').exists()


def test_shell_has_distinct_runs_and_evidence_destinations_and_preserves_console():
    with app.test_client() as client:
        body = client.get('/findings').data.decode()
        demo = client.get('/?demo=1').data.decode()
    nav = body.split('<nav class="orion-nav"')[1].split('</nav>')[0]
    assert nav.count('href="/runs"') == 1
    assert nav.count('href="/runs?view=evidence"') == 1
    assert '>Runs<' in nav and '>Evidence<' in nav
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
    finding = Finding(id='QA-FINDING', title='Untrusted tool request', family='agentic_ai', affected_target='test-agent', applied_control='tool_allowlist', evidence_refs=['QA-RUN'])
    monkeypatch.setattr(findings, 'list_findings', lambda *args: [finding])
    with app.test_client() as client:
        html = client.get('/findings').data.decode()
    headings = html.split('<thead>')[1].split('</thead>')[0]
    assert headings.count('<th scope="col">') == 7
    for label in ['Finding', 'Family', 'Severity', 'Status', 'Target', 'Retest', 'Last Observed']:
        assert label in headings
    assert '<summary>Details</summary>' in html and 'tool_allowlist' in html and 'QA-FINDING' in html
    assert '1 evidence record(s)' in html and '#finding-evidence' in html
    assert 'Agentic AI' in html
    for key in ['status', 'severity', 'retest', 'family', 'target']:
        assert f'data-filter="{key}"' in html and f'data-{key}=' in html
    assert 'js/findings.js' in html


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



def test_navigation_hooks_and_watermark_modes():
    with app.test_client() as client:
        html = client.get('/findings').data.decode()
    assert html.count('id="nav-toggle"') == 1
    assert html.count('id="workflow-navigation"') == 1
    assert 'type="button" aria-label="Close navigation"' in html
    assert 'aria-expanded="true" aria-controls="workflow-navigation"' in html
    assert 'orion-sentinel-watermark' in html
    css = Path('static/css/orion-terminal.css').read_text()
    assert '.orion-sentinel-watermark { display: none;' in css
    assert 'body[data-orion-appearance="classic"] .orion-sentinel-watermark { display: block;' in css


def test_allow_and_deny_are_neutral_independently_of_run_outcome(monkeypatch):
    from orion.evidence import EvidenceStore, ExperimentRecord
    for decision in ['ALLOW', 'DENY', None]:
        with app.test_request_context('/'):
            html = render_template_string("{% from 'components/status.html' import policy_decision %}{{ policy_decision(decision) }}", decision=decision)
        assert '✓' not in html and '×' not in html
        assert ('■' if decision else '?') in html
        assert (decision or 'UNKNOWN') in html
    record = ExperimentRecord(trace_id='QA-POLICY', status='ATTACK_BLOCKED',
        execution_trace=[{'type': 'authorization_event', 'source': 'policy',
            'decision': 'ALLOW', 'reason': 'Recorded decision',
            'required_privilege': 'low', 'effective_privilege': 'low', 'agent_identity': 'fixture'}])
    monkeypatch.setattr(EvidenceStore, 'load', lambda *args: record)
    with app.test_client() as client:
        html = client.get('/runs/QA-POLICY').data.decode()
        assert 'POLICY DECISION' in html and ' ALLOW' in html
        assert 'Security outcome' in html and 'badge-attack_blocked' in html
        record.status = 'UNKNOWN'
        html = client.get('/runs/QA-POLICY').data.decode()
        assert 'badge-unknown' in html and 'badge-attack_blocked' not in html



def test_observation_sidebar_active_states_are_independent():
    from html.parser import HTMLParser

    class PrimaryLinks(HTMLParser):
        def __init__(self):
            super().__init__()
            self.primary = False
            self.links = {}

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'nav' and attrs.get('aria-label') == 'Primary':
                self.primary = True
            elif tag == 'a' and self.primary:
                self.links[attrs['href']] = attrs

        def handle_endtag(self, tag):
            if tag == 'nav':
                self.primary = False

    with app.test_client() as client:
        for path, active in [('/runs', '/runs'), ('/runs?view=runs', '/runs'),
                             ('/runs?view=evidence', '/runs?view=evidence'),
                             ('/runs/QA-MISSING', '/runs')]:
            parser = PrimaryLinks()
            parser.feed(client.get(path).data.decode())
            assert '/runs' in parser.links and '/runs?view=evidence' in parser.links
            assert [href for href, attrs in parser.links.items()
                    if attrs.get('aria-current') == 'page'] == [active]
            assert 'active' in parser.links[active]['class'].split()
            other = '/runs' if active.endswith('evidence') else '/runs?view=evidence'
            assert 'active' not in parser.links[other]['class'].split()
