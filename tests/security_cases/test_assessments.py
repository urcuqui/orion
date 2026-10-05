"""Assessment aggregates references; evidence still decides security state."""
import json
from pathlib import Path
import pytest
from orion.assessments import AssessmentService, AssessmentRepository
from orion import experiments as EXP, plans as PL
from orion.findings import FindingStore
from orion.findings.model import Finding
from orion.evidence import EvidenceStore
from orion.target_analysis import build_assessment_from_summary


def effort(directory, **extra):
    return AssessmentService(directory).create({'name': 'Agent assessment', 'scope': {'surfaces': ['tools']}, **extra})


def plan(directory, approved=True):
    assessment = build_assessment_from_summary({'target': 'http://agent', 'endpoints': ['/chat'],
        'report_markdown': 'ai agent with tool calling and retrieval rag',
        'findings': [{'id': 'f', 'title': 'ml_inference_response', 'severity': 'info'}]})
    result = PL.build_plan_from_target_analysis(assessment)
    if approved:
        PL.approve_plan(result)
    PL.save_plan(result, str(directory))
    return result


def snapshot(directory):
    return {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob('*') if p.is_file()}


def test_create_get_list_update_transitions_and_scope(tmp_path):
    service = AssessmentService(tmp_path)
    a = effort(tmp_path)
    assert a.assessment_id.startswith('ORN-ASMT-') and a.status == 'DRAFT'
    assert service.get(a.assessment_id).scope == {'surfaces': ['tools']}
    assert service.repository.list()[0][0].assessment_id == a.assessment_id
    service.update(a.assessment_id, {'status': 'ACTIVE', 'metadata': {'owner': 'research'}, 'description': 'Bounded effort'})
    service.complete(a.assessment_id)
    assert service.get(a.assessment_id).status == 'COMPLETED'
    service.archive(a.assessment_id)
    with pytest.raises(ValueError):
        service.update(a.assessment_id, {'status': 'ACTIVE'})
    with pytest.raises(ValueError):
        service.update(a.assessment_id, {'name': 'Edited archive'})


@pytest.mark.parametrize('payload', [{'name': ''}, {'name': 'x', 'status': 'SECURE'},
    {'name': 'x', 'scope': []}, {'name': 'x', 'finding_ids': ['../escape']},
    {'name': 'x', 'assessment_id': 'forced'}])
def test_invalid_creation_does_not_write(tmp_path, payload):
    with pytest.raises(ValueError):
        AssessmentService(tmp_path).create(payload)
    assert not list(tmp_path.rglob('*.json'))


def test_explicit_link_is_idempotent_without_lifecycle_changes_or_new_evidence(tmp_path):
    service = AssessmentService(tmp_path); a = effort(tmp_path)
    ws = EXP.create_from_plan(plan(tmp_path), str(tmp_path))
    EXP.run_agentic_attack(ws, {'attack_id': 'ORN-ATTACK-AG-002'}, str(tmp_path))
    original = ws.to_dict()
    files = set(snapshot(tmp_path))
    service.link_experiment(a.assessment_id, ws.experiment_workspace_id)
    service.link_experiment(a.assessment_id, ws.experiment_workspace_id)
    linked = EXP.load_workspace(ws.experiment_workspace_id, str(tmp_path))
    assert linked.assessment_id == a.assessment_id
    for key in ('stages', 'attack_run_id', 'active_experiment_id', 'current_stage', 'plan_id'):
        assert linked.to_dict()[key] == original[key]
    assert service.get(a.assessment_id).experiment_ids == [ws.experiment_workspace_id]
    assert set(snapshot(tmp_path)) == files
    assert EvidenceStore(tmp_path).load(ws.attack_run_id).provenance['assessment_id'] == a.assessment_id
    other = effort(tmp_path)
    with pytest.raises(ValueError):
        service.link_experiment(other.assessment_id, ws.experiment_workspace_id)


def test_full_linked_lifecycle_preserves_finding_and_retest_semantics(tmp_path):
    service = AssessmentService(tmp_path); a = effort(tmp_path, status='ACTIVE')
    ws = service.create_experiment(a.assessment_id, plan(tmp_path).plan_id)
    EXP.run_agentic_attack(ws, {'attack_id': 'ORN-ATTACK-AG-002', 'trials': 3}, str(tmp_path))
    measurement = EXP.measure(ws, str(tmp_path))
    assert measurement['finding']['status'] == 'OBSERVED'
    summary = service.summary(a.assessment_id)
    assert summary['findings']['observed'] == 1 and summary['findings']['confirmed'] == 0
    assert summary['run_records'][0]['provenance']['assessment_id'] == a.assessment_id
    assert summary['finding_records'][0]['provenance']['assessment_id'] == a.assessment_id
    EXP.apply_defense(ws, {'control_id': 'tool_authorization'}, str(tmp_path))
    summary = service.summary(a.assessment_id)
    assert summary['controls']['applied'] == 1 and summary['controls']['awaiting_retest'] == 1
    assert summary['posture']['verified_mitigations'] == 0
    retest = EXP.retest(ws, str(tmp_path))
    assert retest['status'] == 'ATTACK_BLOCKED'
    summary = service.summary(a.assessment_id)
    assert summary['controls']['awaiting_retest'] == 0
    assert summary['retests']['mitigated'] == 1 and summary['experiments']['completed'] == 1
    assert len(summary['run_records']) == 2
    for run in summary['run_records']:
        assert run['provenance']['assessment_id'] == a.assessment_id
        assert run['provenance']['threat_model_id'] == ws.threat_model_id
    before = snapshot(tmp_path)
    service.summary(a.assessment_id)
    assert snapshot(tmp_path) == before


def test_completion_does_not_mitigate_promote_or_claim_security(tmp_path):
    service = AssessmentService(tmp_path); a = effort(tmp_path, status='ACTIVE')
    finding = Finding(title='Hypothesis', applied_control='tool_authorization', status='HYPOTHESIS')
    FindingStore(tmp_path).save(finding)
    service.link_finding(a.assessment_id, finding.id)
    before = FindingStore(tmp_path)._path(finding.id).read_bytes()
    service.complete(a.assessment_id)
    summary = service.summary(a.assessment_id)
    assert summary['assessment']['status'] == 'COMPLETED'
    assert summary['findings']['hypothesis'] == 1
    assert summary['controls']['applied'] == 1 and summary['controls']['awaiting_retest'] == 1
    assert summary['retests']['mitigated'] == 0
    assert 'SECURE' not in json.dumps(summary)
    assert FindingStore(tmp_path)._path(finding.id).read_bytes() == before


def test_multiple_experiments_and_specific_threat_models_survive(tmp_path):
    service = AssessmentService(tmp_path); a = effort(tmp_path)
    p1, p2 = plan(tmp_path), plan(tmp_path)
    ws1 = service.create_experiment(a.assessment_id, p1.plan_id)
    ws2 = service.create_experiment(a.assessment_id, p2.plan_id)
    summary = service.summary(a.assessment_id)
    assert summary['experiments']['total'] == 2
    assert set(summary['assessment']['plan_ids']) == {p1.plan_id, p2.plan_id}
    assert set(summary['assessment']['threat_model_ids']) == {p1.threat_model_id, p2.threat_model_id}
    assert ws1.threat_model_id == p1.threat_model_id and ws2.threat_model_id == p2.threat_model_id


def test_standalone_legacy_workspace_loads_and_runs(tmp_path):
    ws = EXP.create_from_plan(plan(tmp_path), str(tmp_path))
    path = tmp_path / ws.experiment_workspace_id / 'workspace.json'
    data = json.loads(path.read_text()); data.pop('assessment_id'); path.write_text(json.dumps(data))
    loaded = EXP.load_workspace(ws.experiment_workspace_id, str(tmp_path))
    assert loaded.assessment_id is None
    EXP.run_agentic_attack(loaded, {'attack_id': 'ORN-ATTACK-AG-002'}, str(tmp_path))
    EXP.measure(loaded, str(tmp_path))
    assert AssessmentRepository(tmp_path).list() == ([], [])


def test_missing_sources_are_unknown_and_read_only(tmp_path):
    service = AssessmentService(tmp_path); a = effort(tmp_path)
    ws = service.create_experiment(a.assessment_id, plan(tmp_path).plan_id)
    (tmp_path / ws.experiment_workspace_id / 'workspace.json').unlink()
    before = snapshot(tmp_path)
    summary = service.summary(a.assessment_id)
    assert summary['experiments']['total'] is None
    assert summary['findings']['total'] is None and summary['posture']['confirmed'] is None
    assert summary['errors'] and snapshot(tmp_path) == before
    empty = service.summary(effort(tmp_path).assessment_id)
    assert empty['experiments']['total'] == 0 and empty['findings']['total'] == 0


def test_linking_a_finding_preserves_origin_and_allows_shared_membership(tmp_path):
    service = AssessmentService(tmp_path); a, b = effort(tmp_path), effort(tmp_path)
    f = Finding(status='OBSERVED', provenance={'plan_id': 'ORIGINAL'}, evidence_refs=['RUN-1'])
    FindingStore(tmp_path).save(f); before = FindingStore(tmp_path)._path(f.id).read_bytes()
    service.link_finding(a.assessment_id, f.id); service.link_finding(b.assessment_id, f.id)
    assert service.summary(a.assessment_id)['finding_records'][0]['status'] == 'OBSERVED'
    assert FindingStore(tmp_path)._path(f.id).read_bytes() == before


def test_assessment_api_and_pages(monkeypatch, tmp_path):
    import orion.integrations.flask_blueprint as bp
    from app import app
    monkeypatch.setattr(bp, 'ARTIFACT_DIR', str(tmp_path))
    client = app.test_client()
    response = client.post('/api/assessments', json={'name': 'UI assessment', 'scope': {'notes': 'Local lab'}})
    assert response.status_code == 201
    assessment_id = response.json['assessment_id']
    assert client.get('/api/assessments/'+assessment_id).json['name'] == 'UI assessment'
    assert len(client.get('/api/assessments').json['assessments']) == 1
    assert client.patch('/api/assessments/'+assessment_id, json={'status': 'ACTIVE'}).status_code == 200
    assert client.get('/api/assessments/'+assessment_id+'/summary').json['experiments']['total'] == 0
    assert client.get('/assessments').status_code == 200
    detail = client.get('/assessments/'+assessment_id)
    assert detail.status_code == 200 and b'Assessment Completed' in detail.data
    assert client.get('/api/assessments/MISSING').status_code == 404
    assert client.patch('/api/assessments/'+assessment_id, json={'experiment_ids': ['MISSING']}).status_code == 400
    assert client.get('/api/assessments/'+assessment_id).json['experiment_ids'] == []
    assert client.post('/api/assessments', json={'name': 'x', 'target_id': '../escape'}).status_code == 400


def test_linked_proposal_switch_retains_all_explicit_runs(tmp_path):
    service = AssessmentService(tmp_path); a = effort(tmp_path)
    ws = service.create_experiment(a.assessment_id, plan(tmp_path).plan_id)
    EXP.run_agentic_attack(ws, {'attack_id': 'ORN-ATTACK-AG-002'}, str(tmp_path))
    original_run = ws.attack_run_id
    EXP.measure(ws, str(tmp_path))
    original_finding = ws.finding_id
    EXP.select_experiment(ws, ws.active_experiment_id, str(tmp_path))
    EXP.run_agentic_attack(ws, {'attack_id': 'ORN-ATTACK-AG-002'}, str(tmp_path))
    summary = service.summary(a.assessment_id)
    assert {r['trace_id'] for r in summary['run_records']} == {original_run, ws.attack_run_id}
    assert original_finding in summary['assessment']['finding_ids']


def test_creating_from_unapproved_plan_never_runs_or_approves(tmp_path):
    service = AssessmentService(tmp_path); a = effort(tmp_path)
    p = plan(tmp_path, approved=False)
    ws = service.create_experiment(a.assessment_id, p.plan_id)
    assert ws.stages['plan'] == 'PROPOSED' and ws.attack_run_id is None
    assert not PL.load_plan(p.plan_id, str(tmp_path)).approved_by_human
    assert EvidenceStore(tmp_path).list_traces() == []
    with pytest.raises(EXP.StageError):
        EXP.run_agentic_attack(ws, {'attack_id': 'ORN-ATTACK-AG-002'}, str(tmp_path))


def test_missing_direct_finding_evidence_is_unknown_not_zero(tmp_path):
    service = AssessmentService(tmp_path); a = effort(tmp_path)
    f = Finding(status='OBSERVED', evidence_refs=['MISSING-RUN'])
    FindingStore(tmp_path).save(f); service.link_finding(a.assessment_id, f.id)
    summary = service.summary(a.assessment_id)
    assert summary['findings']['total'] is None and summary['findings']['known_total'] == 1
    assert summary['finding_records'][0]['status'] == 'OBSERVED'
    assert any(e['kind'] == 'run' for e in summary['errors'])


def test_applied_control_references_survive_proposal_switch(tmp_path):
    service = AssessmentService(tmp_path); a = effort(tmp_path)
    ws = service.create_experiment(a.assessment_id, plan(tmp_path).plan_id)
    EXP.run_agentic_attack(ws, {'attack_id': 'ORN-ATTACK-AG-002'}, str(tmp_path))
    EXP.measure(ws, str(tmp_path)); EXP.apply_defense(ws, {'control_id': 'tool_authorization'}, str(tmp_path))
    defense_id = ws.defense_id
    EXP.select_experiment(ws, ws.active_experiment_id, str(tmp_path))
    summary = service.summary(a.assessment_id)
    assert summary['control_records'][0]['id'] == defense_id
    assert summary['controls']['applied'] == 1 and summary['controls']['awaiting_retest'] == 1


def test_linking_partial_legacy_evidence_keeps_explicit_unknown(tmp_path):
    service = AssessmentService(tmp_path); a = effort(tmp_path)
    ws = EXP.create_from_plan(plan(tmp_path), str(tmp_path))
    ws.attack_run_id = 'MISSING-RUN'
    EXP.save_workspace(ws, str(tmp_path))
    service.link_experiment(a.assessment_id, ws.experiment_workspace_id)
    summary = service.summary(a.assessment_id)
    assert summary['experiments']['total'] == 1
    assert summary['findings']['total'] is None
    assert any(error['id'] == 'MISSING-RUN' for error in summary['errors'])


def test_scope_profile_conflict_is_rejected_without_changing_relationships(tmp_path):
    service = AssessmentService(tmp_path); a = effort(tmp_path, system_profile_id='ORN-SELF-A')
    p = plan(tmp_path); p.self_profile_id = 'ORN-SELF-B'; PL.save_plan(p, str(tmp_path))
    before = snapshot(tmp_path)
    with pytest.raises(ValueError):
        service.create_experiment(a.assessment_id, p.plan_id)
    assert snapshot(tmp_path) == before


def test_corrupt_assessment_does_not_hide_available_list_records(tmp_path):
    service = AssessmentService(tmp_path); a = effort(tmp_path)
    broken = tmp_path / 'assessments' / 'ORN-ASMT-BROKEN.json'; broken.write_text('[]')
    records, errors = service.repository.list()
    assert [record.assessment_id for record in records] == [a.assessment_id]
    assert errors[0]['assessment_id'] == 'ORN-ASMT-BROKEN'
