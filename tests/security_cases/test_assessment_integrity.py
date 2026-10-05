"""Failure-mode coverage for live Assessment membership, without repair on read."""
import pytest
from orion.assessments import AssessmentService, AssessmentRepository
from orion.assessments.service import record_workspace
from orion.experiments import lifecycle
from orion import experiments as EXP
from orion.evidence import EvidenceStore
from orion.evidence.record import ExperimentRecord
from orion.findings import FindingStore
from orion.findings.model import Finding
from orion.context.store import save_profile
from tests.security_cases.test_assessments import effort, plan, snapshot


def linked(directory):
    service = AssessmentService(directory)
    assessment = effort(directory, status='ACTIVE')
    ws = service.create_experiment(assessment.assessment_id, plan(directory).plan_id)
    return service, assessment, ws


def run(directory, ws, owner=None, workspace=True):
    provenance = {'assessment_id': owner} if owner else {}
    if workspace:
        provenance['experiment_workspace_id'] = ws.experiment_workspace_id
    record = ExperimentRecord(provenance=provenance)
    EvidenceStore(directory).save(record)
    ws.attack_run_id = record.trace_id
    lifecycle.save(ws, str(directory))
    return record


def codes(result):
    return {issue['code'] for issue in result['issues']}


def test_valid_integrity_is_read_only(tmp_path):
    service, a, ws = linked(tmp_path)
    run(tmp_path, ws, a.assessment_id)
    before = snapshot(tmp_path)
    assert service.validate_integrity(a.assessment_id) == {'assessment_id': a.assessment_id, 'valid': True, 'issues': []}
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('owner,expected', [(None, 'EXPERIMENT_BACKREF_MISSING'), ('ORN-ASMT-OTHER', 'EXPERIMENT_ASSESSMENT_MISMATCH')])
def test_workspace_membership_mismatch(tmp_path, owner, expected):
    service, a, ws = linked(tmp_path)
    ws.assessment_id = owner
    lifecycle.save(ws, str(tmp_path))
    before = snapshot(tmp_path)
    result = service.validate_integrity(a.assessment_id)
    assert not result['valid'] and expected in codes(result)
    assert snapshot(tmp_path) == before


def test_reverse_workspace_reference_missing_from_assessment(tmp_path):
    service, a, ws = linked(tmp_path)
    record = service.get(a.assessment_id); record.experiment_ids = []; service.repository.save(record)
    assert 'EXPERIMENT_FORWARDREF_MISSING' in codes(service.validate_integrity(a.assessment_id))


@pytest.mark.parametrize('content,expected', [(None, 'EXPERIMENT_MISSING'), ('{', 'REFERENCE_UNAVAILABLE'), ('[]', 'REFERENCE_UNAVAILABLE')])
def test_missing_or_corrupt_referenced_workspace(tmp_path, content, expected):
    service, a, ws = linked(tmp_path)
    path = tmp_path / ws.experiment_workspace_id / 'workspace.json'
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
    result = service.validate_integrity(a.assessment_id)
    assert not result['valid'] and expected in codes(result)


@pytest.mark.parametrize('content', ['{', '[]', '{"assessment_id":"WRONG"}'])
def test_corrupt_assessment_is_structured_result(tmp_path, content):
    service, a, ws = linked(tmp_path)
    (tmp_path / 'assessments' / (a.assessment_id + '.json')).write_text(content)
    result = service.validate_integrity(a.assessment_id)
    assert not result['valid'] and codes(result) == {'ASSESSMENT_RECORD_CORRUPT'}


def test_run_assessment_mismatch(tmp_path):
    service, a, ws = linked(tmp_path)
    run(tmp_path, ws, 'ORN-ASMT-OTHER')
    result = service.validate_integrity(a.assessment_id)
    assert not result['valid'] and 'RUN_ASSESSMENT_MISMATCH' in codes(result)


def test_run_claims_assessment_while_workspace_is_standalone(tmp_path):
    service, a, ws = linked(tmp_path)
    ws.assessment_id = None
    run(tmp_path, ws, a.assessment_id)
    assert 'RUN_WITHOUT_WORKSPACE_MEMBERSHIP' in codes(service.validate_integrity(a.assessment_id))


def test_reverse_historical_run_checks_workspace_membership(tmp_path):
    service, a, ws = linked(tmp_path)
    run(tmp_path, ws, 'ORN-ASMT-OTHER')
    ws.attack_run_id = None; lifecycle.save(ws, str(tmp_path))
    assert 'RUN_ASSESSMENT_MISMATCH' in codes(service.validate_integrity(a.assessment_id))


def test_legacy_missing_run_provenance_warns_without_invalidating(tmp_path):
    service, a, ws = linked(tmp_path)
    run(tmp_path, ws, workspace=False)
    result = service.validate_integrity(a.assessment_id)
    assert result['valid'] and codes(result) == {'RUN_ASSESSMENT_PROVENANCE_MISSING'}
    assert result['issues'][0]['severity'] == 'WARNING'


@pytest.mark.parametrize('corrupt', [False, True])
def test_unavailable_referenced_run_is_reported(tmp_path, corrupt):
    service, a, ws = linked(tmp_path)
    record = run(tmp_path, ws, a.assessment_id)
    path = tmp_path / record.trace_id / 'experiment.json'
    if corrupt:
        path.write_text('[]')
    else:
        path.unlink()
    result = service.validate_integrity(a.assessment_id)
    assert not result['valid'] and 'REFERENCE_UNAVAILABLE' in codes(result)


def test_run_without_resolvable_workspace_is_reported(tmp_path):
    service, a, ws = linked(tmp_path)
    EvidenceStore(tmp_path).save(ExperimentRecord(provenance={'assessment_id': a.assessment_id, 'experiment_workspace_id': 'MISSING'}))
    assert not service.validate_integrity(a.assessment_id)['valid']


def test_sync_is_idempotent_including_timestamp_and_evidence(tmp_path):
    service, a, ws = linked(tmp_path)
    EXP.run_agentic_attack(ws, {'attack_id': 'ORN-ATTACK-AG-002'}, str(tmp_path))
    EXP.measure(ws, str(tmp_path)); EXP.apply_defense(ws, {'control_id': 'tool_authorization'}, str(tmp_path))
    save_profile('CTX-SYNC', {'analysis_context_id': 'CTX-SYNC'}, str(tmp_path))
    ws.analysis_context_id = 'CTX-SYNC'; lifecycle.save(ws, str(tmp_path))
    record_workspace(ws, str(tmp_path)); before = snapshot(tmp_path)
    record_workspace(ws, str(tmp_path)); record_workspace(ws, str(tmp_path))
    assert snapshot(tmp_path) == before
    assessment = service.get(a.assessment_id)
    assert assessment.experiment_ids == [ws.experiment_workspace_id]
    assert assessment.finding_ids == [ws.finding_id]
    assert len(assessment.control_refs) == 1
    assert len(assessment.plan_ids) == len(assessment.threat_model_ids) == 1
    assert assessment.analysis_context_ids == ['CTX-SYNC']


def test_domain_save_does_not_mutate_assessment_or_evidence(tmp_path):
    service, a, ws = linked(tmp_path)
    record = run(tmp_path, ws)
    before = snapshot(tmp_path)
    lifecycle.save(ws, str(tmp_path))
    after = snapshot(tmp_path)
    before.pop(ws.experiment_workspace_id + '/workspace.json'); after.pop(ws.experiment_workspace_id + '/workspace.json')
    assert before == after
    assert not EvidenceStore(tmp_path).load(record.trace_id).provenance.get('assessment_id')


def test_workspace_write_survives_failed_assessment_sync(monkeypatch, tmp_path):
    service = AssessmentService(tmp_path); a = effort(tmp_path)
    ws = EXP.create_from_plan(plan(tmp_path), str(tmp_path))
    evidence = run(tmp_path, ws)
    ws.assessment_id = a.assessment_id
    def fail(*args):
        raise OSError('assessment write failed')
    with monkeypatch.context() as patch:
        patch.setattr(AssessmentRepository, 'save', fail)
        with pytest.raises(OSError, match='assessment write failed'):
            EXP.save_workspace(ws, str(tmp_path))
    assert EXP.load_workspace(ws.experiment_workspace_id, str(tmp_path)).assessment_id == a.assessment_id
    assert EvidenceStore(tmp_path).load(evidence.trace_id).trace_id == evidence.trace_id
    result = service.validate_integrity(a.assessment_id)
    assert 'EXPERIMENT_FORWARDREF_MISSING' in codes(result)
    assert 'RUN_ASSESSMENT_PROVENANCE_MISSING' in codes(result)


def test_run_provenance_write_failure_preserves_prior_writes(monkeypatch, tmp_path):
    service, a, ws = linked(tmp_path)
    record = run(tmp_path, ws)
    ws.finding_id = 'ORN-FND-LATE'
    FindingStore(tmp_path).save(Finding(id=ws.finding_id))
    before_run = (tmp_path / record.trace_id / 'experiment.json').read_bytes()
    def fail(*args):
        raise OSError('provenance write failed')
    with monkeypatch.context() as patch:
        patch.setattr(EvidenceStore, 'save', fail)
        with pytest.raises(OSError, match='provenance write failed'):
            EXP.save_workspace(ws, str(tmp_path))
    assert ws.finding_id in service.get(a.assessment_id).finding_ids
    assert EXP.load_workspace(ws.experiment_workspace_id, str(tmp_path)).finding_id == ws.finding_id
    assert (tmp_path / record.trace_id / 'experiment.json').read_bytes() == before_run
    assert 'RUN_ASSESSMENT_PROVENANCE_MISSING' in codes(service.validate_integrity(a.assessment_id))


@pytest.mark.parametrize('patch', [{'name': 'changed'}, {'scope': {'notes': 'changed'}}, {'description': 'changed'}, {'experiment_ids': ['NEW']}])
def test_archived_rejects_user_metadata_and_manual_membership(tmp_path, patch):
    service, a, ws = linked(tmp_path); service.archive(a.assessment_id)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError):
        service.update(a.assessment_id, patch)
    assert snapshot(tmp_path) == before


def test_archived_assessment_allows_system_reference_sync_for_existing_experiment(tmp_path):
    service, a, ws = linked(tmp_path); service.archive(a.assessment_id)
    EXP.run_agentic_attack(ws, {'attack_id': 'ORN-ATTACK-AG-002'}, str(tmp_path))
    EXP.measure(ws, str(tmp_path)); EXP.apply_defense(ws, {'control_id': 'tool_authorization'}, str(tmp_path))
    assessment = service.get(a.assessment_id)
    assert assessment.status == 'ARCHIVED' and ws.finding_id in assessment.finding_ids
    assert ws.defense_id in assessment.control_refs
    assert service.validate_integrity(a.assessment_id)['valid']


def test_archived_sync_cannot_attach_new_experiment(tmp_path):
    service, a, ws = linked(tmp_path); service.archive(a.assessment_id)
    other = lifecycle.create_from_plan(plan(tmp_path), str(tmp_path), assessment_id=a.assessment_id)
    with pytest.raises(ValueError, match='new experiment'):
        record_workspace(other, str(tmp_path))
    assert other.experiment_workspace_id not in service.get(a.assessment_id).experiment_ids


def test_shared_findings_remain_live_after_archive(tmp_path):
    service = AssessmentService(tmp_path); a, b = effort(tmp_path), effort(tmp_path)
    f = Finding(status='OBSERVED'); store = FindingStore(tmp_path); store.save(f)
    for assessment in (a, b):
        service.link_finding(assessment.assessment_id, f.id); service.archive(assessment.assessment_id)
    f.status = 'CONFIRMED'; store.save(f)
    for assessment in (a, b):
        assert service.summary(assessment.assessment_id)['finding_records'][0]['status'] == 'CONFIRMED'
    assert len(list(store.dir.glob('*.json'))) == 1


def test_complete_and_archive_do_not_change_finding_control_or_retest(tmp_path):
    service, a, ws = linked(tmp_path)
    EXP.run_agentic_attack(ws, {'attack_id': 'ORN-ATTACK-AG-002'}, str(tmp_path))
    EXP.measure(ws, str(tmp_path)); EXP.apply_defense(ws, {'control_id': 'tool_authorization'}, str(tmp_path)); EXP.retest(ws, str(tmp_path))
    before = snapshot(tmp_path)
    service.complete(a.assessment_id); service.archive(a.assessment_id)
    after = snapshot(tmp_path)
    assessment_path = 'assessments/' + a.assessment_id + '.json'
    before.pop(assessment_path); after.pop(assessment_path)
    assert before == after


def test_integrity_api_returns_structured_failures_without_repair(monkeypatch, tmp_path):
    from app import app
    import orion.integrations.flask_blueprint as bp
    monkeypatch.setattr(bp, 'ARTIFACT_DIR', str(tmp_path))
    service, a, ws = linked(tmp_path); client = app.test_client()
    url = '/api/assessments/' + a.assessment_id + '/integrity'
    assert client.get(url).json['valid']
    ws.assessment_id = None; lifecycle.save(ws, str(tmp_path)); before = snapshot(tmp_path)
    response = client.get(url)
    assert response.status_code == 200 and not response.json['valid']
    assert snapshot(tmp_path) == before


def scoped(directory):
    service, a, ws = linked(directory)
    # Delete the sole workspace relationship to exercise pre-experiment actions.
    record = service.get(a.assessment_id); record.experiment_ids = []
    record.system_profile_id = 'SELF'; record.target_id = 'TARGET'
    save_profile('SELF', {'self_profile_id': 'SELF'}, str(directory))
    save_profile('TARGET', {'target_profile_id': 'TARGET'}, str(directory))
    (directory / ws.experiment_workspace_id / 'workspace.json').unlink()
    service.repository.save(record)
    return service, a, ws


def test_next_action_sees_threat_model_errors_before_plan_action(tmp_path):
    service, a, ws = scoped(tmp_path)
    record = service.get(a.assessment_id); record.threat_model_ids.append('TM-MISSING'); service.repository.save(record)
    assert service.summary(a.assessment_id)['next_action']['label'] == 'Review unavailable references'


def test_next_action_integrity_error_precedes_experiment_action(tmp_path):
    service, a, ws = linked(tmp_path)
    record = service.get(a.assessment_id); record.system_profile_id = 'SELF'; record.target_id = 'TARGET'; service.repository.save(record)
    save_profile('SELF', {'self_profile_id': 'SELF'}, str(tmp_path)); save_profile('TARGET', {'target_profile_id': 'TARGET'}, str(tmp_path))
    ws.assessment_id = 'ORN-ASMT-OTHER'; lifecycle.save(ws, str(tmp_path))
    assert service.summary(a.assessment_id)['next_action']['label'] == 'Review unavailable references'


def test_closed_assessment_next_action_remains_evidence_review_with_errors(tmp_path):
    service, a, ws = linked(tmp_path); service.archive(a.assessment_id)
    (tmp_path / ws.experiment_workspace_id / 'workspace.json').unlink()
    assert service.summary(a.assessment_id)['next_action']['label'] == 'Review recorded evidence'


def test_sync_rejects_ambiguous_run_ownership_without_mutating_references(tmp_path):
    service, a, ws = linked(tmp_path)
    run(tmp_path, ws, 'ORN-ASMT-OTHER')
    ws.finding_id = 'ORN-FND-LATE'
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match='another assessment'):
        record_workspace(ws, str(tmp_path))
    assert snapshot(tmp_path) == before


def test_missing_assessment_integrity_result_and_invalid_identifier(tmp_path):
    service = AssessmentService(tmp_path)
    result = service.validate_integrity('ORN-ASMT-MISSING')
    assert not result['valid'] and codes(result) == {'REFERENCE_UNAVAILABLE'}
    with pytest.raises(ValueError):
        service.validate_integrity('../invalid')


def test_unapproved_plan_next_action_uses_existing_link_flow(tmp_path):
    service, a, ws = scoped(tmp_path)
    from orion import plans
    p = plans.load_plan(ws.plan_id, str(tmp_path)); p.approved_by_human = False; plans.save_plan(p, str(tmp_path))
    action = service.summary(a.assessment_id)['next_action']
    assert action == {'label': 'Create experiment to review plan', 'href': '#assessment-links'}


def test_corrupt_reverse_reference_with_known_membership_is_error(tmp_path):
    service, a, ws = linked(tmp_path)
    assessment = service.get(a.assessment_id); assessment.experiment_ids = []; service.repository.save(assessment)
    import json
    path = tmp_path / ws.experiment_workspace_id / 'workspace.json'
    data = json.loads(path.read_text()); data['stages'] = []; path.write_text(json.dumps(data))
    result = service.validate_integrity(a.assessment_id)
    assert not result['valid'] and 'REFERENCE_UNAVAILABLE' in codes(result)
