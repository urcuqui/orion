"""Read-only checks of explicit Assessment/workspace/Run relationships."""
import json
from pathlib import Path
from .repository import AssessmentRepository, identifier
from .model import STATUSES


def validate_assessment_integrity(assessment_id, base_dir='artifacts'):
    identifier(assessment_id)
    base = Path(base_dir)
    issues = []

    def issue(code, message, severity='ERROR', **refs):
        issues.append(dict(code=code, severity=severity, message=message, **refs))

    def read(path):
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('record must be an object')
        return data

    try:
        assessment = AssessmentRepository(base).get(assessment_id)
        if assessment is None:
            issue('REFERENCE_UNAVAILABLE', 'Assessment does not exist.')
            return {'assessment_id': assessment_id, 'valid': False, 'issues': issues}
        if assessment.status not in STATUSES or not isinstance(assessment.scope, dict) or not isinstance(assessment.control_refs, dict):
            raise ValueError('invalid Assessment status, scope or control references')
        for field in ('experiment_ids', 'finding_ids', 'plan_ids', 'analysis_context_ids', 'threat_model_ids'):
            references = getattr(assessment, field)
            if not isinstance(references, list):
                raise ValueError('invalid Assessment reference list: ' + field)
            for ref in references:
                identifier(ref)
        listed = set(assessment.experiment_ids)
    except (ValueError, TypeError, OSError, AttributeError) as exc:
        issue('ASSESSMENT_RECORD_CORRUPT', str(exc))
        return {'assessment_id': assessment_id, 'valid': False, 'issues': issues}

    workspaces = {}
    runs = {}
    # Scan reverse references as well as the Assessment's forward references.
    try:
        folders = sorted(path for path in base.iterdir() if path.is_dir())
    except OSError as exc:
        issue('REFERENCE_UNAVAILABLE', str(exc))
        folders = []
    workspace_ids = listed | {p.name for p in folders if (p / 'workspace.json').exists()}
    for ws_id in sorted(workspace_ids):
        path = base / ws_id / 'workspace.json'
        if not path.exists():
            issue('EXPERIMENT_MISSING', 'Assessment references a missing workspace.', experiment_id=ws_id)
            continue
        ws = None
        try:
            ws = read(path)
            if ws.get('experiment_workspace_id') != ws_id or not isinstance(ws.get('stages'), dict):
                raise ValueError('invalid workspace identifier or stages')
            if ws.get('assessment_id') is not None:
                identifier(ws['assessment_id'])
            for key in ('attack_run_id', 'retest_run_id'):
                if ws.get(key):
                    identifier(ws[key])
        except (ValueError, TypeError, OSError) as exc:
            issue('REFERENCE_UNAVAILABLE', 'Workspace unreadable: ' + str(exc),
                  severity='ERROR' if ws_id in listed or (ws and ws.get('assessment_id') == assessment_id) else 'WARNING', experiment_id=ws_id)
            continue
        workspaces[ws_id] = ws
        owner = ws.get('assessment_id')
        if ws_id in listed:
            if not owner:
                issue('EXPERIMENT_BACKREF_MISSING', 'Assessment lists workspace without its Assessment back-reference.', experiment_id=ws_id)
            elif owner != assessment_id:
                issue('EXPERIMENT_ASSESSMENT_MISMATCH', 'Workspace references another Assessment.', experiment_id=ws_id)
        elif owner == assessment_id:
            issue('EXPERIMENT_FORWARDREF_MISSING', 'Workspace references Assessment but Assessment does not list it.', experiment_id=ws_id)
        if owner == assessment_id or ws_id in listed:
            for key in ('attack_run_id', 'retest_run_id'):
                if ws.get(key):
                    runs.setdefault(ws[key], set()).add(ws_id)

    catalog_ids = {p.name for p in folders if (p / 'experiment.json').exists()}
    for run_id in sorted(catalog_ids | set(runs)):
        record = None
        try:
            record = read(base / run_id / 'experiment.json')
            provenance = record.get('provenance')
            if provenance is None:
                provenance = {}
            if record.get('trace_id') != run_id or not isinstance(provenance, dict):
                raise ValueError('invalid Run identifier or provenance')
            for key in ('assessment_id', 'experiment_workspace_id'):
                if provenance.get(key):
                    identifier(provenance[key])
        except (ValueError, TypeError, OSError) as exc:
            issue('REFERENCE_UNAVAILABLE', 'Run unreadable: ' + str(exc),
                  severity='ERROR' if run_id in runs or (record and isinstance(record.get('provenance'), dict) and record['provenance'].get('assessment_id') == assessment_id) else 'WARNING', run_id=run_id)
            continue
        owner = provenance.get('assessment_id')
        ws_id = provenance.get('experiment_workspace_id')
        related = set(runs.get(run_id, set()))
        if ws_id in workspaces and (owner == assessment_id or ws_id in listed or workspaces[ws_id].get('assessment_id') == assessment_id):
            related.add(ws_id)
        if not related and owner == assessment_id:
            issue('REFERENCE_UNAVAILABLE', 'Run claims Assessment membership without a resolvable workspace.', run_id=run_id, experiment_id=ws_id)
        for related_id in sorted(related):
            workspace_owner = workspaces[related_id].get('assessment_id')
            if owner and not workspace_owner:
                issue('RUN_WITHOUT_WORKSPACE_MEMBERSHIP', 'Run claims Assessment membership while workspace is standalone.', run_id=run_id, experiment_id=related_id)
            elif owner and owner != workspace_owner:
                issue('RUN_ASSESSMENT_MISMATCH', 'Run and workspace reference different Assessments.', run_id=run_id, experiment_id=related_id)
            elif not owner and workspace_owner:
                issue('RUN_ASSESSMENT_PROVENANCE_MISSING', 'Run has no Assessment provenance; legacy data or incomplete synchronization.', severity='WARNING', run_id=run_id, experiment_id=related_id)
            if ws_id and ws_id != related_id:
                issue('RUN_WORKSPACE_MISMATCH', 'Run references a different workspace.', run_id=run_id, experiment_id=related_id)
    return {'assessment_id': assessment_id, 'valid': not any(i['severity'] == 'ERROR' for i in issues), 'issues': issues}
