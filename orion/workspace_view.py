"""Read-only presentation data for the workspace overview.

Uses the existing lifecycle next_action; never writes artifacts or executes work.
"""
import json
from pathlib import Path

from orion.context.store import load_profile
from orion.evidence import EvidenceStore
from orion.experiments.lifecycle import list_workspaces, load
from orion.findings import list_findings


def overview(base_dir='artifacts'):
    errors = []
    workspaces = list_workspaces(base_dir)
    current = None
    for item in workspaces:
        try:
            current = load(item['experiment_workspace_id'], base_dir)
            if current:
                break
        except (ValueError, OSError, TypeError):
            errors.append('An experiment workspace could not be read.')
    profiles = []
    surfaces = []
    for key, label, filename, endpoint in (
        ('self', 'System', 'self.json', 'know_yourself_page'),
        ('target', 'Target', 'target.json', 'target_analysis_page'),
        ('environment', 'Environment', 'environment.json', 'environment_page'),
    ):
        profile_id = getattr(current, key + '_profile_id', None) if current else None
        data = None
        try:
            if profile_id:
                data = load_profile(profile_id, base_dir)
            elif not current:
                candidates = []
                for path in Path(base_dir).glob('*/' + filename):
                    try:
                        value = json.loads(path.read_text())
                        candidates.append((value.get('updated_at') or value.get('created_at') or '', path.parent.name, value))
                    except (ValueError, OSError):
                        errors.append(f'A {label.lower()} profile could not be read.')
                if candidates:
                    _, profile_id, data = max(candidates, key=lambda v: v[0])
        except (ValueError, OSError):
            errors.append(f'The linked {label.lower()} profile could not be read.')
        if data:
            if key == 'target':
                items = data.get('known_interfaces') or []
            elif key == 'environment':
                items = data.get('endpoints') or []
            else:
                items = [f"{k}: {v}" for k, v in (data.get('capabilities') or {}).items()]
            surfaces.extend({'source': label, 'value': str(item)} for item in items)
        profiles.append({'key': key, 'label': label, 'id': profile_id, 'endpoint': endpoint,
                         'name': (data or {}).get('name') or (data or {}).get('system_type') or (data or {}).get('target_reference'),
                         'available': data is not None})
    findings = [f.to_dict() for f in list_findings(base_dir)]
    runs = EvidenceStore(base_dir).list_summaries()
    if current:
        action = dict(current.to_dict()['next_action'], href='/experiment/' + current.experiment_workspace_id,
                      description='Open the workspace to review the current stage. Execution requires an explicit action.')
    else:
        action = {'label': 'REVIEW ASSESSMENT INPUTS', 'href': '/context',
                  'description': 'Select the profiles for analysis. Analysis proposes experiments; human approval does not execute them.'}
    return {'profiles': profiles, 'current': current.to_dict() if current else None,
            'workspaces': workspaces[:6], 'findings': findings, 'runs': runs[:6],
            'action': action, 'errors': errors, 'surfaces': surfaces[:8]}


def finding_workspace(finding, base_dir='artifacts'):
    """Read a linked workspace only if it still belongs to this finding.

    A stale provenance ID must not offer another experiment's next action.
    This presentation lookup never creates workspaces or changes lifecycle state.
    """
    if not finding:
        return None
    workspace_id = (finding.provenance or {}).get('experiment_workspace_id')
    if not workspace_id:
        return None
    try:
        ws = load(workspace_id, base_dir)
    except (ValueError, OSError, TypeError):
        return None
    if ws and (ws.finding_id == finding.id or ws.attack_run_id in finding.evidence_refs):
        return ws.to_dict()
    return None
