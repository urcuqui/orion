"""Explicit membership and read-only aggregation, not a second security engine."""
from .model import Assessment, STATUSES, now
from .repository import AssessmentRepository, identifier


class AssessmentService:
    def __init__(self, base_dir='artifacts'):
        self.base_dir = str(base_dir)
        self.repository = AssessmentRepository(base_dir)

    def get(self, assessment_id):
        result = self.repository.get(assessment_id)
        if result is None:
            raise LookupError('assessment not found')
        return result

    def create(self, data):
        assessment = Assessment()
        self._apply(assessment, data, creating=True)
        self.repository.save(assessment)
        if data.get('experiment_ids'):
            return self.update(assessment.assessment_id, {'experiment_ids': data['experiment_ids']})
        return assessment

    def update(self, assessment_id, data):
        assessment = self.get(assessment_id)
        self._apply(assessment, data)
        assessment.updated_at = now()
        self.repository.save(assessment)
        # Membership propagation follows validation and never changes stage/state.
        from orion.experiments import load_workspace, save_workspace
        for ws_id in data.get('experiment_ids', []):
            ws = load_workspace(ws_id, self.base_dir)
            ws.assessment_id = assessment_id
            save_workspace(ws, self.base_dir)
        return self.get(assessment_id)

    def _apply(self, assessment, data, creating=False):
        if not isinstance(data, dict):
            raise ValueError('assessment payload must be an object')
        allowed = set(Assessment.__dataclass_fields__) - {'assessment_id', 'created_at', 'updated_at', 'control_refs'}
        if set(data) - allowed:
            raise ValueError('unsupported assessment fields: ' + ', '.join(sorted(set(data) - allowed)))
        for key in ('name', 'description'):
            if key in data and not isinstance(data[key], str):
                raise ValueError(key + ' must be text')
        if not (data.get('name', assessment.name) or '').strip():
            raise ValueError('assessment name is required')
        for key in ('scope', 'metadata'):
            if key in data and not isinstance(data[key], dict):
                raise ValueError(key + ' must be an object')
        scope = data.get('scope', {})
        for key in ('surfaces', 'in_scope_capabilities', 'out_of_scope_capabilities'):
            if key in scope and (not isinstance(scope[key], list) or any(not isinstance(value, str) for value in scope[key])):
                raise ValueError('scope ' + key + ' must be a list of text values')
        for key in ('notes', 'access_model'):
            if key in scope and not isinstance(scope[key], str):
                raise ValueError('scope ' + key + ' must be text')
        for key in ('system_profile_id', 'target_id', 'environment_id'):
            if data.get(key) is not None:
                identifier(data[key])
        reference_fields = ('analysis_context_ids', 'threat_model_ids', 'plan_ids', 'experiment_ids', 'finding_ids')
        for key in reference_fields:
            if key in data:
                if not isinstance(data[key], list):
                    raise ValueError(key + ' must be a list')
                for value in data[key]:
                    identifier(value)
        from orion.experiments import load_workspace
        from orion.findings import FindingStore
        from orion import plans
        from orion.context.store import load_profile
        profile_changed = any(key in data and data[key] != getattr(assessment, key) for key in ('system_profile_id', 'target_id', 'environment_id'))
        for ws_id in set((assessment.experiment_ids if profile_changed else []) + data.get('experiment_ids', [])):
            ws = load_workspace(ws_id, self.base_dir)
            if ws is None:
                raise ValueError('experiment not found: ' + ws_id)
            if ws.assessment_id not in (None, assessment.assessment_id):
                raise ValueError('experiment already belongs to another assessment')
            from orion.evidence import EvidenceStore
            for trace in (ws.attack_run_id, ws.retest_run_id):
                if trace:
                    identifier(trace)
                    try:
                        provenance = EvidenceStore(self.base_dir).load(trace).provenance or {}
                    except (ValueError, TypeError, OSError):
                        continue  # Partial historical evidence remains visible as unavailable.
                    if provenance.get('assessment_id') not in (None, assessment.assessment_id):
                        raise ValueError('run already belongs to another assessment')
            for field, linked in [('system_profile_id', ws.self_profile_id), ('target_id', ws.target_profile_id), ('environment_id', ws.environment_profile_id)]:
                scoped = data.get(field, getattr(assessment, field))
                if scoped and linked and scoped != linked:
                    raise ValueError('experiment conflicts with assessment ' + field)
        for finding_id in data.get('finding_ids', []):
            if FindingStore(self.base_dir).load(finding_id) is None:
                raise ValueError('finding not found: ' + finding_id)
        for plan_id in set((assessment.plan_ids if profile_changed else []) + data.get('plan_ids', [])):
            plan = plans.load_plan(plan_id, self.base_dir)
            if plan is None:
                raise ValueError('plan not found: ' + plan_id)
            for field, linked in [('system_profile_id', plan.self_profile_id), ('target_id', plan.target_profile_id), ('environment_id', plan.environment_profile_id)]:
                scoped = data.get(field, getattr(assessment, field))
                if scoped and linked and scoped != linked:
                    raise ValueError('plan conflicts with assessment ' + field)
        for context_id in data.get('analysis_context_ids', []):
            if load_profile(context_id, self.base_dir) is None:
                raise ValueError('analysis context not found: ' + context_id)
        proposed = data.get('status', assessment.status)
        transitions = {'DRAFT': {'DRAFT', 'ACTIVE', 'ARCHIVED'}, 'ACTIVE': {'ACTIVE', 'COMPLETED', 'ARCHIVED'}, 'COMPLETED': {'COMPLETED', 'ARCHIVED'}, 'ARCHIVED': {'ARCHIVED'}}
        if proposed not in STATUSES or (creating and proposed not in ('DRAFT', 'ACTIVE')) or (not creating and proposed not in transitions[assessment.status]):
            raise ValueError('invalid assessment status transition')
        if assessment.status == 'COMPLETED' and any(set(data.get(k, [])) - set(getattr(assessment, k)) for k in reference_fields):
            raise ValueError('completed assessment cannot add relationships')
        if assessment.status == 'ARCHIVED' and any(k != 'status' for k in data):
            raise ValueError('archived assessment metadata and manual links are read-only')
        for key, value in data.items():
            if key in reference_fields:
                # Additive references retain historical threat models/plans.
                setattr(assessment, key, list(dict.fromkeys(getattr(assessment, key) + value)))
            else:
                setattr(assessment, key, value)
        for plan_id in data.get('plan_ids', []):
            plan = plans.load_plan(plan_id, self.base_dir)
            for key, ref in [('threat_model_ids', plan.threat_model_id), ('analysis_context_ids', plan.analysis_context_id)]:
                if ref and ref not in getattr(assessment, key):
                    getattr(assessment, key).append(ref)

    def link_experiment(self, assessment_id, experiment_id):
        return self.update(assessment_id, {'experiment_ids': [experiment_id]})

    def link_finding(self, assessment_id, finding_id):
        return self.update(assessment_id, {'finding_ids': [finding_id]})

    def complete(self, assessment_id):
        return self.update(assessment_id, {'status': 'COMPLETED'})

    def archive(self, assessment_id):
        return self.update(assessment_id, {'status': 'ARCHIVED'})

    def validate_integrity(self, assessment_id):
        from .integrity import validate_assessment_integrity
        return validate_assessment_integrity(assessment_id, self.base_dir)

    def create_experiment(self, assessment_id, plan_id, proposal_id=None):
        from orion import plans
        from orion.experiments import create_from_plan
        assessment = self.get(assessment_id)
        if assessment.status in ('COMPLETED', 'ARCHIVED'):
            raise ValueError('assessment is closed')
        identifier(plan_id)
        plan = plans.load_plan(plan_id, self.base_dir)
        if plan is None:
            raise ValueError('plan not found')
        from orion.experiments.lifecycle import _proposal_runnable
        if proposal_id:
            proposal = next((p for p in plan.proposals if p.experiment_id == proposal_id), None)
            if proposal is None or not _proposal_runnable(proposal):
                raise ValueError('proposal is not runnable')
        for field, linked in [('system_profile_id', plan.self_profile_id), ('target_id', plan.target_profile_id), ('environment_id', plan.environment_profile_id)]:
            if getattr(assessment, field) and linked and getattr(assessment, field) != linked:
                raise ValueError('plan conflicts with assessment ' + field)
        ws = create_from_plan(plan, self.base_dir, assessment_id=assessment_id)
        if proposal_id:
            # Pick the proposal without approving or running the plan.
            ws.active_experiment_id = proposal_id
            from orion.experiments import save_workspace
            save_workspace(ws, self.base_dir)
        return ws

    def summary(self, assessment_id):
        """Resolve explicit references, retaining unavailable data as unknown."""
        assessment = self.get(assessment_id)
        from orion.experiments import load_workspace, next_action
        from orion.evidence import EvidenceStore
        from orion.findings import FindingStore
        from orion.context.store import load_profile
        from orion import plans
        errors, experiments, runs = [], [], {}
        def resolve(kind, ref, loader):
            try:
                identifier(ref)
                obj = loader(ref)
                if obj is None:
                    raise ValueError('not found')
                return obj
            except (ValueError, TypeError, OSError, KeyError, AttributeError) as exc:
                errors.append({'kind': kind, 'id': ref, 'error': str(exc)})
                return None
        profiles = {key: resolve('profile', ref, lambda value: load_profile(value, self.base_dir)) if ref else None
                    for key, ref in [('system', assessment.system_profile_id), ('target', assessment.target_id), ('environment', assessment.environment_id)]}
        contexts = [resolve('context', ref, lambda value: load_profile(value, self.base_dir)) for ref in assessment.analysis_context_ids]
        resolved_plans = [resolve('plan', ref, lambda value: plans.load_plan(value, self.base_dir)) for ref in assessment.plan_ids]
        evidence = EvidenceStore(self.base_dir)
        for ws_id in assessment.experiment_ids:
            ws = resolve('experiment', ws_id, lambda value: load_workspace(value, self.base_dir))
            if ws:
                experiments.append(ws.to_dict())
                for trace in (ws.attack_run_id, ws.retest_run_id):
                    if trace:
                        record = resolve('run', trace, evidence.load)
                        if record:
                            runs[trace] = record.to_dict()
        # Earlier runs survive proposal switches through explicit Assessment provenance.
        if evidence.base_dir.exists():
            for folder in evidence.base_dir.iterdir():
                if folder.is_dir() and (folder / 'experiment.json').exists():
                    try:
                        record = evidence.load(folder.name)
                        if (record.provenance or {}).get('assessment_id') == assessment_id:
                            runs[record.trace_id] = record.to_dict()
                    except (ValueError, TypeError, OSError, AttributeError):
                        # Unknown membership in a corrupt record cannot be guessed.
                        errors.append({'kind': 'run_catalog', 'id': folder.name, 'error': 'record unavailable; membership unknown'})
        findings_store = FindingStore(self.base_dir)
        finding_ids = set(assessment.finding_ids)
        finding_ids.update(ws['finding_id'] for ws in experiments if ws.get('finding_id'))
        try:
            # Read individually so corrupt files do not silently disappear from the summary.
            for path in findings_store.dir.glob('*.json'):
                try:
                    finding = findings_store.load(path.stem)
                    if finding and ((finding.provenance or {}).get('assessment_id') == assessment_id or set(finding.evidence_refs + finding.retest_refs).intersection(runs)):
                        finding_ids.add(finding.id)
                except (ValueError, TypeError, OSError, AttributeError):
                    errors.append({'kind': 'finding_catalog', 'id': path.stem, 'error': 'record unavailable; membership unknown'})
        except OSError as exc:
            errors.append({'kind': 'finding_catalog', 'id': None, 'error': str(exc)})
        findings = [f.to_dict() for ref in sorted(finding_ids) if (f := resolve('finding', ref, findings_store.load))]
        for finding in findings:
            for trace in finding.get('evidence_refs', []) + finding.get('retest_refs', []):
                if trace not in runs:
                    record = resolve('run', trace, evidence.load)
                    if record:
                        runs[trace] = record.to_dict()
        for ref in assessment.control_refs.values():
            for trace in (ref.get('attack_run_id'), ref.get('retest_run_id')):
                if trace and trace not in runs:
                    record = resolve('run', trace, evidence.load)
                    if record:
                        runs[trace] = record.to_dict()
        controls = [dict(ref, id=ref_id, awaiting_retest=not bool(ref.get('retest_run_id'))) for ref_id, ref in assessment.control_refs.items()]
        covered = {ref.get('finding_id') for ref in controls}
        recorded_controls = {ref['id'] for ref in controls}
        for ws in experiments:
            if ws.get('defense_id') and ws['defense_id'] not in recorded_controls:
                covered.add(ws.get('finding_id'))
                controls.append({'id': ws['defense_id'], 'experiment_id': ws['experiment_workspace_id'], 'finding_id': ws.get('finding_id'), 'control': ws.get('defense_control'), 'implementation': ws.get('defense_implementation'), 'retest_run_id': ws.get('retest_run_id'), 'awaiting_retest': not bool(ws.get('retest_run_id'))})
        for finding in findings:
            if finding.get('applied_control') and finding['id'] not in covered:
                controls.append({'id': None, 'experiment_id': None, 'finding_id': finding['id'], 'control': finding['applied_control'], 'implementation': finding.get('control_implementation'), 'retest_run_id': None, 'awaiting_retest': not bool(finding.get('retest_refs'))})
        unknown_exp = any(e['kind'] == 'experiment' for e in errors)
        unknown_findings = unknown_exp or any(e['kind'] in ('finding', 'finding_catalog', 'run', 'run_catalog') for e in errors)
        def count(value, unknown=False):
            return None if unknown else value
        posture = {'confirmed': count(sum(f['status'] == 'CONFIRMED' for f in findings), unknown_findings),
                   'observed': count(sum(f['status'] == 'OBSERVED' for f in findings), unknown_findings),
                   'high_critical': count(sum(f['severity'] in ('HIGH', 'CRITICAL') for f in findings), unknown_findings),
                   'awaiting_retest': count(sum(c['awaiting_retest'] for c in controls), unknown_findings),
                   'verified_mitigations': count(sum(f.get('retest_status') == 'EFFECTIVE' for f in findings), unknown_findings),
                   'unresolved': count(sum(f['status'] not in ('MITIGATED', 'NOT_REPRODUCIBLE') for f in findings), unknown_findings)}
        threat_models = []
        for ref in assessment.threat_model_ids:
            sources = [c for c in contexts if c and (c.get('threat_model') or {}).get('threat_model_id') == ref]
            sources += [p.to_dict() for p in resolved_plans if p and p.threat_model_id == ref]
            threat_models.append({'id': ref, 'available': bool(sources)})
            if not sources:
                errors.append({'kind': 'threat_model', 'id': ref, 'error': 'source context/plan unavailable'})
        integrity = self.validate_integrity(assessment_id)
        errors.extend({'kind': 'integrity', 'id': issue.get('experiment_id') or issue.get('run_id'), 'error': issue['message']} for issue in integrity['issues'] if issue['severity'] == 'ERROR')
        action = {'label': 'Review assessment', 'href': '/assessments/' + assessment_id}
        if assessment.status in ('COMPLETED', 'ARCHIVED'):
            action['label'] = 'Review recorded evidence'
            action['href'] += '#assessment-experiments'
        elif not assessment.system_profile_id or not assessment.target_id or not assessment.scope:
            action = {'label': 'Define system, target and scope', 'href': '#assessment-edit'}
        elif errors:
            action = {'label': 'Review unavailable references', 'href': '#assessment-provenance'}
        elif not assessment.analysis_context_ids and not assessment.plan_ids:
            from urllib.parse import urlencode
            action = {'label': 'Run analysis', 'href': '/context?' + urlencode({'self_profile_id': assessment.system_profile_id, 'target_profile_id': assessment.target_id, 'environment_profile_id': assessment.environment_id or ''})}
        elif not experiments and resolved_plans and any(p and not p.approved_by_human for p in resolved_plans):
            action = {'label': 'Create experiment to review plan', 'href': '#assessment-links'}
        else:
            pending = next((ws for ws in experiments if ws.get('defense_id') and not ws.get('retest_run_id')), None)
            pending = pending or next((ws for ws in experiments if ws['stages'].get('retest') != 'COMPLETE'), None)
            if pending:
                action = {'label': pending['next_action']['label'], 'href': '/experiment/' + pending['experiment_workspace_id']}
            elif not experiments:
                action = {'label': 'Link or create experiments from a plan', 'href': '#assessment-links'}
            elif assessment.status == 'ACTIVE':
                action = {'label': 'Review scope before completing assessment', 'href': '#assessment-edit'}
        for ws in experiments:
            record = runs.get(ws.get('attack_run_id'), {})
            source_plan = next((p for p in resolved_plans if p and p.plan_id == ws.get('plan_id')), None)
            proposal = next((p for p in source_plan.proposals if p.experiment_id == ws.get('active_experiment_id')), None) if source_plan else None
            ws['family'] = record.get('family') or (proposal.branch if proposal else None)
            ws['result'] = record.get('status')
            ws['retest_result'] = runs.get(ws.get('retest_run_id'), {}).get('status')
        return {'assessment': assessment.to_dict(), 'profiles': profiles, 'next_action': action,
                'experiments': {'total': count(len(experiments), unknown_exp), 'known_total': len(experiments),
                                'planned': count(sum(ws['stages'].get('attack') not in ('RUNNING', 'COMPLETE') for ws in experiments), unknown_exp),
                                'running': count(sum('RUNNING' in ws['stages'].values() for ws in experiments), unknown_exp),
                                'completed': count(sum(ws['stages'].get('retest') == 'COMPLETE' for ws in experiments), unknown_exp)},
                'findings': {'total': count(len(findings), unknown_findings), 'known_total': len(findings),
                             **{state.lower(): count(sum(f['status'] == state for f in findings), unknown_findings) for state in ('CONFIRMED', 'OBSERVED', 'HYPOTHESIS')}},
                'controls': {'applied': count(len(controls), unknown_findings), 'suggested': count(len({c for f in findings for c in f.get('recommended_controls', [])}), unknown_findings), 'awaiting_retest': posture['awaiting_retest']},
                'retests': {'mitigated': posture['verified_mitigations'],
                            'partial': count(sum(f.get('retest_status') == 'PARTIALLY_EFFECTIVE' for f in findings), unknown_findings),
                            'not_mitigated': count(sum(f.get('retest_status') == 'INEFFECTIVE' for f in findings), unknown_findings)},
                'posture': posture, 'experiment_records': experiments, 'finding_records': findings,
                'run_records': list(runs.values()), 'control_records': controls, 'threat_models': threat_models,
                'errors': errors, 'integrity': integrity}


def record_workspace(ws, base_dir):
    """Persist membership after existing domain operations, without status changes."""
    if not ws.assessment_id:
        return
    service = AssessmentService(base_dir)
    assessment = service.get(ws.assessment_id)
    if assessment.status == 'ARCHIVED' and ws.experiment_workspace_id not in assessment.experiment_ids:
        raise ValueError('archived assessment cannot acquire a new experiment through synchronization')
    from orion.evidence import EvidenceStore
    evidence = EvidenceStore(base_dir)
    records = []
    for trace in (ws.attack_run_id, ws.retest_run_id):
        if trace:
            identifier(trace)
            try:
                record = evidence.load(trace)
            except (ValueError, TypeError, OSError):
                continue  # Read-only integrity/summary reports unavailable legacy children.
            provenance = record.provenance or {}
            if provenance.get('assessment_id') not in (None, ws.assessment_id):
                raise ValueError('run belongs to another assessment')
            if provenance.get('experiment_workspace_id') not in (None, ws.experiment_workspace_id):
                raise ValueError('run belongs to another workspace')
            records.append(record)
    before = assessment.to_dict()
    # Lifecycle mutations remain available for linked workspaces; archive is
    # assessment metadata, not an execution authorization mechanism.
    for field, value in [('experiment_ids', ws.experiment_workspace_id), ('plan_ids', ws.plan_id),
                         ('analysis_context_ids', ws.analysis_context_id), ('threat_model_ids', ws.threat_model_id), ('finding_ids', ws.finding_id)]:
        if value and value not in getattr(assessment, field):
            getattr(assessment, field).append(value)
    if ws.defense_id:
        assessment.control_refs[ws.defense_id] = {'experiment_id': ws.experiment_workspace_id, 'finding_id': ws.finding_id, 'control': ws.defense_control, 'implementation': ws.defense_implementation, 'attack_run_id': ws.attack_run_id, 'retest_run_id': ws.retest_run_id}
    if assessment.to_dict() != before:
        assessment.updated_at = now()
        service.repository.save(assessment)
    for record in records:
        provenance = record.provenance or {}
        if provenance.get('assessment_id') != ws.assessment_id or provenance.get('experiment_workspace_id') != ws.experiment_workspace_id:
            record.provenance = {**provenance, 'assessment_id': ws.assessment_id, 'experiment_workspace_id': ws.experiment_workspace_id}
            evidence.save(record)
