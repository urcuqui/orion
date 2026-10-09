"""Public workspace operations followed by explicit Assessment synchronization.

Lifecycle persistence remains independent. Successful writes are retained if
synchronization fails; callers receive the failure and can inspect integrity.
"""
from functools import wraps
from inspect import signature
from . import lifecycle


def sync_assessment_membership(ws, base_dir=lifecycle.DEFAULT_DIR):
    if ws.assessment_id:
        from orion.assessments.service import record_workspace
        record_workspace(ws, base_dir)


def persist_workspace_and_sync_assessment(ws, base_dir=lifecycle.DEFAULT_DIR):
    path = lifecycle.save(ws, base_dir)
    sync_assessment_membership(ws, base_dir)
    return path


def _synchronized(operation):
    parameters = signature(operation)

    @wraps(operation)
    def call(*args, **kwargs):
        bound = parameters.bind(*args, **kwargs)
        bound.apply_defaults()
        result = operation(*args, **kwargs)
        ws = bound.arguments.get('ws', result)
        sync_assessment_membership(ws, bound.arguments['base_dir'])
        return result
    return call


# Preserve the established public signatures and return values. Domain operations
# keep their own stage/error handling; failed operations are not silently repaired.
create_from_plan = _synchronized(lifecycle.create_from_plan)
select_experiment = _synchronized(lifecycle.select_experiment)
run_attack = _synchronized(lifecycle.run_attack)
run_blackbox_attack = _synchronized(lifecycle.run_blackbox_attack)
run_whitebox_attack = _synchronized(lifecycle.run_whitebox_attack)
run_agentic_attack = _synchronized(lifecycle.run_agentic_attack)
run_live_agentic_attack = _synchronized(lifecycle.run_live_agentic_attack)
measure = _synchronized(lifecycle.measure)
apply_defense = _synchronized(lifecycle.apply_defense)
retest = _synchronized(lifecycle.retest)
