# Assessments

An **Experiment** validates one security hypothesis. An **Assessment** coordinates multiple evidence-driven experiments over a defined system, target and scope. It groups references; it does not own the security meaning of its Findings or evidence.

## Model and storage

`orion.assessments` follows Orion's JSON/filesystem pattern. Records live in `artifacts/assessments/ORN-ASMT-….json` and use UUID-backed IDs. Fields are `assessment_id`, `name`, `description`, `status`, creation/update timestamps, optional `system_profile_id`, `target_id`, `environment_id`, explicit `scope`, historical `analysis_context_ids`, `threat_model_ids`, `plan_ids`, `experiment_ids`, `finding_ids`, and `metadata`.

`control_refs` is a service-maintained map of applied defense IDs to catalog implementation, workspace, original Attack Run, Finding and Retest Run references. The existing lifecycle stores only its current defense; retaining these pointers prevents a proposal switch from erasing Assessment-level visibility of earlier applications. No control definitions, evidence bytes, metric snapshots or Finding statuses are copied into the Assessment.

Example (illustrative IDs):

```json
{
  "assessment_id": "ORN-ASMT-0042",
  "name": "Support agent security assessment",
  "description": "Public support interface and privileged support tools",
  "status": "ACTIVE",
  "created_at": "2026-10-05T12:00:00+00:00",
  "updated_at": "2026-10-05T13:00:00+00:00",
  "system_profile_id": "ORN-SELF-001",
  "target_id": "ORN-TARGET-001",
  "environment_id": null,
  "scope": {"surfaces": ["rag", "tools"], "access_model": "black_box", "notes": "Local validation only"},
  "analysis_context_ids": ["ORN-CTX-001"],
  "threat_model_ids": ["TM-001", "TM-002"],
  "plan_ids": ["ORN-PLAN-001", "ORN-PLAN-002"],
  "experiment_ids": ["ORN-EXP-001", "ORN-EXP-002"],
  "finding_ids": ["ORN-FND-001"],
  "metadata": {},
  "control_refs": {}
}
```

Status transitions are explicit: DRAFT → ACTIVE → COMPLETED → ARCHIVED; DRAFT/ACTIVE can also be archived. Same-state updates are supported. Archives reject user metadata edits and manual links; system synchronization from already-linked workspaces remains allowed. Completed Assessments reject new manual relationships. Completion neither mitigates Findings nor certifies security. Assessment status is not an execution authorization mechanism: already-linked workspaces retain their existing lifecycle gates.

Scope is user-supplied, not inferred from an attack or timestamp. The UI accepts surfaces, access model, included/excluded capabilities and notes; the API supports additional scope keys. Missing scope/profile information remains UNKNOWN. A DRAFT can be created with only a name; optional inputs are not fabricated. Activation and completion are explicit analyst declarations, not computed security decisions.

## Relationships

`Assessment → context/threat-model generation/plan → workspace → Run/evidence → Finding → defense → Retest` uses existing IDs. Linking a plan retains its exact context/threat-model references. Creating an Assessment experiment creates a separate existing `ExperimentLifecycle` workspace from a plan, optionally selecting a proposal. It neither approves nor executes the plan. Multiple workspaces can use the same plan independently.

Workspaces have optional `assessment_id`. Run/evidence and newly generated Finding provenance carry this root reference. Retest provenance retains the original attack, exact plan/threat model and applied control. The shared catalog's `ControlImplementation` definitions are not tagged with an Assessment: applied-control references identify their use, not ownership of the definition.

Findings already aggregate corroboration by attack and target. They can therefore be linked into multiple Assessments. A shared Finding retains its original provenance and existing global corroboration/retest status; Assessment membership does not constrain or promote that status. Summary membership also resolves Findings through the explicitly related Run IDs. A workspace may belong to one Assessment; attempts to reassign it are rejected.

Standalone records load as before with `assessment_id=None` or absent provenance. Explicitly linking a historical workspace adds Assessment/Workspace provenance to its currently referenced Run and Retest records, regenerating their existing reports in place. It creates no new evidence directory and does not change metrics, results or artifact files. Earlier records lacking retained relationships cannot be recovered by guessing from names or timestamps.

## Service, API and UI

`AssessmentService` supports create/get/update, idempotent linking, completion/archive, creating existing workspaces from plans and read-only summary resolution. Updates retain historical relationship IDs; PATCH is additive for reference lists, replaces supplied scope/metadata objects, and does not unlink history. Conflicting known system/target/environment references are rejected when attaching workspaces/plans. Linking an independently stored Finding does not alter that Finding.

| Method | URL | Purpose |
|---|---|---|
| GET / POST | `/api/assessments` | List / create |
| GET / PATCH | `/api/assessments/<id>` | Read / metadata, scope, status and explicit links |
| GET | `/api/assessments/<id>/summary` | Resolve related records, factual counts and next action |
| GET | `/api/assessments/<id>/integrity` | Read-only relationship validation |
| POST | `/api/assessments/<id>/experiments` | Create a workspace from `plan_id`, optional `proposal_id` |

PATCH can link `experiment_ids`, `finding_ids`, `plan_ids`, `analysis_context_ids` and `threat_model_ids`. Missing linked experiments/findings/plans/contexts are rejected before an update is persisted. Threat-model IDs are retained as version references and resolved from their source plans/context; unavailable sources are reported explicitly.

The workstation sidebar adds **Assessments** under ASSESS. `/assessments` provides a compact list/create form. `/assessments/<id>` shows Next Action, scope, factual posture, experiment/finding links, applied controls/retests and structured provenance. Its forms edit metadata and link existing objects. Attack execution remains in the existing Experiment workspace. Generate analysis/plans using the existing Analysis/review flow, then explicitly associate their IDs or create a linked workspace from the plan. No hidden UI-driven membership inference occurs.

Assessment Next Action points to missing scope/context, unavailable references, or an existing workspace's deterministic `next_action`. It prioritizes recorded pending retests and never executes an attack/retest or changes status during a page read.

## Summaries

Counts come from resolved records and existing Finding/retest states. Explicitly empty membership can yield `0`. Missing or unreadable related records yield `null` (rendered UNKNOWN), plus `errors` and `known_total`; available records remain visible. Catalog corruption that prevents establishing membership is reported conservatively. Reads never persist or change state.

For an Assessment with one observed Finding, an applied control and no retest, factual sections may look like:

```json
{
  "experiments": {"total": 1, "known_total": 1, "planned": 0, "running": 0, "completed": 0},
  "findings": {"total": 1, "known_total": 1, "confirmed": 0, "observed": 1, "hypothesis": 0},
  "controls": {"applied": 1, "suggested": 3, "awaiting_retest": 1},
  "retests": {"mitigated": 0, "partial": 0, "not_mitigated": 0},
  "posture": {"confirmed": 0, "observed": 1, "high_critical": 1, "awaiting_retest": 1, "verified_mitigations": 0, "unresolved": 1}
}
```

These are examples, not seeded results. Applied controls do not contribute verified mitigations until existing Retest evidence establishes effectiveness.

## Foundation limits

- Filesystem records are atomically replaced individually; cross-object writes have no database transaction or multi-writer guarantee. Single-user local operation follows the existing persistence assumptions.
- Summaries resolve artifact references at read time; there is no index/cache/pagination or historical posture snapshot. Completing/archiving an Assessment preserves references, not frozen copies of future-changing shared Findings.
- Workspaces remain mutable: a superseded pending defense stays visible through its pointers, but restoring an earlier workspace configuration is not implemented. Create separate workspaces for separate proposals to retain their executable lifecycle.
- Older workspace proposal switches may have discarded references before this layer existed. Only surviving explicit references are linkable; no forced migration is attempted.
- Some Assessment-related Findings can include evidence from other Assessments because existing corroboration is global. This remains visible through original Run/Finding provenance.
- Assessment status and linkage provide organization, not RBAC, tenant isolation, authorization or a new safety policy.
- Deferred: security regression products, reproducibility bundles, adaptive retesting, experiment matrices, PDF reporting, scheduling, CI/CD integrations, new attacks and severity models.

## Relationship consistency and integrity validation

`AssessmentService.validate_integrity(id)` (also available as `validate_assessment_integrity(id, base_dir)` in `orion.assessments.integrity`) checks both directions of Assessment/workspace membership and current/historical Run provenance. `GET /api/assessments/<id>/integrity` exposes the same read-only result:

```json
{"assessment_id": "ORN-ASMT-0042", "valid": true, "issues": []}
```

A mismatch is reported with identifiers and deterministic issue codes:

```json
{
  "assessment_id": "ORN-ASMT-0042",
  "valid": false,
  "issues": [{
    "code": "EXPERIMENT_ASSESSMENT_MISMATCH",
    "severity": "ERROR",
    "experiment_id": "ORN-EXP-001",
    "message": "Workspace references another Assessment."
  }]
}
```

Checks include missing/corrupt Assessment and workspace records, missing forward/back-references, conflicting Assessment ownership, Run/workspace ownership conflicts, missing referenced Runs and Runs without resolvable workspaces. Historical Runs are found by explicit provenance, not filenames/timestamps. Catalog records with unreadable content generate warnings when their membership cannot be established. A legacy Run with no Assessment provenance produces `RUN_ASSESSMENT_PROVENANCE_MISSING` as a WARNING: it may predate this layer or indicate incomplete propagation. `valid` means no ERROR issues; warnings remain visible and do not certify complete provenance. Missing Assessments return an invalid structured result; invalid identifier syntax is rejected.

Validation does not save, repair, reassign ownership or change evidence. No reconciliation operation is introduced. Active Assessment Next Action considers all reference/integrity errors before proposing experiment actions; closed Assessments retain evidence review as their next action. The existing deterministic experiment next action remains authoritative inside each workspace.

## Synchronization and failure model

`orion.experiments.lifecycle.save()` now writes only the workspace. The public `orion.experiments` operations invoke the domain operation and then `sync_assessment_membership(ws, base_dir)`. The public `save_workspace()` remains a compatibility alias for `persist_workspace_and_sync_assessment()`, preserving existing application callers. Standalone operations skip synchronization and do not require an Assessment.

Synchronization deduplicates workspace, plan, context, threat-model and Finding IDs, retains control pointers, and annotates the current Run/Retest provenance in place. Repeated synchronization does not rewrite unchanged Assessment timestamps or duplicate references. Conflicting Run ownership is rejected rather than reassigned. Lower-level lifecycle callers intentionally persist independently; callers coordinating an assessed operation must invoke synchronization explicitly or use the public orchestration API.

There are **no cross-object transactions**. If a workspace write succeeds and Assessment persistence fails, the workspace remains saved and the exception reaches the caller; integrity checking reports the missing relationship. If Assessment persistence succeeds and a subsequent Run provenance write fails, those successful writes also remain. The exception is surfaced and validation reports missing/conflicting/unavailable provenance. A missing legacy child remains unavailable and is reported on integrity/summary reads. A domain-operation failure retains its existing lifecycle error handling and successful writes; orchestration does not silently repair it. Successful retry of an unambiguous synchronization is idempotent, but validation itself never performs that retry. No successful Experiment, Evidence or Finding data is deleted or rolled back to simulate a transaction.

## Archived semantics and shared domain objects

**Assessment is a live aggregation of security objects, not an immutable snapshot.** COMPLETED declares that the intended activity is finished. ARCHIVED marks the Assessment as no longer actively operated; its list/detail remain available for reviewing evidence. Archived user edits to name, description, scope, target and manual relationships are blocked. System-derived synchronization may still record late Run provenance, discovered Findings and control/retest pointers from an **already-linked** workspace. It cannot attach a new workspace to an archived Assessment.

Shared Findings and independent Evidence retain their existing domain behavior. A Finding linked into several Assessments remains one record; later corroboration/status updates appear in all of them, including archived ones. Completion/archive do not modify Finding status, Retest results or control verification. ARCHIVED is not an execution authorization gate for already-linked workspaces.

**For immutable historical reproduction, use a future Reproducibility Bundle rather than relying on ARCHIVED state.** That bundle is not implemented here. Corroboration criteria reproduction, success signatures, per-criterion rates and confirmation redesign remain follow-ups for the Security Regression Tests milestone. Severity evolution also remains deferred; this consistency pass changes neither model.
