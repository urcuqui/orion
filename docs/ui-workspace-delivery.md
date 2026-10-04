# Orion analyst workspace — delivery notes

This iteration evolves the Flask/Jinja application in four focused passes: foundation, assessment, validation, and observability. It keeps the existing routes, APIs, CLI, scenarios, and security workflow. No frontend framework, new security methodology, or new execution engine was introduced.

## Navigation and information architecture

- Overview: assessment overview at `/`.
- Assess: System (`/know-yourself`), Target (`/target-analysis`), Environment (`/environment`), Analysis (`/context`). The original Art of War identity remains visible beneath operational labels.
- Validate: Experiments (`/experiment`), Findings (`/findings`).
- Observe: Runs (`/runs`), Evidence (`/runs?view=evidence`). Evidence is an artifact-oriented view of the same persisted records, with links to the originating run, available finding, and experiment. No new destination route or duplicate evidence store was added.

The desktop sidebar uses a marker, weight, surface, border, and `aria-current` for the active page. The compact persistent terminal header shows the actual URL path. Small screens use a wrapping navigation layout. Operational tables scroll locally.

## Design system and reusable components

Semantic tokens cover surfaces, borders, text, identity, status, actions, and spacing. Existing variable names remain aliases, so existing page-specific styles continue to work. UI text uses system sans-serif; technical values, IDs, metrics, and terminal output use monospace. Crimson identifies Orion and important security results, phosphor green supports affirmative state, and amber signals uncertainty or work requiring attention.

Analyst is the default. Classic persists in `localStorage`; `?demo=1` still enables the conference presentation. Classic preserves ASCII, scanlines, and restrained glow. Decorative flicker was removed. Reduced-motion rules apply in both modes.

Added/consolidated components:

- Jinja `page_header` and `next_action_card` in `components/workspace.html`.
- JavaScript next-action, provenance, and execution-boundary renderers for dynamic workspaces.
- Native execution confirmation dialog, reused for live attacks, live agentic retests, and URL probes. Cancel, Escape, focus restoration, and captured execution parameters are supported.
- Existing Jinja/JavaScript status badges now include symbols and explicit text. Missing status and metric values render UNKNOWN. UNKNOWN and NOT_APPLICABLE remain separate.
- Existing metric, provenance, and image comparison components remain in use. Image amplification and underlying measurements are unchanged.
- The actual experiment lifecycle is an ordered stepper showing each backend stage state. Assessment methodology strips indicate workflow orientation and no longer imply completed executions from page location.
- Shared key/value styles moved out of target-specific CSS into the base layer.

Accessibility improvements include landmarks, skip navigation, visible focus, real buttons for assessment menu choices, selected filter state, form labels, live status/error announcements, reduced motion, locally scrolling tables, and native dialogs. This is not a claim of a complete WCAG 2.2 AA audit.

## Before and after

| Surface | Before | After |
| --- | --- | --- |
| Overview | Decorative identity panel and click-only launcher rows | Read-only assessment profiles, deterministic next action, recorded findings, linked attack-surface data, recent experiments and executions; missing security information is explicit |
| Assess | Art of War labels were the main orientation | System / Target / Environment labels, original identity, clear purpose, keyboard-operated choices, and live probe confirmations |
| Analysis | Hidden under Target navigation; profile selection had little context | First-class navigation, explicit input-to-approval flow, live selected-input count, optional missing inputs, preserved source-attributed threat objects and experiment applicability |
| Experiment | ASCII-only lifecycle and plain next-action line | Active proposal heading, backend-state stepper, prominent next action, current-stage review link, execution boundary, precise live confirmation, expandable context/provenance, original/retest evidence links |
| Findings | Compact table without search/filter controls | Search plus status, severity, retest, and family filters; target, control, evidence links, and update timestamp remain visible in a locally scrolling table |
| Finding detail | ID-led summary and dense technical preformatted block | Security title, what happened, impact, proof, action, explicit mitigation-not-verified state, and experiment/retest CTA; technical lineage follows the security interpretation |
| Runs | Runs labeled as Evidence, shared concept | Distinct execution view, search, selected type/result filters, and backend result badges; detail prioritizes result, related finding and metrics ahead of provenance/trace; raw exports move lower |
| Evidence | Another name for the run listing | Artifact-oriented view using existing run records, with artifact downloads and available Finding / Experiment links, bounded concurrent reads, and retained summaries if some records fail |

Threat objects retain their current backend structure rather than inventing per-threat fields. Comparisons retain the backend's metric values and deltas; no unrelated metrics are combined and no fabricated charts were added.

## Minimal backend compatibility changes

`app.py` now supplies the overview with a read-only presenter in `orion/workspace_view.py`. That presenter reads existing profile, experiment, evidence, and finding stores and uses `ExperimentLifecycle.next_action`. It does not save artifacts, approve plans, execute attacks, or change lifecycle state. When a workspace lacks a linked profile, it does not substitute an unrelated recent profile. Without a workspace, recent profiles are explicitly presented as inputs to select in Analysis.

The run-detail template context now also associates a retest trace with a finding through its existing `retest_refs`. This is a read-only join; the finding store, record format, and APIs are unchanged.

## Verification

- Full existing suite plus six presentation regressions: 259 passed; one existing PyTorch test fails during import. The test process also exits with native code 139 after reporting results.
- `python -c 'import torch'` independently reproduces the installed environment's circular-import error involving `torch.utils.data.datapipes.dataframe`. No dependency or security-runner changes were made to mask this.
- Excluding only `test_build_attack_maps_ids_to_real_art_attacks`: 259 passed, one deselected.
- Syntax checks passed for all seven changed JavaScript files; Python compilation and `git diff --check` passed.
- Main application pages returned HTTP 200 and produced no browser JavaScript errors.
- Chromium checks covered 1440, 1280, 1024, 768, and 390 pixel widths without page-level horizontal overflow on the inspected overview/finding surfaces. Populated experiment/provenance, run-detail, and evidence surfaces were also inspected.
- Browser interaction checks passed for skip navigation, keyboard menu choices, appearance persistence, reduced motion, dialog Cancel/Escape/Confirm, evidence search/filter state, and all five lifecycle stages.
- A populated finding was rendered from an in-memory preview fixture to verify the applied-control/no-retest presentation without writing a fake finding to the evidence store.
- No live attack or external target probe was performed for verification.

The existing tests continue to cover approval versus execution, attack gates, defense versus mitigation, retest conclusions, evidence, provenance, CLI, and scenarios. New regressions check read-only overview behavior, exact linked-profile association, preservation of valid partial inputs, distinct security labels, separate Runs/Evidence views, and reverse linkage from a retest run to its finding.

## Preserved semantics

Approval still does not execute an attack. Backend approval gates remain authoritative. Applying a control still does not classify a finding as mitigated. Retest remains the source of the backend's mitigation conclusion. Hypotheses are not promoted by presentation. Evidence and findings remain separate objects with their original identifiers and provenance. Live execution requires explicit user action. No lifecycle transitions, attack families, metrics, policy decisions, or evidence formats changed.

## Remaining UI/UX work

- Overview follows the most recently updated experiment; it does not add an assessment-switching preference system or an aggregate priority engine.
- The existing threat-model schema does not provide every suggested per-threat field. This iteration preserves source-attributed objects; a full normalized per-threat editor is not implemented.
- Findings have no sortable columns or pagination. Evidence supports search and type/result filters, but not dedicated date, experiment, or finding filter controls. Large histories still require record reads for artifact inspection.
- Some existing stores silently skip malformed records, so the UI cannot enumerate every skipped artifact without a separate backend reporting change.
- Existing recon console and other legacy specialty pages inherit the shell, but their internal controls have not all been converted to the new confirmation component. Existing recon approvals are retained.
- Provenance nodes without an existing object-detail route remain identifiers rather than invented destinations. Missing lineage is explicitly unknown.
- Initial workspace measurement and profile deep-link behavior are preserved. This iteration does not redesign those backend or interaction semantics.
- No live execution, GPU/image attack, screen-reader audit, complete contrast audit, or cross-browser certification was performed.

## Out-of-scope backend/security follow-ups

1. **Repair the installed PyTorch environment.** The import failure prevents the ART attack-construction test from running; the full test process also reports a native teardown failure. This is reproducible independently of the UI.
2. **Audit real image/model replay fidelity.** `lifecycle.retest` routes non-agentic attacks through `experiments.runner.replay`, whose `run_scenario` uses the synthetic teaching backend. Live agentic retest has its own endpoint runner. The UI now labels this distinction explicitly; synthetic replay does not verify mitigation on live model weights or a live black-box endpoint. The runners were not changed.
3. **Expose execution constraints if desired.** Isolation, filesystem restrictions, network restrictions, and complete budgets are not uniformly reported by the existing APIs. The UI uses UNKNOWN instead of asserting sandbox guarantees. Implementing isolation or resource control remains outside this iteration.

## Modified and added files

- `app.py`
- `docs/ui-workspace-delivery.md`
- `orion/workspace_view.py`
- `static/css/base.css`
- `static/css/orion-terminal.css`
- `static/css/target-analysis.css`
- `static/js/context.js`
- `static/js/environment.js`
- `static/js/know-yourself.js`
- `static/js/orion.js`
- `static/js/runs.js`
- `static/js/target-analysis.js`
- `static/js/workspace.js`
- `templates/base.html`
- `templates/components/methodology.html`
- `templates/components/metric.html`
- `templates/components/provenance_chain.html`
- `templates/components/status.html`
- `templates/components/workspace.html`
- `templates/context.html`
- `templates/environment.html`
- `templates/experiment.html`
- `templates/finding-detail.html`
- `templates/findings.html`
- `templates/index.html`
- `templates/know-yourself.html`
- `templates/run-detail.html`
- `templates/runs.html`
- `templates/target-analysis.html`
- `tests/security_cases/test_workspace_ui.py`
