# Orion UI polish delivery

Focused polish of `ui_improvement`; the workstation navigation, header, Analyst/Classic modes and lifecycle remain in place.

## Visual changes

Navigation and selections use raised neutral surfaces, stronger borders and phosphor indicators. Approval and applying controls use the primary phosphor treatment. Offensive execution, live recon, live retests, high severity and failed boundaries keep crimson. Routine navigation labels no longer carry terminal brackets; terminal and execution surfaces retain them.

Overview now orders current assessment, Next Action, posture/attack surface, profiles, experiments and activity. Finding detail emphasizes severity, evidence state, affected component and crossed boundary; confidence, retest and IDs are secondary. Corroboration uses definition-list rows, readable rates and explicit confirmation eligibility without promoting the finding state.

Run Detail has one prominent next-action navigation link derived from the persisted workspace's backend `next_action`. A workspace is used only if it still references the displayed attack/retest. Replay is secondary; explanation, new experiment and artifacts are tertiary. Missing/stale workspace context retains a neutral Defend fallback.

Runs show executions: run, available experiment, scenario, target, result, timestamp and available related finding. Evidence shows individual recorded artifacts: artifact/file, type, source, originating run, available experiment/finding, result and available provenance. Unsupported relationship columns are omitted. Record/finding joins use existing read-only APIs in batches of eight; partial failures retain available results. Artifact entries come from the record; no speculative reports are added.

## CSS removed or merged

- Consolidated duplicate `.orion-nav`, `.orion-main`, `.orion-topbar`, `.orion-brand` and footer shell definitions. Removed superseded horizontal navigation, centered 1280px content cap, old navigation separators and conflicting mobile layout rules.
- Consolidated duplicate component declarations for panels, headings, buttons, metrics, badges, provenance and finding grids; grouped responsive shell rules by breakpoint.
- Removed obsolete demo shell hiding/revert rules and the old demo width cap.
- Removed dragon watermark/demo rules, unused CRT toggle styling and unused flicker keyframes. Classic retains scanlines and restrained glow.
- Removed obsolete capability-menu `li.active` styling, red assessment titles, crimson technical IDs and crimson AI-signal indicators.
- Replaced repeated margin patterns with action, summary, filter and spacing classes. Retained unique form sizing and specialist console styles.
- Preserved visible keyboard focus and reduced motion. Fixed mobile report-table overflow and long run-context values.

## Identity

Two variants live in `templates/components/cat.html`: primary on Overview (restrained Analyst, larger/glowing Classic) and a very low-opacity Classic watermark; compact in the sidebar. Both are decorative and hidden from assistive technology. The global watermark is `display:none` in Analyst, including operational pages. Small screens suppress overview/sidebar ASCII. Footer now says AI SECURITY VALIDATION WORKSPACE with the three-part positioning line.

## Verification

- Existing full suite: **261 passed, 1 failed**. The failure is the pre-existing installed Torch/datapipes circular import in `test_build_attack_maps_ids_to_real_art_attacks`. The full runner also exits 139 during native teardown. No dependency repair was made during UI polish.
- Focused UI and lifecycle suite: **47 passed**; includes new regression coverage for backend next action, stale run/workspace relationships, and eligible corroboration remaining OBSERVED.
- Chrome/Playwright exercised Overview → System/Assess → Analysis → plan review/approval → experiment → controlled agentic attack → measurement → OBSERVED finding → Defend → effective Retest → MITIGATED finding → Runs/Evidence. Fixtures used a separate temporary artifact directory; no live endpoint was attacked.
- Populated run/finding/evidence/workspace pages checked at 1440, 1280, 1024, 768 and 390 pixels, without document overflow or browser JavaScript errors. Main assessment pages also checked at desktop/mobile widths. Analyst watermark hidden; Classic watermark shown; reduced motion disables animation.
- Visible 2px focus verified for navigation, inputs, textarea, select, buttons and table actions. Skip-to-content focuses main; dialogs initially focus Cancel and restore focus after Escape. Simulated record failures preserve execution summaries and show explicit partial Evidence state.
- JavaScript syntax and `git diff --check` passed.

## Scope and remaining inconsistencies

No attack, evidence model, security methodology, status promotion rules, approval boundary or lifecycle transition changed. The only server change reads existing workspace context for Run Detail presentation.

Some specialized terminal/recon panels retain older inline sizing/spacing and bracketed execution language. Nested auxiliary metrics still display the backend's raw dictionary representation. Long histories require record joins and have no new pagination; this pass preserves the existing API model. This was a Chrome desktop/mobile and keyboard spot-check, not a complete cross-browser accessibility certification.

## Modified files

- `app.py`
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
- `templates/components/status.html`
- `templates/components/workspace.html`
- `templates/context.html`
- `templates/finding-detail.html`
- `templates/index.html`
- `templates/know-yourself.html`
- `templates/run-detail.html`
- `templates/runs.html`
- `templates/target-analysis.html`
- `tests/security_cases/test_workspace_ui.py`
- `templates/components/cat.html`
- `docs/ui-polish-delivery.md`
