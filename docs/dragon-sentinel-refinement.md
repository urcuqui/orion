# Dragon Sentinel workstation refinement

Incremental refinement of `ui_improvement`; Flask, Jinja, vanilla JavaScript,
existing URLs, appearance storage, experiment transitions, evidence formats and
provenance remain in place. No new frontend dependency or API endpoint.

## What changed

| Surface | Before | After |
| --- | --- | --- |
| Overview | Shared cat artwork and passive Next Action styling | Shared expanded Dragon Sentinel and stronger Next Action styling |
| Analysis Context | Existing analysis workflow | Workflow unchanged; shared shell, appearance and state semantics refined |
| Experiment | Local and external attack buttons both red; access first in boundary | Local buttons amber, external buttons red; external effect, network and authorization first |
| Findings | Eleven columns; filtering inline | Seven primary columns; family beneath title, ID/confidence/control in Details; dedicated filter module with target filter |
| Finding Detail | Passive remediation panel; generic retest link | Prominent remediation, applied/verification state and linked workspace's deterministic Next Action |
| Evidence | Separate Runs and Evidence sidebar destinations | One Evidence destination, with Executions and Artifacts views; `/runs?view=evidence` retained |
| Run Detail | DENY styled green | Neutral policy decisions, separately labeled recorded security outcome; hierarchy and provenance retained |

The compact dragon lives in the shell; expanded art appears on Overview and the
very faint Classic watermark. Branding is outside the art. Artwork is decorative
and excluded from screen-reader text. Analyst is the default. Classic retains
CRT and glow; `?demo=1` selects Classic and retains the existing demo behavior.
Appearance is represented by `data-orion-appearance` on the body.

OBSERVED uses a blue informational badge and a dot. CONFIRMED uses a green
validated badge and a check. HYPOTHESIS, INFERRED, UNKNOWN and NOT_APPLICABLE retain
their separate symbols, labels and styles. ALLOW/DENY badges do not determine the
security outcome.

## Backend and security semantics

The only backend addition is `finding_workspace()` in `orion/workspace_view.py`:
a read-only lookup that verifies the linked workspace still belongs to the
finding, then returns the lifecycle's existing `next_action`. It does not create
or approve plans, execute attacks, apply controls, retest, or change findings.
Missing or stale workspace links produce an explicit unavailable state.

Applied-control presentation also reads the recorded workspace's DEFEND state,
so a newly applied control is visible before the retest updates the finding.
Applying a control never displays automatic mitigation. A retest record without
a finding classification displays UNKNOWN verification.

Live confirmation still uses the existing native dialog. Target, attack, access,
maximum requests, network and authorization are shown before live attacks.
Unknown query counts, isolation and filesystem restrictions remain UNKNOWN.
Escape/Cancel preserve focus and do not execute. No sandbox, authentication,
upload handling, execution backend or policy-enforcement guarantees were added.

## Validation and limitations

- Six new presentation regression tests cover dragon variants, shell/modes,
  policy versus outcome, table density, read-only Next Action and applied-control
  semantics. Existing observation-view label assertions reflect the new tabs.
- `pytest -q -k 'not test_build_attack_maps_ids_to_real_art_attacks'`: 267 passed,
  one deselected. Lifecycle, provenance and finding-policy tests pass.
- The full suite was run. The installed PyTorch package fails importing
  `torch.utils.data.datapipes.dataframe`; the same optional ART construction test
  failed before these changes. The full run also exits with signal 11 in that
  environment. This dependency failure is unresolved; the optional real ART
  construction path is not validated.
- Headless Chrome checks passed at 1440, 1280, 1024 and 768px: filters, appearance
  persistence, local table scrolling, collapsed navigation, Escape/focus return,
  stepper state, boundary order, local/live styling, confirmation and cancellation.
  API execution requests were intercepted; no live target or model was executed.
- JavaScript syntax checks and `git diff --check` passed.
- This is not a full WCAG conformance audit. Existing execution/sandboxing
  limitations remain; the boundary does not imply stronger isolation.

## Files

Modified:

- `app.py`
- `orion/workspace_view.py`
- `static/css/base.css`
- `static/css/orion-terminal.css`
- `static/css/target-analysis.css`
- `static/js/orion.js`
- `static/js/runs.js`
- `static/js/workspace.js`
- `templates/base.html`
- `templates/components/status.html`
- `templates/finding-detail.html`
- `templates/findings.html`
- `templates/index.html`
- `templates/run-detail.html`
- `templates/runs.html`
- `tests/security_cases/test_workspace_ui.py`

Added:

- `templates/components/dragon.html`
- `static/js/findings.js`
- `tests/security_cases/test_sentinel_refinements.py`
- `docs/dragon-sentinel-refinement.md`

Removed: `templates/components/cat.html` (replaced by the Dragon Sentinel macro).
