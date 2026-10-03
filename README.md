# Orion

> Orion is an AI security experimentation framework that combines adversarial
> machine learning, threat modeling, MITRE ATLAS, agent-assisted workflows,
> security tooling, and reproducible evidence.

Orion does not exist to prove that AI can fail. It exists to **understand how it
fails, measure the failure, and improve the system before an adversary does.**

![Orion Demo](demo.gif)

---

## 1. What is Orion?

Orion is a hands-on lab for breaking and defending AI models. It brings a single
methodology to adversarial ML experiments so results are coherent,
reproducible, measurable and evidence-driven — usable by red teams, blue teams
and researchers.

Orion is used in the talk *"Building Orion: Un framework para romper y defender
modelos de IA."*

## 2. Why Orion exists

Most AI-security demos stop at a screenshot of a misclassified image. Orion
insists on the harder, more useful questions: *Against which adversary? By how
much did the model degrade? Does a defense actually help when the same attack is
replayed?*

> **An attack algorithm without a threat model is only an experiment.**

## 3. Methodology

> **Understand → Threat Model → Attack → Measure → Harden → Retest**

| Phase | What happens | Package |
| --- | --- | --- |
| Understand | Build **Self, Target and Environment** context → Analysis Context | `orion.know_yourself`, `orion.target_analysis`, `orion.context` / `libs.recon` |
| Threat Model | Define assets, adversary, surfaces (from all 3 profiles) | `orion.threat_model` |
| Attack | Run a controlled, justified attack | `orion.experiments` |
| Measure | Quantify degradation with metrics | `orion.metrics` |
| Harden | Apply a candidate defense | `orion.defenses` |
| Retest | Replay the attack, compare, report | `orion.experiments` + `orion.evidence` |

Full write-up: [docs/methodology.md](docs/methodology.md).

## The Three Context Pillars

Understanding is built from three **distinct** pillars, each producing a reusable
profile. They are independent and compose progressively (any one may be absent).

### Know Yourself
Understand the AI system itself → **Self Profile** (`orion.know_yourself`).
Traditional ML (model-centric) or Generative AI (system-centric).

### Know Your Target
Define *what* is being assessed and *why* → **Target Profile**
(`orion.context.target`). It **consumes** Self and Environment profiles; it does
not own or duplicate reconnaissance.

### Know the Environment
Map the terrain *around* the target → **Environment Profile**
(`orion.context.environment`): assets, technologies, external & **AI
dependencies**, MCP/tool ecosystem, identity context, trust relationships and a
lightweight topology. **Recon is a data source that populates this profile** — it
is not the environment model.

These profiles converge into an **Analysis Context** (`orion.context`) at
`/context` — pick any combination of Self / Target / Environment (any may be
absent), **[ RUN AGENT ANALYSIS ]** correlates them into a **source-aware** threat
model and applicability (each conclusion tagged `SELF` / `TARGET` /
`ENVIRONMENT`), then **[ APPROVE PLAN ]** hands off to Attack with the full
context (`analysis_context_id`, `self_profile_id`, `target_profile_id`,
`environment_profile_id`) preserved end-to-end. It drives threat modeling and
experiment planning:

```
KNOW YOURSELF ─┐
KNOW YOUR TARGET ─┼─▶ ANALYSIS CONTEXT ─▶ THREAT MODEL ─▶ EXPERIMENT PLAN
KNOW THE ENVIRONMENT ─┘        ↓ HUMAN APPROVAL ↓
                        ATTACK ─▶ MEASURE ─▶ DEFEND ─▶ RETEST ─▶ EVIDENCE
```

> Know Yourself describes the system. Know Your Target defines the objective.
> Know the Environment maps the terrain. Orion correlates the evidence. Humans decide.

## 4. Architecture

```
app.py                     # lightweight Flask web/application layer (routes only)
orion/                     # the methodology framework (logic lives here)
  methodology.py           #   the six phases as code
  threat_model/            #   assets, adversary, boundaries, scenario (no LLM)
  scenarios/               #   YAML scenario loader + validation
  metrics/                 #   reusable metric interfaces
  experiments/             #   scenario -> attack -> measure -> evidence (+ synthetic backend)
  evidence/                #   trace ids, experiment.json, report.md, statuses
  defenses/                #   pluggable hardening mechanisms
  mappings/                #   curated MITRE ATLAS index
  policies/                #   human-oversight / approval (fail closed)
  adversarial/             #   structured adversarial-image experiment (wraps libs.adversarial)
  agents/                  #   agent copilots + boundary guards (wraps libs.agent)
  recon/                   #   'Know the Environment' (wraps libs.recon)
  integrations/            #   Flask blueprint exposing the methodology
libs/, tools/              # existing implementations (preserved)
scenarios/                 # scenario files (e.g. pgd_evasion.yaml)
docs/                      # methodology, agent-role, attack-replay, limitations
tests/security_cases/      # security regression tests
artifacts/                 # generated evidence (git-ignored)
```

Business logic stays out of Flask routes; routes call into `orion.*`.

## 5. Threat model

Threat models are structured, LLM-independent objects
(`orion.threat_model.ThreatModel`) describing the target, the adversary
(knowledge / access / goal / budget / constraints), the assets at stake, and the
attack surfaces. See [docs/methodology.md](docs/methodology.md#2-threat-model--define-the-adversary).

## 6. Adversarial ML

The classic Carlini & Wagner / PGD image flow is preserved and now returns
**both** a visual artifact and measurable evidence
(`orion.adversarial.generate_adversarial_evidence`):

```json
{
  "attack": "PGD",
  "baseline_prediction": "stop_sign",   "baseline_confidence": 0.997,
  "adversarial_prediction": "speed_limit_30", "adversarial_confidence": 0.913,
  "perturbation_linf": 0.028, "attack_success": true
}
```

A deterministic, dependency-free synthetic backend runs the same methodology
without a GPU (used by the CLI and tests).

## 6b. Know Yourself — AI security profiling

> What exactly do I have, how does it behave, what assumptions does it make,
> how fragile is it, and which experiments are actually applicable?

**Know Yourself** (`/know-yourself`) characterizes an AI system *before* any
attack (profiling only — execution belongs to the Attack phase). It auto-detects
the branch and never mixes their security assumptions:

- **Traditional ML** (model-centric): model fingerprint (framework, task,
  input/output, access, gradients, artifact hash/size), input/feature surface,
  model assumptions (`OBSERVED` / `INFERRED` / `UNKNOWN`), control inventory, and
  a robustness-oriented security posture.
- **Generative AI** (system-centric): system fingerprint (provider, RAG, memory,
  tools, MCP, external actions, human approval), prompt/context surface
  (`TRUSTED` / `UNTRUSTED` / `MIXED` / `UNKNOWN`), **trust-boundary inventory**,
  tool/MCP and identity profiles, and a control inventory.

The **security posture** lists each attack as `APPLICABLE` / `CONDITIONAL` /
`NOT_APPLICABLE` with rationale, evidence and prerequisites, using deterministic
rules (white-box gradients → PGD/FGSM/C&W/DeepFool applicable; black-box →
conditional; prompt injection requires an LLM surface; RAG poisoning requires
retrieval; tool/MCP poisoning requires a tool/MCP surface). It then proposes
**recommended next experiments** — proposals only, sent to the Attack phase on
human review, never auto-executed. `UNKNOWN` is a valid answer and never becomes
`NOT_FOUND`. Results export to `artifacts/<trace_id>/know_yourself.json`.
`orion/know_yourself/` keeps the two branches in separate modules.

## 6c. The Experiment lifecycle

> **Analysis proposes. Humans approve. Humans execute.
> Approve Plan ≠ Run Attack. Defense applied ≠ defense effective.
> Retest determines whether posture improved.**

Context pillars end with **[ APPROVE PLAN ]**, which builds a shared
`ExperimentPlan` (`orion/plans/`), records human approval and produces a handoff —
*without executing anything*. **[ OPEN EXPERIMENT ]** then opens the single
**Experiment workspace** (`/experiment/<id>`), a stateful console that runs the
whole lifecycle as **one continuous workflow** (not five separate apps):

```
PLAN → ATTACK → MEASURE → DEFEND → RETEST
```

`ExperimentLifecycle` (`orion/experiments/lifecycle.py`, persisted to
`artifacts/<ORN-EXP-id>/workspace.json`) tracks the stage states, the active
experiment (queue), and `attack_run_id / measurement_id / defense_id /
retest_run_id`. Transitions are **deterministic and fail-closed** (no attack
before an approved plan; no measure before attack; no retest before an applied
defense), and a deterministic next-action engine drives the UI — never the LLM.
The ATTACK stage starts from the **attacker's access level**, which is the
operational form of the threat model's `adversary.knowledge` and is **derived**
from the plan's origin — Know Yourself (you own the model) → **white-box** weights;
an external live service → **black-box** query-only — then shown for the analyst to
**confirm or override** (never a silent decision). The chosen level drives an
**attack catalog** (`orion/adversarial/catalog.py`), each entry shipping **base
tuning** so a run is one click away while every parameter stays editable:

* **White-box** → real gradient attacks (torch + ART) on the weights:
  **C&W (L2)**, **PGD (L∞)**, **FGSM** — `orion.adversarial.run_adversarial_experiment`.
  The class count is **inferred from the checkpoint** (`infer_num_outputs`), not
  demanded from the user.
* **Black-box** → **decision-based query evasion**
  (`orion.adversarial.run_blackbox_evasion`): L∞-bounded perturbed inputs to the
  live endpoint, measuring the decision change under a bounded query budget.

A **demo runner** (synthetic robustness via `scenarios/pgd_evasion.yaml`,
reproducible without a GPU) remains for the no-weights/no-GPU path. Attack
execution stays **explicit** (`[ RUN ATTACK ]`, with a confirm step before any
real queries to a live endpoint). Image attacks then render the
[Image Attack Comparison](#image-attack-comparison) immediately. Retest replays the **same attack**
under the new posture and the BEFORE/AFTER comparison is a *result* of retest.
Provenance links the whole chain
(`self → target → environment → context → threat_model → plan → experiment →
run → defense → retest`). Legacy `/attack`, `/measure`, `/defend` reuse this one
workspace (no parallel UX); **Evidence** stays globally accessible.

## Image Attack Comparison

When an attack operates on images, Orion shows the attack the way an analyst
reasons about it — **side by side**:

```
ORIGINAL            ADVERSARIAL          PERTURBATION / DIFFERENCE MAP
prediction: cat     prediction: dog      L∞ = 0.0312  ·  L2 = 1.456
confidence: 0.98    confidence: 0.71     absolute difference  [ amplify ]
```

The third panel is the **perturbation / difference map** (`|adversarial −
original|`), never a "mask" unless a real segmentation mask exists. The
comparison is a **reusable, type-aware artifact**, not a one-off view:

* **One source of truth.** A single Jinja component
  (`templates/components/image_attack_comparison.html`) renders it server-side on
  **Evidence** (`/runs/<id>`), and the matching client renderer
  (`Orion.imageComparison`) renders the *identical* markup inside the Experiment
  console at the **Attack**, **Measure** and **Retest** stages — so it appears
  **immediately after an image attack runs**, not only in Evidence.
* **Type-aware.** Tabular, text, agentic and network experiments carry no image
  artifacts, so the component renders **nothing** for them — no empty panels.
* **Honest semantics.** The prediction change is explicit (**CLASSIFICATION
  CHANGED** vs **ATTACK FAILED / UNCHANGED**), and attack success comes from the
  experiment's own metric (`attack_success`), **not** from how different the
  images look.
* **Display vs measurement.** Images are denormalized/clipped **for display
  only**; the reported `perturbation_linf` / `perturbation_l2` are the **raw**
  metric values, shown verbatim. The **[ amplify ]** control scales the
  difference image ×8 **for visibility only** (labelled display-only) and never
  touches the numbers. Click any panel to inspect the raw artifact.
* **Retest before/after.** Retest shows the original adversarial example
  *before* defense and the hardened replay *after*; if the hardened replay
  produced no new image, it says **NO NEW IMAGE ARTIFACT** rather than
  fabricating one.
* **Conference mode.** Append `?demo=1` to make the comparison prominent.

## 7. AI agents

Agents are **copilots, not oracles**: they plan and explain; metrics and
experiments decide the result. Boundaries are enforced in code. See
[docs/agent-role.md](docs/agent-role.md).

## 8. Know the Environment (`/environment`)

The third context pillar. Reconnaissance (Playwright, Nuclei, MCP tools, SSE,
mock mode, approvals, screenshots, reports) is **one capability** here — a data
source that populates the **Environment Profile** (assets, technologies, external
& AI dependencies, MCP/tools, identity, trust relationships, topology). Build a
profile from a completed recon run or a direct URL probe, then
**[ SEND TO TARGET ANALYSIS ]** passes the `environment_profile_id` onward — no
data is copied by hand.

## 8b. Know Your Target — evidence-grounded Target Analysis

> **Recon collects. Orion interprets. Humans decide.**

The **Know Your Target** workflow (`/target-analysis`) defines the Target Profile
and interprets evidence (from a recon run, a direct probe, or a loaded
**Environment Profile**), following the core principle:

> **Evidence → applicability → hypothesis → test → finding** (never
> *generic knowledge → finding*).

It **consumes** the Environment Profile (`?environment_profile_id=…`); it does not
own reconnaissance. Preferred flow: **Environment Profile → Analyze → Threat
Model → Experiment Plan → Human review**.

Every conclusion is classified `OBSERVED` / `INFERRED` / `HYPOTHESIS` /
`NOT_APPLICABLE`, carries evidence references + confidence + rationale, and is
checked by a deterministic validation layer (`orion.target_analysis`). A
hypothesis is never promoted to a finding; a suggested experiment is not evidence
of a vulnerability.

### AI Surface Detection

AI-specific threats are only surfaced when recon actually observed an AI/ML
surface. Detection is deterministic and strength-scored:

- **STRONG signals** (confirm on their own): classic ML model-serving routes
  (`/v1/models`, `/v2/models`, `/invocations`) and stacks (TensorFlow Serving,
  TorchServe, Triton, KServe, BentoML, MLflow, scikit-learn/XGBoost); LLM routes
  (`/v1/chat/completions`, `/v1/embeddings`); SDKs (OpenAI/Anthropic/LangChain);
  MCP server metadata; model artifacts (`.pt`, `.pth`, `.onnx`, `.safetensors`,
  `.joblib`); and prediction-shaped JSON responses (`predictions` /
  `probabilities` / `logits` / `softmax` / `signature_name`).
- **MEDIUM signals** (need corroboration): `/predict`, `/inference`, `rag`,
  `embedding`, `vector database`, `machine learning`, chatbot/assistant UIs.
- **WEAK signals** (never confirm alone): generic `/chat`, `/model`, `/tool`,
  `/assistant`, bare "ai"/"chat" text.

Classification: `CONFIRMED` = ≥1 strong **or** ≥2 medium; `POSSIBLE` = 1 medium
or ≥2 weak; `NOT_OBSERVED` otherwise. A lone weak route (`/chat`, `/model`, …)
can **never** confirm AI. For a traditional web app, Orion reports
`AI SURFACE: NOT_OBSERVED`, maps **no** ATLAS techniques, and **abstains** —
"insufficient evidence to determine whether this target exposes an AI/ML attack
surface" — which is a useful result, not a failure. Generic web findings
(WordPress, login forms, APIs) are **never** converted into prompt injection,
model extraction, adversarial ML or RAG poisoning without AI-specific evidence.

> **Recon is real by default** for live, reproducible results. **Know The
> Environment → Live Recon** connects to the target now (GET + an optional
> harmless active test request) and normalizes the response into the Environment
> Profile (`POST /api/environment/from-url {"url":…, "active":true}`) — no mock,
> no LLM dependency. `--mock` is an opt-in safe simulation (deterministic
> fixtures that never reach the target, so it yields `NOT_OBSERVED` on a real
> app). The active probe only applies to systems you are authorized to test.

### Honest threat model

Orion does not invent adversary properties. Only what recon can establish is
marked `[OBSERVED]` (e.g. network reachability → `REMOTE_PUBLIC`); goal,
knowledge and budget default to `UNDEFINED` / `UNKNOWN` and require analyst
input.

## 9. MITRE ATLAS

Orion uses **MITRE ATLAS** for AI-specific threats (and only references MITRE
ATT&CK where a real ATT&CK mapping exists). Scenarios declare ATLAS mappings;
unknown technique IDs are flagged as unverified rather than invented
(`orion.mappings`).

## 10. Evidence

Every run writes `artifacts/<trace_id>/` with `experiment.json`, `report.md`
and images. Statuses: `NO_ATTACK`, `ATTACK_SUCCESS`, `ATTACK_BLOCKED`,
`PARTIALLY_MITIGATED`.

## 11. Attack replay

Replay the same attack before and after hardening and compare. See
[docs/attack-replay.md](docs/attack-replay.md).

## 12. Quick start

```bash
git clone https://github.com/urcuqui/orion.git
cd orion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # full stack (torch, langgraph, ...)
# For just the methodology + CLI + tests you only need pyyaml + pytest:
# pip install pyyaml pytest

# Run the methodology end-to-end (no GPU required):
python -m orion run scenarios/pgd_evasion.yaml --mode baseline
python -m orion run scenarios/pgd_evasion.yaml --mode attack
python -m orion replay <attack_trace_id> --mode hardened
python -m orion compare <attack_trace_id> <hardened_trace_id>

# Or launch the web UI:
python app.py   # http://localhost:5000
```

## 13. Scenarios

Scenarios are declarative YAML that bind the whole methodology together
(target, adversary, attack, metrics, defenses, ATLAS mappings). Example:
[scenarios/pgd_evasion.yaml](scenarios/pgd_evasion.yaml). Validate one with
`python -m orion validate scenarios/pgd_evasion.yaml`.

## 14. Web UI

A retro **adversarial-intelligence console**: black background, crimson borders,
phosphor-green status, amber warnings, monospace type, subtle CRT scanlines
(toggle in the top bar), an ASCII Orion dragon, and a command prompt. All
sections share one design system (`base.html` + `static/css/base.css` +
`orion-terminal.css`).

Navigation (Art of War structure):

Top-level navigation is three groups — **CONTEXT**, **EXPERIMENTATION**,
**OBSERVABILITY**:

- **Dashboard** — command center: system status, dragon, grouped `[01]`–`[05]`
  menu, recent activity (real run data), `orion@security:~$`.
- **CONTEXT**
  - **Know Yourself** — Traditional ML / Generative AI profiling → Self Profile.
  - **Know Your Target** — Target Profile + Agent Analysis + Threat Model
    (consumes the Environment Profile; does not own recon).
  - **Know The Environment** — reconnaissance + Environment Profile (assets, AI
    dependencies, MCP/tools, trust relationships, topology).
- **EXPERIMENTATION**
  - **Experiment** — the single workspace for Plan / Attack / Measure / Defend /
    Retest. Attack, Measure and Defend are **stages**, not top-level apps; Retest
    is its own stage.
- **OBSERVABILITY**
  - **Evidence** — runs, reports, comparisons, provenance (cross-cutting).

Conference mode: append `?demo=1` to enlarge type, hide secondary controls and
decorative CRT, and emphasize target type, AI surface, observations, suggested
experiments, threat model and results.

The Flask app is preserved: adversarial image generation, AI chat, streaming
(SSE), and the reconnaissance workflow with human approval. The methodology is
also exposed under `/orion/*` and a JSON API under `/api/*` (`/api/runs`,
`/api/agent/analyze-recon`, `/api/recon/runs`, …). Legacy URLs such as
`/red-pill.html` redirect to the current Red Team landing page.

## 15. CLI

```bash
python -m orion list
python -m orion run <scenario.yaml> [--mode baseline|attack|hardened]
python -m orion replay <trace_id> [--mode ...]
python -m orion report <trace_id>
python -m orion compare <baseline_trace> <hardened_trace>
python -m orion validate <scenario.yaml>
```

## 16. Limitations

Orion measures failure under specific conditions; it does not certify security.
Read [docs/limitations.md](docs/limitations.md) before drawing conclusions.

## 17. Responsible use

This project is for research and education. Use it only against systems you are
authorized to test. Sensitive actions require human approval and fail closed.
Report results honestly, always stating the threat model and limitations.

---

Christian Camilo Urcuqui López
