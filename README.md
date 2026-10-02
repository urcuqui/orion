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
| Understand | Know the target & environment (recon) | `orion.recon` / `libs.recon` |
| Threat Model | Define assets, adversary, surfaces | `orion.threat_model` |
| Attack | Run a controlled, justified attack | `orion.experiments` |
| Measure | Quantify degradation with metrics | `orion.metrics` |
| Harden | Apply a candidate defense | `orion.defenses` |
| Retest | Replay the attack, compare, report | `orion.experiments` + `orion.evidence` |

Full write-up: [docs/methodology.md](docs/methodology.md).

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

## 7. AI agents

Agents are **copilots, not oracles**: they plan and explain; metrics and
experiments decide the result. Boundaries are enforced in code. See
[docs/agent-role.md](docs/agent-role.md).

## 8. Know the Environment / Recon

The reconnaissance workflow is the *Understand* phase. It takes an objective,
target, iteration limit and approval settings, and can use Playwright, Nuclei
and MCP tools — always requiring human approval for sensitive actions. Its UI
extends the shared Orion terminal design system (`base.html`).

## 8b. Know Your Target — evidence-grounded Target Analysis

> **Recon collects. Orion interprets. Humans decide.**

The **Know Your Target** workflow (`/target-analysis`) turns recon evidence into
an interpretation, following the core principle:

> **Evidence → applicability → hypothesis → test → finding** (never
> *generic knowledge → finding*).

Preferred flow: **Recon → Analyze with Orion → Threat Model → Experiment Plan →
Human review** (also available standalone, or as a one-click *Guided Analysis*).

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

> **Mock recon never reaches the target.** It returns deterministic fixtures, so
> analyzing a real app in mock mode yields `NOT_OBSERVED`. To analyze a *running*
> service, either run recon with real tools (uncheck `--mock`) or use **Analyze a
> running URL directly** in Know Your Target (`POST /api/target-analysis/probe`),
> a bounded, GET-only probe that collects real evidence (e.g. an exposed
> `/mcp_tools` or `/v1/chat/completions` endpoint) and feeds the same analyzer.

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

- **Dashboard** — command center: system status, dragon, `[01]`–`[05]` menu,
  recent activity (real run data), `orion@security:~$`.
- **Know Yourself** — adversarial ML (model robustness).
- **Know Your Target** — Recon + Agent Analysis + Threat Model + Guided Analysis.
- **Know The Environment** — reconnaissance (now in the shared design system).
- **Attack** — red-team hub.
- **Evidence** — runs, reports, comparisons.

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
