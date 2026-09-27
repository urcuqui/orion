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
and MCP tools — always requiring human approval for sensitive actions.

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

The Flask app is preserved: adversarial image generation, AI chat, streaming
(SSE), and the "Know the Environment" reconnaissance workflow with human
approval. The methodology is additionally exposed under `/orion/*`
(`/orion/methodology`, `/orion/scenarios`, `/orion/run`, `/orion/compare`).

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
