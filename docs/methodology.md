# The Orion Methodology

Orion turns ad-hoc AI security experiments into a repeatable, **evidence-driven**
process that is the same across Traditional ML, Generative AI and Agentic AI:

```
UNDERSTAND     Know Yourself · Know Your Target · Know the Environment
                                   ↓
               Analysis Context → Threat Model → Experiment Plan
                                   ↓
                         ⟨ Human Approval ⟩
                                   ↓
OPERATIONS     Attack → Measure → Defend → Retest
                                   ↓
ANALYSIS       Finding   (the security interpretation of the evidence)
                                   ↓
OBSERVABILITY  Evidence  (the full provenance chain)
```

Each phase is first-class in code (`orion.methodology.Phase`) and every scenario
and experiment declares which phase it belongs to.

> **An attack algorithm without a threat model is only an experiment.**
> **Approve Plan ≠ Run Attack. Control applied ≠ system secured.**

---

## 1. Understand — *three context pillars*

**Understand** is built from three distinct pillars, each producing a reusable
profile — not from reconnaissance alone:

- **Know Yourself** → Self Profile (`orion.know_yourself`): the AI system itself
  (Traditional ML, model-centric; or Generative AI, system-centric).
- **Know Your Target** → Target Profile (`orion.context.target`): *what* is being
  assessed and *why*.
- **Know the Environment** → Environment Profile (`orion.context.environment`):
  the terrain around the target. Reconnaissance (`libs/recon`, `orion.recon`) is a
  **data source that populates this one pillar**, subject to human approval for
  sensitive actions (`orion.policies`) — it is not the whole Understand phase.

The three converge into an **Analysis Context** (`orion.context`) which drives the
threat model and the experiment plan.

## 2. Threat Model — *Define the adversary*

Decide **who** the adversary is and **what** is at stake, independently of any
LLM. See `orion.threat_model`.

- **Assets** — model integrity, prediction reliability, training data, model
  weights, inference API, confidentiality, availability.
- **Adversary** — knowledge (`none|limited|full`), access
  (`black_box|gray_box|white_box`), goal, budget, constraints.
- **Attack surfaces** — training pipeline, dataset, feature pipeline, inference
  API, model artifact, feedback loop.

```json
{
  "target": { "task": "image_classification", "access": "white_box" },
  "adversary": { "goal": "targeted_misclassification", "knowledge": "full", "budget": "low" }
}
```

## 2b. Experiment Plan — *select, then approve*

Applicable attack hypotheses are drawn from the **unified catalog**
(`orion.catalog.attacks`) and assembled into an `ExperimentPlan` (`orion.plans`).
A human **approves** the plan; approval opens the Experiment Workspace with the
experiment loaded and ready — it does **not** execute anything.

## 3. Attack — *Execute a controlled attack*

Run an attack that is **justified by the threat model**. The same stage covers
every family, each from the one catalog:

- **Traditional ML** — real white-box (torch + ART) FGSM / PGD / C&W on the
  weights, or decision-based black-box evasion against a live endpoint
  (`orion.adversarial`); access level (white/black-box) is derived and confirmed.
- **Generative AI / Agentic AI** — experiments run through a tool-enabled agent
  whose tool selection comes from a model (`orion.agentic.agent`). **Direct** and
  **Indirect** Prompt Injection are distinct execution paths with different trust
  boundaries (user channel vs untrusted retrieved content), and tool/privilege
  abuse is observed through the agent's actual tool calls and authorization
  decisions. Discovered AI endpoints also become **live targets** via
  `orion.agentic.run_live_prompt_injection` (OWASP LLM payload catalog). Attack
  success is decided by the catalog's **executable success criteria**; results are
  derived from the recorded trace, never asserted.

> **Influence is not impact.** An agent can be compromised (it follows an injected
> instruction) without a security boundary being crossed. The finding is the
> observable boundary crossing, not the influence.

Execution is always explicit (a human presses RUN); analysis never silently
becomes attack execution.

## 4. Measure — *Quantify impact (family-aware)*

Attacks produce **metrics, not screenshots**. The Metric Catalog
(`orion.catalog.metrics`) is **not** image-adversarial only; each metric declares
its family and whether higher or lower is *better*:

- Traditional ML — clean/robust accuracy, attack success rate, L∞ / L2, query count.
- Generative AI — refusal, policy-violation, secret-leakage, instruction-following rates.
- Agentic AI — unauthorized-tool-call, privilege-boundary-violation, approval-bypass rates.

## 5. Defend — *Apply a candidate control*

Apply a control from the Control Catalog (`orion.catalog.controls` /
`orion.defenses`). Orion separates the **ControlDefinition** (the concept) from the
**ControlImplementation** (how a specific environment enforces it, with a declared
`coverage` of `PARTIAL` / `SUBSTANTIAL` / `FULL`): a gateway injection filter is a
`PARTIAL` implementation of instruction provenance, not full coverage. Selecting a
control and applying its implementation does **not** mark a finding mitigated.

> **A control is not validated until the attack is replayed.**

## 6. Retest — *Replay and compare*

Replay the **same** attack under the new posture and compare before/after
(`orion.experiments.replay` / `compare` for ML; re-run with the control for
agentic). The control is then classified from the measured result:
`INEFFECTIVE` / `PARTIALLY_EFFECTIVE` / `EFFECTIVE`.

```
Attack Success Rate: 100% → 0%    (agentic, tool_authorization → EFFECTIVE)
Robust Accuracy:     51.7% → 78.4%  (image, adversarial_training)
```

## 7. Finding — *interpret the evidence*

A **Finding** (`orion.findings`) is the security interpretation of the evidence —
a first-class object between experimentation and reporting:

> **Evidence = what happened. Finding = the security meaning of what happened.**

A single successful run is **OBSERVED**. Promotion to **CONFIRMED** requires a
**corroboration policy** (`CorroborationPolicy`: default 3 independent trials,
≥0.66 success rate) reproduced against the same target/criterion — not merely two
evidence refs. The structured classification (severity, boundary crossed, observed
action) is deterministic from the attack definition + evidence + success-criteria
evaluation, not LLM-generated. A retest sets `retest_status` honestly
(`INEFFECTIVE` / `PARTIALLY_EFFECTIVE` / `EFFECTIVE`) and moves the finding to
`MITIGATED` / `PARTIALLY_MITIGATED` or leaves it unmitigated. Every finding points
back to its evidence and shows the full provenance chain.

---

## Evidence and status

Every run writes an evidence directory `artifacts/<trace_id>/` containing
`experiment.json`, `report.md` and any images (`orion.evidence`). Each
experiment ends with an explicit status:

| Status | Meaning |
| --- | --- |
| `NO_ATTACK` | No attack executed (baseline). |
| `ATTACK_SUCCESS` | The attack met its goal. |
| `ATTACK_BLOCKED` | A defense fully neutralized the attack. |
| `PARTIALLY_MITIGATED` | A defense reduced but did not eliminate the impact. |

## Running the methodology

```bash
python -m orion list
python -m orion run scenarios/pgd_evasion.yaml --mode baseline
python -m orion run scenarios/pgd_evasion.yaml --mode attack
python -m orion replay <attack_trace_id> --mode hardened
python -m orion compare <attack_trace_id> <hardened_trace_id>
```

See also: [attack-replay.md](attack-replay.md), [agent-role.md](agent-role.md),
[limitations.md](limitations.md).

## Assessment container

An Experiment validates one hypothesis; an Assessment coordinates multiple
validation experiments against an explicit system/target/scope. It retains
references to exact context, threat-model generations, plans, workspaces, Runs,
Findings and applied control/Retest records. It adds a root to provenance, not a
new methodology phase or a security decision engine.

Assessment Completed ≠ Secure. Existing Finding corroboration and Retest
effectiveness remain authoritative. [Assessment foundation](assessments.md).
