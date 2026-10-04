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
- **Generative AI / Agentic AI** — controlled, deterministic lab experiments for
  prompt injection and tool/privilege abuse (`orion.agentic`), with a full
  per-trial execution trace. Results are derived from the trace, never asserted.

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
`orion.defenses`): input preprocessing, confidence threshold, adversarial training
(ML); and tool authorization, least privilege, destination allowlist, instruction
provenance, context isolation, human approval (agentic). Orion never asserts a
system is secure.

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

A single successful run is **OBSERVED** (never auto-CONFIRMED); corroboration
promotes it to **CONFIRMED**. A retest sets `retest_status` honestly and moves the
finding to `MITIGATED` / `PARTIALLY_MITIGATED` or leaves it unmitigated. Every
finding points back to its evidence and shows the full provenance chain.

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
