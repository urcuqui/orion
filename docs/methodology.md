# The Orion Methodology

Orion turns ad-hoc AI security experiments into a repeatable process:

> **Understand → Threat Model → Attack → Measure → Harden → Retest**

Each phase is first-class in code (`orion.methodology.Phase`) and every scenario
and experiment declares which phase it belongs to.

> **An attack algorithm without a threat model is only an experiment.**

---

## 1. Understand — *Know the Environment*

Know the target and the environment before touching it: the ML task, the data,
the model artifact, the inference surface, and the surrounding system.

- In Orion this is the **reconnaissance** workflow ("Know the Environment"),
  implemented in `libs/recon` and re-exported as `orion.recon`.
- Inputs: objective, target, max iterations, approval settings.
- Tools: Playwright, Nuclei, MCP tools — all subject to human approval for
  sensitive actions (`orion.policies`).
- Output: structured evidence about the environment.

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

## 3. Attack — *Execute a controlled attack*

Run an attack that is **justified by the threat model** (e.g. a white-box PGD
evasion against an image classifier). Attacks are declared in scenario files and
run through `orion.experiments.run_scenario(..., mode="attack")`.

## 4. Measure — *Quantify degradation*

Attacks produce **metrics, not screenshots** (`orion.metrics`):

- clean accuracy, robust accuracy, attack success rate
- perturbation magnitude (L∞ / L2), confidence shift
- class-level degradation, attack cost / query count

Single-example experiments report per-example metrics; batch experiments report
aggregates.

## 5. Harden — *Apply a candidate defense*

Apply a pluggable defense (`orion.defenses`): adversarial training, input
preprocessing, confidence/anomaly checks, rate limiting, input validation,
monitoring hooks. Orion never asserts a model is secure.

> **A defense is not validated until the attack is replayed.**

## 6. Retest — *Replay and compare*

Replay the **same** attack against the hardened system and compare before/after
(`orion.experiments.replay` / `compare`). Report the trade-offs and the
limitations; never claim universal robustness.

```
Clean Accuracy:      94.2% → 91.1%
Robust Accuracy:     51.7% → 78.4%
Attack Success Rate: 43.8% → 19.6%
```

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
