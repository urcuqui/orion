# Attack Replay

Replay is what turns a defense from a claim into evidence. Orion replays the
**same attack** against a system before and after hardening and compares the
measured degradation.

> **A defense is not validated until the attack is replayed.**

## What a replay holds constant

A replay reuses, as far as possible:

- the same **target** (task, access, model version)
- the same **attack technique** and **parameters**
- the same **malicious input / payload** (deterministic dataset seed)

The only thing that changes between the attack run and the hardened run is the
**defense configuration**.

## Modes

Every experiment runs in one of three modes:

| Mode | Meaning | Typical status |
| --- | --- | --- |
| `baseline` | model on clean inputs, no attack | `NO_ATTACK` |
| `attack` | run the attack, measure degradation | `ATTACK_SUCCESS` / `ATTACK_BLOCKED` |
| `hardened` | apply defenses, then replay the attack | `ATTACK_BLOCKED` / `PARTIALLY_MITIGATED` / `ATTACK_SUCCESS` |

## CLI

```bash
# 1. Understand the clean baseline
python -m orion run scenarios/pgd_evasion.yaml --mode baseline

# 2. Attack and measure
python -m orion run scenarios/pgd_evasion.yaml --mode attack
#   -> trace_id: 20260927T...-abcd1234

# 3. Harden and retest by replaying the SAME attack with defenses on
python -m orion replay 20260927T...-abcd1234 --mode hardened
#   -> trace_id: 20260927T...-ef567890

# 4. Compare before vs after
python -m orion compare 20260927T...-abcd1234 20260927T...-ef567890
```

You can also run the hardened configuration directly from the scenario
(`--mode hardened`); `replay` is preferred when you want to prove that a
*specific prior attack trace* is mitigated.

## How replay is reconstructed

`orion.experiments.replay(trace_id)` loads `artifacts/<trace_id>/experiment.json`
and rebuilds the scenario from it — same technique, parameters, target, threat
model, ATLAS mappings and declared defenses — then re-runs it. Because the
synthetic backend is deterministic (fixed dataset seed), an `attack`→`attack`
replay reproduces identical metrics, which the security regression tests assert
(`test_replay_uses_same_parameters`).

## Reading the comparison

```
Robust Accuracy:     0.75 → 1.0   (delta +0.25)
Attack Success Rate: 0.25 → 0.0   (delta -0.25)
Clean Accuracy:      1.0  → 1.0   (delta  0.0)   <- legitimate behaviour preserved
```

A good mitigation improves a robustness metric **without** destroying clean
accuracy. Always read both. And remember: one mitigated attack does not prove
general robustness — see [limitations.md](limitations.md).
