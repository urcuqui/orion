# The Role of AI Agents in Orion

Orion uses AI agents (the supervisor/subagent workflow in `libs/agent`, exposed
as `orion.agents`) as **copilots**, not as security oracles.

> **The agent can plan and explain. Metrics and experiments determine the result.**

## Agents may

- suggest experiments
- explain results
- correlate findings with MITRE ATLAS
- summarize evidence
- recommend candidate controls
- assist with report generation

## Agents must NOT be presented as

- deciding whether a model is secure
- replacing quantitative evaluation
- proving robustness

## Why

An LLM produces plausible text; it does not measure degradation. A statement
like "the model is now secure" is a **verdict**, and verdicts in Orion come from
the metrics/evidence layer (`orion.metrics`, `orion.evidence`) — from a replayed
attack and its numbers — never from an agent's narrative.

## Enforced in code

`orion.agents.boundaries.assert_not_security_verdict(text)` guards the boundary
where agent output would otherwise be treated as a conclusion. It raises
`AgentBoundaryError` if the text asserts security/robustness/safety, so agent
prose can never silently override a measured result.

```python
from orion.agents.boundaries import assert_not_security_verdict

summary = assert_not_security_verdict(agent_summary)  # raises on a verdict claim
```

## Practical guidance

- Let the agent draft the plan and the report; let the experiment decide the
  status (`ATTACK_SUCCESS`, `ATTACK_BLOCKED`, ...).
- Any control an agent recommends is a **candidate** until it is applied and the
  attack is replayed (see [attack-replay.md](attack-replay.md)).
- Treat ATLAS mappings suggested by an agent as approximate until verified
  against the curated index (`orion.mappings.validate_mappings`).

## Assessments

Agents may help describe scope and explain factual Assessment summaries. They
do not choose Assessment membership implicitly, promote Finding status, declare
controls effective or certify that a completed Assessment is secure. Linking,
activation, completion and archiving are explicit operations; Next Action uses
recorded context and the existing deterministic workspace lifecycle. The
Assessment is a reference container, not a replacement security oracle. See
[Assessments](assessments.md).
