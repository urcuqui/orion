# Limitations — what Orion does NOT prove

Orion is an instrument for understanding and measuring AI failure, not a
certificate of security. Be explicit about the boundaries of every result.

## Orion does not prove security

- **Passing a scenario does not prove a model is secure.** It shows that one
  attack, with one configuration, on one dataset, did (or did not) succeed.
- **One attack does not represent all adversaries.** A different goal,
  knowledge level, access level or budget can change the outcome entirely.
- **Metrics depend on the selected data.** Change the dataset, the seed or the
  sample size and the numbers change. Aggregate metrics hide per-class and
  per-example variance.
- **Defenses may fail under adaptive attacks.** A defense evaluated against a
  fixed attack can be defeated by an attacker who adapts to it. Robustness
  against *this* attack is not robustness in general.
- **A mitigation is local, not universal.** Improving robust accuracy against
  PGD says nothing about, say, patch attacks or data poisoning.

## Orion's synthetic backend is a teaching tool

The default, dependency-free backend (`orion.experiments.synthetic`) is a
deterministic *teaching* model. Its purpose is to make the full
Understand→Retest loop reproducible without a GPU. Its numbers illustrate the
methodology; they are **not** measurements of any real model. Use the
torch/ART-backed path (`orion.adversarial.generate_adversarial_evidence`) for
real models.

## LLM / agent limitations

- **LLM-generated suggestions may be incorrect.** Plans, explanations and
  recommended controls from agents are candidates to verify, not conclusions.
- **Agents do not decide security.** See [agent-role.md](agent-role.md). The
  verdict comes from metrics and replayed experiments.

## Tooling limitations

- **Automated tools require human validation.** Reconnaissance and scanning
  tools (Playwright, Nuclei, MCP tools) produce findings that a human must
  confirm. Sensitive actions require approval and fail closed
  (`orion.policies`).
- **ATLAS mappings may be approximate.** Orion ships a curated *subset* of the
  MITRE ATLAS matrix. Unknown technique IDs are flagged as unverified rather
  than silently accepted; consult <https://atlas.mitre.org/> for authority.

## How to report responsibly

Every evidence report ends with a limitations section and the reminder that a
result is scoped to its scenario. When you summarize:

- state the threat model the result is scoped to;
- state the data and parameters used;
- prefer "the attack succeeded / was mitigated under these conditions" over
  "the model is (in)secure".
