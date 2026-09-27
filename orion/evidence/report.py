"""Render an ExperimentRecord to a human-readable Markdown report."""
from __future__ import annotations

from typing import Any, Dict

from orion.evidence.record import ExperimentRecord


def _kv_table(data: Dict[str, Any]) -> str:
    if not data:
        return "_none_\n"
    lines = ["| Field | Value |", "| --- | --- |"]
    for k, v in data.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def render_report(record: ExperimentRecord) -> str:
    r = record
    lines = []
    lines.append(f"# Orion Experiment Report — `{r.trace_id}`")
    lines.append("")
    lines.append(f"- **Timestamp:** {r.timestamp}")
    lines.append(f"- **Scenario:** {r.scenario_name or '(ad-hoc)'}")
    lines.append(f"- **Methodology phase:** {r.phase}")
    lines.append(f"- **Mode:** {r.mode}")
    lines.append(f"- **Model version:** {r.model_version}")
    lines.append(f"- **Final status:** **{r.status}**")
    lines.append("")

    lines.append("## Target")
    lines.append(_kv_table(r.target))

    lines.append("## Threat model")
    lines.append(_kv_table(r.threat_model))

    lines.append("## Attack")
    lines.append(f"- **Technique:** {r.attack_technique}")
    lines.append("")
    lines.append("**Parameters**")
    lines.append("")
    lines.append(_kv_table(r.parameters))

    lines.append("## Results")
    lines.append("**Baseline**")
    lines.append("")
    lines.append(_kv_table(r.baseline_result))
    lines.append("**Adversarial**")
    lines.append("")
    lines.append(_kv_table(r.adversarial_result))

    lines.append("## Metrics")
    if r.metrics:
        mlines = ["| Metric | Value |", "| --- | --- |"]
        for k, v in r.metrics.items():
            if isinstance(v, dict) and "value" in v:
                mlines.append(f"| {k} | {v['value']} {v.get('unit', '')} |")
            else:
                mlines.append(f"| {k} | {v} |")
        lines.append("\n".join(mlines))
    else:
        lines.append("_No metrics recorded._")
    lines.append("")

    lines.append("## MITRE ATLAS mappings")
    if r.mitre_atlas:
        alines = ["| Technique ID | Tactic | Technique | Confidence | Verified |", "| --- | --- | --- | --- | --- |"]
        for m in r.mitre_atlas:
            alines.append(
                f"| {m.get('technique_id','')} | {m.get('tactic','')} | {m.get('technique','')} "
                f"| {m.get('confidence','')} | {'yes' if m.get('known') else 'NO (unverified)'} |"
            )
        lines.append("\n".join(alines))
    else:
        lines.append("_No ATLAS mappings declared._")
    lines.append("")

    lines.append("## Controls tested")
    if r.controls_tested:
        for c in r.controls_tested:
            status = "BLOCKED" if c.get("blocked") else "did not block"
            lines.append(f"- **{c.get('name','')}** — {status}. {c.get('notes','')}")
    else:
        lines.append("_No controls applied in this run._")
    lines.append("")

    if r.artifacts:
        lines.append("## Artifacts")
        for label, rel in r.artifacts.items():
            lines.append(f"- **{label}:** `{rel}`")
        lines.append("")

    lines.append("## Limitations")
    if r.limitations:
        for lim in r.limitations:
            lines.append(f"- {lim}")
    else:
        lines.append("- Results depend on the selected data and parameters; this is not a proof of security.")
    lines.append("")
    lines.append("> Passing (or failing) a scenario does not prove a model is secure or insecure in general. "
                 "See `docs/limitations.md`.")
    lines.append("")
    return "\n".join(lines)
