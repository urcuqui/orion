"""Deterministic, evidence-driven target interpretation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# In-memory registry of built assessments + human approval state, keyed by the
# recon run_id (or a synthetic key for context-only analyses).
ANALYSES: Dict[str, Dict[str, Any]] = {}


def display_run_id(run_id: str, kind: str = "RECON") -> str:
    """Human-facing id, e.g. ORN-RECON-A1B2."""
    tail = (run_id or "0000").replace("-", "")[:4].upper()
    return f"ORN-{kind}-{tail}"


# --------------------------------------------------------------------------- #
# 1. Summarize recon evidence (from the in-memory recon session events)
# --------------------------------------------------------------------------- #
def summarize_recon(run_id: str) -> Optional[Dict[str, Any]]:
    """Extract a structured summary from a recon run's events.

    Returns None if the run is unknown. Reads the existing recon session; it
    does not re-run anything or fabricate data.
    """
    try:
        from libs.recon import web as recon_web
    except Exception:  # pragma: no cover - recon optional
        return None

    session = recon_web.get_run(run_id)
    if session is None:
        return None

    with session.lock:
        events = list(session.events)
        status = session.status
        final_markdown = session.final_markdown

    objective = target = ""
    findings: Dict[str, Dict[str, Any]] = {}
    endpoints: List[str] = []
    auth_indicators: List[str] = []
    screenshots: List[str] = []
    report_markdown = final_markdown or ""

    for ev in events:
        etype = ev.get("type")
        if etype == "start":
            objective = ev.get("objective", "") or objective
            target = ev.get("target", "") or target
        elif etype == "evaluate":
            for f in ev.get("findings", []) or []:
                fid = f.get("id") or f.get("title")
                if fid:
                    findings[fid] = f
        elif etype in ("execute", "browser_tool") or ev.get("browser_tool"):
            for ep in ev.get("api_endpoints", []) or []:
                if ep not in endpoints:
                    endpoints.append(ep)
            for ai in ev.get("auth_indicators", []) or []:
                if ai not in auth_indicators:
                    auth_indicators.append(ai)
            shot = ev.get("screenshot_filename")
            if shot and shot not in screenshots:
                screenshots.append(shot)
        elif etype == "report":
            report_markdown = ev.get("report_markdown") or report_markdown

    findings_list = list(findings.values())
    return {
        "run_id": run_id,
        "display_id": display_run_id(run_id),
        "status": status,
        "objective": objective,
        "target": target or "unknown",
        "endpoints": endpoints,
        "auth_indicators": auth_indicators,
        "screenshots": screenshots,
        "findings": findings_list,
        "report_markdown": report_markdown,
        "counts": {
            "endpoints": len(endpoints),
            "findings": len(findings_list),
            "screenshots": len(screenshots),
            "auth_flows": len(auth_indicators),
        },
    }


# --------------------------------------------------------------------------- #
# 2. Propose a threat model (PROPOSED until human-approved)
# --------------------------------------------------------------------------- #
def propose_threat_model(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a candidate threat model from recon evidence via explicit rules."""
    endpoints = summary.get("endpoints", [])
    auth = summary.get("auth_indicators", [])

    assets = ["model_integrity", "prediction_reliability"]
    surfaces = []
    if endpoints:
        assets.append("inference_api")
        surfaces.append("inference_api")
    if auth:
        assets.append("confidentiality")
    surfaces.append("model_artifact")

    tm = {
        "status": "PROPOSED",
        "target": {
            "task": "ml_service",
            "access": "black_box",
            "model_name": summary.get("target", "unknown"),
        },
        "adversary": {
            # External recon implies a remote, partial-knowledge adversary.
            "goal": "evasion",
            "knowledge": "limited",
            "access": "black_box",
            "budget": "medium",
        },
        "assets": assets,
        "surfaces": sorted(set(surfaces)),
        "rationale": (
            f"Derived from recon of {summary.get('target','the target')}: "
            f"{summary['counts']['endpoints']} endpoint(s), "
            f"{summary['counts']['auth_flows']} auth indicator(s), "
            f"{summary['counts']['findings']} finding(s)."
        ),
    }
    return tm


# --------------------------------------------------------------------------- #
# 3. Threat hypotheses + candidate experiments (rule-based, ATLAS-mapped)
# --------------------------------------------------------------------------- #
def threat_hypotheses(summary: Dict[str, Any]) -> List[Dict[str, str]]:
    hyps: List[Dict[str, str]] = []
    # Adversarial input is always relevant for an ML target.
    hyps.append({"severity": "HIGH", "name": "Adversarial input manipulation",
                 "rationale": "ML inference targets are susceptible to crafted inputs."})
    if summary.get("endpoints"):
        hyps.append({"severity": "MEDIUM", "name": "Inference API abuse",
                     "rationale": f"{summary['counts']['endpoints']} endpoint(s) observed."})
    if summary.get("auth_indicators"):
        hyps.append({"severity": "MEDIUM", "name": "Authentication / authorization weakness",
                     "rationale": "Authentication flow(s) observed during recon."})
    # Any high/critical recon finding becomes a hypothesis.
    for f in summary.get("findings", []):
        sev = (f.get("severity") or "info").upper()
        if sev in ("CRITICAL", "HIGH"):
            hyps.append({"severity": sev, "name": f.get("title", "recon finding"),
                         "rationale": (f.get("description") or "")[:160]})
    return hyps


def propose_experiments(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Candidate experiments derived from evidence, mapped to MITRE ATLAS."""
    from orion.mappings import validate_mappings

    exps: List[Dict[str, Any]] = []

    def atlas(ids):
        return [m.to_dict() for m in validate_mappings([{"technique_id": i} for i in ids])]

    exps.append({
        "id": "01", "name": "Adversarial input robustness (PGD / C&W)", "risk": "LOW",
        "target": "local model", "scenario": "scenarios/pgd_evasion.yaml",
        "atlas": atlas(["AML.T0043.000", "AML.T0015"]),
        "rationale": "Measure degradation under bounded perturbations.",
    })
    if summary.get("endpoints"):
        exps.append({
            "id": "02", "name": "API boundary validation", "risk": "MEDIUM",
            "target": summary.get("target", "unknown"), "scenario": None,
            "atlas": atlas(["AML.T0040"]),
            "rationale": "Validate inputs/outputs at the inference API boundary.",
        })
    exps.append({
        "id": str(len(exps) + 1).zfill(2), "name": "Model behavior analysis", "risk": "LOW",
        "target": "local model", "scenario": None, "atlas": [],
        "rationale": "Characterize confidence/decision behavior under probing.",
    })
    exps.append({
        "id": str(len(exps) + 1).zfill(2), "name": "Reconnaissance extension", "risk": "LOW",
        "target": summary.get("target", "unknown"), "scenario": None, "atlas": [],
        "rationale": "Expand coverage of the observed attack surface.",
    })
    return exps


# --------------------------------------------------------------------------- #
# 4. Full assessment
# --------------------------------------------------------------------------- #
def build_assessment(run_id: str) -> Optional[Dict[str, Any]]:
    summary = summarize_recon(run_id)
    if summary is None:
        return None
    tm = propose_threat_model(summary)
    hyps = threat_hypotheses(summary)
    exps = propose_experiments(summary)
    atlas_count = sum(len(e.get("atlas", [])) for e in exps)
    assessment = {
        "source": "recon",
        "recon_run": summary["display_id"],
        "recon_run_id": run_id,
        "target": summary["target"],
        "summary": summary,
        "threat_model": tm,
        "threat_hypotheses": hyps,
        "suggested_experiments": exps,
        "atlas_count": atlas_count,
        "next_action": "Human review required",
        "approved": {"threat_model": False, "experiment_plan": False},
        "limitations": [
            "Proposed artifacts are derived from recon evidence by rules and are not confirmed.",
            "LLM/agent narrative (if any) is advisory, not a security verdict.",
        ],
    }
    ANALYSES[run_id] = assessment
    return assessment


def build_assessment_from_context(context: str, target: str = "manual-context") -> Dict[str, Any]:
    """Analysis without recon: a minimal, clearly-proposed assessment from text."""
    text = (context or "").lower()
    summary = {
        "run_id": "", "display_id": display_run_id("manual", "CTX"), "status": "manual",
        "objective": "manual target context", "target": target,
        "endpoints": ["(declared)"] if any(k in text for k in ("api", "endpoint", "service")) else [],
        "auth_indicators": ["(declared)"] if any(k in text for k in ("auth", "login", "token", "credential")) else [],
        "screenshots": [], "findings": [], "report_markdown": "",
        "counts": {"endpoints": 0, "findings": 0, "screenshots": 0, "auth_flows": 0},
    }
    summary["counts"]["endpoints"] = len(summary["endpoints"])
    summary["counts"]["auth_flows"] = len(summary["auth_indicators"])
    tm = propose_threat_model(summary)
    assessment = {
        "source": "context",
        "recon_run": None,
        "recon_run_id": None,
        "target": target,
        "context": context,
        "summary": summary,
        "threat_model": tm,
        "threat_hypotheses": threat_hypotheses(summary),
        "suggested_experiments": propose_experiments(summary),
        "atlas_count": 0,
        "next_action": "Human review required",
        "approved": {"threat_model": False, "experiment_plan": False},
        "limitations": [
            "No recon evidence loaded; this assessment is from user-provided context only.",
        ],
    }
    key = "ctx:" + target
    ANALYSES[key] = assessment
    return assessment


def approve_threat_model(run_id: str, approved: bool = True) -> Optional[Dict[str, Any]]:
    a = ANALYSES.get(run_id)
    if a is None:
        a = build_assessment(run_id)
    if a is None:
        return None
    a["approved"]["threat_model"] = bool(approved)
    a["threat_model"]["status"] = "APPROVED" if approved else "REJECTED"
    return a
