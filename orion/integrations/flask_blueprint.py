"""A Flask blueprint exposing the Orion methodology (read-only + run).

Register it from ``app.py``::

    from orion.integrations.flask_blueprint import orion_bp
    app.register_blueprint(orion_bp)

Route handlers are thin: all logic lives in the ``orion`` package.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from orion.evidence import EvidenceStore
from orion.experiments import compare as compare_traces
from orion.experiments import run_scenario
from orion.methodology import PHASES, describe_phase
from orion.scenarios.loader import list_scenarios, load_scenario

orion_bp = Blueprint("orion", __name__, url_prefix="/orion")


@orion_bp.get("/methodology")
def methodology():
    return jsonify({
        "phases": [{"phase": p.value, "order": p.order, "description": describe_phase(p)} for p in PHASES],
        "principle": "An attack algorithm without a threat model is only an experiment.",
    })


@orion_bp.get("/scenarios")
def scenarios():
    return jsonify({"scenarios": list_scenarios("scenarios")})


@orion_bp.post("/run")
def run():
    payload = request.get_json(silent=True) or {}
    scenario_path = payload.get("scenario")
    mode = payload.get("mode", "attack")
    if not scenario_path:
        return jsonify({"error": "scenario is required"}), 400
    try:
        scenario = load_scenario(scenario_path)
        record = run_scenario(scenario, mode=mode)
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the UI
        return jsonify({"error": str(exc)}), 400
    return jsonify(record.to_dict())


@orion_bp.get("/traces")
def traces():
    return jsonify({"traces": EvidenceStore("artifacts").list_traces()})


@orion_bp.get("/report/<trace_id>")
def report(trace_id: str):
    store = EvidenceStore("artifacts")
    path = store.trace_dir(trace_id) / "report.md"
    if not path.exists():
        return jsonify({"error": "unknown trace_id"}), 404
    return jsonify({"trace_id": trace_id, "report_markdown": path.read_text(encoding="utf-8")})


@orion_bp.get("/compare")
def compare():
    before = request.args.get("before")
    after = request.args.get("after")
    if not before or not after:
        return jsonify({"error": "before and after trace ids are required"}), 400
    try:
        result = compare_traces(before, after)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(result)
