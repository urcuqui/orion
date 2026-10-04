"""Black-box, query-based evasion against a *live* inference endpoint.

Real attack (authorized targets only): send a clean input, read the model's
decision, then send L-infinity-bounded perturbed inputs and observe whether the
decision changes. Decision-based — no gradients, no model weights. Bounded query
budget. Produces real, measurable evidence (baseline vs adversarial decision,
perturbation, query count).

The decision is extracted heuristically from the response (JSON prediction shape
or result terms in HTML); see ``_signature``. This measures an *observable*
behavioural change under small perturbation, and its limitations are recorded in
the evidence.
"""
from __future__ import annotations

import hashlib
import io
import json as _json
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse

from orion.evidence import EvidenceStore, ExperimentRecord, ExperimentStatus
from orion.metrics import MetricResult


def _json_decision(obj: Any, depth: int = 0) -> Optional[str]:
    """Extract a stable decision string from a JSON prediction response."""
    if depth > 4:
        return None
    if isinstance(obj, dict):
        for k in ("label", "class", "prediction", "result", "category", "name"):
            if k in obj and isinstance(obj[k], (str, int, float)):
                return f"{k}={obj[k]}"
        for k in ("probabilities", "scores", "logits", "confidences"):
            v = obj.get(k)
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                return f"argmax={max(range(len(v)), key=lambda i: v[i])}"
        for k in ("predictions", "outputs", "data", "results"):
            if k in obj:
                d = _json_decision(obj[k], depth + 1)
                if d:
                    return d
        # Fall back to the first scalar-ish value.
        for v in obj.values():
            d = _json_decision(v, depth + 1)
            if d:
                return d
    if isinstance(obj, list) and obj:
        return _json_decision(obj[0], depth + 1)
    return None


_RESULT_TERMS = ["no faces", "no face", "faces detected", "face detected", "detected",
                 "real", "fake", "genuine", "spoof", "positive", "negative"]


def _signature(text: str, content_type: str) -> Tuple[str, Any]:
    """A decision signature robust to echoed inputs (base64) and volatile markup."""
    ct = (content_type or "").lower()
    if "json" in ct:
        try:
            d = _json_decision(_json.loads(text))
            if d is not None:
                return ("json", d)
        except Exception:  # noqa: BLE001
            pass
    t = re.sub(r"data:[^;,]+;base64,[A-Za-z0-9+/=\s]+", " ", text or "")   # drop echoed images
    t = re.sub(r"<script[\s\S]*?</script>", " ", t, flags=re.I)
    t = re.sub(r"<style[\s\S]*?</style>", " ", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    low = re.sub(r"\s+", " ", t).lower().strip()
    terms = [kw for kw in _RESULT_TERMS if kw in low]
    counts = re.findall(r"(\d+)\s*(faces?|objects?|detections?|classes?)", low)
    if terms or counts:
        return ("text", (tuple(sorted(set(terms))), tuple(counts)))
    return ("hash", hashlib.sha1(low[:2000].encode()).hexdigest())


def _load_image_array(image_path: str):
    import numpy as np
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img).astype("float32") / 255.0


def _array_to_png_bytes(arr):
    import numpy as np
    from PIL import Image
    a = (np.clip(arr, 0.0, 1.0) * 255.0).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


def run_blackbox_evasion(url: str, path: str = "/", field: str = "image",
                         image_path: str = "static/fake/0001_00_00_01_0.jpg",
                         epsilon: float = 0.05, max_queries: int = 20, seed: int = 1337,
                         base_dir: str = "artifacts", output_dir: str = "static/adversarial",
                         provenance: Optional[Dict[str, Any]] = None) -> ExperimentRecord:
    """Decision-based black-box evasion via bounded random L-inf search."""
    import numpy as np
    import requests

    parsed = urlparse(url if "://" in url else "http://" + url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Provide an http(s) URL for the target service.")
    target = urljoin(f"{parsed.scheme}://{parsed.netloc}/", path.lstrip("/"))
    if not Path(image_path).exists():
        raise FileNotFoundError(f"test image not found: {image_path}")

    clean = _load_image_array(image_path)
    rng = np.random.default_rng(seed)

    def query(arr) -> Tuple[Tuple[str, Any], Any]:
        png = _array_to_png_bytes(arr)
        r = requests.post(target, files={field: ("input.png", png, "image/png")}, timeout=10)
        return _signature(r.text, r.headers.get("Content-Type", "")), png

    baseline_sig, _ = query(clean)
    queries = 1
    success = False
    adv = clean
    adv_sig = baseline_sig
    best_linf = 0.0
    # Escalate epsilon across restarts within the query budget.
    levels = [epsilon, min(epsilon * 2, 0.25), min(epsilon * 4, 0.5)]
    for lvl in levels:
        restarts = max(1, (max_queries - queries) // len(levels))
        for _ in range(restarts):
            if queries >= max_queries:
                break
            noise = rng.uniform(-lvl, lvl, size=clean.shape).astype("float32")
            cand = np.clip(clean + noise, 0.0, 1.0)
            sig, _ = query(cand)
            queries += 1
            if sig != baseline_sig:
                success, adv, adv_sig = True, cand, sig
                best_linf = float(np.max(np.abs(cand - clean)))
                break
        if success:
            break

    # Persist evidence + images.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "original.png").write_bytes(_array_to_png_bytes(clean))
    (Path(output_dir) / "output_art.png").write_bytes(_array_to_png_bytes(adv))
    (Path(output_dir) / "difference.png").write_bytes(_array_to_png_bytes(np.abs(adv - clean)))

    record = ExperimentRecord(
        scenario_name="blackbox-evasion",
        phase="attack", mode="attack", family="traditional_ml",
        target={"task": "image_classification", "access": "black_box",
                "model_name": f"{parsed.netloc}{path}", "model_version": "live-service"},
        model_version="live-service",
        threat_model={"target": {"task": "image_classification", "access": "black_box"},
                      "adversary": {"goal": "evasion", "knowledge": "limited",
                                    "access": "black_box", "budget": "low"}},
        attack_technique="Black-box query evasion (decision-based)",
        parameters={"endpoint": target, "field": field, "epsilon": epsilon,
                    "max_queries": max_queries, "seed": seed,
                    "attack_id": "ORN-ATTACK-EVA-BB"},
        baseline_result={"decision": f"{baseline_sig[0]}:{baseline_sig[1]}"},
        adversarial_result={"decision": f"{adv_sig[0]}:{adv_sig[1]}"},
        metrics={
            "attack_success": {"value": success},
            "query_count": MetricResult("query_count", queries, "queries").to_dict(),
            "perturbation_linf": MetricResult("perturbation_linf", round(best_linf, 4)).to_dict(),
            "decision_changed": {"value": success},
        },
        status=(ExperimentStatus.ATTACK_SUCCESS if success else ExperimentStatus.ATTACK_BLOCKED).value,
        limitations=[
            "Decision-based black-box: success = observable change in the service's "
            "decision signature under a bounded perturbation; the signature is parsed "
            "heuristically from the response.",
            "Bounded query budget; absence of a flip is not proof of robustness.",
            "Live attack against the specified authorized endpoint only.",
        ],
        notes=f"Live black-box query evasion against {target} ({queries} queries).",
        provenance=provenance or {},
    )
    EvidenceStore(base_dir).save(record, images={
        "original": str(Path(output_dir) / "original.png"),
        "adversarial": str(Path(output_dir) / "output_art.png"),
        "difference": str(Path(output_dir) / "difference.png"),
    })
    return record
