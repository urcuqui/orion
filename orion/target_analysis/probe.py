"""Direct URL probe: gather *real* evidence from a running service.

Bridges "I have an app at a URL" and evidence-grounded analysis without a full
recon run. Performs bounded, **GET-only** requests to the target and a set of
well-known endpoints (LLM, agent, and — importantly — classic ML model-serving
endpoints), inspects response shapes and server headers, follows endpoints the
home page references, and builds a recon-style summary that
``build_assessment_from_summary`` interprets.

Detecting a plain ML classifier service (no MCP/LLM) is explicitly supported:
model-serving routes (`/predict`, `/v1/models`, `/v2/models`, `/invocations`),
serving-stack headers (TensorFlow Serving, TorchServe, Triton, KServe, BentoML),
and prediction-shaped JSON (`predictions` / `probabilities` / `logits` /
`softmax` / `signature_name`) are all strong, inspectable signals.

Authorized use only: GET-only, read-only, request-capped, no payloads.
"""
from __future__ import annotations

import json as _json
import re
from typing import Any, Dict, List
from urllib.parse import urljoin, urlparse

# Well-known endpoints worth checking. GET-only (POST-only routes answer 405,
# which still proves the route exists).
PROBE_PATHS = [
    "/",
    # ML model-serving / inference
    "/predict", "/predictions", "/inference", "/infer", "/invocations",
    "/classify", "/score", "/api/predict",
    "/v1/models", "/v2/models", "/v2/health/ready", "/ping", "/metrics",
    # LLM / agent
    "/mcp_tools", "/chat", "/chat_stream",
    "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/agent",
    # API docs / health
    "/openapi.json", "/swagger.json", "/docs", "/api", "/health", "/healthz",
]

# Serving stacks commonly announced in the Server header.
_SERVING_HEADERS = ["tensorflow serving", "torchserve", "triton", "kserve",
                    "seldon", "bentoml", "mlflow", "gunicorn/ml", "uvicorn"]

# JSON keys that indicate a prediction / model-serving response.
_STRONG_PRED_KEYS = ["predictions", "probabilities", "logits", "softmax",
                     "signature_name", "model_version_status", "model_version",
                     "num_classes"]
_WEAK_PRED_KEYS = ["class", "label", "confidence", "score", "classes", "instances", "inputs"]

# A minimal valid 1x1 PNG, used as a harmless probe upload (active mode only).
import base64 as _b64  # noqa: E402
_MIN_PNG = _b64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
# Result terms that appear in an inference response but not a blank upload form.
_INFERENCE_RESULT_TERMS = [
    "detected", "no face", "faces found", "no faces", "bounding", "confidence",
    "probability", "probabilities", "prediction", "landmark", "recognized",
    "class", "label", "score",
]

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_FORM_RE = re.compile(r"<form\b([^>]*)>(.*?)</form>", re.IGNORECASE | re.DOTALL)
_FILE_INPUT_RE = re.compile(r"""<input\b[^>]*type\s*=\s*['"]?file['"]?[^>]*>""", re.IGNORECASE)
_NAME_RE = re.compile(r"""name\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE)
_METHOD_RE = re.compile(r"""method\s*=\s*['"]?([a-z]+)['"]?""", re.IGNORECASE)
_ACTION_RE = re.compile(r"""action\s*=\s*['"]([^'"]*)['"]""", re.IGNORECASE)
_REF_RE = re.compile(r"""(?:action|href|src)\s*=\s*['"](/[^'"\s>]+)['"]"""
                     r"""|fetch\(\s*['"](/[^'"\s)]+)['"]"""
                     r"""|url\s*:\s*['"](/[^'"\s]+)['"]""", re.IGNORECASE)


def _text_from_html(body: str, limit: int = 5000) -> str:
    title = ""
    m = _TITLE_RE.search(body or "")
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
    stripped = re.sub(r"<style[\s\S]*?</style>", " ", body or "", flags=re.IGNORECASE)
    # Keep <script> text: inline JS often references the predict endpoint.
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped)
    return (title + " " + stripped)[:limit]


def _extract_refs(body: str) -> List[str]:
    """Pull referenced paths (form actions, fetch URLs, src/href) from HTML/JS."""
    refs: List[str] = []
    for m in _REF_RE.finditer(body or ""):
        path = m.group(1) or m.group(2) or m.group(3)
        if path and path not in refs and not path.startswith("//"):
            # Ignore static asset noise.
            if not re.search(r"\.(css|png|jpg|jpeg|gif|svg|ico|woff2?|ttf)(\?|$)", path, re.IGNORECASE):
                refs.append(path)
    return refs[:12]


def _ml_response_tokens(snippet: str) -> List[str]:
    """Return signal tokens if a JSON body looks like a model prediction."""
    try:
        data = _json.loads(snippet)
    except Exception:
        return []
    keys = set()

    def walk(obj, depth=0):
        if depth > 4:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                keys.add(str(k).lower())
                walk(v, depth + 1)
        elif isinstance(obj, list):
            for v in obj[:5]:
                walk(v, depth + 1)
    walk(data)

    strong = [k for k in _STRONG_PRED_KEYS if k in keys]
    weak = [k for k in _WEAK_PRED_KEYS if k in keys]
    if strong or (("class" in keys or "label" in keys) and ("confidence" in keys or "score" in keys)):
        return ["ml_prediction_response", "softmax" if "softmax" in keys else "probabilities"]
    if len(weak) >= 2:
        return ["ml_prediction_response"]
    return []


def _parse_upload_forms(html: str) -> List[Dict[str, Any]]:
    """Find POST forms that accept a file upload (action, method, file field)."""
    forms: List[Dict[str, Any]] = []
    for m in _FORM_RE.finditer(html or ""):
        attrs, inner = m.group(1), m.group(2)
        if not _FILE_INPUT_RE.search(inner):
            continue
        method = (_METHOD_RE.search(attrs).group(1).lower() if _METHOD_RE.search(attrs) else "get")
        if method != "post":
            continue
        action_m = _ACTION_RE.search(attrs)
        action = action_m.group(1) if action_m else ""
        file_field = "image"
        fin = _FILE_INPUT_RE.search(inner)
        nm = _NAME_RE.search(fin.group(0)) if fin else None
        if nm:
            file_field = nm.group(1)
        forms.append({"action": action or "/", "method": "post", "file_field": file_field})
    return forms


# Inference endpoints worth an active POST, with safe test payloads per family.
# CV/tabular predictors accept an image or a tiny numeric tensor; LLM endpoints a
# short prompt; embedding endpoints a short input. All payloads are benign.
_CV_POST_PATHS = ["/predict", "/predictions", "/inference", "/infer", "/invocations",
                  "/classify", "/score", "/api/predict"]
_LLM_POST_PATHS = ["/v1/chat/completions", "/v1/completions", "/generate", "/chat", "/chat_stream"]
_EMB_POST_PATHS = ["/v1/embeddings", "/embeddings"]


def _post_attempts(path: str):
    """Ordered (kind, payload) attempts for an inference endpoint. 'files' uses a
    1×1 PNG; 'json' uses a minimal tensor/prompt."""
    if path in _LLM_POST_PATHS:
        return [("json", {"model": "test", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}),
                ("json", {"prompt": "ping", "max_tokens": 1}),
                ("json", {"input": "ping"})]
    if path in _EMB_POST_PATHS:
        return [("json", {"input": "ping"}), ("json", {"inputs": ["ping"]})]
    # CV / tabular / generic predictor
    return [("files", {"image": ("probe.png", _MIN_PNG, "image/png")}),
            ("files", {"file": ("probe.png", _MIN_PNG, "image/png")}),
            ("json", {"instances": [[0.0, 0.0, 0.0]]}),
            ("json", {"inputs": [[0.0, 0.0, 0.0]]}),
            ("json", {"data": [0.0, 0.0, 0.0]})]


def _inference_tokens_from_response(home_text: str, ctype: str, body: str) -> List[str]:
    """Behavioural ML signals from an active POST response."""
    tokens: List[str] = []
    ct = (ctype or "").lower()
    if "json" in ct:
        tokens += _ml_response_tokens(body)
    low = (body or "")[:5000].lower()
    # An image returned in response to an image upload = model output (e.g. CV).
    if "image/" in ct:
        tokens.append("ml_inference_response")
    # Result-specific terms that a blank upload form would not contain.
    home_low = (home_text or "").lower()
    for term in _INFERENCE_RESULT_TERMS:
        if term in low and term not in home_low:
            tokens.append("ml_inference_response")
            break
    return list(dict.fromkeys(tokens))


def build_probe_summary(url: str, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a recon-style summary from probe observations (pure/testable)."""
    endpoints: List[str] = []
    findings: List[Dict[str, Any]] = []
    report_parts: List[str] = [f"# Direct URL probe of {url}"]

    def add_finding(title: str):
        findings.append({"title": title, "severity": "info",
                         "description": f"Observed via direct URL probe of {url}."})

    for o in observations:
        status = o.get("status")
        path = o.get("path", "")
        exists = status is not None and status not in (404, 501, 502, 503)
        if not exists:
            continue  # never log attempted-but-absent paths into the signal text
        if path and path != "/" and path not in endpoints:
            endpoints.append(path)
        server = (o.get("server") or "").strip()
        ctype = (o.get("content_type") or "").strip()
        report_parts.append(f"- {path} -> {status} {server} {ctype}".rstrip())
        # Serving-stack header -> evidence-backed finding.
        low_server = server.lower()
        for h in _SERVING_HEADERS:
            if h in low_server:
                add_finding(f"server stack: {h}")
        snippet = o.get("snippet")
        if snippet:
            report_parts.append(snippet)
            if "json" in ctype.lower():
                for tok in _ml_response_tokens(snippet):
                    add_finding(tok)
        # Behavioural tokens (active POST) -> evidence-backed findings so they
        # carry an evidence id and can satisfy attack prerequisites.
        for tok in o.get("tokens", []) or []:
            add_finding(tok)

    auth = [p for p in endpoints if any(k in p for k in ("login", "auth", "signin"))]
    return {
        "target": url,
        "objective": "direct URL probe",
        "endpoints": endpoints,
        "auth_indicators": auth,
        "screenshots": [],
        "findings": findings,
        "report_markdown": "\n".join(report_parts),
        "counts": {"endpoints": len(endpoints), "findings": len(findings),
                   "screenshots": 0, "auth_flows": len(auth)},
    }


def probe_url(url: str, timeout: float = 3.0, max_requests: int = 32,
              active: bool = False) -> Dict[str, Any]:
    """Probe ``url`` + well-known/referenced endpoints.

    GET-only by default. With ``active=True`` (a sensitive action — opt-in,
    authorized targets only), it also submits a harmless 1x1 test image to any
    discovered POST upload form and inspects the response for model-inference
    behaviour (e.g. a face detector returning results). It never sends real
    user data — only the synthetic test image.
    """
    parsed = urlparse(url if "://" in url else "http://" + url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Provide an http(s) URL, e.g. http://127.0.0.1:5001")
    base = f"{parsed.scheme}://{parsed.netloc}"

    import requests

    def _get(path: str) -> Dict[str, Any]:
        target = urljoin(base + "/", path.lstrip("/"))
        obs: Dict[str, Any] = {"path": path if path != "/" else "/"}
        try:
            r = requests.get(target, timeout=timeout, allow_redirects=True)
            obs["status"] = r.status_code
            obs["server"] = r.headers.get("Server", "")
            obs["content_type"] = r.headers.get("Content-Type", "")
            ct = obs["content_type"].lower()
            if "html" in ct:
                obs["_raw"] = r.text
                obs["snippet"] = _text_from_html(r.text)
            elif "json" in ct:
                obs["snippet"] = (r.text or "")[:2000]
        except Exception as exc:  # noqa: BLE001
            obs["status"] = None
            obs["error"] = str(exc)
        return obs

    observations: List[Dict[str, Any]] = []
    # 1. Base page first, so we can follow the endpoints it references.
    home = _get("/")
    observations.append(home)
    discovered = _extract_refs(home.get("_raw", "")) if home.get("_raw") else []

    # 2. Known paths + discovered refs, de-duplicated, capped.
    seen = {"/"}
    queue = [p for p in (PROBE_PATHS[1:] + discovered) if not (p in seen or seen.add(p))]
    for path in queue[: max_requests - 1]:
        observations.append(_get(path))

    # 3. Active mode (authorized targets only): send harmless test inputs to
    #    (a) discovered upload forms and (b) common inference endpoints, to elicit
    #    a prediction response — covering ML services that don't expose inference
    #    on the home page.
    if active:
        home_text = home.get("snippet", "")

        def _record_post(label_path, resp, body):
            ctype = resp.headers.get("Content-Type", "")
            toks = _inference_tokens_from_response(home_text, ctype, body)
            observations.append({"path": "POST " + label_path, "status": resp.status_code,
                                 "content_type": ctype, "snippet": (body or "")[:1500],
                                 "tokens": toks})
            return toks

        # (a) discovered upload forms on the home page
        if home.get("_raw"):
            for form in _parse_upload_forms(home["_raw"])[:3]:
                target = urljoin(base + "/", form["action"].lstrip("/"))
                try:
                    r = requests.post(target, files={form["file_field"]: ("probe.png", _MIN_PNG, "image/png")},
                                      timeout=timeout, allow_redirects=True)
                    body = r.text if "image/" not in r.headers.get("Content-Type", "").lower() else ""
                    _record_post(form["action"] or "/", r, body)
                except Exception:  # noqa: BLE001
                    pass

        # (b) common inference endpoints — try safe payloads; record the first
        #     informative response (prediction shape or 2xx) per endpoint.
        inference_paths = _CV_POST_PATHS + _LLM_POST_PATHS + _EMB_POST_PATHS
        for path in inference_paths[: max_requests]:
            target = urljoin(base + "/", path.lstrip("/"))
            for kind, payload in _post_attempts(path):
                try:
                    if kind == "files":
                        r = requests.post(target, files=payload, timeout=timeout, allow_redirects=True)
                    else:
                        r = requests.post(target, json=payload, timeout=timeout, allow_redirects=True)
                except Exception:  # noqa: BLE001
                    break  # unreachable — stop this path
                if r.status_code in (404, 405, 501, 502, 503):
                    break  # route/method not present — not an active inference endpoint
                body = r.text if "image/" not in r.headers.get("Content-Type", "").lower() else ""
                toks = _inference_tokens_from_response(home_text, r.headers.get("Content-Type", ""), body)
                if toks or r.status_code < 400:
                    observations.append({"path": "POST " + path, "status": r.status_code,
                                         "content_type": r.headers.get("Content-Type", ""),
                                         "snippet": (body or "")[:1500], "tokens": toks})
                    break  # informative response found — next endpoint

    return build_probe_summary(base, observations)
