"""Traditional ML profiling: model-centric (model, features, predictions, robustness)."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from orion.know_yourself.common import (
    Item, OBSERVED, INFERRED, UNKNOWN, ENABLED, PARTIAL, NOT_FOUND, NOT_APPLICABLE,
    HIGH, MEDIUM, LOW,
)

_EXT_FRAMEWORK = {
    ".pt": "pytorch", ".pth": "pytorch", ".safetensors": "pytorch/hf",
    ".onnx": "onnx", ".h5": "keras/tensorflow", ".keras": "keras",
    ".pb": "tensorflow", ".joblib": "scikit-learn", ".pkl": "scikit-learn",
    ".xgb": "xgboost", ".cbm": "catboost",
}


@dataclass
class ModelFingerprint:
    system_type: str = "traditional_ml"
    model_name: str = "unknown"
    model_type: str = "unknown"          # image_classifier / tabular_classifier / regressor...
    framework: str = "unknown"
    framework_version: str = "unknown"
    task: str = "unknown"
    input_type: str = "unknown"
    input_shape: Optional[list] = None
    output_type: str = "unknown"
    num_classes: Optional[int] = None
    artifact: Optional[str] = None
    model_hash: Optional[str] = None
    model_size_bytes: Optional[int] = None
    access: str = "unknown"
    supports_gradients: bool = False
    inference_method: str = "unknown"    # local / http_api
    preprocessing: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def build_model_fingerprint(summary: Dict[str, Any], descriptor: Optional[Dict[str, Any]],
                            caps: Dict[str, Any]) -> ModelFingerprint:
    d = descriptor or {}
    fp = ModelFingerprint()
    fp.model_name = d.get("model_name", summary.get("target", "unknown"))
    fp.access = caps.get("access", "unknown")
    fp.supports_gradients = bool(caps.get("supports_gradients"))
    fp.input_type = d.get("input_type", caps.get("input_type", "unknown"))
    fp.task = d.get("task", _infer_task(summary, fp.input_type))
    fp.model_type = d.get("model_type", _infer_model_type(fp.task, fp.input_type))
    fp.inference_method = "local" if (d.get("artifact") or d.get("model_path")) else (
        "http_api" if summary.get("endpoints") else "unknown")
    fp.preprocessing = list(d.get("preprocessing", []))
    fp.input_shape = d.get("input_shape")
    fp.num_classes = d.get("num_classes")
    fp.output_type = d.get("output_type", "class_label" if "class" in fp.model_type else "unknown")
    fp.framework_version = d.get("framework_version", "unknown")

    # Framework from an artifact file (no model loading required).
    artifact = d.get("artifact") or d.get("model_path")
    if artifact:
        fp.artifact = artifact
        ext = os.path.splitext(artifact)[1].lower()
        fp.framework = d.get("framework", _EXT_FRAMEWORK.get(ext, "unknown"))
        try:
            if os.path.exists(artifact):
                fp.model_size_bytes = os.path.getsize(artifact)
                fp.model_hash = _sha256(artifact)
        except OSError:
            pass
    else:
        # Framework from serving evidence (report text).
        fp.framework = d.get("framework", _framework_from_text(summary))
    return fp


def _sha256(path: str, limit: int = 64 * 1024 * 1024) -> Optional[str]:
    try:
        h = hashlib.sha256()
        read = 0
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk or read > limit:
                    break
                h.update(chunk)
                read += len(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _framework_from_text(summary: Dict[str, Any]) -> str:
    hay = str(summary.get("report_markdown", "")).lower()
    for kw, fw in (("tensorflow serving", "tensorflow"), ("torchserve", "pytorch"),
                   ("triton", "triton"), ("pytorch", "pytorch"), ("tensorflow", "tensorflow"),
                   ("keras", "keras"), ("scikit-learn", "scikit-learn"), ("sklearn", "scikit-learn"),
                   ("onnx", "onnx"), ("xgboost", "xgboost")):
        if kw in hay:
            return fw
    return "unknown"


def _infer_task(summary: Dict[str, Any], input_type: str) -> str:
    hay = str(summary.get("report_markdown", "")).lower()
    if "face detection" in hay or "object detection" in hay:
        return "object_detection"
    if "classification" in hay or "classifier" in hay or input_type == "image":
        return "image_classification" if input_type == "image" else "classification"
    return "unknown"


def _infer_model_type(task: str, input_type: str) -> str:
    if "detection" in task:
        return "object_detector"
    if input_type == "image":
        return "image_classifier"
    if input_type == "tabular":
        return "tabular_classifier"
    if "classif" in task:
        return "classifier"
    return "unknown"


# --------------------------------------------------------------------------- #
# Input & feature surface + model assumptions
# --------------------------------------------------------------------------- #
def input_surface(fp: ModelFingerprint, descriptor: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    d = descriptor or {}
    return {
        "input_type": fp.input_type,
        "input_shape": fp.input_shape if fp.input_shape is not None else "UNKNOWN",
        "valid_range": d.get("valid_range", "UNKNOWN"),
        "normalization": d.get("normalization", "UNKNOWN"),
        "preprocessing": fp.preprocessing or "UNKNOWN",
        "categorical_features": d.get("categorical_features", "UNKNOWN"),
        "missing_value_handling": d.get("missing_value_handling", "UNKNOWN"),
        "ood_behavior": d.get("ood_behavior", "UNKNOWN"),
    }


def model_assumptions(fp: ModelFingerprint, caps: Dict[str, Any],
                      descriptor: Optional[Dict[str, Any]]) -> List[Item]:
    d = descriptor or {}
    items: List[Item] = []
    if fp.preprocessing:
        items.append(Item("Input preprocessing: " + ", ".join(fp.preprocessing), OBSERVED, HIGH,
                          rationale="Declared in the model descriptor."))
    else:
        items.append(Item("Input preprocessing/normalization", UNKNOWN, confidence="NONE",
                          rationale="Not observed; preprocessing must be confirmed before crafting inputs."))
    if fp.input_type == "image":
        items.append(Item("Model expects in-distribution RGB images", INFERRED, MEDIUM,
                          rationale="Inferred from image input type."))
    if caps.get("access") == "white_box":
        items.append(Item("White-box access / gradients available", OBSERVED, HIGH,
                          evidence=["access:white_box"], rationale="Model artifact is available locally."))
    elif caps.get("access") == "black_box":
        items.append(Item("Black-box access (query only)", OBSERVED, HIGH,
                          rationale="Reached as a remote service."))
    # Always-unknown provenance assumptions unless explicitly provided.
    if d.get("training_data_known") is None:
        items.append(Item("Training-data provenance", UNKNOWN, confidence="NONE",
                          rationale="Not provided; relevant to privacy/poisoning risk."))
    if d.get("adversarial_training") is None:
        items.append(Item("Adversarial training used", UNKNOWN, confidence="NONE",
                          rationale="Not provided; affects robustness expectations."))
    return items


# --------------------------------------------------------------------------- #
# Control inventory (UNKNOWN never becomes NOT_FOUND)
# --------------------------------------------------------------------------- #
_TRAD_CONTROLS = [
    "adversarial_training", "input_validation", "confidence_thresholds",
    "anomaly_detection", "ood_detection", "monitoring", "rate_limiting",
    "model_versioning", "artifact_integrity", "signed_models", "rollback_support",
]


def control_inventory(descriptor: Optional[Dict[str, Any]], summary: Dict[str, Any]) -> List[Dict[str, str]]:
    d = (descriptor or {}).get("controls", {}) or {}
    hay = str(summary.get("report_markdown", "")).lower()
    out: List[Dict[str, str]] = []
    for c in _TRAD_CONTROLS:
        if c in d:
            status = str(d[c]).upper()
            if status not in (ENABLED, PARTIAL, NOT_FOUND, NOT_APPLICABLE, UNKNOWN):
                status = ENABLED if d[c] else NOT_FOUND
        elif c.replace("_", " ") in hay:
            status = PARTIAL
        else:
            status = UNKNOWN  # never guess NOT_FOUND
        out.append({"control": c, "status": status})
    return out
