"""Structured adversarial-image experiment (torch + ART).

Wraps the classic Carlini & Wagner / PGD flow so an experiment returns a
structured, measurable result such as::

    {
      "attack": "PGD",
      "baseline_prediction": "stop_sign",
      "baseline_confidence": 0.997,
      "adversarial_prediction": "speed_limit_30",
      "adversarial_confidence": 0.913,
      "perturbation_linf": 0.028,
      "attack_success": true
    }

Torch/ART are optional. When they are unavailable this module raises a clear
error and callers fall back to the deterministic synthetic backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:  # torch/timm/ART are optional heavy deps.
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    TORCH_AVAILABLE = False


@dataclass
class AdversarialImageResult:
    attack: str
    baseline_prediction: str
    baseline_confidence: float
    adversarial_prediction: str
    adversarial_confidence: float
    perturbation_linf: float
    attack_success: bool
    perturbation_l2: float = 0.0
    labels: Dict[int, str] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack": self.attack,
            "baseline_prediction": self.baseline_prediction,
            "baseline_confidence": self.baseline_confidence,
            "adversarial_prediction": self.adversarial_prediction,
            "adversarial_confidence": self.adversarial_confidence,
            "perturbation_linf": self.perturbation_linf,
            "perturbation_l2": self.perturbation_l2,
            "attack_success": self.attack_success,
            "artifacts": self.artifacts,
        }


def generate_adversarial_evidence(
    weights_path: str,
    num_outputs: int,
    image_path: str,
    labels: Optional[Dict[int, str]] = None,
    attack: str = "CarliniL2",
    output_dir: str = "static/adversarial",
) -> AdversarialImageResult:
    """Run a structured adversarial-image experiment.

    Produces both the visual artifact (adversarial PNG + difference map) and a
    measurable :class:`AdversarialImageResult`. Requires torch + ART + timm.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "torch is not available; use the synthetic backend "
            "(orion.experiments.run_scenario) or install the ML extras."
        )

    import os
    import torch
    import timm
    from PIL import Image
    from torchvision import transforms
    from art.attacks.evasion import CarliniL2Method
    from art.estimators.classification import PyTorchClassifier

    labels = labels or {0: "fake", 1: "real"}
    device = torch.device("cpu")

    model = timm.create_model("vit_base_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
    model.head = torch.nn.Linear(model.head.in_features, int(num_outputs))
    model = model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        nb_classes=len(labels),
        input_shape=(3, 224, 224),
    )

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    unnormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1 / s for s in [0.229, 0.224, 0.225]],
    )

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0)

    def _predict(tensor) -> tuple:
        with torch.no_grad():
            logits = model(tensor.to(device))
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(probs.argmax().item())
            return idx, float(probs[idx].item())

    base_idx, base_conf = _predict(x)

    atk = CarliniL2Method(classifier)
    adv = atk.generate(x.numpy())
    adv_t = torch.from_numpy(adv)
    adv_idx, adv_conf = _predict(adv_t)

    linf = float((adv_t - x).abs().max().item())
    l2 = float((adv_t - x).norm().item())

    os.makedirs(output_dir, exist_ok=True)
    artifacts: Dict[str, str] = {}
    orig_pil = transforms.functional.to_pil_image(unnormalize(x)[0].clamp(0, 1))
    adv_pil = transforms.functional.to_pil_image(unnormalize(adv_t)[0].clamp(0, 1))
    diff_pil = transforms.functional.to_pil_image((adv_t - x).abs()[0].clamp(0, 1))
    for name, pil in (("original", orig_pil), ("output_art", adv_pil), ("difference", diff_pil)):
        path = os.path.join(output_dir, f"{name}.png")
        pil.save(path)
        artifacts[name] = path

    return AdversarialImageResult(
        attack=attack,
        baseline_prediction=labels.get(base_idx, str(base_idx)),
        baseline_confidence=round(base_conf, 4),
        adversarial_prediction=labels.get(adv_idx, str(adv_idx)),
        adversarial_confidence=round(adv_conf, 4),
        perturbation_linf=round(linf, 4),
        perturbation_l2=round(l2, 4),
        attack_success=base_idx != adv_idx,
        labels=labels,
        artifacts=artifacts,
    )
