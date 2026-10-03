"""Visual comparison of image attacks — Original · Adversarial · Perturbation.

The comparison is a *reusable, type-aware artifact*: it renders for image
attacks in Attack, Measure, Retest and Evidence, and renders NOTHING for
non-image experiments (no empty panels). Raw perturbation metrics are never
normalized — only the displayed images are.
"""
import pathlib
import re

import pytest
from flask import render_template_string

import app as orion_app
from orion.evidence import EvidenceStore, ExperimentRecord

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _render(record, trace_id="ORN-RUN-TEST", demo=False):
    tmpl = ('{% from "components/image_attack_comparison.html" import image_attack_comparison %}'
            "{{ image_attack_comparison(record, trace_id, demo=demo) }}")
    with orion_app.app.test_request_context():
        return render_template_string(tmpl, record=record, trace_id=trace_id, demo=demo)


def _image_record(**over):
    kw = dict(
        scenario_name="pgd-evasion", attack_technique="PGD evasion",
        status="ATTACK_SUCCESS",
        artifacts={"original": "original.png", "adversarial": "output_art.png",
                   "difference": "difference.png"},
        baseline_result={"prediction": "cat", "confidence": 0.98},
        adversarial_result={"prediction": "dog", "confidence": 0.71},
        metrics={"attack_success": {"value": True},
                 "perturbation_linf": {"value": 0.0312749},
                 "perturbation_l2": {"value": 1.4567821}},
    )
    kw.update(over)
    return ExperimentRecord(**kw)


# ---------------------------- reusable component ---------------------------- #
def test_component_file_exists():
    assert (ROOT / "templates" / "components" / "image_attack_comparison.html").exists()


def test_renders_three_panels_for_image_attack():
    html = _render(_image_record())
    assert "img-triptych" in html
    assert html.count("img-panel") == 3
    assert "Original" in html and "Adversarial" in html
    # Third panel is a perturbation / difference map — never called "Mask".
    assert "Perturbation" in html and "Difference map" in html
    assert "Mask" not in html


# ------------------------------- type-aware --------------------------------- #
def test_non_image_experiment_renders_nothing():
    rec = ExperimentRecord(scenario_name="prompt-injection", attack_technique="Prompt injection",
                           status="ATTACK_SUCCESS",
                           metrics={"attack_success": {"value": True}},
                           artifacts={})  # no image artifacts
    html = _render(rec).strip()
    assert "img-triptych" not in html
    assert "img-panel" not in html


def test_tabular_attack_shows_no_empty_image_panels():
    rec = ExperimentRecord(scenario_name="tabular-evasion", attack_technique="HopSkipJump",
                           artifacts={"report": "report.md"})  # non-image artifact only
    html = _render(rec).strip()
    assert "img-triptych" not in html


# ----------------------------- prediction change ---------------------------- #
def test_classification_changed_is_explicit():
    html = _render(_image_record())
    assert "CLASSIFICATION CHANGED" in html


def test_attack_failed_when_prediction_unchanged():
    rec = _image_record(status="ATTACK_BLOCKED",
                        adversarial_result={"prediction": "cat", "confidence": 0.95},
                        metrics={"attack_success": {"value": False},
                                 "perturbation_linf": {"value": 0.02}})
    html = _render(rec)
    assert "ATTACK FAILED" in html or "UNCHANGED" in html
    assert "CLASSIFICATION CHANGED" not in html


# --------------------- success from semantics, not pixels ------------------- #
def test_success_badge_driven_by_metric_not_visual():
    # Visually different images but attack_success False -> no success badge.
    rec = _image_record(status="ATTACK_BLOCKED",
                        metrics={"attack_success": {"value": False},
                                 "perturbation_linf": {"value": 0.03}})
    html = _render(rec)
    assert "badge-attack_success" not in html


# --------------------- raw perturbation values preserved -------------------- #
def test_perturbation_values_not_normalized():
    # The displayed L-inf / L2 are the raw metric values, verbatim (no rounding,
    # no rescaling to 0..1 for the *numbers*).
    html = _render(_image_record())
    assert "0.0312749" in html
    assert "1.4567821" in html


def test_decision_labels_supported_for_blackbox():
    # Black-box runs carry {decision: ...} instead of {prediction: ...}.
    rec = _image_record(attack_technique="Black-box query evasion",
                        baseline_result={"decision": "real"},
                        adversarial_result={"decision": "fake"},
                        metrics={"attack_success": {"value": True},
                                 "perturbation_linf": {"value": 0.05}})
    html = _render(rec)
    assert "real" in html and "fake" in html
    assert "CLASSIFICATION CHANGED" in html


# ------------------------------- amplify / zoom ----------------------------- #
def test_amplify_control_is_display_only():
    html = _render(_image_record())
    assert "img-amp" in html and "amplify" in html


def test_images_link_for_inspection():
    html = _render(_image_record())
    # Each image is wrapped in an inspect link to the raw artifact.
    assert 'href="/api/artifacts/ORN-RUN-TEST/original.png"' in html
    assert 'href="/api/artifacts/ORN-RUN-TEST/difference.png"' in html


def test_demo_mode_adds_prominence_class():
    assert "img-compare demo" in _render(_image_record(), demo=True)
    assert "img-compare demo" not in _render(_image_record(), demo=False)


# ------------------------------ evidence route ------------------------------ #
@pytest.fixture
def saved_run():
    """Save a record to the live store the page route reads, then clean up."""
    import shutil
    store = EvidenceStore("artifacts")
    created = []

    def _save(rec):
        store.save(rec)
        created.append(rec.trace_id)
        return rec

    yield _save
    for tid in created:
        shutil.rmtree(store.trace_dir(tid), ignore_errors=True)


def test_run_detail_renders_comparison_for_image_run(saved_run):
    rec = saved_run(_image_record())
    body = orion_app.app.test_client().get("/runs/" + rec.trace_id).data.decode()
    assert "img-triptych" in body
    assert "Perturbation" in body


def test_run_detail_no_empty_panels_for_non_image_run(saved_run):
    rec = saved_run(ExperimentRecord(scenario_name="prompt-injection", status="ATTACK_SUCCESS",
                                     metrics={"attack_success": {"value": True}}))
    body = orion_app.app.test_client().get("/runs/" + rec.trace_id).data.decode()
    assert "img-triptych" not in body


# --------------------- shared JS renderer is wired in ----------------------- #
def test_js_renderer_defined_and_type_aware():
    src = (ROOT / "static" / "js" / "orion.js").read_text()
    assert "Orion.imageComparison" in src
    assert "Orion.wireImageAmplify" in src
    # Type-aware guard: bail out when there is no image artifact.
    assert "if (!art.original && !art.adversarial) return" in src


def test_console_injects_comparison_in_attack_measure_retest():
    src = (ROOT / "static" / "js" / "workspace.js").read_text()
    assert src.count("injectComparison(") >= 3  # attack, measure, retest(before/after)
    assert "renderRetestImages" in src
    assert "NO NEW IMAGE ARTIFACT" in src
