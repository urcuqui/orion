"""Target-analysis properties: evidence-driven, PROPOSED until human-approved."""
from orion.target_analysis import (
    build_assessment_from_context,
    approve_threat_model,
    propose_experiments,
)


def test_threat_model_is_proposed_until_approved():
    a = build_assessment_from_context("payments-api with an inference endpoint and token auth",
                                      target="payments-api")
    # The proposed threat model must not be treated as fact.
    assert a["threat_model"]["status"] == "PROPOSED"
    assert a["approved"]["threat_model"] is False
    assert a["next_action"] == "Human review required"


def test_experiments_are_evidence_driven_and_atlas_mapped():
    # Context mentioning an endpoint yields an API experiment; otherwise not.
    with_api = propose_experiments({"target": "t", "endpoints": ["/x"], "auth_indicators": [],
                                    "findings": [], "counts": {"endpoints": 1}})
    without_api = propose_experiments({"target": "t", "endpoints": [], "auth_indicators": [],
                                       "findings": [], "counts": {"endpoints": 0}})
    names_with = [e["name"] for e in with_api]
    names_without = [e["name"] for e in without_api]
    assert any("API boundary" in n for n in names_with)
    assert not any("API boundary" in n for n in names_without)
    # ATLAS mappings on the adversarial experiment come from the curated index.
    adv = next(e for e in with_api if "Adversarial" in e["name"])
    assert adv["atlas"] and all(m["known"] for m in adv["atlas"])


def test_context_analysis_states_no_recon_limitation():
    a = build_assessment_from_context("some model", target="m")
    assert any("no recon evidence" in l.lower() for l in a["limitations"])
