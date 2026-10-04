"""The unified catalogs are the single source of truth across families."""
import pytest

from orion.catalog import attacks as A, controls as C, metrics as M


# ------------------------------- attacks ------------------------------------ #
def test_three_families_present():
    assert set(A.FAMILIES) == {"traditional_ml", "generative_ai", "agentic_ai"}
    for fam in A.FAMILIES:
        assert A.by_family(fam), f"no attacks for {fam}"


def test_recommended_initial_catalog_present():
    names = {a.name for a in A.all_attacks()}
    assert {"FGSM", "PGD", "C&W (L2)", "Black-box Evasion"} <= names         # traditional ml
    assert {"Direct Prompt Injection", "Indirect Prompt Injection"} <= names  # genai
    assert {"Tool Poisoning", "Privilege / Tool Abuse"} <= names              # agentic


def test_attacks_resolve_by_alias_and_legacy_id():
    assert A.get("FGSM").id == "ORN-ATTACK-EVA-FGSM"
    assert A.get("indirect prompt injection").id == "ORN-ATTACK-PI-002"
    assert A.get("pgd_evasion").id == "ORN-ATTACK-EVA-PGD"      # legacy experiment id


def test_every_attack_has_the_common_schema():
    for a in A.all_attacks():
        d = a.to_dict()
        for k in ("family", "applicable_targets", "required_access", "success_criteria",
                  "supported_metrics", "recommended_controls", "framework_mappings"):
            assert k in d, f"{a.id} missing {k}"
        assert a.success_criteria and a.supported_metrics


def test_access_level_view_is_a_projection_not_a_duplicate():
    # The adversarial access-level catalog is derived from the unified one.
    from orion.adversarial import catalog as AC
    wb = {x["catalog_id"] for x in AC.attacks_for(AC.WHITE_BOX)}
    assert "ORN-ATTACK-EVA-PGD" in wb and "ORN-ATTACK-EVA-FGSM" in wb


# -------------------------------- metrics ----------------------------------- #
def test_metrics_cover_all_families():
    for fam in A.FAMILIES:
        assert M.for_family(fam), f"no metrics for {fam}"


def test_metric_for_attack_matches_declaration():
    ids = {m.id for m in M.for_attack("ORN-ATTACK-AG-002")}
    assert "privilege_boundary_violation_rate" in ids
    assert "unauthorized_tool_call_rate" in ids


def test_metric_direction_and_improvement():
    assert M.direction_of("attack_success_rate") == M.LOWER_BETTER
    assert M.direction_of("robust_accuracy") == M.HIGHER_BETTER
    assert M.is_improvement("attack_success_rate", 1.0, 0.0) is True
    assert M.is_improvement("attack_success_rate", 1.0, 1.0) is False
    assert M.is_improvement("robust_accuracy", 0.5, 0.9) is True


# -------------------------------- controls ---------------------------------- #
def test_controls_cover_agentic_vectors():
    ids = {c.id for c in C._CONTROLS}
    assert {"tool_authorization", "least_privilege", "destination_allowlist",
            "instruction_provenance", "context_isolation", "human_approval"} <= ids


def test_controls_for_attack_are_the_recommended_ones():
    ids = {c.id for c in C.for_attack("ORN-ATTACK-PI-002")}
    assert "instruction_provenance" in ids and "context_isolation" in ids
