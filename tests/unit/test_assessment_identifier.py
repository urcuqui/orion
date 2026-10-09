"""Unit tests for assessment id validation + integrity error paths.

Identifier validation is a security boundary: every cross-reference an integrity
check follows must be a well-formed id, never attacker-influenced path material.
"""
import pytest

from orion.assessments.repository import identifier
from orion.assessments.integrity import validate_assessment_integrity


# --------------------------- identifier validation -------------------------- #
def test_identifier_accepts_well_formed_ids():
    assert identifier("ORN-ASSESS-1A2B") == "ORN-ASSESS-1A2B"
    assert identifier("ORN-RUN_01") == "ORN-RUN_01"


@pytest.mark.parametrize("bad", ["", "has space", "../etc", "a/b", "é", "-leading", 123, None])
def test_identifier_rejects_malformed_ids(bad):
    with pytest.raises((ValueError, TypeError)):
        identifier(bad)


def test_identifier_rejects_overlong():
    with pytest.raises(ValueError):
        identifier("A" * 200)


# ----------------------------- integrity paths ------------------------------ #
def test_integrity_unknown_assessment_is_invalid(tmp_path):
    report = validate_assessment_integrity("ORN-ASSESS-MISSING", base_dir=str(tmp_path))
    assert report["valid"] is False
    assert any(i["code"] == "REFERENCE_UNAVAILABLE" for i in report["issues"])


def test_integrity_rejects_malformed_assessment_id(tmp_path):
    # A malformed id must be rejected before any filesystem traversal.
    with pytest.raises(ValueError):
        validate_assessment_integrity("../escape", base_dir=str(tmp_path))
