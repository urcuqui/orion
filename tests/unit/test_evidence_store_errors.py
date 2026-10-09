"""Unit tests for orion.evidence.store — roundtrip + error/missing paths."""
import json

import pytest

from orion.evidence import EvidenceStore, ExperimentRecord


def test_load_missing_trace_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        EvidenceStore(str(tmp_path)).load("20990101T000000-deadbeef")


def test_save_then_load_roundtrip(tmp_path):
    store = EvidenceStore(str(tmp_path))
    rec = ExperimentRecord(scenario_name="unit", status="ATTACK_BLOCKED",
                           metrics={"attack_success_rate": {"value": 0.0}})
    store.save(rec)
    back = store.load(rec.trace_id)
    assert back.trace_id == rec.trace_id and back.status == "ATTACK_BLOCKED"


def test_malformed_experiment_json_raises_on_load(tmp_path):
    store = EvidenceStore(str(tmp_path))
    (store.trace_dir("BADRUN")).mkdir(parents=True)
    (store.trace_dir("BADRUN") / "experiment.json").write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        store.load("BADRUN")


def test_list_summaries_skips_malformed_records(tmp_path):
    store = EvidenceStore(str(tmp_path))
    store.save(ExperimentRecord(scenario_name="good", status="ATTACK_SUCCESS"))
    (store.trace_dir("BADRUN")).mkdir(parents=True)
    (store.trace_dir("BADRUN") / "experiment.json").write_text("{bad", encoding="utf-8")
    summaries = store.list_summaries()
    # The bad record must not break the listing; only the good one appears.
    assert len(summaries) == 1 and summaries[0]["scenario"] == "good"


def test_list_traces_missing_base_dir_is_empty(tmp_path):
    assert EvidenceStore(str(tmp_path / "nope")).list_traces() == []
