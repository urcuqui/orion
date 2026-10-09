"""Coverage for the Orion CLI (orion.cli.main) — end-to-end over a tmp store."""
import pytest

from orion import cli
from orion.evidence import EvidenceStore


def test_list_scenarios_and_empty_traces(tmp_path, capsys):
    assert cli.main(["--artifacts", str(tmp_path), "list"]) == 0
    out = capsys.readouterr().out
    assert "Scenarios in scenarios/" in out and "pgd" in out.lower()


def test_validate_ok_and_invalid(tmp_path):
    assert cli.main(["validate", "scenarios/pgd_evasion.yaml"]) == 0
    assert cli.main(["validate", "no/such.yaml"]) == 1


def test_run_replay_report_compare_roundtrip(tmp_path, capsys):
    art = str(tmp_path)
    assert cli.main(["--artifacts", art, "run", "scenarios/pgd_evasion.yaml", "--mode", "attack"]) == 0
    traces = EvidenceStore(art).list_traces()
    assert len(traces) == 1
    attack = traces[0]

    assert cli.main(["--artifacts", art, "report", attack]) == 0        # report exists
    assert cli.main(["--artifacts", art, "replay", attack, "--mode", "hardened"]) == 0
    hardened = [t for t in EvidenceStore(art).list_traces() if t != attack][0]

    capsys.readouterr()
    assert cli.main(["--artifacts", art, "compare", attack, hardened]) == 0
    assert "->" in capsys.readouterr().out

    # 'list' now reports evidence traces too; missing report returns 1.
    assert cli.main(["--artifacts", art, "list"]) == 0
    assert cli.main(["--artifacts", art, "report", "UNKNOWN-TRACE"]) == 1


def test_missing_subcommand_errors(tmp_path):
    with pytest.raises(SystemExit):
        cli.main([])          # subcommand is required
