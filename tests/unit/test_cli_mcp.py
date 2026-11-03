from __future__ import annotations

import json

import pytest

from incident_agent.app import Components
from incident_agent.cli import main
from incident_agent.domain import EvidenceSource


def test_cli_triage(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "triage",
            "--service",
            "ledger-service",
            "--title",
            "ORA-00060 errors",
            "--severity",
            "SEV2",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["retrieval_mode"] == "keyword"
    assert [n["node"] for n in out["audit_trail"]] == ["plan", "gather", "reason"]


def test_cli_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark"]) == 0
    assert "precedent_recall=1.00" in capsys.readouterr().out


async def test_mcp_server_exposes_domain_tool(local: Components) -> None:
    pytest.importorskip("mcp")
    from incident_agent.tools.mcp_server import build_mcp_server

    server = build_mcp_server(local.servers[EvidenceSource.TICKETS])
    tools = await server.list_tools()
    assert [t.name for t in tools] == ["search_tickets"]
