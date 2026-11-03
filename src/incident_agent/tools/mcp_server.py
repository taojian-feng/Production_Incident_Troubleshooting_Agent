"""Expose a domain tool server over the Model Context Protocol.

One MCP server per domain (``--domain logs|tickets|knowledge``), mirroring the
design's domain-based governance: an MCP client granted only the knowledge
server cannot reach logs. Requires the ``mcp`` extra (``pip install
incident-agent[mcp]``); run with ``python -m incident_agent.tools.mcp_server``.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import TYPE_CHECKING, Any

from incident_agent.domain.enums import Severity
from incident_agent.domain.models import Alert
from incident_agent.retrieval.hybrid import plan_retrieval

if TYPE_CHECKING:
    from incident_agent.tools.base import ToolServer


def build_mcp_server(server: ToolServer) -> Any:  # noqa: ANN401 - optional untyped dependency
    """Wrap ``server`` as an MCP server with a single typed search tool."""
    from mcp.server.mcpserver import MCPServer  # noqa: PLC0415 - optional dependency

    app = MCPServer(f"incident-{server.domain.value}")

    @app.tool(
        name=f"search_{server.domain.value}",
        description=f"Search the {server.domain.value} domain for evidence about an alert.",
    )
    async def search(service: str, title: str, description: str = "") -> list[dict[str, Any]]:
        alert = Alert(service=service, title=title, description=description, severity=Severity.SEV3)
        evidence = await server.gather(alert, plan_retrieval(alert))
        return [
            {"ref": e.ref, "title": e.title, "snippet": e.snippet, "score": e.score}
            for e in evidence
        ]

    return app


def main() -> None:  # pragma: no cover - thin CLI wrapper
    """Run one domain server over stdio using the bundled sample data."""
    from incident_agent.app import build_local_components  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["logs", "tickets", "knowledge"], required=True)
    args = parser.parse_args()
    servers = asyncio.run(build_local_components()).servers
    selected = next(s for d, s in servers.items() if d.value == args.domain)
    build_mcp_server(selected).run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
