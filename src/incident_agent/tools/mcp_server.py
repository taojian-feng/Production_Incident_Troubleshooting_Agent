"""Expose a domain tool server over the Model Context Protocol.

One MCP server per domain (logs, tickets, knowledge), mirroring the design's
domain-based governance. In AWS each runs as its own Lambda with its own IAM
role, so a client granted only the knowledge server cannot reach logs.

For local use, ``python -m incident_agent.tools.mcp_server --domain knowledge``
serves the bundled sample data over stdio.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx2
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from incident_agent.domain.enums import Severity
from incident_agent.domain.models import Alert, McpEvidence, McpEvidenceBatch
from incident_agent.retrieval.hybrid import plan_retrieval

if TYPE_CHECKING:
    from incident_agent.tools.base import ToolServer

_FORWARDED_REQUEST_HEADERS = ("accept", "content-type", "mcp-protocol-version", "mcp-session-id")
_RETURNED_RESPONSE_HEADERS = ("content-type", "mcp-session-id")


def tool_name(domain: str) -> str:
    """The one tool each domain server offers."""
    return f"search_{domain}"


def build_mcp_server(server: ToolServer) -> MCPServer[Any]:
    """Wrap ``server`` as an MCP server with a single typed search tool."""
    app: MCPServer[Any] = MCPServer(f"incident-{server.domain.value}")

    @app.tool(
        name=tool_name(server.domain.value),
        description=f"Search the {server.domain.value} domain for evidence about an alert.",
    )
    async def search(
        service: str,
        title: str,
        description: str = "",
        severity: Severity = Severity.SEV3,
        triggered_at: datetime | None = None,
        log_group: str | None = None,
    ) -> McpEvidenceBatch:
        alert = Alert(
            service=service,
            title=title,
            description=description,
            severity=severity,
            triggered_at=triggered_at or datetime.now(UTC),
            log_group=log_group,
        )
        evidence = await server.gather(alert, plan_retrieval(alert))
        return McpEvidenceBatch(
            items=tuple(
                McpEvidence(
                    source=e.source,
                    ref=e.ref,
                    title=e.title,
                    snippet=e.snippet,
                    score=e.score,
                    uri=e.uri,
                )
                for e in evidence
            )
        )

    return app


async def handle_http_event(server: ToolServer, event: dict[str, Any]) -> dict[str, Any]:
    """Serve one Lambda Function URL request with the SDK's own HTTP transport.

    Stateless and JSON-only, because a Lambda may handle each request in a different
    environment. IAM authenticates callers at the Function URL, and the host check is
    off because the Function URL's hostname is the only way in.
    """
    app = build_mcp_server(server).streamable_http_app(
        json_response=True,
        stateless_http=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    headers = {key.lower(): value for key, value in event.get("headers", {}).items()}
    raw_body = event.get("body") or ""
    body = base64.b64decode(raw_body) if event.get("isBase64Encoded") else raw_body.encode()
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://mcp.internal"
        ) as client,
    ):
        response = await client.request(
            event["requestContext"]["http"]["method"],
            event.get("rawPath") or "/mcp",
            content=body,
            headers={k: headers[k] for k in _FORWARDED_REQUEST_HEADERS if k in headers},
        )
    return {
        "statusCode": response.status_code,
        "headers": {
            k: response.headers[k] for k in _RETURNED_RESPONSE_HEADERS if k in response.headers
        },
        "body": response.text,
    }


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
