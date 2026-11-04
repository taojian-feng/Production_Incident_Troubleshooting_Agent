"""MCP path: the real SDK client and server over the same HTTP shape Lambda receives."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

import httpx2
import pytest
from botocore.credentials import ReadOnlyCredentials

from incident_agent.core.errors import ToolCallError
from incident_agent.domain import Alert, EvidenceSource, RetrievalPlan, Severity
from incident_agent.retrieval.hybrid import plan_retrieval
from incident_agent.tools.base import RawEvidence
from incident_agent.tools.mcp_client import LambdaUrlAuth, McpToolServer
from incident_agent.tools.mcp_server import handle_http_event

if TYPE_CHECKING:
    from incident_agent.app import Components
    from incident_agent.tools.base import ToolServer

URL = "https://abc123.lambda-url.us-east-1.on.aws/mcp"


class LambdaTransport(httpx2.AsyncBaseTransport):
    """Turns each client request into a Function URL event, as AWS does."""

    def __init__(self, server: ToolServer, *, encode_body: bool = False) -> None:
        self.server = server
        self.encode_body = encode_body
        self.methods: list[str] = []

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        body = await request.aread()
        self.methods.append(request.method)
        event: dict[str, Any] = {
            "rawPath": request.url.path,
            "headers": dict(request.headers),
            "requestContext": {"http": {"method": request.method}},
            "body": base64.b64encode(body).decode() if self.encode_body else body.decode(),
            "isBase64Encoded": self.encode_body,
        }
        reply = await handle_http_event(self.server, event)
        return httpx2.Response(
            reply["statusCode"], headers=reply["headers"], content=reply["body"].encode()
        )


def _client(server: ToolServer, *, encode_body: bool = False) -> McpToolServer:
    transport = LambdaTransport(server, encode_body=encode_body)
    return McpToolServer(
        domain=server.domain,
        url=URL,
        client_factory=lambda: httpx2.AsyncClient(transport=transport),
    )


@pytest.mark.parametrize("encode_body", [False, True])
async def test_remote_server_returns_the_same_evidence_as_calling_it_directly(
    local: Components, payments_alert: Alert, encode_body: bool
) -> None:
    server = local.servers[EvidenceSource.TICKETS]
    plan = plan_retrieval(payments_alert)
    direct = await server.gather(payments_alert, plan)
    remote = await _client(server, encode_body=encode_body).gather(payments_alert, plan)
    assert remote == direct
    assert remote


async def test_alert_details_reach_the_server_unchanged(payments_alert: Alert) -> None:
    received: list[Alert] = []

    class Recording:
        domain = EvidenceSource.LOGS

        async def gather(self, alert: Alert, plan: RetrievalPlan) -> list[RawEvidence]:
            received.append(alert)
            return []

    alert = payments_alert.model_copy(
        update={"severity": Severity.SEV1, "log_group": "/app/custom", "description": "d"}
    )
    await _client(Recording()).gather(alert, plan_retrieval(alert))
    assert received[0].triggered_at == alert.triggered_at
    assert received[0].severity is Severity.SEV1
    assert received[0].log_group == "/app/custom"
    assert received[0].description == "d"


async def test_a_failing_server_raises_so_the_registry_can_degrade(payments_alert: Alert) -> None:
    class Broken:
        domain = EvidenceSource.LOGS

        async def gather(self, alert: Alert, plan: RetrievalPlan) -> list[RawEvidence]:
            raise ConnectionError("cloudwatch is down")

    with pytest.raises(ToolCallError, match="logs server reported an error"):
        await _client(Broken()).gather(payments_alert, plan_retrieval(payments_alert))


async def test_agent_runs_end_to_end_with_every_domain_behind_mcp(
    local: Components, payments_alert: Alert
) -> None:
    direct, _ = await local.agent.run(payments_alert)
    original = local.registry.servers
    local.registry.servers = {domain: _client(s) for domain, s in original.items()}
    remote, _ = await local.agent.run(payments_alert)
    assert {e.ref for e in remote.evidence} == {e.ref for e in direct.evidence}
    assert remote.status == direct.status
    assert not local.registry.failures


def test_requests_are_signed_for_the_lambda_service() -> None:
    credentials = ReadOnlyCredentials("AKIDEXAMPLE", "secret", "session-token")
    auth = LambdaUrlAuth("us-east-1", credentials=lambda: credentials)
    request = httpx2.Request("POST", URL, content=b"{}")
    signed = next(auth.auth_flow(request))
    authorization = signed.headers["Authorization"]
    assert "AKIDEXAMPLE/" in authorization
    assert "/us-east-1/lambda/aws4_request" in authorization
    assert signed.headers["X-Amz-Security-Token"] == "session-token"
