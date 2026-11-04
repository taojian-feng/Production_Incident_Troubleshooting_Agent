"""Call a remote domain server over MCP, as if it were local.

:class:`McpToolServer` satisfies the same ``ToolServer`` Protocol as the in-process
servers, so the agent and its access policy do not change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import boto3
import httpx2
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from incident_agent.core.errors import ToolCallError
from incident_agent.domain.models import McpEvidenceBatch
from incident_agent.tools.base import RawEvidence
from incident_agent.tools.mcp_server import tool_name

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from botocore.credentials import ReadOnlyCredentials

    from incident_agent.domain.enums import EvidenceSource
    from incident_agent.domain.models import Alert
    from incident_agent.domain.state import RetrievalPlan

_SIGNED_HEADERS = ("Authorization", "X-Amz-Date", "X-Amz-Security-Token")
MCP_TIMEOUT_SECONDS = 25.0


def _ambient_credentials() -> ReadOnlyCredentials:  # pragma: no cover - needs AWS
    credentials = boto3.Session().get_credentials()
    if credentials is None:
        msg = "no AWS credentials available to sign MCP requests"
        raise ToolCallError(msg)
    return credentials.get_frozen_credentials()


class LambdaUrlAuth(httpx2.Auth):
    """Sign each request for a Lambda Function URL that uses IAM authentication."""

    requires_request_body = True

    def __init__(
        self, region: str, credentials: Callable[[], ReadOnlyCredentials] = _ambient_credentials
    ) -> None:
        self.region = region
        self.credentials = credentials

    def auth_flow(
        self, request: httpx2.Request
    ) -> Generator[httpx2.Request, httpx2.Response, None]:
        """Add a SigV4 signature, using fresh credentials for each request."""
        signed = AWSRequest(method=request.method, url=str(request.url), data=request.content)
        SigV4Auth(self.credentials(), "lambda", self.region).add_auth(signed)
        for name in _SIGNED_HEADERS:
            if name in signed.headers:
                request.headers[name] = signed.headers[name]
        yield request


@dataclass(slots=True)
class McpToolServer:
    """``ToolServer`` backed by a remote MCP server."""

    domain: EvidenceSource
    url: str
    client_factory: Callable[[], httpx2.AsyncClient]

    async def gather(self, alert: Alert, plan: RetrievalPlan) -> list[RawEvidence]:
        """Call the domain's search tool. The server derives its own plan from the alert."""
        del plan
        arguments = {
            "service": alert.service,
            "title": alert.title,
            "description": alert.description,
            "severity": alert.severity.value,
            "triggered_at": alert.triggered_at.isoformat(),
            "log_group": alert.log_group,
        }
        async with (
            self.client_factory() as client,
            # Stateless server: there is no session to close afterwards.
            streamable_http_client(self.url, http_client=client, terminate_on_close=False) as (
                read,
                write,
                *_,
            ),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(tool_name(self.domain.value), arguments)
        text = "".join(block.text for block in result.content if block.type == "text")
        if result.is_error:
            msg = f"{self.domain.value} server reported an error: {text[:300]}"
            raise ToolCallError(msg)
        return [
            RawEvidence(
                source=item.source,
                ref=item.ref,
                title=item.title,
                snippet=item.snippet,
                score=item.score,
                uri=item.uri,
            )
            for item in McpEvidenceBatch.model_validate_json(text).items
        ]
