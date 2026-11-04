"""Lambda: one domain's MCP server behind an IAM-authenticated Function URL.

The same code runs for logs, tickets and knowledge. ``IA_MCP_DOMAIN`` selects the
domain, and each function has its own role holding only that domain's permissions.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from typing import TYPE_CHECKING, Any

import boto3

from incident_agent.app import build_domain_server
from incident_agent.config import Settings
from incident_agent.core.refresh import RefreshingValue
from incident_agent.domain.enums import EvidenceSource
from incident_agent.storage.repositories import S3IndexSnapshot
from incident_agent.tools.mcp_server import handle_http_event

if TYPE_CHECKING:
    from incident_agent.tools.base import ToolServer

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@functools.cache
def _server() -> RefreshingValue[ToolServer]:  # pragma: no cover - needs AWS
    domain = EvidenceSource(os.environ["IA_MCP_DOMAIN"])
    settings = Settings.from_env()

    def build() -> ToolServer:
        return build_domain_server(domain, settings)

    if domain is not EvidenceSource.KNOWLEDGE:
        return RefreshingValue(version=lambda: None, build=build)
    client = boto3.client("s3", region_name=settings.region)
    return RefreshingValue(
        version=S3IndexSnapshot(client, settings.knowledge_bucket).version, build=build
    )


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:  # pragma: no cover
    """Lambda entry point."""
    del context
    return asyncio.run(handle_http_event(_server().get(), event))
