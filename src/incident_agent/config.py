"""Runtime configuration from environment variables (twelve-factor).

A frozen dataclass rather than a module of globals: it is constructed once,
validated once, passed explicitly, and trivially overridden in tests with
``dataclasses.replace``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from incident_agent.domain.enums import EvidenceSource, ModelTier

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class Settings:
    """All tunables in one immutable object."""

    region: str
    knowledge_bucket: str
    incidents_table: str
    reports_table: str
    feedback_table: str
    notify_topic_arn: str | None = None
    model_overrides: Mapping[ModelTier, str] | None = None
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dim: int = 512
    bedrock_rps: float = 2.0
    max_usd_per_run: float = 0.75
    log_group_prefix: str = "/app/"
    anthropic_key_secret: str | None = None
    mcp_urls: Mapping[EvidenceSource, str] | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Read ``IA_*`` variables; required ones raise ``KeyError`` with their name."""
        env = os.environ if env is None else env
        overrides = {
            tier: env[f"IA_MODEL_{tier.name}"]
            for tier in ModelTier
            if env.get(f"IA_MODEL_{tier.name}")
        }
        mcp_urls = {
            domain: env[f"IA_MCP_{domain.name}_URL"]
            for domain in EvidenceSource
            if env.get(f"IA_MCP_{domain.name}_URL")
        }
        return cls(
            region=env.get("AWS_REGION", "us-east-1"),
            knowledge_bucket=env["IA_KNOWLEDGE_BUCKET"],
            incidents_table=env["IA_INCIDENTS_TABLE"],
            reports_table=env["IA_REPORTS_TABLE"],
            feedback_table=env["IA_FEEDBACK_TABLE"],
            notify_topic_arn=env.get("IA_NOTIFY_TOPIC_ARN") or None,
            model_overrides=overrides or None,
            embedding_model_id=env.get("IA_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"),
            embedding_dim=int(env.get("IA_EMBEDDING_DIM", "512")),
            bedrock_rps=float(env.get("IA_BEDROCK_RPS", "2.0")),
            max_usd_per_run=float(env.get("IA_MAX_USD_PER_RUN", "0.75")),
            log_group_prefix=env.get("IA_LOG_GROUP_PREFIX", "/app/"),
            anthropic_key_secret=env.get("IA_ANTHROPIC_KEY_SECRET") or None,
            mcp_urls=mcp_urls or None,
        )
