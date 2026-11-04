"""Claude through the first-party Claude API, for accounts without Bedrock access.

Same request and response handling as :class:`ClaudeBedrockProvider`; only the
client and the model ids differ. The API key comes from the caller (AWS Secrets
Manager in production) and is never read from configuration or logged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from incident_agent.core.rate_limit import AsyncTokenBucket
from incident_agent.domain.enums import ModelTier
from incident_agent.llm.base import ModelSpec
from incident_agent.llm.bedrock import ClaudeBedrockProvider

if TYPE_CHECKING:
    from collections.abc import Mapping


def default_api_models(
    overrides: Mapping[ModelTier, str] | None = None,
) -> dict[ModelTier, ModelSpec]:
    """Claude tiers on the Claude API, with list prices per million tokens."""
    ids = {
        ModelTier.LIGHT: "claude-haiku-5-5",
        ModelTier.STANDARD: "claude-sonnet-5-5",
        ModelTier.HEAVY: "claude-opus-5-5",
        **(overrides or {}),
    }
    return {
        ModelTier.LIGHT: ModelSpec(
            "anthropic", ids[ModelTier.LIGHT], ModelTier.LIGHT, 0.10, 0.50, "low"
        ),
        ModelTier.STANDARD: ModelSpec(
            "anthropic", ids[ModelTier.STANDARD], ModelTier.STANDARD, 2.0, 10.0, "medium"
        ),
        ModelTier.HEAVY: ModelSpec(
            "anthropic", ids[ModelTier.HEAVY], ModelTier.HEAVY, 4.0, 20.0, "high"
        ),
    }


class ClaudeApiProvider(ClaudeBedrockProvider):
    """``LLMProvider`` implementation for the first-party Claude API."""

    @classmethod
    def from_api_key(cls, api_key: str, requests_per_second: float = 2.0) -> ClaudeApiProvider:
        """Build a client; retries stay in our own policy, as on Bedrock."""
        from anthropic import AsyncAnthropic  # noqa: PLC0415

        client = AsyncAnthropic(api_key=api_key, max_retries=0, timeout=90.0)
        return cls(client, AsyncTokenBucket(rate=requests_per_second, capacity=4))
