"""AWS Bedrock adapters.

* :class:`ClaudeBedrockProvider` - Claude on Amazon Bedrock through the official
  Anthropic SDK's Bedrock client (``AsyncAnthropicBedrockMantle``). Model ids
  carry the ``anthropic.`` prefix, e.g. ``anthropic.claude-sonnet-5-5``.
* :class:`TitanEmbedder` - Amazon Titan Text Embeddings v2 through
  ``bedrock-runtime`` ``InvokeModel``.

Both share our retry policy and a token bucket sized to the account's Bedrock
quota. SDK-level retries are disabled (``max_retries=0``) so retries are not
multiplied across layers.
"""

from __future__ import annotations

import asyncio
import json
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from incident_agent.core.errors import LLMOutputError
from incident_agent.core.rate_limit import AsyncTokenBucket, gather_bounded
from incident_agent.core.retry import BackoffPolicy, async_retry
from incident_agent.domain.enums import ModelTier
from incident_agent.domain.values import TokenUsage
from incident_agent.llm.base import LLMRequest, LLMResponse, Message, ModelSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt
    from anthropic import AsyncAnthropicBedrockMantle
    from anthropic.types import MessageParam, OutputConfigParam
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

_LLM_RETRY = BackoffPolicy(max_attempts=4, base_delay_s=1.0, max_delay_s=20.0)


def default_bedrock_models(
    overrides: Mapping[ModelTier, str] | None = None,
) -> dict[ModelTier, ModelSpec]:
    """Claude tiers on Bedrock. Prices are list prices; Bedrock bills separately.

    Override ids per tier with env vars (see ``config.Settings``) when a region
    only offers specific models or you use cross-region inference profiles.
    """
    ids = {
        ModelTier.LIGHT: "anthropic.claude-haiku-5-5",
        ModelTier.STANDARD: "anthropic.claude-sonnet-5-5",
        ModelTier.HEAVY: "anthropic.claude-opus-5-5",
        **(overrides or {}),
    }
    return {
        ModelTier.LIGHT: ModelSpec(
            "bedrock", ids[ModelTier.LIGHT], ModelTier.LIGHT, 0.10, 0.50, "low"
        ),
        ModelTier.STANDARD: ModelSpec(
            "bedrock", ids[ModelTier.STANDARD], ModelTier.STANDARD, 2.0, 10.0, "medium"
        ),
        ModelTier.HEAVY: ModelSpec(
            "bedrock", ids[ModelTier.HEAVY], ModelTier.HEAVY, 4.0, 20.0, "high"
        ),
    }


def _to_param(message: Message) -> MessageParam:
    """Map our provider-neutral message onto the SDK's TypedDict (narrowed by role)."""
    if message.role == "user":
        return {"role": "user", "content": message.content}
    return {"role": "assistant", "content": message.content}


class ClaudeBedrockProvider:
    """``LLMProvider`` implementation for Claude on Bedrock."""

    def __init__(
        self,
        client: AsyncAnthropicBedrockMantle,
        limiter: AsyncTokenBucket | None = None,
    ) -> None:
        self._client = client
        self._limiter = limiter or AsyncTokenBucket(rate=2.0, capacity=4)

    @classmethod
    def from_region(cls, region: str, requests_per_second: float = 2.0) -> ClaudeBedrockProvider:
        """Build with AWS credentials from the default chain (Lambda role)."""
        from anthropic import AsyncAnthropicBedrockMantle  # noqa: PLC0415

        client = AsyncAnthropicBedrockMantle(aws_region=region, max_retries=0, timeout=90.0)
        return cls(client, AsyncTokenBucket(rate=requests_per_second, capacity=4))

    @async_retry(policy=_LLM_RETRY)
    async def complete(self, model: ModelSpec, request: LLMRequest) -> LLMResponse:
        """One Messages API call; JSON-schema-constrained when ``request.json_schema``."""
        output_config: OutputConfigParam = {"effort": model.effort}
        if request.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        messages = [_to_param(m) for m in request.messages]

        async with self._limiter:
            response = await self._client.messages.create(
                model=model.model_id,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=messages,
                output_config=output_config,
            )

        if response.stop_reason == "refusal":
            msg = f"model {model.model_id} declined the request"
            raise LLMOutputError(msg)
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            usage=TokenUsage(response.usage.input_tokens, response.usage.output_tokens),
            model_id=model.model_id,
            stop_reason=str(response.stop_reason),
        )


class TitanEmbedder:
    """``EmbeddingProvider`` for ``amazon.titan-embed-text-v2:0``.

    Titan v2 embeds one text per request, so a batch fans out with
    :func:`gather_bounded`: concurrent, but never more than ``concurrency`` in
    flight and never faster than the token bucket allows.
    """

    def __init__(
        self,
        client: BedrockRuntimeClient,
        *,
        model_id: str = "amazon.titan-embed-text-v2:0",
        dim: int = 512,
        concurrency: int = 8,
        limiter: AsyncTokenBucket | None = None,
    ) -> None:
        self._client = client
        self.model_id = model_id
        self.dim = dim
        self._concurrency = concurrency
        self._limiter = limiter or AsyncTokenBucket(rate=20.0, capacity=20)

    @async_retry(policy=_LLM_RETRY)
    async def _embed_one(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text[:20_000], "dimensions": self.dim, "normalize": True})
        # boto3 is synchronous: run it in a worker thread so the event loop keeps
        # serving the other in-flight embeddings (ruff's ASYNC rules catch the
        # mistake of calling it directly).
        response = await asyncio.to_thread(
            self._client.invoke_model,
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        return list(payload["embedding"])

    async def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Embed ``texts`` concurrently, preserving order."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = await gather_bounded(
            (partial(self._embed_one, t) for t in texts),
            limit=self._concurrency,
            limiter=self._limiter,
        )
        return np.asarray(vectors, dtype=np.float32)
