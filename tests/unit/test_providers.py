"""Bedrock adapters against in-process fakes (no network)."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from incident_agent.core.errors import LLMOutputError
from incident_agent.core.rate_limit import AsyncTokenBucket
from incident_agent.domain import ModelTier
from incident_agent.llm.anthropic_api import ClaudeApiProvider, default_api_models
from incident_agent.llm.base import LLMRequest, Message
from incident_agent.llm.bedrock import ClaudeBedrockProvider, TitanEmbedder, default_bedrock_models


@dataclass
class _Block:
    type: str
    text: str = ""


@dataclass
class _Usage:
    input_tokens: int = 120
    output_tokens: int = 40


@dataclass
class _Message:
    content: list[_Block]
    stop_reason: str = "end_turn"
    usage: _Usage = field(default_factory=_Usage)


class FakeMessages:
    def __init__(self, response: _Message) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> _Message:
        self.kwargs = kwargs
        return self.response


class FakeAnthropic:
    def __init__(self, response: _Message) -> None:
        self.messages = FakeMessages(response)


REQUEST = LLMRequest(
    system="sys",
    messages=(Message("user", "hi"), Message("assistant", "a"), Message("user", "again")),
    json_schema={"type": "object"},
)


class TestClaudeBedrockProvider:
    async def test_maps_request_and_response(self) -> None:
        fake = FakeAnthropic(_Message([_Block("thinking"), _Block("text", '{"ok": 1}')]))
        provider = ClaudeBedrockProvider(fake, AsyncTokenBucket(rate=100))  # type: ignore[arg-type]
        spec = default_bedrock_models()[ModelTier.STANDARD]
        response = await provider.complete(spec, REQUEST)

        sent = fake.messages.kwargs
        assert sent["model"] == "anthropic.claude-sonnet-5-5"
        assert sent["output_config"] == {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": {"type": "object"}},
        }
        assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "user"]
        assert "temperature" not in sent
        assert response.text == '{"ok": 1}'
        assert (response.usage.input_tokens, response.usage.output_tokens) == (120, 40)

    async def test_refusal_raises(self) -> None:
        fake = FakeAnthropic(_Message([], stop_reason="refusal"))
        provider = ClaudeBedrockProvider(fake)  # type: ignore[arg-type]
        with pytest.raises(LLMOutputError, match="declined"):
            await provider.complete(default_bedrock_models()[ModelTier.LIGHT], REQUEST)


class TestClaudeApiProvider:
    async def test_sends_first_party_model_id(self) -> None:
        fake = FakeAnthropic(_Message([_Block("text", '{"ok": 1}')]))
        provider = ClaudeApiProvider(fake, AsyncTokenBucket(rate=100))  # type: ignore[arg-type]
        response = await provider.complete(default_api_models()[ModelTier.HEAVY], REQUEST)

        assert fake.messages.kwargs["model"] == "claude-opus-5-5"
        assert fake.messages.kwargs["output_config"]["effort"] == "high"
        assert response.text == '{"ok": 1}'

    def test_overrides_replace_one_tier(self) -> None:
        models = default_api_models({ModelTier.LIGHT: "claude-haiku-4-5"})
        assert models[ModelTier.LIGHT].model_id == "claude-haiku-4-5"
        assert models[ModelTier.STANDARD].model_id == "claude-sonnet-5-5"

    def test_from_api_key_disables_sdk_retries(self) -> None:
        provider = ClaudeApiProvider.from_api_key("test-key", requests_per_second=5.0)
        assert provider._client.max_retries == 0


class FakeBedrockRuntime:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.bodies: list[dict[str, Any]] = []

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        body = json.loads(kwargs["body"])
        self.bodies.append(body)
        vec = [float(len(body["inputText"]))] + [0.0] * (self.dim - 1)
        return {"body": io.BytesIO(json.dumps({"embedding": vec}).encode())}


class TestTitanEmbedder:
    async def test_embeds_concurrently_in_order(self) -> None:
        client = FakeBedrockRuntime(dim=4)
        embedder = TitanEmbedder(client, dim=4, limiter=AsyncTokenBucket(rate=1000, capacity=100))  # type: ignore[arg-type]
        out = await embedder.embed(["a", "bbb", "cc"])
        assert out.shape == (3, 4)
        assert out.dtype == np.float32
        assert out[:, 0].tolist() == [1.0, 3.0, 2.0]
        assert client.bodies[0] == {"inputText": "a", "dimensions": 4, "normalize": True}
        assert (await embedder.embed([])).shape == (0, 4)
