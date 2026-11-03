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
