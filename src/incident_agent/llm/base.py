"""Provider-neutral LLM interfaces.

``typing.Protocol`` gives structural typing: any class with a matching
``complete`` / ``embed`` method satisfies the interface without inheriting from
it. That is how we stay vendor-neutral (design section 4.1): Bedrock, Azure
OpenAI or Gemini adapters only have to match the shape, and tests use a fake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt

    from incident_agent.domain.enums import ModelTier
    from incident_agent.domain.values import TokenUsage


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A routable model with its price card (USD per million tokens)."""

    provider: str
    model_id: str
    tier: ModelTier
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"

    def cost(self, usage: TokenUsage) -> float:
        """Dollar cost of ``usage`` on this model."""
        return (
            usage.input_tokens * self.input_usd_per_mtok
            + usage.output_tokens * self.output_usd_per_mtok
        ) / 1_000_000


@dataclass(frozen=True, slots=True)
class Message:
    """One chat turn."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """Provider-neutral request. ``json_schema`` asks for structured output."""

    system: str
    messages: tuple[Message, ...]
    max_tokens: int = 4096
    json_schema: dict[str, Any] | None = field(default=None, hash=False)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Provider-neutral response."""

    text: str
    usage: TokenUsage
    model_id: str
    stop_reason: str = "end_turn"


class LLMProvider(Protocol):
    """Anything that can complete a request against a given model."""

    async def complete(self, model: ModelSpec, request: LLMRequest) -> LLMResponse:
        """Run one completion."""
        ...


class EmbeddingProvider(Protocol):
    """Anything that can embed a batch of texts into a ``(n, dim)`` matrix."""

    dim: int

    async def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Embed texts; row i corresponds to ``texts[i]``."""
        ...
