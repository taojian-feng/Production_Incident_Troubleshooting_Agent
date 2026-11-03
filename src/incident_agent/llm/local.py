"""Deterministic offline providers for tests, CI and the local demo.

They satisfy the same Protocols as the Bedrock adapters, so the whole agent
runs end-to-end with no AWS account and no network - and produces identical
output on every run, which is what you want from a test double.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING

import numpy as np

from incident_agent.domain.enums import StepKind
from incident_agent.domain.values import TokenUsage
from incident_agent.llm.base import LLMRequest, LLMResponse, ModelSpec
from incident_agent.retrieval.text import tokenize

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

_EVIDENCE_LINE = re.compile(r"^\[(E\d+)\]\s*\((\w+)\)\s*(.*)$", re.MULTILINE)


class HashingEmbedder:
    """Feature-hashing bag-of-words embedder (signed, L2-normalised).

    Not semantic, but stable and dependency-free; shared vocabulary still gives
    sensible nearest neighbours on the sample corpus.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> npt.NDArray[np.float32]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            vec[bucket] += 1.0 if digest[4] & 1 else -1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    async def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Embed texts (synchronous math, async signature to match the Protocol)."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])


class ScriptedLLM:
    """Builds a grounded triage answer directly from the evidence in the prompt.

    Mirrors the contract the real model is held to: every step cites evidence
    ids that appear in the prompt. Set ``hallucinate=True`` to emit an uncited
    claim and exercise the citation validator.
    """

    def __init__(self, *, hallucinate: bool = False) -> None:
        self.hallucinate = hallucinate
        self.calls: list[tuple[str, LLMRequest]] = []

    async def complete(self, model: ModelSpec, request: LLMRequest) -> LLMResponse:
        """Return a schema-valid JSON triage built from ``[E#]`` evidence lines."""
        self.calls.append((model.model_id, request))
        prompt = request.messages[-1].content
        evidence = _EVIDENCE_LINE.findall(prompt)
        by_source: dict[str, list[tuple[str, str]]] = {}
        for eid, source, text in evidence:
            by_source.setdefault(source, []).append((eid, text))

        logs = by_source.get("logs", [])
        tickets = by_source.get("tickets", [])
        # Prefer actionable sections (Mitigation / Resolution) for the recommendation.
        knowledge = sorted(
            by_source.get("knowledge", []),
            key=lambda item: not re.search(r"mitigation|resolution", item[1], re.IGNORECASE),
        )
        first = evidence[0][0] if evidence else None

        steps = [
            {
                "kind": StepKind.SUMMARY.value,
                "content": "Alert correlates with: " + (logs[0][1][:160] if logs else "no logs"),
                "citations": [logs[0][0]] if logs else ([first] if first else []),
            },
            {
                "kind": StepKind.PRECEDENT.value,
                "content": "Closest precedent: "
                + (tickets[0][1][:160] if tickets else "none found"),
                "citations": [tickets[0][0]] if tickets else [],
            },
            {
                "kind": StepKind.RECOMMENDATION.value,
                "content": "Follow runbook: "
                + (knowledge[0][1][:200] if knowledge else "escalate"),
                "citations": [knowledge[0][0]] if knowledge else [],
            },
        ]
        if self.hallucinate:
            steps.append(
                {
                    "kind": StepKind.RECOMMENDATION.value,
                    "content": "Restart the mainframe bridge INC9999999",
                    "citations": [],
                }
            )
        text = json.dumps({"steps": steps})
        usage = TokenUsage(input_tokens=len(prompt) // 4, output_tokens=len(text) // 4)
        return LLMResponse(text=text, usage=usage, model_id=model.model_id)
