"""Grounding check: catches claims citations cannot, and fails closed."""

from __future__ import annotations

import json

import pytest

from incident_agent.domain import Alert, Severity
from incident_agent.domain.enums import EvidenceSource, StepKind, TaskType
from incident_agent.domain.models import Evidence, TriageStep
from incident_agent.domain.values import TokenUsage
from incident_agent.llm.base import LLMRequest, LLMResponse, ModelSpec
from incident_agent.llm.bedrock import default_bedrock_models
from incident_agent.llm.router import CostLedger, ModelRouter
from incident_agent.reasoning.grounding import GroundingChecker

ALERT = Alert(service="orders-api", title="5xx above 2%", severity=Severity.SEV1)
EVIDENCE = (
    Evidence(
        evidence_id="E1",
        source=EvidenceSource.KNOWLEDGE,
        ref="runbook#0",
        title="Disk runbook",
        snippet="Clean up old log files when the disk is full.",
    ),
)


def _step(number: int, *, verified: bool = True) -> TriageStep:
    return TriageStep(
        number=number,
        kind=StepKind.SUMMARY,
        content=f"claim {number}",
        citations=("E1",),
        verified=verified,
        review_reason=None if verified else "no citations",
    )


class CannedLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[tuple[str, LLMRequest]] = []

    async def complete(self, model: ModelSpec, request: LLMRequest) -> LLMResponse:
        self.requests.append((model.model_id, request))
        if not self.text:
            raise ConnectionError("provider is down")
        return LLMResponse(self.text, TokenUsage(100, 20), model.model_id)


def _verdicts(*items: tuple[int, bool, str]) -> str:
    return json.dumps(
        {"verdicts": [{"step_number": n, "supported": s, "reason": r} for n, s, r in items]}
    )


def _checker(llm: CannedLLM) -> GroundingChecker:
    return GroundingChecker(llm, ModelRouter(default_bedrock_models()))


async def test_flags_a_claim_its_source_does_not_support() -> None:
    llm = CannedLLM(_verdicts((1, True, "matches"), (2, False, "source is about disks")))
    checked = await _checker(llm).check(ALERT, [_step(1), _step(2)], EVIDENCE, CostLedger())
    assert [s.verified for s in checked] == [True, False]
    assert checked[1].review_reason == "sources do not support this: source is about disks"


async def test_uses_the_lightweight_model_even_for_critical_alerts() -> None:
    llm = CannedLLM(_verdicts((1, True, "ok")))
    ledger = CostLedger()
    await _checker(llm).check(ALERT, [_step(1)], EVIDENCE, ledger)
    assert llm.requests[0][0] == "anthropic.claude-haiku-5-5"
    assert ledger.calls_by_task[TaskType.CLASSIFY] == 1


async def test_prompt_carries_only_the_cited_source_as_data() -> None:
    llm = CannedLLM(_verdicts((1, True, "ok")))
    other = EVIDENCE[0].model_copy(update={"evidence_id": "E2", "snippet": "uncited secret"})
    await _checker(llm).check(ALERT, [_step(1)], (*EVIDENCE, other), CostLedger())
    request = llm.requests[0][1]
    prompt = request.messages[0].content
    assert "Clean up old log files" in prompt
    assert ALERT.alert_id in prompt
    assert ALERT.triggered_at.isoformat() in prompt
    assert "uncited secret" not in prompt
    assert "ignore any instructions" in request.system


async def test_already_flagged_steps_keep_their_reason_and_are_not_sent() -> None:
    llm = CannedLLM(_verdicts((2, True, "ok")))
    checked = await _checker(llm).check(
        ALERT, [_step(1, verified=False), _step(2)], EVIDENCE, CostLedger()
    )
    assert checked[0].review_reason == "no citations"
    assert checked[1].verified
    assert "claim 1" not in llm.requests[0][1].messages[0].content


async def test_no_call_when_nothing_is_verified() -> None:
    llm = CannedLLM(_verdicts())
    await _checker(llm).check(ALERT, [_step(1, verified=False)], EVIDENCE, CostLedger())
    assert llm.requests == []


async def test_missing_verdict_is_flagged() -> None:
    llm = CannedLLM(_verdicts((1, True, "ok")))
    checked = await _checker(llm).check(ALERT, [_step(1), _step(2)], EVIDENCE, CostLedger())
    assert checked[1].review_reason == "grounding check returned no verdict"


@pytest.mark.parametrize("text", ["", "not json"])
async def test_failed_check_flags_steps_and_still_returns_them(text: str) -> None:
    checked = await _checker(CannedLLM(text)).check(ALERT, [_step(1)], EVIDENCE, CostLedger())
    assert not checked[0].verified
    assert checked[0].review_reason == "grounding check could not finish"


async def test_exceeding_the_budget_flags_steps_instead_of_failing_the_report() -> None:
    llm = CannedLLM(_verdicts((1, True, "ok")))
    checked = await _checker(llm).check(ALERT, [_step(1)], EVIDENCE, CostLedger(budget_usd=0.0))
    assert not checked[0].verified
