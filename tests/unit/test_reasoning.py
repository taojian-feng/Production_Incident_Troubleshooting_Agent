from __future__ import annotations

import json

import pytest

from incident_agent.core.errors import BudgetExceededError, LLMOutputError
from incident_agent.domain import (
    Alert,
    Evidence,
    EvidenceSource,
    LLMTriageStep,
    ModelTier,
    Severity,
    StepKind,
    TaskType,
    TokenUsage,
)
from incident_agent.llm.base import LLMRequest, LLMResponse, ModelSpec
from incident_agent.llm.bedrock import default_bedrock_models
from incident_agent.llm.router import CostLedger, ModelRouter
from incident_agent.reasoning.citations import validate_steps
from incident_agent.reasoning.context import ContextBudget, render_evidence, select_evidence
from incident_agent.reasoning.engine import ReasoningEngine
from incident_agent.tools.base import RawEvidence

EVIDENCE = [
    Evidence(
        evidence_id="E1",
        source=EvidenceSource.TICKETS,
        ref="INC0041877",
        title="INC0041877 pool exhausted",
        snippet="resolution: rolled back",
    ),
    Evidence(
        evidence_id="E2",
        source=EvidenceSource.LOGS,
        ref="log:1",
        title="25x",
        snippet="ERROR ERR_GATEWAY_TIMEOUT",
    ),
]


def step(kind: StepKind, content: str, *cites: str) -> LLMTriageStep:
    return LLMTriageStep(kind=kind, content=content, citations=list(cites))


class TestCitations:
    def test_grounded_steps_verify(self) -> None:
        steps = validate_steps(
            [
                step(StepKind.RECOMMENDATION, "Roll back as in INC0041877", "E1"),
                step(StepKind.SUMMARY, "ERR_GATEWAY_TIMEOUT spike", "E2"),
                step(StepKind.PRECEDENT, "INC0041877 matches", "E1", "E1"),
            ],
            EVIDENCE,
        )
        assert [s.kind for s in steps] == [
            StepKind.SUMMARY,
            StepKind.PRECEDENT,
            StepKind.RECOMMENDATION,
        ]
        assert [s.number for s in steps] == [1, 2, 3]
        assert all(s.verified for s in steps)
        assert steps[1].citations == ("E1",)  # duplicate citation collapsed

    @pytest.mark.parametrize(
        ("bad", "reason"),
        [
            (step(StepKind.SUMMARY, "something"), "no citations"),
            (step(StepKind.SUMMARY, "x", "E9"), "unknown evidence"),
            (step(StepKind.SUMMARY, "like INC0099999", "E1"), "INC0099999"),
            (step(StepKind.SUMMARY, "ORA-00060 seen", "E2"), "ORA-00060"),
        ],
    )
    def test_ungrounded_steps_need_review(self, bad: LLMTriageStep, reason: str) -> None:
        (first, *_rest) = validate_steps([bad], EVIDENCE)
        assert not first.verified
        assert first.review_reason is not None
        assert reason in first.review_reason

    def test_missing_steps_are_added_unverified(self) -> None:
        steps = validate_steps([step(StepKind.SUMMARY, "ERR_GATEWAY_TIMEOUT", "E2")], EVIDENCE)
        assert [s.kind for s in steps] == [
            StepKind.SUMMARY,
            StepKind.PRECEDENT,
            StepKind.RECOMMENDATION,
        ]
        assert [s.verified for s in steps] == [True, False, False]


class TestContext:
    def test_round_robin_per_source_and_char_budget(self) -> None:
        raw = [
            RawEvidence(EvidenceSource.LOGS, f"log:{i}", "t", "x" * 10, score=i) for i in range(5)
        ] + [RawEvidence(EvidenceSource.TICKETS, "INC0000001", "t", "y" * 10, score=0.1)]
        selected = select_evidence(raw, ContextBudget(per_source=2, max_chars=1000))
        assert [e.ref for e in selected] == ["log:4", "INC0000001", "log:3"]
        assert [e.evidence_id for e in selected] == ["E1", "E2", "E3"]
        tight = select_evidence(raw, ContextBudget(per_source=2, max_chars=25))
        assert len(tight) == 2
        assert render_evidence(selected).splitlines()[0].startswith("[E1] (logs)")


class TestRouterAndLedger:
    def test_routes_by_task_and_escalates_critical_reasoning(self) -> None:
        router = ModelRouter(default_bedrock_models())
        assert router.route(TaskType.CLASSIFY).tier is ModelTier.LIGHT
        assert router.route(TaskType.REASON, Severity.SEV3).tier is ModelTier.STANDARD
        assert router.route(TaskType.REASON, Severity.SEV1).tier is ModelTier.HEAVY
        assert router.route(TaskType.REASON).model_id.startswith("anthropic.claude-")

    def test_router_requires_all_tiers(self) -> None:
        models = default_bedrock_models()
        del models[ModelTier.HEAVY]
        with pytest.raises(ValueError, match="heavy"):
            ModelRouter(models)

    def test_overrides(self) -> None:
        models = default_bedrock_models({ModelTier.LIGHT: "us.anthropic.claude-haiku-5-5"})
        assert models[ModelTier.LIGHT].model_id == "us.anthropic.claude-haiku-5-5"

    def test_ledger_tracks_and_enforces_budget(self) -> None:
        spec = default_bedrock_models()[ModelTier.HEAVY]
        ledger = CostLedger(budget_usd=0.05)
        ledger.record(TaskType.REASON, spec, TokenUsage(1_000, 500))
        assert ledger.summary().usd == pytest.approx((1_000 * 4 + 500 * 20) / 1e6)
        assert ledger.calls_by_task[TaskType.REASON] == 1
        with pytest.raises(BudgetExceededError):
            ledger.record(TaskType.REASON, spec, TokenUsage(10_000, 2_000))


class SequenceLLM:
    """Returns canned texts in order; records requests."""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.requests: list[LLMRequest] = []

    async def complete(self, model: ModelSpec, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(self.texts.pop(0), TokenUsage(100, 50), model.model_id)


GOOD = json.dumps(
    {
        "steps": [
            {"kind": "summary", "content": "ERR_GATEWAY_TIMEOUT spike", "citations": ["E2"]},
            {"kind": "precedent", "content": "INC0041877", "citations": ["E1"]},
            {"kind": "recommendation", "content": "Roll back", "citations": ["E1"]},
        ]
    }
)


class TestEngine:
    @pytest.fixture
    def alert(self) -> Alert:
        return Alert(service="payments-api", title="504s", severity=Severity.SEV1)

    async def test_happy_path_uses_schema_and_heavy_model(self, alert: Alert) -> None:
        llm = SequenceLLM(GOOD)
        engine = ReasoningEngine(llm, ModelRouter(default_bedrock_models()))
        ledger = CostLedger()
        steps = await engine.triage(alert, EVIDENCE, ledger)
        assert all(s.verified for s in steps)
        request = llm.requests[0]
        assert request.json_schema is not None
        assert request.json_schema["additionalProperties"] is False
        assert "[E1] (tickets)" in request.messages[0].content
        assert set(ledger.tokens_by_model) == {"anthropic.claude-opus-5-5"}

    async def test_repairs_invalid_json_once(self, alert: Alert) -> None:
        llm = SequenceLLM('{"steps": "oops"}', GOOD)
        engine = ReasoningEngine(llm, ModelRouter(default_bedrock_models()))
        ledger = CostLedger()
        steps = await engine.triage(alert, EVIDENCE, ledger)
        assert len(steps) == 3
        assert len(llm.requests[1].messages) == 3
        assert "did not match the schema" in llm.requests[1].messages[-1].content
        assert sum(ledger.calls_by_task.values()) == 2

    async def test_raises_after_second_invalid_output(self, alert: Alert) -> None:
        engine = ReasoningEngine(SequenceLLM("nope", "{}"), ModelRouter(default_bedrock_models()))
        with pytest.raises(LLMOutputError):
            await engine.triage(alert, EVIDENCE, CostLedger())
