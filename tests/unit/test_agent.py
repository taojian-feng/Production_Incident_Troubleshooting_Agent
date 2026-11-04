from __future__ import annotations

import io
import json

import pytest

from incident_agent.app import Components, build_local_components
from incident_agent.core.errors import ToolAccessDeniedError
from incident_agent.domain import (
    AgentState,
    Alert,
    EvidenceSource,
    ReportStatus,
    RetrievalPlan,
    Severity,
)
from incident_agent.llm.local import ScriptedLLM
from incident_agent.observability.metrics import MetricsLogger, record_triage
from incident_agent.orchestration.graph import END, StateGraph
from incident_agent.retrieval.hybrid import plan_retrieval
from incident_agent.tools.base import RawEvidence


class TestEndToEnd:
    async def test_grounded_report(self, local: Components, payments_alert: Alert) -> None:
        report, state = await local.agent.run(payments_alert)
        assert report.status is ReportStatus.VALIDATED
        sources = {e.source for e in report.evidence}
        assert sources == set(EvidenceSource)
        assert "INC0041877" in {e.ref for e in report.evidence}
        assert [r.node for r in state.history] == ["plan", "gather", "reason"]
        assert report.cost.usd > 0
        assert set(report.cost.by_model) == {
            "anthropic.claude-opus-5-5",  # writes the report for this SEV1
            "anthropic.claude-haiku-5-5",  # checks that its sources support it
        }
        assert sum(report.cost.by_model.values()) == pytest.approx(report.cost.usd)

    async def test_hallucinated_step_flags_report(self, payments_alert: Alert) -> None:
        components = await build_local_components(llm=ScriptedLLM(hallucinate=True))
        report, _ = await components.agent.run(payments_alert)
        assert report.status is ReportStatus.NEEDS_REVIEW
        flagged = [s for s in report.steps if not s.verified]
        assert len(flagged) == 1
        assert flagged[0].review_reason == "no citations"

    async def test_grounding_check_flags_a_step_and_routes_the_report_to_review(
        self, payments_alert: Alert
    ) -> None:
        components = await build_local_components(llm=ScriptedLLM(unsupported_steps=frozenset({1})))
        report, _ = await components.agent.run(payments_alert)
        assert report.status == ReportStatus.NEEDS_REVIEW
        reason = report.steps[0].review_reason
        assert reason == "sources do not support this: scripted verdict"
        assert all(step.verified for step in report.steps[1:])

    async def test_tool_metrics_count_only_the_current_run(
        self, local: Components, payments_alert: Alert
    ) -> None:
        counts = []
        for _ in range(2):
            report, _state = await local.agent.run(payments_alert)
            stream = io.StringIO()
            record_triage(MetricsLogger({"Service": "s"}, stream=stream), report, local.registry)
            counts.append(json.loads(stream.getvalue())["ToolCallSuccess"])
        assert counts == [3, 3]

    async def test_tool_failure_degrades_gracefully(
        self, local: Components, payments_alert: Alert
    ) -> None:
        class Broken:
            domain = EvidenceSource.LOGS

            async def gather(self, alert: Alert, plan: RetrievalPlan) -> list[RawEvidence]:
                raise ConnectionError("logs down")

        local.registry.servers = {**local.registry.servers, EvidenceSource.LOGS: Broken()}
        report, _ = await local.agent.run(payments_alert)
        assert EvidenceSource.LOGS not in {e.source for e in report.evidence}
        assert local.registry.failures[EvidenceSource.LOGS] == 1
        assert report.steps

    async def test_unknown_service_without_evidence(self, local: Components) -> None:
        local.registry.role = "nobody"
        report, state = await local.agent.run(Alert(service="ghost", title="???"))
        assert [r.node for r in state.history] == ["plan", "gather", "no_evidence"]
        assert report.status is ReportStatus.NEEDS_REVIEW

    async def test_access_policy_is_enforced(
        self, local: Components, payments_alert: Alert
    ) -> None:
        local.registry.role = "analyst-readonly"
        plan = plan_retrieval(payments_alert)
        with pytest.raises(ToolAccessDeniedError):
            await local.registry.call(EvidenceSource.LOGS, payments_alert, plan)
        report, _ = await local.agent.run(payments_alert)
        assert EvidenceSource.LOGS not in {e.source for e in report.evidence}

    async def test_metrics_emitted_as_emf(self, local: Components, payments_alert: Alert) -> None:
        report, _ = await local.agent.run(payments_alert)
        stream = io.StringIO()
        record_triage(
            MetricsLogger({"Service": "payments-api"}, stream=stream), report, local.registry
        )
        doc = json.loads(stream.getvalue())
        names = {m["Name"] for m in doc["_aws"]["CloudWatchMetrics"][0]["Metrics"]}
        assert {"TriageLatency", "CostMicroUSD", "ToolCallSuccess", "NeedsReview"} <= names
        assert doc["Service"] == "payments-api"
        assert doc["report_id"] == report.report_id


class TestGraph:
    async def test_conditional_routing_and_audit(self) -> None:
        async def a(state: AgentState) -> None:
            state.notes["a"] = "1"

        async def b(state: AgentState) -> None:
            state.notes["b"] = "1"

        graph = (
            StateGraph(entry="a")
            .add_node("a", a)
            .add_node("b", b)
            .add_conditional_edge("a", lambda s: "b" if s.notes.get("a") else END)
        )
        state = await graph.run(AgentState(alert=Alert(service="s", title="t")))
        assert [r.node for r in state.history] == ["a", "b"]
        assert state.history[0].snapshot["notes"] == {"a": "1"}  # snapshot taken after a

    async def test_cycle_guard_and_duplicate_names(self) -> None:
        async def loop(state: AgentState) -> None:
            return None

        graph = StateGraph(entry="x", max_steps=3).add_node("x", loop).add_edge("x", "x")
        with pytest.raises(RuntimeError, match="exceeded"):
            await graph.run(AgentState(alert=Alert(service="s", title="t", severity=Severity.SEV4)))
        with pytest.raises(ValueError, match="duplicate"):
            graph.add_node("x", loop)
