"""The orchestration agent: plan -> gather (parallel) -> reason -> report.

::

    plan ──> gather ──(evidence?)──> reason ──> END
                          └── no ──> no_evidence ──> END
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from incident_agent.core.timing import Stopwatch
from incident_agent.domain.enums import EvidenceSource, RetrievalMode, StepKind
from incident_agent.domain.models import TriageReport, TriageStep
from incident_agent.domain.state import AgentState
from incident_agent.llm.router import CostLedger
from incident_agent.orchestration.graph import END, StateGraph
from incident_agent.reasoning.context import ContextBudget, select_evidence
from incident_agent.retrieval.hybrid import plan_retrieval

if TYPE_CHECKING:
    from incident_agent.domain.models import Alert
    from incident_agent.reasoning.engine import ReasoningEngine
    from incident_agent.tools.base import RawEvidence, ToolRegistry


_LOG_SIGNALS: Final = 3
_LOG_SIGNAL_CHARS: Final = 200


def _with_log_signals(alert: Alert, log_evidence: list[RawEvidence]) -> Alert:
    """Return a copy of ``alert`` for searching, with its top log errors added.

    Used only to search. The model still sees the original alert and the logs as evidence.
    """
    signals = [e.snippet.partition(" ")[2][:_LOG_SIGNAL_CHARS] for e in log_evidence]
    if not signals:
        return alert
    text = "\n".join([alert.description, *signals[:_LOG_SIGNALS]]).strip()
    return alert.model_copy(update={"description": text})


@dataclass(slots=True)
class TriageAgent:
    """Composes tool servers, the reasoning engine and FinOps into one run."""

    tools: ToolRegistry
    reasoner: ReasoningEngine
    budget: ContextBudget = ContextBudget()
    max_usd_per_run: float = 0.75

    async def run(self, alert: Alert) -> tuple[TriageReport, AgentState]:
        """Triage one alert; returns the report and the full audited state."""
        ledger = CostLedger(budget_usd=self.max_usd_per_run)
        graph = self._build_graph(ledger)
        with Stopwatch() as watch:
            state = await graph.run(AgentState(alert=alert))
        plan = state.plan
        report = TriageReport(
            alert=alert,
            retrieval_mode=plan.mode if plan else RetrievalMode.HYBRID,
            steps=tuple(state.steps),
            evidence=tuple(state.evidence),
            cost=ledger.summary(),
            latency_ms=round(watch.elapsed_ms, 1),
        )
        return report, state

    def _build_graph(self, ledger: CostLedger) -> StateGraph:
        async def plan(state: AgentState) -> None:
            state.plan = plan_retrieval(state.alert)

        async def gather(state: AgentState) -> None:
            assert state.plan is not None  # noqa: S101 - graph ordering invariant
            allowed = self.tools.allowed()
            # Logs run first: an alarm's own text is often generic, and the errors it
            # points to are the best search terms for past incidents and runbooks.
            logs_branch = state.fork()
            log_evidence = (
                await self._gather_domain(EvidenceSource.LOGS, logs_branch)
                if EvidenceSource.LOGS in allowed
                else []
            )
            search_alert = _with_log_signals(state.alert, log_evidence)
            state.plan = plan_retrieval(search_alert)

            domains = sorted(allowed - {EvidenceSource.LOGS}, key=lambda d: d.value)
            # Each branch works on its own deep-copied fork so concurrent
            # branches can never mutate shared lists (see domain.state).
            branches = [state.fork() for _ in domains]
            for branch in branches:
                branch.alert = search_alert
            results = await asyncio.gather(
                *(self._gather_domain(d, b) for d, b in zip(domains, branches, strict=True))
            )
            raw: list[RawEvidence] = [*log_evidence, *(item for batch in results for item in batch)]
            state.evidence = select_evidence(raw, self.budget)
            # Collects per-branch errors back into the parent.
            state.merge_evidence(logs_branch, *branches)
            state.notes["tool_successes"] = str(dict(self.tools.successes))

        async def reason(state: AgentState) -> None:
            state.steps = await self.reasoner.triage(state.alert, state.evidence, ledger)

        async def no_evidence(state: AgentState) -> None:
            state.steps = [
                TriageStep(
                    number=1,
                    kind=StepKind.SUMMARY,
                    content="No evidence was found in logs, tickets or runbooks; "
                    "escalate to the service owner.",
                    citations=(),
                    verified=False,
                    review_reason="no evidence retrieved",
                )
            ]

        return (
            StateGraph(entry="plan")
            .add_node("plan", plan)
            .add_node("gather", gather)
            .add_node("reason", reason)
            .add_node("no_evidence", no_evidence)
            .add_edge("plan", "gather")
            .add_conditional_edge("gather", lambda s: "reason" if s.evidence else "no_evidence")
            .add_edge("reason", END)
            .add_edge("no_evidence", END)
        )

    async def _gather_domain(self, domain: EvidenceSource, branch: AgentState) -> list[RawEvidence]:
        assert branch.plan is not None  # noqa: S101
        try:
            return await self.tools.call(domain, branch.alert, branch.plan)
        except Exception as exc:  # access denied etc.: record on the branch, keep going
            branch.errors.append(f"{domain.value}: {exc}")
            return []
