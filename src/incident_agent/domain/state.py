"""Mutable agent state passed through the orchestration graph.

This is the one deliberately *mutable* structure in the core. Nodes enrich it
step by step. Two copy semantics matter here and are easy to get wrong:

* ``copy.copy(state)`` (shallow) creates a new ``AgentState`` whose ``evidence``
  list, ``notes`` dict, etc. are the *same objects* as the original. If two
  parallel branches each get a shallow copy and ``append`` evidence, they
  corrupt each other and the parent.
* ``copy.deepcopy(state)`` recursively copies containers, so each branch owns
  its own lists. ``deepcopy`` is not free (it also re-allocates frozen pydantic
  leaves), so we pay for it only at fan-out points (``fork``) and for the
  audit trail (``snapshot``), never per log line.

See ``tests/unit/test_state.py`` for an executable demonstration of the
shallow-copy aliasing bug this design avoids.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from incident_agent.domain.enums import RetrievalMode
    from incident_agent.domain.models import Alert, Evidence, TriageStep


@dataclass(slots=True)
class RetrievalPlan:
    """Output of the query planner: what to search and how."""

    mode: RetrievalMode
    query: str
    identifiers: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ()
    ticket_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class NodeRecord:
    """Audit entry written after every graph node."""

    node: str
    duration_ms: float
    snapshot: dict[str, Any]


@dataclass(slots=True)
class AgentState:
    """Everything the agent knows about one alert while triaging it."""

    alert: Alert
    plan: RetrievalPlan | None = None
    evidence: list[Evidence] = field(default_factory=list)
    steps: list[TriageStep] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    history: list[NodeRecord] = field(default_factory=list)

    def fork(self) -> AgentState:
        """Independent copy for a parallel branch (deep, but without history)."""
        history, self.history = self.history, []
        try:
            return copy.deepcopy(self)
        finally:
            self.history = history

    def merge_evidence(self, *branches: AgentState) -> None:
        """Merge evidence found by parallel branches, de-duplicated by source ref.

        Uses a ``set`` of already-seen refs for O(1) membership checks, keeping
        first-seen order (``dict``/list order is insertion order).
        """
        seen: set[str] = {e.ref for e in self.evidence}
        for branch in branches:
            for item in branch.evidence:
                if item.ref not in seen:
                    seen.add(item.ref)
                    self.evidence.append(item)
            self.errors.extend(branch.errors)

    def snapshot(self) -> dict[str, Any]:
        """A JSON-friendly, fully detached view of the state for the audit log."""
        return copy.deepcopy(
            {
                "plan": None
                if self.plan is None
                else {"mode": self.plan.mode.value, "query": self.plan.query},
                "evidence_ids": [e.evidence_id for e in self.evidence],
                "steps": len(self.steps),
                "notes": self.notes,
                "errors": self.errors,
            }
        )
