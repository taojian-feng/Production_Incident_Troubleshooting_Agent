"""A small, typed state-graph runtime (LangGraph-style, no framework dependency).

Nodes are async functions ``(AgentState) -> None`` that enrich the state. Edges
are either fixed or chosen by a router function. After every node the runtime
appends a deep-copied snapshot to ``state.history`` so engineers can audit the
exact path the agent took (design section 3).

We keep this in-house because the graph is small and the audit/snapshot
semantics are a compliance requirement we want to own; the node functions
port 1:1 to LangGraph if the workflow grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from incident_agent.core.timing import Stopwatch
from incident_agent.domain.state import NodeRecord

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from incident_agent.domain.state import AgentState

    Node = Callable[[AgentState], Awaitable[None]]
    Router = Callable[[AgentState], str]

END: Final = "__end__"


@dataclass(slots=True)
class StateGraph:
    """Directed graph of async nodes with optional conditional edges."""

    entry: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, str] = field(default_factory=dict)
    routers: dict[str, Router] = field(default_factory=dict)
    max_steps: int = 25

    def add_node(self, name: str, fn: Node) -> StateGraph:
        """Register a node (fluent)."""
        if name in self.nodes or name == END:
            msg = f"duplicate or reserved node name {name!r}"
            raise ValueError(msg)
        self.nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> StateGraph:
        """Unconditional transition."""
        self.edges[src] = dst
        return self

    def add_conditional_edge(self, src: str, router: Router) -> StateGraph:
        """Transition decided at runtime by ``router(state) -> node name``."""
        self.routers[src] = router
        return self

    def _next(self, current: str, state: AgentState) -> str:
        if current in self.routers:
            return self.routers[current](state)
        return self.edges.get(current, END)

    async def run(self, state: AgentState) -> AgentState:
        """Execute from ``entry`` until ``END``; ``max_steps`` guards against loops."""
        current = self.entry
        for _ in range(self.max_steps):
            if current == END:
                return state
            node = self.nodes[current]
            with Stopwatch() as watch:
                await node(state)
            state.history.append(NodeRecord(current, round(watch.elapsed_ms, 2), state.snapshot()))
            current = self._next(current, state)
        msg = f"graph exceeded {self.max_steps} steps (cycle?)"
        raise RuntimeError(msg)
