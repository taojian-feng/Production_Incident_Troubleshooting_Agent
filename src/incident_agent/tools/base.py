"""Domain-governed tool servers (design section 2.3).

Each business domain (logs, tickets, knowledge) is its own tool server with
its own IAM role scope, instead of one monolithic server. Inside the Lambda
runtime the agent calls them in-process through :class:`ToolRegistry`, which
enforces a role -> domain access policy and records per-tool success metrics.
``tools.mcp_server`` exposes the very same objects over MCP for external
clients.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from incident_agent.core.errors import ToolAccessDeniedError
from incident_agent.core.timing import Stopwatch
from incident_agent.domain.enums import EvidenceSource

if TYPE_CHECKING:
    from collections.abc import Mapping

    from incident_agent.domain.models import Alert
    from incident_agent.domain.state import RetrievalPlan

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RawEvidence:
    """Evidence before it is numbered ``E1..En`` for the prompt."""

    source: EvidenceSource
    ref: str
    title: str
    snippet: str
    score: float = 0.0
    uri: str | None = None


class ToolServer(Protocol):
    """One business domain's read-only tool surface."""

    @property
    def domain(self) -> EvidenceSource:
        """The domain this server governs."""
        ...

    async def gather(self, alert: Alert, plan: RetrievalPlan) -> list[RawEvidence]:
        """Return evidence relevant to ``alert`` following ``plan``."""
        ...


# Least privilege: which agent roles may call which domains.
DEFAULT_POLICY: Mapping[str, frozenset[EvidenceSource]] = {
    "triage": frozenset(EvidenceSource),
    "analyst-readonly": frozenset({EvidenceSource.KNOWLEDGE, EvidenceSource.TICKETS}),
}


@dataclass(slots=True)
class ToolRegistry:
    """Routes calls to domain servers, enforcing the access policy."""

    servers: Mapping[EvidenceSource, ToolServer]
    role: str = "triage"
    policy: Mapping[str, frozenset[EvidenceSource]] = field(default_factory=lambda: DEFAULT_POLICY)
    successes: Counter[EvidenceSource] = field(default_factory=Counter)
    failures: Counter[EvidenceSource] = field(default_factory=Counter)
    latency_ms: dict[EvidenceSource, float] = field(default_factory=dict)

    def allowed(self) -> frozenset[EvidenceSource]:
        """Domains this registry's role may call."""
        return self.policy.get(self.role, frozenset())

    async def call(
        self, domain: EvidenceSource, alert: Alert, plan: RetrievalPlan
    ) -> list[RawEvidence]:
        """Call one domain. Tool failures degrade to "no evidence", never crash triage."""
        if domain not in self.allowed():
            msg = f"role {self.role!r} may not call domain {domain.value!r}"
            raise ToolAccessDeniedError(msg)
        server = self.servers[domain]
        with Stopwatch() as watch:
            try:
                result = await server.gather(alert, plan)
            except Exception:
                logger.exception("tool domain %s failed", domain.value)
                self.failures[domain] += 1
                result = []
            else:
                self.successes[domain] += 1
        self.latency_ms[domain] = watch.elapsed_ms
        return result
