"""Task-based model routing and FinOps cost tracking (design section 4.1).

Cheap, fast models handle classification and extraction; the expensive model
is reserved for multi-step reasoning on severe incidents. Every call is
recorded in a :class:`CostLedger` that enforces a per-run dollar budget.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from incident_agent.core.errors import BudgetExceededError
from incident_agent.domain.enums import ModelTier, Severity, TaskType
from incident_agent.domain.models import CostSummary

if TYPE_CHECKING:
    from collections.abc import Mapping

    from incident_agent.domain.values import TokenUsage
    from incident_agent.llm.base import ModelSpec

# Base routing table, wrapped in MappingProxyType so it is read-only at runtime.
TASK_TIERS: Mapping[TaskType, ModelTier] = MappingProxyType(
    {
        TaskType.CLASSIFY: ModelTier.LIGHT,
        TaskType.EXTRACT: ModelTier.LIGHT,
        TaskType.SUMMARIZE: ModelTier.STANDARD,
        TaskType.REASON: ModelTier.STANDARD,
    }
)


@dataclass(frozen=True, slots=True)
class ModelRouter:
    """Pick a :class:`ModelSpec` for a task, escalating for critical incidents."""

    models: Mapping[ModelTier, ModelSpec]

    def __post_init__(self) -> None:
        missing = set(ModelTier) - set(self.models)
        if missing:
            msg = f"router is missing tiers: {sorted(missing)}"
            raise ValueError(msg)

    def route(self, task: TaskType, severity: Severity = Severity.SEV3) -> ModelSpec:
        """Return the model for ``task``; SEV1/SEV2 reasoning goes to the HEAVY tier."""
        tier = TASK_TIERS[task]
        if task is TaskType.REASON and severity.is_critical:
            tier = ModelTier.HEAVY
        return self.models[tier]


@dataclass(slots=True)
class CostLedger:
    """Accumulates tokens and dollars per model and per task for one triage run."""

    budget_usd: float = 1.0
    tokens_by_model: Counter[str] = field(default_factory=Counter)
    usd_by_model: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls_by_task: Counter[TaskType] = field(default_factory=Counter)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_usd(self) -> float:
        """Spend so far."""
        return sum(self.usd_by_model.values())

    def record(self, task: TaskType, model: ModelSpec, usage: TokenUsage) -> float:
        """Record one call; raise :class:`BudgetExceededError` past the budget."""
        cost = model.cost(usage)
        self.tokens_by_model[model.model_id] += usage.total
        self.usd_by_model[model.model_id] += cost
        self.calls_by_task[task] += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        if self.total_usd > self.budget_usd:
            msg = f"triage spend ${self.total_usd:.4f} exceeded budget ${self.budget_usd:.2f}"
            raise BudgetExceededError(msg)
        return cost

    def summary(self) -> CostSummary:
        """Immutable roll-up for the report."""
        return CostSummary(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            usd=round(self.total_usd, 6),
            by_model={k: round(v, 6) for k, v in self.usd_by_model.items()},
        )
