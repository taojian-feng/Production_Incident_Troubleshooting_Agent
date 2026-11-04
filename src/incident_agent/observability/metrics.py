"""CloudWatch Embedded Metric Format (EMF) emitter.

Printing one EMF JSON line to stdout in Lambda creates CloudWatch metrics
asynchronously - no ``PutMetricData`` API calls, no added latency, no extra
IAM permissions. The same line is a structured log you can query in Logs
Insights (e.g. join cost to report id).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TextIO

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from incident_agent.domain.models import TriageReport
    from incident_agent.tools.base import ToolRegistry

Unit = Literal["Count", "Milliseconds", "None"]
NAMESPACE = "IncidentAgent"


@dataclass(slots=True)
class MetricsLogger:
    """Collects metrics + properties, then flushes one EMF document."""

    dimensions: Mapping[str, str]
    namespace: str = NAMESPACE
    stream: TextIO = field(default_factory=lambda: sys.stdout)
    clock: Callable[[], float] = time.time
    _metrics: dict[str, tuple[float, Unit]] = field(default_factory=dict)
    _properties: dict[str, object] = field(default_factory=dict)

    def put(self, name: str, value: float, unit: Unit = "None") -> None:
        """Set a metric value (last write wins within one flush)."""
        self._metrics[name] = (value, unit)

    def prop(self, key: str, value: object) -> None:
        """High-cardinality context (report id, model id) as a log property, not a dimension."""
        self._properties[key] = value

    def flush(self) -> dict[str, object]:
        """Write the EMF line and reset. Returns the document (handy in tests)."""
        doc: dict[str, object] = {
            "_aws": {
                "Timestamp": int(self.clock() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": self.namespace,
                        "Dimensions": [list(self.dimensions)],
                        "Metrics": [{"Name": n, "Unit": u} for n, (_, u) in self._metrics.items()],
                    }
                ],
            },
            **self.dimensions,
            **{name: value for name, (value, _) in self._metrics.items()},
            **self._properties,
        }
        self.stream.write(json.dumps(doc, default=str) + "\n")
        self._metrics.clear()
        self._properties.clear()
        return doc


def record_triage(metrics: MetricsLogger, report: TriageReport, tools: ToolRegistry) -> None:
    """Standard metric set for one triage run (drives the CloudWatch dashboard)."""
    metrics.put("TriageLatency", report.latency_ms, "Milliseconds")
    metrics.put("InputTokens", report.cost.input_tokens, "Count")
    metrics.put("OutputTokens", report.cost.output_tokens, "Count")
    metrics.put("CostMicroUSD", round(report.cost.usd * 1_000_000), "Count")
    metrics.put("EvidenceItems", len(report.evidence), "Count")
    metrics.put("UnverifiedSteps", sum(not s.verified for s in report.steps), "Count")
    metrics.put("NeedsReview", int(report.status == "needs_review"), "Count")
    successes, failures = tools.drain_counts()
    metrics.put("ToolCallSuccess", successes, "Count")
    metrics.put("ToolCallFailure", failures, "Count")
    metrics.prop("report_id", report.report_id)
    metrics.prop("alert_id", report.alert.alert_id)
    metrics.prop("retrieval_mode", report.retrieval_mode.value)
    metrics.prop("cost_by_model", report.cost.by_model)
    metrics.flush()
