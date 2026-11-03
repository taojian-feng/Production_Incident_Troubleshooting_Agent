"""Ground-truth benchmarking and the feedback -> eval-set loop (design section 4.2).

Each :class:`GroundTruthCase` pairs an alert with what a correct triage must
contain (the right precedent tickets, key remediation phrases). The benchmark
runs cases concurrently (bounded) and reports retrieval recall, keyword
coverage, citation validity, cost and latency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from statistics import fmean
from typing import TYPE_CHECKING

from pydantic import Field, TypeAdapter

from incident_agent.core.rate_limit import gather_bounded
from incident_agent.domain.models import Alert, FrozenModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from incident_agent.domain.models import Feedback, TriageReport
    from incident_agent.orchestration.agent import TriageAgent


class GroundTruthCase(FrozenModel):
    """A verified incident resolution used as a test case."""

    case_id: str
    alert: Alert
    expected_precedents: tuple[str, ...] = ()
    expected_keywords: tuple[str, ...] = Field(default=(), description="Lower-case phrases.")


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Scores for one case."""

    case_id: str
    precedent_recall: float
    keyword_coverage: float
    all_steps_verified: bool
    usd: float
    latency_ms: float


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Aggregate scores."""

    cases: tuple[CaseResult, ...]

    @property
    def precedent_recall(self) -> float:
        """Mean recall of expected precedent ticket ids in the evidence."""
        return fmean(c.precedent_recall for c in self.cases) if self.cases else 0.0

    @property
    def keyword_coverage(self) -> float:
        """Mean share of expected phrases present in the recommendations."""
        return fmean(c.keyword_coverage for c in self.cases) if self.cases else 0.0

    @property
    def verified_rate(self) -> float:
        """Share of cases whose every step passed citation validation."""
        return fmean(c.all_steps_verified for c in self.cases) if self.cases else 0.0

    @property
    def total_usd(self) -> float:
        """Total spend of the run."""
        return sum(c.usd for c in self.cases)


def load_cases(path: Path) -> list[GroundTruthCase]:
    """Parse and validate a JSON array of cases."""
    adapter = TypeAdapter(list[GroundTruthCase])
    return adapter.validate_json(path.read_bytes())


def score(case: GroundTruthCase, report: TriageReport) -> CaseResult:
    """Score one report against its ground truth."""
    evidence_refs = {e.ref for e in report.evidence}
    expected = set(case.expected_precedents)
    recall = len(expected & evidence_refs) / len(expected) if expected else 1.0

    # What the engineer sees: step text plus the evidence each step cites.
    cited_text = " ".join(
        report.evidence[int(c[1:]) - 1].snippet.lower()
        for step in report.steps
        for c in step.citations
        if 0 < int(c[1:]) <= len(report.evidence)
    )
    rec_text = " ".join(s.content.lower() for s in report.steps) + " " + cited_text
    hits = [k for k in case.expected_keywords if k.lower() in rec_text]
    coverage = len(hits) / len(case.expected_keywords) if case.expected_keywords else 1.0
    return CaseResult(
        case_id=case.case_id,
        precedent_recall=recall,
        keyword_coverage=coverage,
        all_steps_verified=all(s.verified for s in report.steps),
        usd=report.cost.usd,
        latency_ms=report.latency_ms,
    )


async def run_benchmark(
    agent: TriageAgent, cases: Sequence[GroundTruthCase], concurrency: int = 4
) -> BenchmarkReport:
    """Run all cases with at most ``concurrency`` triages in flight."""

    async def one(case: GroundTruthCase) -> CaseResult:
        report, _ = await agent.run(case.alert)
        return score(case, report)

    results = await gather_bounded((partial(one, c) for c in cases), limit=concurrency)
    return BenchmarkReport(tuple(results))


def feedback_to_cases(
    pairs: Iterable[tuple[TriageReport, Feedback]],
) -> list[GroundTruthCase]:
    """Turn corrected thumbs-down feedback into new candidate eval cases.

    Analysts' corrections become ``expected_keywords`` after human curation;
    this closes the continuous-improvement loop in the design.
    """
    cases: list[GroundTruthCase] = []
    for report, feedback in pairs:
        if feedback.rating != "down" or not feedback.correction:
            continue
        cases.append(
            GroundTruthCase(
                case_id=f"fb-{report.report_id}",
                alert=report.alert,
                expected_keywords=(feedback.correction.lower()[:80],),
            )
        )
    return cases


def dump_cases(cases: Iterable[GroundTruthCase]) -> str:
    """Serialise cases for review in a PR to the eval set."""
    return json.dumps([c.model_dump(mode="json") for c in cases], indent=2)
