from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from incident_agent.analytics.incidents import (
    UNOWNED,
    feedback_quality,
    incidents_frame,
    mttr_by_service,
    unmatched_services,
)
from incident_agent.app import Components
from incident_agent.domain import Alert, Feedback, HistoricalIncident
from incident_agent.evaluation.benchmark import (
    GroundTruthCase,
    dump_cases,
    feedback_to_cases,
    load_cases,
    run_benchmark,
    score,
)


@pytest.fixture
def services(samples: Path) -> pd.DataFrame:
    return pd.read_csv(samples / "services.csv")


class TestAnalytics:
    def test_mttr_by_service(
        self, incidents: list[HistoricalIncident], services: pd.DataFrame
    ) -> None:
        # The service catalog contains a duplicate row; incidents contain a re-synced ticket.
        frame = incidents_frame([*incidents, incidents[0]])
        assert frame["incident_id"].is_unique
        summary = mttr_by_service(frame, services).set_index("service")

        payments = summary.loc["payments-api"]
        assert payments["incidents"] == 3
        assert payments["mttr_median_min"] == pytest.approx(38.0)  # 38, 105, 20 minutes
        assert payments["sla_breach_rate"] == pytest.approx(round(2 / 3, 2))

        ledger = summary.loc["ledger-service"]
        assert ledger["incidents"] == 2
        assert ledger["resolved"] == 1  # the open incident is excluded, not zero-filled

    def test_unknown_service_is_labelled_not_dropped(
        self, incidents: list[HistoricalIncident], services: pd.DataFrame
    ) -> None:
        orphan = incidents[0].model_copy(update={"incident_id": "INC0000009", "service": "fx-api"})
        frame = incidents_frame([*incidents, orphan])
        summary = mttr_by_service(frame, services)
        assert summary.loc[summary["service"] == "fx-api", "team"].item() == UNOWNED
        assert unmatched_services(frame, services) == ["fx-api"]

    def test_merge_validation_catches_dirty_catalog(
        self, incidents: list[HistoricalIncident]
    ) -> None:
        catalog = pd.DataFrame({"service": ["payments-api"], "team": ["a"], "sla_minutes": [30]})
        out = mttr_by_service(incidents_frame(incidents), catalog)
        assert set(out["team"]) == {"a", UNOWNED}

    async def test_feedback_quality(self, local: Components, payments_alert: Alert) -> None:
        r1, _ = await local.agent.run(payments_alert)
        r2, _ = await local.agent.run(Alert(service="auth-service", title="401 spike"))
        feedback = [
            Feedback(report_id=r1.report_id, rating="up"),
            Feedback(report_id=r1.report_id, rating="down", correction="also check RDS proxy"),
        ]
        table = feedback_quality([r1, r2], feedback).set_index("retrieval_mode")
        assert table["reports"].sum() == 2
        assert table["feedback"].sum() == 2
        assert table.loc[r1.retrieval_mode.value, "thumbs_up_rate"] == pytest.approx(0.5)


class TestBenchmark:
    async def test_benchmark_on_ground_truth(self, local: Components, samples: Path) -> None:
        cases = load_cases(samples / "ground_truth.json")
        result = await run_benchmark(local.agent, cases, concurrency=2)
        assert [c.case_id for c in result.cases] == [c.case_id for c in cases]
        assert result.precedent_recall == pytest.approx(1.0)
        assert result.verified_rate == pytest.approx(1.0)
        assert result.keyword_coverage >= 0.5
        assert result.total_usd > 0

    async def test_score_and_feedback_loop(self, local: Components, payments_alert: Alert) -> None:
        report, _ = await local.agent.run(payments_alert)
        case = GroundTruthCase(
            case_id="c",
            alert=payments_alert,
            expected_precedents=("INC0041877", "INC0000000"),
            expected_keywords=("gateway",),
        )
        result = score(case, report)
        assert result.precedent_recall == pytest.approx(0.5)
        assert result.keyword_coverage == pytest.approx(1.0)

        down = Feedback(report_id=report.report_id, rating="down", correction="Check RDS proxy")
        up = Feedback(report_id=report.report_id, rating="up")
        new_cases = feedback_to_cases([(report, down), (report, up)])
        assert [c.expected_keywords for c in new_cases] == [("check rds proxy",)]
        assert '"case_id"' in dump_cases(new_cases)
