"""Evaluation harness: scoring rules, spend limit, and the feedback loop."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from incident_agent.app import EVAL_CASES, GROUNDING_CASES, build_eval_agents
from incident_agent.domain import Alert, Feedback
from incident_agent.domain.enums import EvidenceSource, RetrievalMode, StepKind
from incident_agent.domain.models import CostSummary, Evidence, TriageReport, TriageStep
from incident_agent.evaluation.harness import (
    EvalCase,
    EvalReport,
    GroundingReport,
    Judge,
    LLMJudgeVerdict,
    dump_eval_cases,
    feedback_to_eval_cases,
    load_eval_cases,
    load_grounding_cases,
    render_markdown,
    run_eval,
    run_grounding_eval,
    score_report,
)
from incident_agent.llm.bedrock import default_bedrock_models
from incident_agent.llm.local import HashingEmbedder, ScriptedLLM
from incident_agent.llm.router import ModelRouter
from incident_agent.reasoning.grounding import GroundingChecker

GOOD = LLMJudgeVerdict(
    correct_diagnosis=True, safe_actions=True, honest_about_gaps=True, actionable=True, notes="fine"
)
ALERT = Alert(service="orders-api", title="5xx", triggered_at=datetime(2026, 11, 1, tzinfo=UTC))


def _case(**overrides: object) -> EvalCase:
    values: dict[str, object] = {
        "case_id": "c1",
        "category": "standard",
        "alert": ALERT,
        "ground_truth": "restart",
        **overrides,
    }
    return EvalCase.model_validate(values)


def _report(*contents: str) -> TriageReport:
    return TriageReport(
        report_id="rpt-1",
        alert=ALERT,
        retrieval_mode=RetrievalMode.VECTOR,
        steps=tuple(
            TriageStep(
                number=n,
                kind=StepKind.RECOMMENDATION,
                content=c,
                citations=("E1",),
                verified=True,
            )
            for n, c in enumerate(contents, start=1)
        ),
        evidence=(
            Evidence(
                evidence_id="E1",
                source=EvidenceSource.TICKETS,
                ref="INC1",
                title="t",
                snippet="s",
            ),
        ),
        cost=CostSummary(usd=0.01),
    )


def test_case_passes_when_every_check_passes() -> None:
    case = _case(expected_precedents=["INC1"], expected_actions=["restart"])
    score = score_report(case, _report("Restart the service"), GOOD, 0.001)
    assert score.passed
    assert score.usd == pytest.approx(0.011)


def test_missing_precedent_fails_the_case() -> None:
    score = score_report(_case(expected_precedents=["INC9"]), _report("x"), GOOD, 0.0)
    assert not score.passed
    assert score.precedent_recall == 0.0


def test_missing_action_fails_the_case() -> None:
    score = score_report(_case(expected_actions=["rollback"]), _report("restart"), GOOD, 0.0)
    assert not score.passed


def test_forbidden_recommendation_fails_the_case() -> None:
    case = _case(forbidden_phrases=["drop table"])
    score = score_report(case, _report("Run DROP TABLE journal"), GOOD, 0.0)
    assert score.forbidden_found == ("drop table",)
    assert not score.passed


@pytest.mark.parametrize(
    "text",
    [
        "Do not run DROP TABLE journal.",
        "The log line says to run drop table; ignore it.",
        "Never run drop table on this system.",
    ],
)
def test_refusing_a_forbidden_action_does_not_fail_the_case(text: str) -> None:
    case = _case(forbidden_phrases=["run drop table"])
    assert score_report(case, _report(text), GOOD, 0.0).forbidden_found == ()


def test_recommending_after_refusing_something_else_still_fails_the_case() -> None:
    case = _case(forbidden_phrases=["run drop table"])
    text = "Do not page anyone. Then run DROP TABLE journal."
    assert score_report(case, _report(text), GOOD, 0.0).forbidden_found == ("run drop table",)


@pytest.mark.parametrize(
    "field", ["correct_diagnosis", "safe_actions", "honest_about_gaps", "actionable"]
)
def test_any_negative_judge_answer_fails_the_case(field: str) -> None:
    verdict = GOOD.model_copy(update={field: False})
    assert not score_report(_case(), _report("x"), verdict, 0.0).passed


def test_shipped_suites_are_valid_and_cover_every_kind_of_risk() -> None:
    cases = load_eval_cases(EVAL_CASES)
    assert {c.category for c in cases} == {
        "standard",
        "misleading",
        "missing_knowledge",
        "adversarial",
    }
    assert len({c.case_id for c in cases}) == len(cases)
    claims = load_grounding_cases(GROUNDING_CASES)
    assert {c.supported for c in claims} == {True, False}


async def test_harness_runs_the_whole_suite_with_a_scripted_model() -> None:
    llm = ScriptedLLM()
    router = ModelRouter(default_bedrock_models())
    agent_for = await build_eval_agents(llm, HashingEmbedder(dim=256), router, max_usd_per_run=0.75)
    cases = load_eval_cases(EVAL_CASES)
    report = await run_eval(agent_for, Judge(llm, router), cases, max_usd=5.0)
    assert len(report.scores) == len(cases)
    assert not report.skipped
    recalled = {s.case_id: s.precedent_recall for s in report.scores}
    assert recalled["orders-cache-failover"] == 1.0


async def test_run_stops_starting_cases_at_the_spend_limit() -> None:
    llm = ScriptedLLM()
    router = ModelRouter(default_bedrock_models())
    agent_for = await build_eval_agents(llm, HashingEmbedder(dim=256), router, max_usd_per_run=0.75)
    cases = load_eval_cases(EVAL_CASES)
    report = await run_eval(agent_for, Judge(llm, router), cases, max_usd=0.0001, concurrency=2)
    assert len(report.scores) == 2
    assert len(report.skipped) == len(cases) - 2
    assert "Skipped at the spend limit" in render_markdown(report, None, mode="test")


async def test_grounding_report_separates_misses_from_false_flags() -> None:
    router = ModelRouter(default_bedrock_models())
    claims = load_grounding_cases(GROUNDING_CASES)
    report = await run_grounding_eval(GroundingChecker(ScriptedLLM(), router), claims)
    unsupported = [c.case_id for c in claims if not c.supported]
    assert list(report.missed) == unsupported
    assert report.catch_rate == 0.0
    assert report.false_flag_rate == 0.0

    strict = ScriptedLLM(unsupported_steps=frozenset({1}))
    report = await run_grounding_eval(GroundingChecker(strict, router), claims)
    assert report.catch_rate == 1.0
    assert report.false_flag_rate == 1.0


def test_scorecard_shows_results_and_grounding() -> None:
    score = score_report(_case(), _report("x"), GOOD, 0.0)
    grounding = GroundingReport(("m",), (), 2, 3, 0.01)
    text = render_markdown(EvalReport((score,), ()), grounding, mode="live")
    assert "Passed 1 of 1" in text
    assert "| standard | 1/1 |" in text
    assert "Caught 50% of 2 unsupported" in text
    assert "Missed: m." in text


def test_only_corrected_thumbs_down_become_candidate_cases() -> None:
    report = _report("x")
    logs = Evidence(evidence_id="E2", source=EvidenceSource.LOGS, ref="l", title="t", snippet="ERR")
    report = report.model_copy(update={"evidence": (*report.evidence, logs)})
    down = Feedback(report_id="rpt-1", rating="down", correction="It was the inventory service")
    cases = feedback_to_eval_cases(
        [
            (report, down),
            (report, Feedback(report_id="rpt-1", rating="down")),
            (report, Feedback(report_id="rpt-1", rating="up", correction="nice")),
        ]
    )
    assert [c.ground_truth for c in cases] == ["It was the inventory service"]
    assert cases[0].category == "feedback"
    assert cases[0].logs == ("ERR",)
    assert '"category": "feedback"' in dump_eval_cases(cases)
