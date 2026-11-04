"""Command line: run the agent locally on the sample data, or run the benchmark.

incident-agent triage --service payments-api --title "504s after deploy" --severity SEV1
incident-agent benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import boto3

from incident_agent.app import (
    EVAL_CASES,
    GROUNDING_CASES,
    SAMPLES,
    build_eval_agents,
    build_local_components,
)
from incident_agent.domain.enums import Severity
from incident_agent.domain.models import Alert
from incident_agent.evaluation.benchmark import load_cases, run_benchmark
from incident_agent.evaluation.harness import (
    EvalReport,
    GroundingReport,
    Judge,
    dump_eval_cases,
    feedback_to_eval_cases,
    load_eval_cases,
    load_grounding_cases,
    render_markdown,
    run_eval,
    run_grounding_eval,
)
from incident_agent.llm.anthropic_api import ClaudeApiProvider, default_api_models
from incident_agent.llm.bedrock import TitanEmbedder, default_bedrock_models
from incident_agent.llm.local import HashingEmbedder, ScriptedLLM
from incident_agent.llm.router import ModelRouter
from incident_agent.reasoning.grounding import GroundingChecker
from incident_agent.storage.repositories import DynamoFeedbackStore, DynamoReportStore

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


async def _triage(args: argparse.Namespace) -> int:
    components = await build_local_components()
    alert = Alert(
        service=args.service,
        title=args.title,
        description=args.description,
        severity=Severity(args.severity),
    )
    report, state = await components.agent.run(alert)
    out = report.model_dump(mode="json")
    out["audit_trail"] = [{"node": r.node, "ms": r.duration_ms} for r in state.history]
    sys.stdout.write(json.dumps(out, indent=2) + "\n")
    return 0


async def _benchmark(_: argparse.Namespace) -> int:
    components = await build_local_components()
    result = await run_benchmark(components.agent, load_cases(SAMPLES / "ground_truth.json"))
    for case in result.cases:
        sys.stdout.write(
            f"{case.case_id:<22} recall={case.precedent_recall:.2f} "
            f"keywords={case.keyword_coverage:.2f} verified={case.all_steps_verified}\n"
        )
    sys.stdout.write(
        f"\nprecedent_recall={result.precedent_recall:.2f} "
        f"keyword_coverage={result.keyword_coverage:.2f} "
        f"verified_rate={result.verified_rate:.2f}\n"
    )
    return 0


async def _eval(args: argparse.Namespace) -> int:
    report, grounding = await (_eval_live(args) if args.live else _eval_offline())
    mode = "live model" if args.live else "offline, scripted model: scores are not meaningful"
    scorecard = render_markdown(report, grounding, mode=mode)
    sys.stdout.write(scorecard)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(scorecard, encoding="utf-8")
    if not report.scores or report.skipped:
        return 1
    return 0 if report.pass_rate >= args.min_pass_rate else 1


async def _eval_offline() -> tuple[EvalReport, GroundingReport]:
    llm = ScriptedLLM()
    router = ModelRouter(default_bedrock_models())
    agent_for = await build_eval_agents(llm, HashingEmbedder(dim=256), router, max_usd_per_run=0.75)
    report = await run_eval(agent_for, Judge(llm, router), load_eval_cases(EVAL_CASES), max_usd=1.0)
    grounding = await run_grounding_eval(
        GroundingChecker(llm, router), load_grounding_cases(GROUNDING_CASES)
    )
    return report, grounding


async def _eval_live(  # pragma: no cover - spends money on real models
    args: argparse.Namespace,
) -> tuple[EvalReport, GroundingReport]:
    settings = boto3.Session(region_name=args.region)
    secret = settings.client("secretsmanager").get_secret_value(SecretId=args.secret)
    llm = ClaudeApiProvider.from_api_key(secret["SecretString"].strip(), requests_per_second=2.0)
    router = ModelRouter(default_api_models())
    embedder = TitanEmbedder(settings.client("bedrock-runtime"), dim=512)
    agent_for = await build_eval_agents(llm, embedder, router, max_usd_per_run=0.75)
    report = await run_eval(
        agent_for, Judge(llm, router), load_eval_cases(EVAL_CASES), max_usd=args.max_usd
    )
    grounding = await run_grounding_eval(
        GroundingChecker(llm, router), load_grounding_cases(GROUNDING_CASES)
    )
    return report, grounding


async def _export_feedback(args: argparse.Namespace) -> int:  # pragma: no cover - needs AWS
    dynamo = boto3.resource("dynamodb", region_name=args.region)
    reports = DynamoReportStore(dynamo.Table(args.reports_table))
    entries = DynamoFeedbackStore(dynamo.Table(args.feedback_table)).scan_all()
    pairs = [
        (report, entry) for entry in entries if (report := reports.get(entry.report_id)) is not None
    ]
    cases = feedback_to_eval_cases(pairs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump_eval_cases(cases), encoding="utf-8")
    sys.stdout.write(f"wrote {len(cases)} candidate cases to {args.out}; review before adding\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``incident-agent`` console script."""
    parser = argparse.ArgumentParser(prog="incident-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    triage = sub.add_parser("triage", help="triage one alert against the sample data")
    triage.add_argument("--service", required=True)
    triage.add_argument("--title", required=True)
    triage.add_argument("--description", default="")
    triage.add_argument("--severity", default="SEV3", choices=[s.value for s in Severity])
    triage.set_defaults(func=_triage)
    bench = sub.add_parser("benchmark", help="run the ground-truth benchmark")
    bench.set_defaults(func=_benchmark)
    evaluate = sub.add_parser("eval", help="score the agent on labelled synthetic incidents")
    evaluate.add_argument("--live", action="store_true", help="use real models; this costs money")
    evaluate.add_argument("--secret", default="incident-agent/anthropic-api-key")
    evaluate.add_argument("--region", default="us-east-1")
    evaluate.add_argument("--max-usd", type=float, default=2.0)
    evaluate.add_argument("--min-pass-rate", type=float, default=0.0)
    evaluate.add_argument("--out", type=Path, default=Path("build/eval/scorecard.md"))
    evaluate.set_defaults(func=_eval)
    export = sub.add_parser("export-feedback", help="turn corrections into candidate eval cases")
    export.add_argument("--reports-table", required=True)
    export.add_argument("--feedback-table", required=True)
    export.add_argument("--region", default="us-east-1")
    export.add_argument("--out", type=Path, default=Path("build/eval/candidates.json"))
    export.set_defaults(func=_export_feedback)
    args = parser.parse_args(argv)
    command: Callable[[argparse.Namespace], Coroutine[Any, Any, int]] = args.func
    return asyncio.run(command(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
