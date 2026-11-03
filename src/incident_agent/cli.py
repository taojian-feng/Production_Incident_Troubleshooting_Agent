"""Command line: run the agent locally on the sample data, or run the benchmark.

incident-agent triage --service payments-api --title "504s after deploy" --severity SEV1
incident-agent benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import TYPE_CHECKING, Any

from incident_agent.app import SAMPLES, build_local_components
from incident_agent.domain.enums import Severity
from incident_agent.domain.models import Alert
from incident_agent.evaluation.benchmark import load_cases, run_benchmark

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
    args = parser.parse_args(argv)
    command: Callable[[argparse.Namespace], Coroutine[Any, Any, int]] = args.func
    return asyncio.run(command(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
