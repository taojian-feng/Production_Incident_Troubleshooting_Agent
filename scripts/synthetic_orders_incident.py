"""Exercise the deployed agent with a synthetic orders-api service.

Run with AWS credentials in the environment:

* ``load-knowledge`` uploads the runbooks and past incidents one file at a time.
* ``setup`` creates the ``/app/orders-api`` log group and the ``ia-orders-api-5xx`` alarm.
* ``fire`` writes failover error logs and a 5xx metric spike, which trips the alarm.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from datetime import UTC, datetime, timedelta
from itertools import cycle
from pathlib import Path
from uuid import uuid4

import boto3

STACK = "IncidentAgent-dev"
SERVICE = "orders-api"
LOG_GROUP = f"/app/{SERVICE}"
ALARM = f"ia-{SERVICE}-5xx"
NAMESPACE = "Synthetic/OrdersApi"
METRIC = "Http5xxRate"
DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic" / SERVICE
INDEX_KEY = "index/snapshot.zip"
ERRORS = (
    "ERROR ERR_CACHE_TIMEOUT RedisCommandTimeoutException: command timed out after 500ms",
    "ERROR READONLY You can't write against a read only replica. host=orders-cache-002",
    "ERROR POST /v1/checkout status=503 cause=ERR_CACHE_TIMEOUT",
)
LOG_LINES = 120
SECONDS_PER_LINE = 3

logger = logging.getLogger(__name__)


def _knowledge_bucket() -> str:
    stack = boto3.client("cloudformation").describe_stacks(StackName=STACK)["Stacks"][0]
    return next(o["OutputValue"] for o in stack["Outputs"] if o["OutputKey"] == "KnowledgeBucket")


def _upload_and_wait(bucket: str, path: Path, timeout_s: int = 120) -> None:
    """Upload one file and wait for its re-index, so snapshot writes never overlap."""
    s3 = boto3.client("s3")
    before = s3.head_object(Bucket=bucket, Key=INDEX_KEY)["LastModified"]
    s3.upload_file(str(path), bucket, f"knowledge/{path.name}")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(3)
        if s3.head_object(Bucket=bucket, Key=INDEX_KEY)["LastModified"] != before:
            logger.info("indexed %s", path.name)
            return
    msg = f"timed out waiting for {path.name} to be indexed"
    raise TimeoutError(msg)


def load_knowledge() -> None:
    """Upload the synthetic runbooks and past incidents."""
    bucket = _knowledge_bucket()
    for runbook in sorted((DATA / "knowledge").glob("*.md")):
        _upload_and_wait(bucket, runbook)
    incidents = json.loads((DATA / "incidents.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        for incident in incidents:
            path = Path(tmp) / f"{incident['incident_id']}.json"
            path.write_text(json.dumps(incident))
            _upload_and_wait(bucket, path)


def setup() -> None:
    """Create the log group and the alarm."""
    logs = boto3.client("logs")
    if not logs.describe_log_groups(logGroupNamePrefix=LOG_GROUP)["logGroups"]:
        logs.create_log_group(logGroupName=LOG_GROUP)
    logs.put_retention_policy(logGroupName=LOG_GROUP, retentionInDays=7)
    boto3.client("cloudwatch").put_metric_alarm(
        AlarmName=ALARM,
        AlarmDescription=f"service={SERVICE}\nseverity=SEV2\nlog_group={LOG_GROUP}",
        Namespace=NAMESPACE,
        MetricName=METRIC,
        Statistic="Average",
        Period=60,
        EvaluationPeriods=1,
        Threshold=2.0,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
    )
    logger.info("created %s and %s", LOG_GROUP, ALARM)


def fire() -> None:
    """Write error logs and the metric spike that trips the alarm."""
    logs = boto3.client("logs")
    stream = f"{SERVICE}/task-{uuid4().hex[:8]}"
    logs.create_log_stream(logGroupName=LOG_GROUP, logStreamName=stream)
    now = datetime.now(UTC)
    start = now - timedelta(seconds=LOG_LINES * SECONDS_PER_LINE)
    messages = cycle(ERRORS)
    events = [
        {
            "timestamp": int((start + timedelta(seconds=i * SECONDS_PER_LINE)).timestamp() * 1000),
            "message": f"{next(messages)} request_id={uuid4().hex[:8]}",
        }
        for i in range(LOG_LINES)
    ]
    logs.put_log_events(logGroupName=LOG_GROUP, logStreamName=stream, logEvents=events)
    boto3.client("cloudwatch").put_metric_data(
        Namespace=NAMESPACE,
        MetricData=[{"MetricName": METRIC, "Value": 6.5, "Unit": "Percent", "Timestamp": now}],
    )
    logger.info("wrote %d log lines and a 5xx spike; the alarm trips within minutes", LOG_LINES)


def main() -> None:
    """Run one subcommand."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    commands = {"load-knowledge": load_knowledge, "setup": setup, "fire": fire}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=commands)
    commands[parser.parse_args().command]()


if __name__ == "__main__":
    main()
