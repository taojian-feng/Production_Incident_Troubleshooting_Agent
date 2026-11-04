"""Lambda: CloudWatch Logs subscription -> knowledge base.

Each error pattern seen for the first time becomes one small document, so the knowledge
base records which errors a service has produced. Repeats of a known pattern write
nothing, so a noisy service cannot flood the index. The live logs themselves are still
read directly by the logs domain.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError

from incident_agent.domain.values import LogEvent
from incident_agent.ingestion.log_stream import is_error

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@dataclass(frozen=True, slots=True)
class LogPatternDocument:
    """A first-seen error pattern, ready to store."""

    key: str
    body: str


def parse_subscription_event(event: dict[str, Any]) -> list[LogEvent]:
    """Decode a subscription delivery. Control messages hold no log events."""
    payload = json.loads(gzip.decompress(base64.b64decode(event["awslogs"]["data"])))
    if payload.get("messageType") != "DATA_MESSAGE":
        return []
    return [
        LogEvent(
            event_id=item["id"],
            timestamp=datetime.fromtimestamp(item["timestamp"] / 1000, tz=UTC),
            log_group=payload["logGroup"],
            log_stream=payload.get("logStream", ""),
            message=item["message"],
        )
        for item in payload["logEvents"]
    ]


def pattern_documents(events: list[LogEvent], log_group_prefix: str) -> list[LogPatternDocument]:
    """One document per distinct error pattern in ``events``."""
    documents: dict[str, LogPatternDocument] = {}
    for event in events:
        if not is_error(event):
            continue
        service = event.log_group.removeprefix(log_group_prefix)
        digest = hashlib.blake2b(f"{service}:{event.signature}".encode(), digest_size=8).hexdigest()
        key = f"knowledge/log-{service}-{digest}.md"
        documents.setdefault(
            key,
            LogPatternDocument(
                key=key,
                body=(
                    f"---\nservice: {service}\ndoc_type: log_pattern\n---\n"
                    f"# Error pattern in {service}\n\n## Pattern\n{event.signature}\n\n"
                    f"## First seen\n{event.timestamp.isoformat()} in {event.log_group}\n"
                ),
            ),
        )
    return list(documents.values())


def store_new_patterns(
    client: S3Client, bucket: str, documents: list[LogPatternDocument]
) -> list[str]:
    """Write documents that do not exist yet, returning the keys written."""
    written: list[str] = []
    for document in documents:
        try:
            client.put_object(
                Bucket=bucket, Key=document.key, Body=document.body.encode(), IfNoneMatch="*"
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in {"PreconditionFailed", "412"}:
                raise
        else:
            written.append(document.key)
    return written


def handler(event: dict[str, Any], context: object) -> dict[str, int]:  # pragma: no cover
    """Lambda entry point."""
    del context
    import boto3  # noqa: PLC0415

    from incident_agent.config import Settings  # noqa: PLC0415

    settings = Settings.from_env()
    documents = pattern_documents(parse_subscription_event(event), settings.log_group_prefix)
    client = boto3.client("s3", region_name=settings.region)
    written = store_new_patterns(client, settings.knowledge_bucket, documents)
    logger.info(json.dumps({"patterns": len(documents), "new": len(written)}))
    return {"patterns": len(documents), "new": len(written)}
