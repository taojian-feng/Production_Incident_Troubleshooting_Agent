"""Streaming log processing with generators.

CloudWatch log groups for a busy payments service produce millions of events
per hour. Nothing here loads a whole result set: each function consumes an
iterator and yields, so memory stays O(page size) regardless of volume, and
stages compose like a Unix pipeline::

    events = iter_cloudwatch_events(client, group, start, end)
    errors = (e for e in dedupe(events) if is_error(e))
    top = top_signatures(errors, k=5)
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import islice
from typing import TYPE_CHECKING, Any

from incident_agent.domain.values import LogEvent

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator, Mapping
    from pathlib import Path

    from mypy_boto3_logs import CloudWatchLogsClient

logger = logging.getLogger(__name__)

_ERROR_RE = re.compile(r"\b(ERROR|FATAL|CRITICAL|Exception|Traceback|5\d\d)\b")


def read_jsonl(path: Path) -> Generator[LogEvent, None, None]:
    """Stream a JSON-lines log export one line at a time.

    The ``with`` block guarantees the file handle is closed even if the
    consumer stops iterating early (generator ``close()`` runs the ``finally``
    of the context manager). Malformed lines are counted and skipped, not fatal.
    """
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                yield LogEvent(
                    event_id=str(raw.get("eventId") or f"{path.name}:{lineno}"),
                    timestamp=datetime.fromtimestamp(raw["timestamp"] / 1000, tz=UTC),
                    log_group=raw.get("logGroup", path.stem),
                    log_stream=raw.get("logStream", ""),
                    message=raw["message"],
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                skipped += 1
    if skipped:
        logger.warning("skipped %d malformed lines in %s", skipped, path)


def iter_cloudwatch_events(
    client: CloudWatchLogsClient,
    log_group: str,
    start: datetime,
    end: datetime,
    filter_pattern: str = "",
    max_events: int = 10_000,
) -> Iterator[LogEvent]:
    """Page through ``FilterLogEvents`` lazily, stopping at ``max_events``.

    The boto3 paginator itself is lazy; ``islice`` adds a hard cap so a noisy
    log group can never blow the Lambda's memory or the LLM's context budget.
    """
    paginator = client.get_paginator("filter_log_events")
    pages = paginator.paginate(
        logGroupName=log_group,
        startTime=int(start.timestamp() * 1000),
        endTime=int(end.timestamp() * 1000),
        filterPattern=filter_pattern,
        PaginationConfig={"PageSize": 1000},
    )

    def events() -> Iterator[LogEvent]:
        for page in pages:
            for raw in page.get("events", []):
                yield _from_cloudwatch(raw, log_group)

    yield from islice(events(), max_events)


def _from_cloudwatch(raw: Mapping[str, Any], log_group: str) -> LogEvent:
    return LogEvent(
        event_id=str(raw.get("eventId", "")),
        timestamp=datetime.fromtimestamp(raw.get("timestamp", 0) / 1000, tz=UTC),
        log_group=log_group,
        log_stream=str(raw.get("logStreamName", "")),
        message=str(raw.get("message", "")).rstrip(),
    )


def is_error(event: LogEvent) -> bool:
    """Heuristic error filter used when CloudWatch filter patterns are not set."""
    return _ERROR_RE.search(event.message) is not None


def dedupe(events: Iterable[LogEvent]) -> Iterator[LogEvent]:
    """Drop exact duplicates (same stream + message) while streaming.

    A ``set`` of tuples gives O(1) membership checks. Tuples, unlike lists, are
    hashable, which is why the key is ``(stream, message)`` and not a list.
    """
    seen: set[tuple[str, str]] = set()
    for event in events:
        key = (event.log_stream, event.message)
        if key not in seen:
            seen.add(key)
            yield event


def top_signatures(events: Iterable[LogEvent], k: int = 5) -> list[tuple[str, int, LogEvent]]:
    """Most frequent error signatures with one representative event each.

    ``Counter`` does the frequency count in one pass; a plain dict keeps the
    first event per signature as the exemplar we will cite (``setdefault``
    only stores the first one).
    """
    counts: Counter[str] = Counter()
    exemplar: dict[str, LogEvent] = {}
    for event in events:
        sig = event.signature
        counts[sig] += 1
        exemplar.setdefault(sig, event)
    return [(sig, n, exemplar[sig]) for sig, n in counts.most_common(k)]


def group_by_stream(events: Iterable[LogEvent]) -> dict[str, list[LogEvent]]:
    """Bucket events per log stream (``defaultdict`` avoids key-exists checks)."""
    buckets: defaultdict[str, list[LogEvent]] = defaultdict(list)
    for event in events:
        buckets[event.log_stream].append(event)
    return dict(buckets)
