"""Lambda: S3 ``ObjectCreated`` under ``knowledge/`` -> re-index -> new snapshot.

Real-time synchronisation (design section 2.1): every new runbook, post-mortem,
ticket or log pattern triggers this function. Concurrent uploads are safe: a writer
that loses the race reloads the latest snapshot and applies its documents again.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

from incident_agent.core.errors import SnapshotConflictError
from incident_agent.domain.models import HistoricalIncident, SourceDocument
from incident_agent.ingestion.loaders import parse_markdown

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from mypy_boto3_s3 import S3Client

    from incident_agent.ingestion.pipeline import IngestionPipeline, IngestStats
    from incident_agent.retrieval.keyword import BM25Index
    from incident_agent.retrieval.vector import InMemoryVectorIndex
    from incident_agent.storage.repositories import S3IndexSnapshot
    from incident_agent.tools.tickets import IncidentRepository

logger = logging.getLogger()

MAX_WRITE_ATTEMPTS = 20
WRITE_RETRY_BASE_S = 0.2
WRITE_RETRY_MAX_S = 5.0


def iter_s3_documents(
    client: S3Client, event: dict[str, Any], incidents: IncidentRepository | None = None
) -> Iterator[SourceDocument]:
    """Yield documents from an S3 event; ``.json`` incident exports also upsert the table."""
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
        path = PurePosixPath(key)
        if path.suffix not in {".md", ".json"}:
            # Check before downloading: binary objects must never reach .decode().
            logger.info("skipping unsupported object %s", key)
            continue
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        if path.suffix == ".md":
            yield parse_markdown(path.stem, body, uri=f"s3://{bucket}/{key}")
        else:
            incident = HistoricalIncident.model_validate_json(body)
            if incidents is not None:
                incidents.put(incident)
            yield incident.to_document()


async def reindex(pipeline: IngestionPipeline, docs: Iterable[SourceDocument]) -> IngestStats:
    """Thin async wrapper so the handler and tests share one code path."""
    return await pipeline.ingest(docs)


async def update_snapshot(
    snapshot: S3IndexSnapshot,
    pipeline_for: Callable[[BM25Index, InMemoryVectorIndex], IngestionPipeline],
    docs: Sequence[SourceDocument],
    *,
    dim: int,
    max_attempts: int = MAX_WRITE_ATTEMPTS,
) -> IngestStats:
    """Apply ``docs`` to the latest snapshot, starting over if another writer got there first."""
    for attempt in range(max_attempts):
        vector, keyword, version = snapshot.load_versioned(dim)
        stats = await reindex(pipeline_for(keyword, vector), docs)
        if snapshot.save_if_unchanged(vector, version):
            return stats
        delay = min(WRITE_RETRY_BASE_S * 2**attempt, WRITE_RETRY_MAX_S)
        # Random jitter keeps concurrent writers from colliding again in lockstep.
        await asyncio.sleep(delay * (0.5 + secrets.randbelow(100) / 200))
    msg = f"index snapshot still conflicted after {max_attempts} attempts"
    raise SnapshotConflictError(msg)


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:  # pragma: no cover
    """Lambda entry point."""
    del context
    import boto3  # noqa: PLC0415

    from incident_agent.app import ingest_pipeline_factory  # noqa: PLC0415
    from incident_agent.config import Settings  # noqa: PLC0415
    from incident_agent.storage.repositories import S3IndexSnapshot  # noqa: PLC0415
    from incident_agent.tools.tickets import DynamoIncidentRepository  # noqa: PLC0415

    settings = Settings.from_env()
    s3 = boto3.client("s3", region_name=settings.region)
    table = boto3.resource("dynamodb", region_name=settings.region).Table(settings.incidents_table)
    docs = list(iter_s3_documents(s3, event, DynamoIncidentRepository(table)))
    stats = asyncio.run(
        update_snapshot(
            S3IndexSnapshot(s3, settings.knowledge_bucket),
            ingest_pipeline_factory(settings),
            docs,
            dim=settings.embedding_dim,
        )
    )
    logger.info(json.dumps({"ingested": stats.documents, "chunks": stats.chunks}))
    return {"documents": stats.documents, "chunks": stats.chunks}
