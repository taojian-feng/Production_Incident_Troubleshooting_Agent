"""Lambda: S3 ``ObjectCreated`` under ``knowledge/`` -> re-index -> new snapshot.

Real-time synchronisation (design section 2.1): every new runbook,
post-mortem or exported incident triggers this function. Reserved
concurrency is 1 (see infra) so two uploads can't race on the snapshot.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

from incident_agent.core.refresh import RefreshingValue
from incident_agent.domain.models import HistoricalIncident, SourceDocument
from incident_agent.ingestion.loaders import parse_markdown

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mypy_boto3_s3 import S3Client

    from incident_agent.ingestion.pipeline import IngestionPipeline, IngestStats
    from incident_agent.tools.tickets import IncidentRepository

logger = logging.getLogger()


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


async def reindex(pipeline: IngestionPipeline, docs: Iterator[SourceDocument]) -> IngestStats:
    """Thin async wrapper so the handler and tests share one code path."""
    return await pipeline.ingest(docs)


@functools.cache
def _pipeline() -> RefreshingValue[IngestionPipeline]:  # pragma: no cover - needs AWS
    import boto3  # noqa: PLC0415

    from incident_agent.app import build_ingest_pipeline  # noqa: PLC0415
    from incident_agent.config import Settings  # noqa: PLC0415
    from incident_agent.storage.repositories import S3IndexSnapshot  # noqa: PLC0415

    settings = Settings.from_env()
    client = boto3.client("s3", region_name=settings.region)
    return RefreshingValue(
        version=S3IndexSnapshot(client, settings.knowledge_bucket).version,
        build=lambda: build_ingest_pipeline(settings),
    )


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:  # pragma: no cover
    """Lambda entry point."""
    del context
    import boto3  # noqa: PLC0415

    from incident_agent.config import Settings  # noqa: PLC0415
    from incident_agent.storage.repositories import S3IndexSnapshot  # noqa: PLC0415
    from incident_agent.tools.tickets import DynamoIncidentRepository  # noqa: PLC0415

    settings = Settings.from_env()
    s3 = boto3.client("s3", region_name=settings.region)
    table = boto3.resource("dynamodb", region_name=settings.region).Table(settings.incidents_table)
    pipeline = _pipeline().get()
    docs = iter_s3_documents(s3, event, DynamoIncidentRepository(table))
    stats = asyncio.run(reindex(pipeline, docs))
    S3IndexSnapshot(s3, settings.knowledge_bucket).save(pipeline.vector)
    logger.info(json.dumps({"ingested": stats.documents, "chunks": stats.chunks}))
    return {"documents": stats.documents, "chunks": stats.chunks}
