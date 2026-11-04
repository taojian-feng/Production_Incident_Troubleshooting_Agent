"""Storage and tool adapters against moto-mocked DynamoDB, S3 and CloudWatch Logs."""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import boto3
import numpy as np
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from moto import mock_aws

from incident_agent.app import Components
from incident_agent.core.errors import SnapshotConflictError
from incident_agent.domain import Alert, Feedback, HistoricalIncident, RetrievalMode, Severity
from incident_agent.domain.models import SourceDocument
from incident_agent.domain.state import RetrievalPlan
from incident_agent.domain.values import Chunk
from incident_agent.handlers.ingest import iter_s3_documents, reindex, update_snapshot
from incident_agent.handlers.log_sync import LogPatternDocument, store_new_patterns
from incident_agent.ingestion.pipeline import IngestionPipeline
from incident_agent.llm.local import HashingEmbedder
from incident_agent.retrieval.keyword import BM25Index
from incident_agent.retrieval.vector import InMemoryVectorIndex
from incident_agent.storage.repositories import (
    DynamoFeedbackStore,
    DynamoReportStore,
    S3IndexSnapshot,
)
from incident_agent.tools.logs import CloudWatchLogsServer
from incident_agent.tools.tickets import DynamoIncidentRepository, IncidentHistoryServer

pytestmark = pytest.mark.integration
REGION = "us-east-1"


@pytest.fixture
def aws() -> Iterator[None]:
    with mock_aws():
        yield


def _table(name: str, key: list[tuple[str, str]], gsi: dict[str, Any] | None = None) -> Any:
    dynamo = boto3.resource("dynamodb", region_name=REGION)
    attrs = {k for k, _ in key} | ({"service", "opened_at"} if gsi else set())
    kwargs: dict[str, Any] = {
        "TableName": name,
        "KeySchema": [{"AttributeName": k, "KeyType": t} for k, t in key],
        "AttributeDefinitions": [{"AttributeName": a, "AttributeType": "S"} for a in attrs],
        "BillingMode": "PAY_PER_REQUEST",
    }
    if gsi:
        kwargs["GlobalSecondaryIndexes"] = [gsi]
    return dynamo.create_table(**kwargs)


@pytest.mark.usefixtures("aws")
class TestDynamo:
    def test_incident_repository_and_server(self, incidents: list[HistoricalIncident]) -> None:
        table = _table(
            "incidents",
            [("incident_id", "HASH")],
            {
                "IndexName": "by_service",
                "KeySchema": [
                    {"AttributeName": "service", "KeyType": "HASH"},
                    {"AttributeName": "opened_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        )
        repo = DynamoIncidentRepository(table)
        for incident in incidents:
            repo.put(incident)
        assert [i.incident_id for i in repo.get_many(["INC0040550", "INC0000000"])] == [
            "INC0040550"
        ]
        recent = repo.recent_for_service("payments-api", 2)
        assert [i.incident_id for i in recent] == ["INC0043001", "INC0041877"]

    async def test_ticket_server_exact_id_first(self, incidents: list[HistoricalIncident]) -> None:
        from incident_agent.tools.tickets import InMemoryIncidentRepository

        server = IncidentHistoryServer(InMemoryIncidentRepository.of(incidents))
        alert = Alert(service="payments-api", title="Same as INC0039012? gateway timeout")
        plan = RetrievalPlan(RetrievalMode.HYBRID, alert.text, ticket_ids=("INC0039012",))
        evidence = await server.gather(alert, plan)
        assert evidence[0].ref == "INC0039012"
        assert evidence[0].score == 1.0
        assert "INC0039012" not in [e.ref for e in evidence[1:]]

    async def test_report_and_feedback_roundtrip(
        self, local: Components, payments_alert: Alert
    ) -> None:
        reports = DynamoReportStore(_table("reports", [("report_id", "HASH")]))
        feedback = DynamoFeedbackStore(
            _table("feedback", [("report_id", "HASH"), ("created_at", "RANGE")])
        )
        report, _ = await local.agent.run(payments_alert)
        reports.save(report)
        loaded = reports.get(report.report_id)
        assert loaded == report  # floats survived the Decimal round-trip
        assert reports.get("missing") is None

        feedback.add(
            Feedback(report_id=report.report_id, rating="up", user="arn:aws:iam::1:user/a")
        )
        assert [f.rating for f in feedback.for_report(report.report_id)] == ["up"]
        assert [f.rating for f in feedback.scan_all()] == ["up"]


@pytest.mark.usefixtures("aws")
class TestS3:
    def test_snapshot_version_changes_when_a_new_snapshot_is_saved(self) -> None:
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket="kb-bucket")
        snapshot = S3IndexSnapshot(s3, "kb-bucket")
        assert snapshot.version() is None

        index = InMemoryVectorIndex(dim=2)
        assert snapshot.save_if_unchanged(index, None)
        empty_version = snapshot.version()
        assert empty_version is not None

        chunk = Chunk(chunk_id="c1", doc_id="d1", text="x")
        index.upsert([chunk], np.array([[1.0, 0.0]], dtype=np.float32))
        assert snapshot.save_if_unchanged(index, empty_version)
        assert snapshot.version() != empty_version

    def test_a_writer_with_a_stale_version_cannot_overwrite_a_newer_snapshot(self) -> None:
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket="kb-bucket")
        snapshot = S3IndexSnapshot(s3, "kb-bucket")
        index = InMemoryVectorIndex(dim=2)
        assert snapshot.save_if_unchanged(index, None)
        stale = snapshot.version()

        chunk = Chunk(chunk_id="c1", doc_id="d1", text="x")
        index.upsert([chunk], np.array([[1.0, 0.0]], dtype=np.float32))
        assert snapshot.save_if_unchanged(index, stale)
        assert not snapshot.save_if_unchanged(InMemoryVectorIndex(dim=2), stale)
        assert not snapshot.save_if_unchanged(InMemoryVectorIndex(dim=2), None)
        vectors, _keyword = snapshot.load(dim=2)
        assert len(vectors) == 1

    async def test_concurrent_uploads_both_end_up_in_the_index(self) -> None:
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket="kb-bucket")
        snapshot = S3IndexSnapshot(s3, "kb-bucket")
        embedder = HashingEmbedder(dim=64)
        mine = SourceDocument(doc_id="mine", title="Mine", body="# Mine\n## Fix\nrestart it")
        builds = 0

        def pipeline_for(keyword: BM25Index, vector: InMemoryVectorIndex) -> IngestionPipeline:
            nonlocal builds
            builds += 1
            if builds == 1:
                # Another writer saves after this one loaded, so this one's write must fail.
                rival, _keyword, version = snapshot.load_versioned(64)
                chunk = Chunk(chunk_id="rival#0", doc_id="rival", text="roll back")
                rival.upsert([chunk], np.ones((1, 64), dtype=np.float32))
                assert snapshot.save_if_unchanged(rival, version)
            return IngestionPipeline(embedder, keyword, vector)

        stats = await update_snapshot(snapshot, pipeline_for, [mine], dim=64)

        assert stats.documents == 1
        assert builds == 2
        vectors, _keyword = snapshot.load(dim=64)
        assert {chunk.doc_id for chunk in vectors.chunks()} == {"mine", "rival"}

    async def test_update_gives_up_after_repeated_conflicts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket="kb-bucket")
        snapshot = S3IndexSnapshot(s3, "kb-bucket")
        monkeypatch.setattr(snapshot.__class__, "save_if_unchanged", lambda *_args: False)
        monkeypatch.setattr("incident_agent.handlers.ingest.WRITE_RETRY_BASE_S", 0.0)
        doc = SourceDocument(doc_id="d", title="D", body="# D\n## Fix\nretry")
        with pytest.raises(SnapshotConflictError):
            await update_snapshot(
                snapshot,
                lambda keyword, vector: IngestionPipeline(HashingEmbedder(64), keyword, vector),
                [doc],
                dim=64,
                max_attempts=2,
            )

    def test_new_log_patterns_are_stored_once(self) -> None:
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket="kb-bucket")
        document = LogPatternDocument(key="knowledge/log-a-1.md", body="first")
        assert store_new_patterns(s3, "kb-bucket", [document]) == [document.key]
        repeat = LogPatternDocument(key=document.key, body="second")
        assert store_new_patterns(s3, "kb-bucket", [repeat]) == []
        stored = s3.get_object(Bucket="kb-bucket", Key=document.key)["Body"].read()
        assert stored == b"first"

    def test_snapshot_version_raises_when_access_is_denied(self) -> None:
        s3 = boto3.client("s3", region_name=REGION)
        denied = ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject")
        with Stubber(s3) as stub:
            stub.add_client_error(
                "head_object", service_error_code=denied.response["Error"]["Code"]
            )
            with pytest.raises(ClientError):
                S3IndexSnapshot(s3, "kb-bucket").version()

    async def test_snapshot_and_s3_ingest(self, local: Components, samples: Any) -> None:
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket="kb-bucket")
        snapshot = S3IndexSnapshot(s3, "kb-bucket")

        empty_vectors, empty_keyword = snapshot.load(dim=256)
        assert len(empty_vectors) == len(empty_keyword) == 0

        s3.put_object(
            Bucket="kb-bucket",
            Key="knowledge/new+runbook.md",
            Body=(
                b"---\nservice: fx-api\n---\n"
                b"# FX rates stale\n## Mitigation\nrestart the rates poller"
            ),
        )
        incident = HistoricalIncident(
            incident_id="INC0050000",
            service="fx-api",
            title="FX stale",
            summary="rates stale",
            opened_at=datetime(2026, 11, 1, tzinfo=UTC),
        )
        s3.put_object(
            Bucket="kb-bucket", Key="knowledge/INC0050000.json", Body=incident.model_dump_json()
        )
        s3.put_object(Bucket="kb-bucket", Key="knowledge/image.png", Body=b"\x89PNG")
        event = {
            "Records": [
                {"s3": {"bucket": {"name": "kb-bucket"}, "object": {"key": key}}}
                for key in (
                    "knowledge/new%2Brunbook.md",
                    "knowledge/INC0050000.json",
                    "knowledge/image.png",
                )
            ]
        }

        class Sink:
            def __init__(self) -> None:
                self.items: list[HistoricalIncident] = []

            def put(self, i: HistoricalIncident) -> None:
                self.items.append(i)

            def get_many(self, ids: Any) -> list[HistoricalIncident]:
                return []

            def recent_for_service(self, service: str, limit: int) -> list[HistoricalIncident]:
                return []

        sink = Sink()
        before = len(local.pipeline.vector)
        stats = await reindex(local.pipeline, iter_s3_documents(s3, event, sink))
        assert stats.documents == 2
        assert [i.incident_id for i in sink.items] == ["INC0050000"]
        assert len(local.pipeline.vector) > before

        assert snapshot.save_if_unchanged(local.pipeline.vector, snapshot.version())
        vectors, keyword = snapshot.load(dim=256)
        assert len(vectors) == len(keyword) == len(local.pipeline.vector)
        assert keyword.search("rates poller")[0].chunk.doc_id == "new+runbook"


@pytest.mark.usefixtures("aws")
class TestCloudWatchLogs:
    async def test_logs_server_reads_window_and_ranks_signatures(self) -> None:
        logs = boto3.client("logs", region_name=REGION)
        logs.create_log_group(logGroupName="/app/payments-api")
        logs.create_log_stream(logGroupName="/app/payments-api", logStreamName="task-1")
        now_ms = int(time.time() * 1000)
        messages = [
            "ERROR ERR_GATEWAY_TIMEOUT latency=9001ms id=aa11bb22",
            "ERROR ERR_GATEWAY_TIMEOUT latency=8001ms id=cc33dd44",
            "ERROR HikariPool-1 - Connection is not available after 3000ms",
            "INFO all good",
        ]
        logs.put_log_events(
            logGroupName="/app/payments-api",
            logStreamName="task-1",
            logEvents=[{"timestamp": now_ms + i, "message": m} for i, m in enumerate(messages)],
        )
        server = CloudWatchLogsServer(logs)
        alert = Alert(service="payments-api", title="504s", severity=Severity.SEV1)
        plan = RetrievalPlan(RetrievalMode.HYBRID, "q")  # moto ignores filter patterns
        evidence = await server.gather(alert, plan)
        assert evidence[0].title.startswith("2x")
        assert "ERR_GATEWAY_TIMEOUT" in evidence[0].snippet
        assert all(e.ref.startswith("log:") for e in evidence)
        assert all("INFO" not in e.snippet for e in evidence)
