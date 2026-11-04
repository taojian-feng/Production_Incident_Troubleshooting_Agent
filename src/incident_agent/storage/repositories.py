"""Persistence for reports, feedback and the search index snapshot."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol

from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from incident_agent.domain.models import Feedback, TriageReport
from incident_agent.retrieval.keyword import BM25Index
from incident_agent.retrieval.vector import InMemoryVectorIndex

if TYPE_CHECKING:
    from mypy_boto3_dynamodb.service_resource import Table
    from mypy_boto3_s3 import S3Client

_WRITE_CONFLICT_CODES = {"PreconditionFailed", "412", "ConditionalRequestConflict", "409"}


def _to_dynamo(model: TriageReport | Feedback) -> dict[str, Any]:
    """DynamoDB rejects floats; round-trip through JSON with ``Decimal``."""
    result: dict[str, Any] = json.loads(model.model_dump_json(), parse_float=Decimal)
    return result


class ReportStore(Protocol):
    """Report persistence seam."""

    def save(self, report: TriageReport) -> None:
        """Persist a report."""
        ...

    def get(self, report_id: str) -> TriageReport | None:
        """Load a report by id."""
        ...


class FeedbackStore(Protocol):
    """Feedback persistence seam."""

    def add(self, feedback: Feedback) -> None:
        """Persist one feedback entry."""
        ...

    def for_report(self, report_id: str) -> list[Feedback]:
        """All feedback for a report."""
        ...


@dataclass(slots=True)
class DynamoReportStore:
    """``reports`` table, ``pk=report_id``."""

    table: Table

    def save(self, report: TriageReport) -> None:
        """Persist a report (status is a computed field, stored for querying)."""
        self.table.put_item(Item=_to_dynamo(report))

    def get(self, report_id: str) -> TriageReport | None:
        """Load and re-validate; ``status`` is recomputed, so drop the stored copy."""
        item = self.table.get_item(Key={"report_id": report_id}).get("Item")
        if item is None:
            return None
        item.pop("status", None)
        return TriageReport.model_validate_json(json.dumps(item, default=str))


@dataclass(slots=True)
class DynamoFeedbackStore:
    """``feedback`` table, ``pk=report_id, sk=created_at``."""

    table: Table

    def add(self, feedback: Feedback) -> None:
        """Persist one feedback entry."""
        self.table.put_item(Item=_to_dynamo(feedback))

    def scan_all(self) -> list[Feedback]:
        """Every feedback entry, for exporting candidate evaluation cases."""
        response = self.table.scan()
        items = list(response["Items"])
        while "LastEvaluatedKey" in response:
            response = self.table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            items.extend(response["Items"])
        return [Feedback.model_validate_json(json.dumps(i, default=str)) for i in items]

    def for_report(self, report_id: str) -> list[Feedback]:
        """All feedback for a report, oldest first."""
        items = self.table.query(KeyConditionExpression=Key("report_id").eq(report_id))["Items"]
        return [Feedback.model_validate_json(json.dumps(i, default=str)) for i in items]


@dataclass(slots=True)
class InMemoryReportStore:
    """Dict-backed report store for local runs."""

    reports: dict[str, TriageReport] = field(default_factory=dict)

    def save(self, report: TriageReport) -> None:
        """Persist a report."""
        self.reports[report.report_id] = report

    def get(self, report_id: str) -> TriageReport | None:
        """Load a report by id."""
        return self.reports.get(report_id)


@dataclass(slots=True)
class S3IndexSnapshot:
    """One S3 object holding the vector matrix and chunk metadata; BM25 is rebuilt on load.

    A single object means a reader can never see half an update, and writers use
    conditional puts so concurrent updates cannot overwrite each other.
    """

    client: S3Client
    bucket: str
    key: str = "index/snapshot.zip"

    def version(self) -> str | None:
        """ETag of the snapshot; ``None`` if no snapshot exists."""
        try:
            return self.client.head_object(Bucket=self.bucket, Key=self.key)["ETag"]
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                return None
            raise

    def save_if_unchanged(self, index: InMemoryVectorIndex, version: str | None) -> bool:
        """Write only if the snapshot is still at ``version``; ``False`` if someone else wrote."""
        npz, jsonl = index.to_bytes()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("vectors.npz", npz)
            archive.writestr("chunks.jsonl", jsonl)
        body = buffer.getvalue()
        try:
            if version is None:
                self.client.put_object(Bucket=self.bucket, Key=self.key, Body=body, IfNoneMatch="*")
            else:
                self.client.put_object(Bucket=self.bucket, Key=self.key, Body=body, IfMatch=version)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in _WRITE_CONFLICT_CODES:
                return False
            raise
        return True

    def load_versioned(self, dim: int) -> tuple[InMemoryVectorIndex, BM25Index, str | None]:
        """Download the snapshot with its version; an empty index if none exists yet."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                return InMemoryVectorIndex(dim), BM25Index(), None
            raise
        with zipfile.ZipFile(io.BytesIO(response["Body"].read())) as archive:
            vectors = InMemoryVectorIndex.from_bytes(
                archive.read("vectors.npz"), archive.read("chunks.jsonl")
            )
        keyword = BM25Index()
        keyword.add(vectors.chunks())
        return vectors, keyword, response["ETag"]

    def load(self, dim: int) -> tuple[InMemoryVectorIndex, BM25Index]:
        """Download the snapshot; an empty index if none exists yet."""
        vectors, keyword, _version = self.load_versioned(dim)
        return vectors, keyword
