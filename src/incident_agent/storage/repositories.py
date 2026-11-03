"""Persistence for reports, feedback and the search index snapshot."""

from __future__ import annotations

import json
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
    """Stores the vector matrix + chunk metadata in S3; rebuilds BM25 on load."""

    client: S3Client
    bucket: str
    prefix: str = "index/"

    def save(self, index: InMemoryVectorIndex) -> None:
        """Upload ``vectors.npz`` and ``chunks.jsonl``."""
        npz, jsonl = index.to_bytes()
        self.client.put_object(Bucket=self.bucket, Key=f"{self.prefix}vectors.npz", Body=npz)
        self.client.put_object(Bucket=self.bucket, Key=f"{self.prefix}chunks.jsonl", Body=jsonl)

    def load(self, dim: int) -> tuple[InMemoryVectorIndex, BM25Index]:
        """Download the snapshot; an empty index if none exists yet."""
        try:
            npz = self.client.get_object(Bucket=self.bucket, Key=f"{self.prefix}vectors.npz")
            jsonl = self.client.get_object(Bucket=self.bucket, Key=f"{self.prefix}chunks.jsonl")
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                return InMemoryVectorIndex(dim), BM25Index()
            raise
        vectors = InMemoryVectorIndex.from_bytes(npz["Body"].read(), jsonl["Body"].read())
        keyword = BM25Index()
        keyword.add(vectors.chunks())
        return vectors, keyword
