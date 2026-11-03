from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from incident_agent.app import SAMPLES, Components, build_local_components
from incident_agent.domain import Alert, HistoricalIncident, Severity
from incident_agent.ingestion.loaders import load_incidents


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a test reach real AWS."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def samples() -> Path:
    return SAMPLES


@pytest.fixture
def incidents(samples: Path) -> list[HistoricalIncident]:
    return load_incidents(samples / "incidents.json")


@pytest.fixture
def payments_alert() -> Alert:
    return Alert(
        service="payments-api",
        title="payments-api 504 rate above 3%",
        description="p99 latency 4.2s, ERR_GATEWAY_TIMEOUT climbing after deploy",
        severity=Severity.SEV1,
        triggered_at=datetime(2026, 11, 3, 16, 25, tzinfo=UTC),
    )


@pytest.fixture
async def local() -> Components:
    return await build_local_components()
