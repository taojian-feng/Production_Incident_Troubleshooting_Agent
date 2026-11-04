"""Composition root: the only place that wires concrete implementations together.

Everything else depends on Protocols, so the same agent runs against AWS
(``build_aws_components``) or fully offline (``build_local_components``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from incident_agent.domain.enums import EvidenceSource
from incident_agent.ingestion.loaders import iter_markdown_dir, load_incidents
from incident_agent.ingestion.log_stream import read_jsonl
from incident_agent.ingestion.pipeline import IngestionPipeline
from incident_agent.llm.local import HashingEmbedder, ScriptedLLM
from incident_agent.llm.router import ModelRouter
from incident_agent.orchestration.agent import TriageAgent
from incident_agent.reasoning.engine import ReasoningEngine
from incident_agent.reasoning.grounding import GroundingChecker
from incident_agent.retrieval.hybrid import HybridRetriever
from incident_agent.retrieval.keyword import BM25Index
from incident_agent.retrieval.vector import InMemoryVectorIndex
from incident_agent.tools.base import ToolRegistry
from incident_agent.tools.knowledge import KnowledgeServer
from incident_agent.tools.logs import StaticLogsServer
from incident_agent.tools.tickets import IncidentHistoryServer, InMemoryIncidentRepository

if TYPE_CHECKING:
    from collections.abc import Mapping

    import boto3

    from incident_agent.config import Settings
    from incident_agent.llm.base import EmbeddingProvider, LLMProvider
    from incident_agent.tools.base import ToolServer

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


@dataclass(frozen=True, slots=True)
class Components:
    """The wired object graph."""

    agent: TriageAgent
    pipeline: IngestionPipeline
    servers: Mapping[EvidenceSource, ToolServer]
    registry: ToolRegistry


def _assemble(
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    router: ModelRouter,
    keyword: BM25Index,
    vector: InMemoryVectorIndex,
    logs: ToolServer,
    tickets: ToolServer,
    max_usd_per_run: float,
) -> Components:
    retriever = HybridRetriever(keyword=keyword, vector=vector, embedder=embedder)
    servers: dict[EvidenceSource, ToolServer] = {
        EvidenceSource.LOGS: logs,
        EvidenceSource.TICKETS: tickets,
        EvidenceSource.KNOWLEDGE: KnowledgeServer(retriever),
    }
    registry = ToolRegistry(servers=servers)
    agent = TriageAgent(
        tools=registry,
        reasoner=ReasoningEngine(
            llm=llm, router=router, grounding=GroundingChecker(llm=llm, router=router)
        ),
        max_usd_per_run=max_usd_per_run,
    )
    pipeline = IngestionPipeline(embedder=embedder, keyword=keyword, vector=vector)
    return Components(agent=agent, pipeline=pipeline, servers=servers, registry=registry)


async def build_local_components(
    samples: Path = SAMPLES, llm: LLMProvider | None = None
) -> Components:
    """Offline stack over ``data/samples`` with deterministic providers."""
    from incident_agent.llm.bedrock import default_bedrock_models  # noqa: PLC0415

    embedder = HashingEmbedder(dim=256)
    incidents = load_incidents(samples / "incidents.json")
    components = _assemble(
        llm=llm or ScriptedLLM(),
        embedder=embedder,
        router=ModelRouter(default_bedrock_models()),
        keyword=BM25Index(),
        vector=InMemoryVectorIndex(dim=embedder.dim),
        logs=StaticLogsServer(list(read_jsonl(samples / "logs.jsonl"))),
        tickets=IncidentHistoryServer(InMemoryIncidentRepository.of(incidents)),
        max_usd_per_run=0.75,
    )
    docs = [*iter_markdown_dir(samples / "knowledge"), *(i.to_document() for i in incidents)]
    await components.pipeline.ingest(docs)
    return components


def _claude_backend(  # pragma: no cover - needs AWS
    settings: Settings, session: boto3.Session
) -> tuple[LLMProvider, ModelRouter]:
    """Claude API when a key secret is configured, otherwise Claude on Bedrock."""
    from incident_agent.llm.anthropic_api import (  # noqa: PLC0415
        ClaudeApiProvider,
        default_api_models,
    )
    from incident_agent.llm.bedrock import (  # noqa: PLC0415
        ClaudeBedrockProvider,
        default_bedrock_models,
    )

    if settings.anthropic_key_secret:
        secret = session.client("secretsmanager").get_secret_value(
            SecretId=settings.anthropic_key_secret
        )
        return (
            ClaudeApiProvider.from_api_key(secret["SecretString"].strip(), settings.bedrock_rps),
            ModelRouter(default_api_models(settings.model_overrides)),
        )
    return (
        ClaudeBedrockProvider.from_region(settings.region, settings.bedrock_rps),
        ModelRouter(default_bedrock_models(settings.model_overrides)),
    )


def build_aws_components(settings: Settings) -> Components:  # pragma: no cover - needs AWS
    """Production stack: Bedrock, CloudWatch Logs, DynamoDB, S3 index snapshot."""
    import boto3  # noqa: PLC0415

    from incident_agent.llm.bedrock import TitanEmbedder  # noqa: PLC0415
    from incident_agent.storage.repositories import S3IndexSnapshot  # noqa: PLC0415
    from incident_agent.tools.logs import CloudWatchLogsServer  # noqa: PLC0415
    from incident_agent.tools.tickets import DynamoIncidentRepository  # noqa: PLC0415

    session = boto3.Session(region_name=settings.region)
    embedder = TitanEmbedder(
        session.client("bedrock-runtime"),
        model_id=settings.embedding_model_id,
        dim=settings.embedding_dim,
    )
    vector, keyword = S3IndexSnapshot(session.client("s3"), settings.knowledge_bucket).load(
        settings.embedding_dim
    )
    prefix = settings.log_group_prefix
    llm, router = _claude_backend(settings, session)
    return _assemble(
        llm=llm,
        embedder=embedder,
        router=router,
        keyword=keyword,
        vector=vector,
        logs=CloudWatchLogsServer(
            session.client("logs"), default_log_group_for=lambda svc: f"{prefix}{svc}"
        ),
        tickets=IncidentHistoryServer(
            DynamoIncidentRepository(session.resource("dynamodb").Table(settings.incidents_table))
        ),
        max_usd_per_run=settings.max_usd_per_run,
    )
