"""Composition root: the only place that wires concrete implementations together.

Everything else depends on Protocols, so the same agent runs against AWS
(``build_aws_components``) or fully offline (``build_local_components``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
import httpx2

from incident_agent.domain.enums import EvidenceSource
from incident_agent.evaluation.harness import log_events
from incident_agent.ingestion.loaders import iter_markdown_dir, load_incidents
from incident_agent.ingestion.log_stream import read_jsonl
from incident_agent.ingestion.pipeline import IngestionPipeline
from incident_agent.llm.anthropic_api import ClaudeApiProvider, default_api_models
from incident_agent.llm.bedrock import (
    ClaudeBedrockProvider,
    TitanEmbedder,
    default_bedrock_models,
)
from incident_agent.llm.local import HashingEmbedder, ScriptedLLM
from incident_agent.llm.router import ModelRouter
from incident_agent.orchestration.agent import TriageAgent
from incident_agent.reasoning.engine import ReasoningEngine
from incident_agent.reasoning.grounding import GroundingChecker
from incident_agent.retrieval.hybrid import HybridRetriever
from incident_agent.retrieval.keyword import BM25Index
from incident_agent.retrieval.vector import InMemoryVectorIndex
from incident_agent.storage.repositories import S3IndexSnapshot
from incident_agent.tools.base import ToolRegistry
from incident_agent.tools.knowledge import KnowledgeServer
from incident_agent.tools.logs import CloudWatchLogsServer, StaticLogsServer
from incident_agent.tools.mcp_client import MCP_TIMEOUT_SECONDS, LambdaUrlAuth, McpToolServer
from incident_agent.tools.tickets import (
    DynamoIncidentRepository,
    IncidentHistoryServer,
    InMemoryIncidentRepository,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from incident_agent.config import Settings
    from incident_agent.evaluation.harness import EvalCase
    from incident_agent.llm.base import EmbeddingProvider, LLMProvider
    from incident_agent.orchestration.agent import TriageAgent as TriageAgentType
    from incident_agent.tools.base import ToolServer

DATA = Path(__file__).resolve().parents[2] / "data"
SAMPLES = DATA / "samples"
SYNTHETIC = DATA / "synthetic" / "orders-api"
EVAL_CASES = DATA / "eval" / "cases.json"
GROUNDING_CASES = DATA / "eval" / "grounding_cases.json"


@dataclass(frozen=True, slots=True)
class Runtime:
    """What a triage run needs: the agent and the registry holding its tool metrics."""

    agent: TriageAgent
    registry: ToolRegistry


@dataclass(frozen=True, slots=True)
class Components(Runtime):
    """The full in-process object graph, used locally and by tests."""

    pipeline: IngestionPipeline
    servers: Mapping[EvidenceSource, ToolServer]


def _runtime(
    llm: LLMProvider,
    router: ModelRouter,
    servers: Mapping[EvidenceSource, ToolServer],
    max_usd_per_run: float,
) -> Runtime:
    registry = ToolRegistry(servers=servers)
    agent = TriageAgent(
        tools=registry,
        reasoner=ReasoningEngine(
            llm=llm, router=router, grounding=GroundingChecker(llm=llm, router=router)
        ),
        max_usd_per_run=max_usd_per_run,
    )
    return Runtime(agent=agent, registry=registry)


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
    runtime = _runtime(llm, router, servers, max_usd_per_run)
    pipeline = IngestionPipeline(embedder=embedder, keyword=keyword, vector=vector)
    return Components(
        agent=runtime.agent, registry=runtime.registry, pipeline=pipeline, servers=servers
    )


async def build_local_components(
    samples: Path = SAMPLES, llm: LLMProvider | None = None
) -> Components:
    """Offline stack over ``data/samples`` with deterministic providers."""
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


async def build_eval_agents(  # pragma: no cover - exercised by the eval command and its tests
    llm: LLMProvider, embedder: EmbeddingProvider, router: ModelRouter, *, max_usd_per_run: float
) -> Callable[[EvalCase], TriageAgentType]:
    """Index the sample and synthetic knowledge once; each case gets its own logs."""
    incidents = [
        *load_incidents(SAMPLES / "incidents.json"),
        *load_incidents(SYNTHETIC / "incidents.json"),
    ]
    keyword = BM25Index()
    vector = InMemoryVectorIndex(dim=embedder.dim)
    tickets = IncidentHistoryServer(InMemoryIncidentRepository.of(incidents))

    def components(logs: ToolServer) -> Components:
        return _assemble(llm, embedder, router, keyword, vector, logs, tickets, max_usd_per_run)

    documents = [
        *iter_markdown_dir(SAMPLES / "knowledge"),
        *iter_markdown_dir(SYNTHETIC / "knowledge"),
        *(incident.to_document() for incident in incidents),
    ]
    await components(StaticLogsServer([])).pipeline.ingest(documents)
    return lambda case: components(StaticLogsServer(log_events(case))).agent


def _claude_backend(  # pragma: no cover - needs AWS
    settings: Settings, session: boto3.Session
) -> tuple[LLMProvider, ModelRouter]:
    """Claude API when a key secret is configured, otherwise Claude on Bedrock."""
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


def _aws_index(  # pragma: no cover - needs AWS
    settings: Settings, session: boto3.Session
) -> tuple[TitanEmbedder, BM25Index, InMemoryVectorIndex]:
    embedder = TitanEmbedder(
        session.client("bedrock-runtime"),
        model_id=settings.embedding_model_id,
        dim=settings.embedding_dim,
    )
    vector, keyword = S3IndexSnapshot(session.client("s3"), settings.knowledge_bucket).load(
        settings.embedding_dim
    )
    return embedder, keyword, vector


def build_domain_server(  # pragma: no cover - needs AWS
    domain: EvidenceSource, settings: Settings
) -> ToolServer:
    """The server behind one domain's MCP Lambda, holding only that domain's clients."""
    session = boto3.Session(region_name=settings.region)
    if domain is EvidenceSource.LOGS:
        prefix = settings.log_group_prefix
        return CloudWatchLogsServer(
            session.client("logs"), default_log_group_for=lambda svc: f"{prefix}{svc}"
        )
    if domain is EvidenceSource.TICKETS:
        table = session.resource("dynamodb").Table(settings.incidents_table)
        return IncidentHistoryServer(DynamoIncidentRepository(table))
    embedder, keyword, vector = _aws_index(settings, session)
    return KnowledgeServer(HybridRetriever(keyword=keyword, vector=vector, embedder=embedder))


def ingest_pipeline_factory(  # pragma: no cover - needs AWS
    settings: Settings,
) -> Callable[[BM25Index, InMemoryVectorIndex], IngestionPipeline]:
    """How the ingest function builds a pipeline around each freshly loaded snapshot."""
    embedder = TitanEmbedder(
        boto3.client("bedrock-runtime", region_name=settings.region),
        model_id=settings.embedding_model_id,
        dim=settings.embedding_dim,
    )
    return lambda keyword, vector: IngestionPipeline(
        embedder=embedder, keyword=keyword, vector=vector
    )


def build_aws_components(settings: Settings) -> Runtime:  # pragma: no cover - needs AWS
    """Production triage: Claude plus the three domains reached over MCP."""
    if not settings.mcp_urls:
        msg = "IA_MCP_LOGS_URL, IA_MCP_TICKETS_URL and IA_MCP_KNOWLEDGE_URL must be set"
        raise KeyError(msg)
    session = boto3.Session(region_name=settings.region)
    llm, router = _claude_backend(settings, session)
    servers: dict[EvidenceSource, ToolServer] = {
        domain: McpToolServer(
            domain=domain,
            url=url,
            client_factory=lambda: httpx2.AsyncClient(
                auth=LambdaUrlAuth(settings.region), timeout=MCP_TIMEOUT_SECONDS
            ),
        )
        for domain, url in settings.mcp_urls.items()
    }
    return _runtime(llm, router, servers, settings.max_usd_per_run)
