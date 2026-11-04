from __future__ import annotations

import numpy as np
import pytest

from incident_agent.domain import Alert, Chunk, RetrievalMode, SearchHit
from incident_agent.llm.local import HashingEmbedder
from incident_agent.retrieval.hybrid import HybridRetriever, plan_retrieval, reciprocal_rank_fusion
from incident_agent.retrieval.keyword import BM25Index
from incident_agent.retrieval.text import extract_error_codes, extract_ticket_ids, tokenize
from incident_agent.retrieval.vector import InMemoryVectorIndex
from incident_agent.tools.knowledge import KnowledgeServer

CHUNKS = [
    Chunk("a", "d1", "gateway timeout ERR_GATEWAY_TIMEOUT from acquirer", {"service": "pay"}),
    Chunk("b", "d1", "connection pool exhausted after deploy", {"service": "pay"}),
    Chunk("c", "d2", "deadlock detected ORA-00060 in ledger", {"service": "ledger"}),
    Chunk("d", "d3", "token signing key rotation jwks", {"service": "auth"}),
]


class TestText:
    def test_tokenize_keeps_identifiers(self) -> None:
        assert "err_gateway_timeout" in tokenize("Saw ERR_GATEWAY_TIMEOUT now")
        assert "the" not in tokenize("the pool")

    def test_identifier_extraction(self) -> None:
        text = "See INC0041877 and PAY-12, ORA-00060, HTTP 503, ERR_GATEWAY_TIMEOUT, INC0041877"
        assert extract_ticket_ids(text) == ("INC0041877", "PAY-12")
        assert extract_error_codes(text) == ("ORA-00060", "HTTP 503", "ERR_GATEWAY_TIMEOUT")


class TestBM25:
    def test_ranks_relevant_first_and_filters(self) -> None:
        index = BM25Index()
        index.add(CHUNKS)
        assert index.search("deadlock ledger")[0].chunk.chunk_id == "c"
        assert index.search("deadlock", filters={"service": "pay"}) == []

    def test_exact_identifier_boost(self) -> None:
        index = BM25Index()
        index.add(CHUNKS)
        hits = index.search("timeout pool", exact_terms=["ERR_GATEWAY_TIMEOUT"])
        assert hits[0].chunk.chunk_id == "a"

    def test_upsert_and_remove(self) -> None:
        index = BM25Index()
        index.add(CHUNKS)
        index.add([Chunk("a", "d1", "completely different words")])
        assert len(index) == 4
        assert all(h.chunk.chunk_id != "a" for h in index.search("gateway"))
        index.remove("a")
        assert "a" not in index
        assert BM25Index().search("x") == []


class TestVectorIndex:
    async def test_cosine_search_filters_and_roundtrip(self) -> None:
        embedder = HashingEmbedder(64)
        index = InMemoryVectorIndex(64)
        index.upsert(CHUNKS, await embedder.embed([c.text for c in CHUNKS]))
        (query,) = await embedder.embed(["deadlock ledger ORA-00060"])
        assert index.search(query, k=1)[0].chunk.chunk_id == "c"
        assert [h.chunk.chunk_id for h in index.search(query, filters={"service": "auth"})] == ["d"]

        restored = InMemoryVectorIndex.from_bytes(*index.to_bytes())
        assert len(restored) == 4
        assert restored.search(query, k=1)[0].chunk == index.search(query, k=1)[0].chunk

    def test_upsert_overwrites_existing_rows(self) -> None:
        index = InMemoryVectorIndex(2)
        index.upsert(CHUNKS[:1], np.array([[1.0, 0.0]], dtype=np.float32))
        index.upsert(CHUNKS[:1], np.array([[0.0, 1.0]], dtype=np.float32))
        assert len(index) == 1
        hit = index.search(np.array([0.0, 1.0], dtype=np.float32), k=1)[0]
        assert hit.score == pytest.approx(1.0)

    def test_shape_validation(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            InMemoryVectorIndex(3).upsert(CHUNKS[:1], np.zeros((1, 2), dtype=np.float32))
        assert InMemoryVectorIndex(3).search(np.zeros(3, dtype=np.float32)) == []


class TestHybrid:
    @pytest.mark.parametrize(
        ("title", "description", "mode"),
        [
            ("ORA-00060 errors", "", RetrievalMode.KEYWORD),
            ("Customers cannot log in", "tokens rejected across services", RetrievalMode.VECTOR),
            (
                "ERR_GATEWAY_TIMEOUT rising",
                "card authorisations slow for most merchants since the deploy this afternoon "
                "and the acquirer dashboard looks normal so far for every region we checked",
                RetrievalMode.HYBRID,
            ),
        ],
    )
    def test_plan_mode_from_alert_structure(
        self, title: str, description: str, mode: RetrievalMode
    ) -> None:
        plan = plan_retrieval(Alert(service="s", title=title, description=description))
        assert plan.mode is mode

    def test_rrf_rewards_agreement(self) -> None:
        a, b, c = (SearchHit(ch, 1.0, "keyword") for ch in CHUNKS[:3])
        fused = reciprocal_rank_fusion([[a, b], [b, c]])
        assert fused[0].chunk.chunk_id == "b"
        assert {h.retriever for h in fused} == {"hybrid"}

    async def test_retriever_falls_back_when_filter_empties_results(self) -> None:
        embedder = HashingEmbedder(64)
        keyword, vector = BM25Index(), InMemoryVectorIndex(64)
        keyword.add(CHUNKS)
        vector.upsert(CHUNKS, await embedder.embed([c.text_for_embedding for c in CHUNKS]))
        retriever = HybridRetriever(keyword, vector, embedder)
        plan = plan_retrieval(Alert(service="x", title="ORA-00060 deadlock in ledger jobs"))
        hits = await retriever.search(plan, k=2, filters={"service": "unknown"})
        assert hits
        assert hits[0].chunk.chunk_id == "c"


def test_snapshot_load_rejects_vectors_and_chunks_from_different_writes() -> None:
    index = InMemoryVectorIndex(dim=2)
    index.upsert(
        [Chunk(chunk_id="c1", doc_id="d1", text="one")],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    old_npz, old_jsonl = index.to_bytes()
    index.upsert(
        [Chunk(chunk_id="c2", doc_id="d2", text="two")],
        np.array([[0.0, 1.0]], dtype=np.float32),
    )
    new_npz, _ = index.to_bytes()

    assert len(InMemoryVectorIndex.from_bytes(old_npz, old_jsonl)) == 1
    with pytest.raises(ValueError, match="snapshot mismatch"):
        InMemoryVectorIndex.from_bytes(new_npz, old_jsonl)


async def test_knowledge_search_adds_the_fix_from_the_best_matching_runbook() -> None:
    def chunk(chunk_id: str, doc_id: str, section: str, text: str) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            metadata={"section": section, "service": "orders-api"},
        )

    chunks = [
        chunk("tax#symptoms", "tax", "Symptoms", "ERR_TAX_QUOTE_STALE on checkout"),
        chunk("tax#mitigation", "tax", "Mitigation", "Rerun the sync job"),
        chunk("tax#escalation", "tax", "Escalation", "Page the secondary"),
        chunk("other#mitigation", "other", "Mitigation", "Restart the cache"),
    ]
    embedder = HashingEmbedder(dim=64)
    keyword = BM25Index()
    keyword.add(chunks)
    vector = InMemoryVectorIndex(dim=64)
    vector.upsert(chunks, await embedder.embed([c.text for c in chunks]))
    server = KnowledgeServer(HybridRetriever(keyword=keyword, vector=vector, embedder=embedder))
    alert = Alert(service="orders-api", title="ERR_TAX_QUOTE_STALE on checkout")

    evidence = await server.gather(alert, plan_retrieval(alert))

    refs = [e.ref for e in evidence]
    assert refs[0] == "tax#symptoms"
    assert "tax#mitigation" in refs
    assert "other#mitigation" not in refs
    assert "tax#escalation" not in refs
