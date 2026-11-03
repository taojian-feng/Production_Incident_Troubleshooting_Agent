from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from incident_agent.domain import LogEvent, SourceDocument
from incident_agent.ingestion.chunking import ChunkingConfig, chunk_document, iter_sections
from incident_agent.ingestion.loaders import iter_markdown_dir, parse_markdown
from incident_agent.ingestion.log_stream import (
    dedupe,
    group_by_stream,
    is_error,
    read_jsonl,
    top_signatures,
)
from incident_agent.ingestion.pipeline import IngestionPipeline
from incident_agent.llm.local import HashingEmbedder
from incident_agent.retrieval.keyword import BM25Index
from incident_agent.retrieval.vector import InMemoryVectorIndex

NOW = datetime(2026, 11, 3, tzinfo=UTC)


class TestChunking:
    def test_sections_carry_heading_breadcrumb(self) -> None:
        md = "# Doc\nintro\n## A\na text\n### A1\ndeep\n## B\nb text"
        paths = [s.path for s in iter_sections(md)]
        assert paths == [("Doc",), ("Doc", "A"), ("Doc", "A", "A1"), ("Doc", "B")]

    def test_windows_overlap_and_ids_are_deterministic(self) -> None:
        words = " ".join(f"w{i}" for i in range(50))
        doc = SourceDocument(doc_id="d", title="Doc", body=f"# Doc\n## Steps\n{words}")
        cfg = ChunkingConfig(size=20, overlap=5, min_words=1)
        chunks = list(chunk_document(doc, cfg))
        assert [c.chunk_id for c in chunks] == ["d#steps:0", "d#steps:1", "d#steps:2"]
        first, second = chunks[0].text.split(), chunks[1].text.split()
        assert first[-5:] == second[:5]  # 5-word overlap
        assert chunks[-1].text.split()[-1] == "w49"  # nothing dropped at the tail
        assert list(chunk_document(doc, cfg)) == chunks

    def test_metadata_and_embedding_header(self) -> None:
        doc = SourceDocument(
            doc_id="rb",
            title="Runbook X",
            body="# Runbook X\n## Mitigation\nroll back now please",
            service="payments-api",
        )
        (chunk,) = chunk_document(doc, ChunkingConfig(min_words=1))
        assert chunk.metadata["service"] == "payments-api"
        assert chunk.metadata["section"] == "Mitigation"
        assert chunk.text_for_embedding.startswith("[Runbook X > Mitigation]\n")

    def test_invalid_overlap(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            ChunkingConfig(size=10, overlap=10)


class TestLoaders:
    def test_front_matter(self) -> None:
        doc = parse_markdown("x", "---\nservice: auth\ndoc_type: postmortem\n---\n# Title\nbody")
        assert (doc.service, doc.doc_type, doc.title) == ("auth", "postmortem", "Title")
        assert not doc.body.startswith("---")

    def test_sample_dir(self, samples: Path) -> None:
        docs = list(iter_markdown_dir(samples / "knowledge"))
        assert {d.service for d in docs} == {"payments-api", "ledger-service", "auth-service"}


class TestLogStream:
    def test_read_jsonl_streams_and_skips_malformed(self, samples: Path) -> None:
        events = read_jsonl(samples / "logs.jsonl")
        first = next(events)  # lazy: nothing else read yet
        assert isinstance(first, LogEvent)
        assert sum(1 for _ in events) == 80  # 82 lines - 1 consumed - 1 malformed

    def test_read_jsonl_closes_file_on_early_stop(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        path.write_text(json.dumps({"timestamp": 0, "message": "m"}) + "\n")
        gen = read_jsonl(path)
        next(gen)
        gen.close()  # runs the `with` block's exit

    def test_dedupe_is_streaming(self) -> None:
        a = LogEvent("1", NOW, "/g", "boom", "s1")
        dup = LogEvent("2", NOW, "/g", "boom", "s1")
        other_stream = LogEvent("3", NOW, "/g", "boom", "s2")
        assert [e.event_id for e in dedupe(iter([a, dup, other_stream]))] == ["1", "3"]

    def test_top_signatures_counts_and_keeps_first_exemplar(self, samples: Path) -> None:
        errors = (e for e in read_jsonl(samples / "logs.jsonl") if is_error(e))
        top = top_signatures(errors, k=2)
        assert [count for _, count, _ in top] == sorted((c for _, c, _ in top), reverse=True)
        sig, _count, exemplar = top[0]
        assert exemplar.signature == sig

    def test_group_by_stream(self) -> None:
        events = [LogEvent(str(i), NOW, "/g", "m", f"s{i % 2}") for i in range(5)]
        groups = group_by_stream(events)
        assert {k: len(v) for k, v in groups.items()} == {"s0": 3, "s1": 2}


class TestPipeline:
    async def test_ingest_indexes_both_and_upserts(self, samples: Path) -> None:
        embedder = HashingEmbedder(64)
        pipeline = IngestionPipeline(
            embedder, BM25Index(), InMemoryVectorIndex(64), batch_size=2, concurrency=2
        )
        docs = list(iter_markdown_dir(samples / "knowledge"))
        stats = await pipeline.ingest(docs)
        assert stats.documents == 4
        assert stats.chunks == len(pipeline.keyword) == len(pipeline.vector)
        again = await pipeline.ingest(docs[:1])
        assert again.documents == 1
        assert len(pipeline.vector) == stats.chunks  # same ids: replaced, not duplicated
