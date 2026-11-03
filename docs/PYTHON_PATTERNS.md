# Python data structures and patterns: where to find them

A guided tour of the Python techniques this codebase relies on, each tied to a real use
in the agent rather than a toy example. Every entry names the file and symbol, and most
have a test that pins the behaviour.

## 1. Core data structures

| Concept | Where | What it does here |
| --- | --- | --- |
| `dict` as hash map, O(1) lookup | `retrieval/keyword.py` `BM25Index._postings` | Inverted index `term -> {chunk_id: tf}`; scoring touches only chunks that contain a query term. |
| `collections.defaultdict` | `retrieval/hybrid.py` `reciprocal_rank_fusion`; `ingestion/log_stream.py` `group_by_stream`; `reasoning/context.py` `select_evidence` | Accumulate scores / buckets without "if key not in" checks. |
| `collections.Counter` | `ingestion/log_stream.py` `top_signatures`; `retrieval/keyword.py` `add`; `llm/router.py` `CostLedger` | Frequency of error signatures, term frequencies, tokens per model; `most_common(k)`. |
| `set` for membership / dedupe | `ingestion/log_stream.py` `dedupe`; `domain/state.py` `merge_evidence`; `tools/tickets.py` term overlap | O(1) "seen before?" while streaming; set intersection for term overlap. |
| `frozenset` | `tools/base.py` `DEFAULT_POLICY`; `core/errors.py` `TRANSIENT_AWS_CODES` | Immutable, hashable sets for policy tables. |
| `tuple` vs `list` | `dedupe` keys `(stream, message)`; `Section.path`; `TriageReport.steps` | Tuples are hashable (usable as dict/set keys) and signal "fixed"; lists for things that grow. |
| `dict.fromkeys` for ordered dedupe | `retrieval/text.py` `extract_ticket_ids` | Remove duplicates while keeping first-seen order. |
| `heapq.nlargest` | `retrieval/keyword.py` `search` | Top-k in O(n log k) instead of sorting everything. |
| `MappingProxyType` | `domain/values.py` `Chunk.metadata`; `llm/router.py` `TASK_TIERS` | Read-only view so a "frozen" object is frozen all the way down. |
| NumPy vectorisation | `retrieval/vector.py` `search` | One matrix-vector product scores every chunk; `argpartition` gives O(n) top-k. |

Tests: `tests/unit/test_retrieval.py`, `tests/unit/test_ingestion.py::TestLogStream`.

## 2. Mutability and copying

| Concept | Where |
| --- | --- |
| Shallow vs deep copy | `domain/state.py` `AgentState.fork` (deep) and the docstring explaining why `copy.copy` would alias the `evidence` list across parallel branches. |
| Executable demonstration of the aliasing bug | `tests/unit/test_domain.py::TestStateCopySemantics::test_shallow_copy_aliases_nested_lists` |
| Detached audit snapshots | `AgentState.snapshot` deep-copies so later mutation can't rewrite history. |
| Immutable derivation | `TriageReport.model_copy(update=...)` in `handlers/triage.py`. |
| Mutable defaults done right | every `field(default_factory=...)`; ruff `B006`/`RUF009` enforce it. |

## 3. Memory efficiency and streaming

| Concept | Where |
| --- | --- |
| Generators (`yield`) for large files | `ingestion/log_stream.py` `read_jsonl` streams line by line and still closes the file if the consumer stops early. |
| Lazy pagination with a hard cap | `iter_cloudwatch_events`: boto3 paginator + `itertools.islice`. |
| Generator pipelines | `dedupe` -> `is_error` filter -> `top_signatures`, each stage O(1) memory. |
| Batching a stream | `ingestion/pipeline.py` uses `itertools.batched` (twice: chunks into embedding batches, batches into concurrency windows). |
| `__slots__` | all hot-path dataclasses use `slots=True`; `Stopwatch` declares `__slots__`. Test: `test_slots_reduce_per_instance_memory`. |

## 4. Context managers

| Style | Where |
| --- | --- |
| Class-based `__enter__` / `__exit__` | `core/timing.py` `Stopwatch` (records duration and whether the block raised, never swallows). |
| Generator-based `@contextmanager` | `core/timing.py` `timed` (emits a metric in `finally`, so failures are measured too). |
| Async context manager `__aenter__` / `__aexit__` | `core/rate_limit.py` `AsyncTokenBucket` (`async with bucket:` before every Bedrock call). |
| Resource cleanup | `read_jsonl`'s `with path.open()` inside a generator. |

## 5. RAG chunking

`ingestion/chunking.py`:

* Splits Markdown on headings first and keeps the heading **breadcrumb** (`Runbook > Mitigation`).
* Slides a fixed word window with overlap inside each section (`ChunkingConfig(size, overlap)`,
  validated in `__post_init__`).
* Attaches metadata (title, section, doc type, service, position) used for filtering,
  citation and as an embedding header (`Chunk.text_for_embedding`).
* Deterministic chunk ids (`doc#section-slug:n`) make re-ingestion an upsert.

Tests: `tests/unit/test_ingestion.py::TestChunking`.

## 6. pandas

`analytics/incidents.py`:

* `merge(..., validate="many_to_one", indicator=True)` makes join cardinality explicit
  and surfaces key mismatches (`unmatched_services`).
* `drop_duplicates` on both sides: re-synced tickets and a dirty service catalog.
* `fillna` deliberately: unknown owner -> `"unowned"`, missing SLA -> catalog median;
  unresolved incidents stay `NaN` and are *excluded* from MTTR, not zero-filled.
* Vectorised arithmetic: datetime column subtraction, `np.where` for SLA breach,
  boolean-to-int columns; no row loops.
* `groupby().agg(...)` with named aggregations including a quantile.

Tests: `tests/unit/test_analytics_eval.py::TestAnalytics`.

## 7. Async concurrency with rate limiting

| Concept | Where |
| --- | --- |
| `asyncio.gather` fan-out | `orchestration/agent.py` gathers all tool domains in parallel. |
| Bounded concurrency | `core/rate_limit.py` `gather_bounded` (semaphore + gather, results in input order). Used for Titan embeddings, ingestion and the benchmark. |
| Throughput limiting | `AsyncTokenBucket` sized to Bedrock quotas, shared by all calls of a provider. |
| Blocking SDK in async code | `asyncio.to_thread` around boto3 in `llm/bedrock.py`, `tools/logs.py`, `tools/tickets.py`. |

Tests: `tests/unit/test_core.py::TestRateLimit` (peak concurrency is asserted).

## 8. Retry with exponential backoff

`core/retry.py`:

* `@retry` and `@async_retry` decorators, typed with `ParamSpec` so decorated functions
  keep their signatures.
* Full-jitter exponential backoff (`BackoffPolicy`, frozen and validated).
* Transient classification in one place (`core/errors.py` `is_transient`): AWS throttling
  codes and 5xx, Anthropic SDK `RateLimitError` / 5xx / connection errors, timeouts.
* Cancellation is never retried.

Applied to Claude-on-Bedrock and Titan calls in `llm/bedrock.py`.
Tests: `tests/unit/test_core.py::TestRetry`.

## 9. Typing techniques

`Protocol` seams, PEP 695 generics, `ParamSpec`, `Self`, `Literal`, `TypedDict`
narrowing for the Anthropic SDK (`llm/bedrock.py` `_to_param`), pydantic
`Annotated[str, StringConstraints(...)]` and `TypeAdapter` for validating a whole JSON
array at once (`ingestion/loaders.py`).
