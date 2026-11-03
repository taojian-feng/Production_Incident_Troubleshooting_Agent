# Architecture

How each section of the system design is realised in code and on AWS.

## Request flow

1. **Alert.** A CloudWatch alarm whose name starts with `ia-` enters `ALARM`. EventBridge
   routes the state-change event to the **triage Lambda**. Analysts can also submit an
   alert through `POST /alerts`, which invokes the same Lambda asynchronously.
   The alarm description carries `service=`, `severity=` and `log_group=` lines, parsed by
   `Alert.from_cloudwatch_alarm`.
2. **Plan** (`retrieval/hybrid.py` `plan_retrieval`). Deterministic, no LLM spend: alerts
   dominated by identifiers (ticket ids, error codes) use keyword search, prose-only
   alerts use vector search, mixed alerts use both with reciprocal rank fusion.
3. **Gather** (`orchestration/agent.py`). The three domain tool servers run concurrently,
   each on its own forked copy of the state, through `ToolRegistry`, which enforces the
   role -> domain policy and records success/failure/latency.
4. **Context engineering** (`reasoning/context.py`). At most N items per domain,
   round-robin across domains, under a character budget, numbered `E1..En`.
5. **Reason** (`reasoning/engine.py`). The router picks a model by task and severity;
   the request carries a JSON schema generated from `LLMTriageOutput`. Output is
   re-validated with pydantic, with one repair attempt.
6. **Validate** (`reasoning/citations.py`). Plain code checks every step cites known
   evidence and that every ticket id / error code it mentions appears in what it cites.
   Missing steps are added as unverified. Any unverified step -> `needs_review`.
7. **Persist and notify.** Report to DynamoDB, EMF metrics to CloudWatch, summary to SNS.
   The analyst UI reads `GET /reports/{id}` and posts feedback.

## Design section -> implementation

| Design | Implementation |
| --- | --- |
| 2.1 Ingestion, unified store | `ingestion/` chunking + pipeline; vector matrix and chunk metadata snapshot in S3 (`storage/repositories.py` `S3IndexSnapshot`), incidents in DynamoDB. |
| 2.1 Real-time sync | S3 `ObjectCreated` on `knowledge/` -> ingest Lambda (reserved concurrency 1) re-indexes and writes a new snapshot; `.json` incident exports also upsert the incidents table. |
| 2.2 Hybrid search | `retrieval/keyword.py` (BM25 + exact-id boost), `retrieval/vector.py` (cosine), `retrieval/hybrid.py` (planner + RRF). |
| 2.3 Domain MCP servers | `tools/` one server per domain behind a `ToolServer` Protocol and an access policy; `tools/mcp_server.py` exposes each one as its own MCP server for external clients. In Lambda they are called in-process to avoid a per-request subprocess. |
| 3 Context engineering | `reasoning/context.py` |
| 3 Enforced citation validation | `reasoning/citations.py` |
| 3 Structured step-by-step reasoning | `StepKind` order SUMMARY -> PRECEDENT -> RECOMMENDATION, JSON-schema output, `TriageStep` numbering. |
| 4.1 Model flexibility | `LLMProvider` / `EmbeddingProvider` Protocols; Bedrock adapters in `llm/bedrock.py`. An Azure OpenAI or Gemini adapter only needs to implement `complete`. |
| 4.1 Task-based routing | `llm/router.py`: classify/extract -> Haiku 5.5, summarise/reason -> Sonnet 5.5, reasoning on SEV1/SEV2 -> Opus 5.5. Per-run budget in `CostLedger`. |
| 4.2 Telemetry | `observability/metrics.py` EMF: latency, tokens, cost, tool success/failure, unverified steps; CloudWatch dashboard and alarms in the CDK stack; X-Ray tracing on all functions. |
| 4.2 Ground-truth benchmarking | `evaluation/benchmark.py`, `data/samples/ground_truth.json`, `make bench` in CI. |
| 4.2 Feedback loop | `POST /reports/{id}/feedback` -> DynamoDB; `feedback_to_cases` turns corrected thumbs-down into candidate eval cases; `analytics.feedback_quality` reports thumbs-up rate by retrieval mode. |
| 5 Snowflake | Not wired in this iteration: the `IncidentRepository` Protocol is the seam for a Snowflake-backed implementation. |

## AWS resources (infra/stacks/incident_agent_stack.py)

| Resource | Purpose | Notes |
| --- | --- | --- |
| S3 bucket | `knowledge/` uploads, `index/` snapshot | versioned, private, SSE, TLS-only |
| DynamoDB `Incidents` | ticket history, GSI `by_service (service, opened_at)` | on-demand, PITR |
| DynamoDB `Reports`, `Feedback` | reports; feedback keyed `(report_id, created_at)` | on-demand, PITR |
| Lambda `Triage` | the agent | Python 3.13 arm64, 1769 MB, 5 min, DLQ, X-Ray |
| Lambda `Ingest` | index sync | reserved concurrency 1 |
| Lambda `Api` | analyst API | 29 s |
| EventBridge rule | alarm -> triage | prefix-filtered on `ia-` |
| HTTP API | `/alerts`, `/reports/{id}`, `/reports/{id}/feedback` | IAM authorisation |
| SNS topic | on-call notification | TLS-only publish |
| CloudWatch | dashboard, error and DLQ alarms | metrics from EMF |

IAM is per function: Bedrock `InvokeModel` only on the three named Claude models, Titan
v2 and inference profiles; `logs:FilterLogEvents` only under the configured log-group
prefix; table and bucket-prefix grants scoped to what each function does.

## Scaling notes

The NumPy index comfortably serves tens of thousands of chunks per Lambda. Beyond that,
implement the `VectorStore` Protocol on OpenSearch Serverless or S3 Vectors; nothing
else changes. Ingest serialises snapshot writes with reserved concurrency 1; a higher
upload rate would move to an SQS-batched ingest.
