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
3. **Gather** (`orchestration/agent.py`). Logs run first, and their top errors are added to
   the search terms and the plan is recalculated, because an alarm's own text is often
   generic. Tickets and knowledge then run concurrently, each on its own forked copy of the
   state. Every call goes through `ToolRegistry`, which enforces the role -> domain policy and
   records success/failure/latency.
4. **Context engineering** (`reasoning/context.py`). At most N items per domain,
   round-robin across domains, under a character budget, numbered `E1..En`.
5. **Reason** (`reasoning/engine.py`). The router picks a model by task and severity;
   the request carries a JSON schema generated from `LLMTriageOutput`. Output is
   re-validated with pydantic, with one repair attempt.
6. **Validate** (`reasoning/citations.py`). Plain code checks every step cites known
   evidence and that every ticket id / error code it mentions appears in what it cites.
   Missing steps are added as unverified. Any unverified step -> `needs_review`.
7. **Ground** (`reasoning/grounding.py`). A lightweight model reads each step that passed
   and its cited sources, and flags any claim the sources do not support. If the check
   cannot finish, the steps are flagged and the report is still delivered.
8. **Persist and notify.** Report to DynamoDB, EMF metrics to CloudWatch, plain text
   report email to SNS (`handlers/notification.py`).
   The Analyst UI (`ui/`) lists and reads reports through `/api/*` and posts feedback.

## Design section -> implementation

| Design | Implementation |
| --- | --- |
| 2.1 Ingestion, unified store | `ingestion/` chunking + pipeline; vector matrix and chunk metadata in one S3 snapshot (`storage/repositories.py` `S3IndexSnapshot`), incidents in DynamoDB. |
| 2.1 Real-time sync | Uploads to `knowledge/` trigger the ingest Lambda, which updates one S3 snapshot with a conditional write, and the knowledge server reloads it on its next invocation. Jira and ServiceNow webhooks (`POST /tickets/*`) are normalised and written to `knowledge/`, and `.json` incidents also upsert the incidents table. A CloudWatch Logs subscription adds each first-seen error pattern. Live logs are read directly by the logs domain. |
| 2.2 Hybrid search | `retrieval/keyword.py` (BM25 + exact-id boost), `retrieval/vector.py` (cosine), `retrieval/hybrid.py` (planner + RRF). |
| 2.3 Domain MCP servers | One Lambda per domain (`handlers/mcp_lambda.py`) serves MCP over an IAM-authenticated Function URL with a role holding only that domain's access. Triage calls them through `tools/mcp_client.py` and holds no direct access to logs, incidents or the index. Locally, tests and the eval harness use the same servers in-process. |
| 3 Context engineering | `reasoning/context.py` |
| 3 Enforced citation validation | `reasoning/citations.py` for citations and identifiers, then `reasoning/grounding.py` for whether the sources support each claim. |
| 3 Structured step-by-step reasoning | `StepKind` order SUMMARY -> PRECEDENT -> RECOMMENDATION, JSON-schema output, `TriageStep` numbering. |
| 4.1 Model flexibility | `LLMProvider` / `EmbeddingProvider` Protocols; Bedrock adapters in `llm/bedrock.py` and a Claude API adapter in `llm/anthropic_api.py`. No Azure OpenAI or Gemini adapter exists yet; each only needs to implement `complete`. |
| 4.1 Task-based routing | `llm/router.py`: reasoning -> Sonnet 5.5, reasoning on SEV1/SEV2 -> Opus 5.5, grounding (classify) -> Haiku 5.5. The extract and summarise routes exist but have no callers yet. Per-run budget in `CostLedger`. |
| 4.2 Telemetry | `observability/metrics.py` EMF: latency, tokens, cost, tool success/failure, unverified steps; CloudWatch dashboard and alarms in the CDK stack; X-Ray tracing on all functions. |
| 4.2 Ground-truth benchmarking | `evaluation/harness.py` scores real models on `data/eval/`; see `docs/EVALUATION.md`. `evaluation/benchmark.py` is a fast offline retrieval check in CI. |
| 4.2 Feedback loop | `POST /reports/{id}/feedback` -> DynamoDB; `incident-agent export-feedback` turns corrected thumbs-down into candidate eval cases for review; `analytics.feedback_quality` reports thumbs-up rate by retrieval mode. |
| 5 Snowflake | Not wired in this iteration: the `IncidentRepository` Protocol is the seam for a Snowflake-backed implementation. |

## AWS resources (infra/stacks/incident_agent_stack.py)

| Resource | Purpose | Notes |
| --- | --- | --- |
| S3 bucket | `knowledge/` uploads, `index/` snapshot | versioned, private, SSE, TLS-only |
| DynamoDB `Incidents` | ticket history, GSI `by_service (service, opened_at)` | on-demand, PITR |
| DynamoDB `Reports`, `Feedback` | reports; feedback keyed `(report_id, created_at)` | on-demand, PITR |
| Lambda `Triage` | the agent | Python 3.13 arm64, 1769 MB, 5 min, DLQ, X-Ray |
| Lambda `Ingest` | index sync | conditional S3 writes make concurrent uploads safe |
| Lambda `LogSync` | first-seen error patterns to knowledge | only with `log_sync_groups` context |
| Lambda `Api` | analyst API | 29 s |
| Lambdas `LogsMcp`, `TicketsMcp`, `KnowledgeMcp` | domain MCP servers | Function URLs with IAM auth, 30 s, one role per domain |
| EventBridge rule | alarm -> triage | prefix-filtered on `ia-` |
| Analyst UI | `ui/` static page on a private S3 bucket behind CloudFront, which also proxies `/api/*`. Cognito hosted sign-in with PKCE, admin-created accounts only | `/api/*` routes need a sign-in token |
| HTTP API | `/alerts`, `/reports/{id}`, `/reports/{id}/feedback`, `/tickets/jira`, `/tickets/servicenow` | IAM authorisation |
| SNS topic | on-call notification | TLS-only publish |
| CloudWatch | dashboard, error and DLQ alarms | metrics from EMF |
| AWS Budget | account-wide monthly spend alert | only with `budget_email` and `monthly_budget_usd` context |

IAM is per function: Bedrock `InvokeModel` only on the three named Claude models, Titan
v2 and inference profiles; `bedrock-mantle:CreateInference` for Claude on Bedrock, or
read access to one named secret when using a Claude API key. Each MCP server's role holds
only its domain's access; `logs:FilterLogEvents` only under the configured log-group
prefix; table and bucket-prefix grants scoped to what each function does.

## Scaling notes

The NumPy index comfortably serves tens of thousands of chunks per Lambda. Beyond that,
implement the `VectorStore` Protocol on OpenSearch Serverless or S3 Vectors; nothing
else changes. Each ingest rewrites the whole snapshot, and a sustained high upload rate
would cause repeated write conflicts, so it would move to an SQS-batched ingest.
