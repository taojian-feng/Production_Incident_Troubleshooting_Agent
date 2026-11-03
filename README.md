# Production Incident Troubleshooting Agent

An agent that turns a production alert into a **step-by-step, cited triage report**
in seconds: it pulls the relevant error logs, the closest past incidents and the right
runbook sections, asks Claude on Amazon Bedrock to reason over *only* that evidence, and
then **deterministically verifies every citation** before an engineer sees the result.
Anything it cannot tie to a source is flagged for human review.

Built on AWS (Lambda, Bedrock, DynamoDB, S3, CloudWatch, EventBridge, API Gateway) and
deployed with the AWS CDK. Implements the attached system design; see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the mapping from design section to code.

```
CloudWatch alarm ─▶ EventBridge ─▶ triage λ ─┬─▶ logs domain      (CloudWatch Logs)
                                             ├─▶ tickets domain   (DynamoDB incident history)
                                             ├─▶ knowledge domain (hybrid BM25 + vector search)
                                             ▼
                                   reasoning engine (Claude on Bedrock, JSON schema output)
                                             ▼
                                   citation validator ─▶ report (DynamoDB) ─▶ SNS / analyst API
```

## Try it in 30 seconds (no AWS account needed)

```bash
uv sync --all-extras --group dev --group infra
make demo     # triage a sample SEV1 against bundled logs, tickets and runbooks
make bench    # ground-truth benchmark: precedent recall, keyword coverage, citation validity
```

The offline mode swaps Bedrock for deterministic stand-ins (`HashingEmbedder`,
`ScriptedLLM`) that satisfy the same `Protocol`s, so the whole pipeline (planning,
parallel tool calls, context budgeting, citation validation, FinOps ledger, audit trail)
runs exactly as in production.

## Quality gates

| Gate | Command | Enforced in |
| --- | --- | --- |
| Formatting + 20+ lint rule families (bugbear, bandit, async, naive-datetime, perf, docstrings, annotations...) | `make lint` | pre-commit, CI |
| `mypy --strict` on source **and** tests, pydantic plugin | `make typecheck` | pre-commit, CI |
| 115 unit + moto-mocked AWS integration tests, 85% coverage floor (currently ~98%) | `make test` | CI |
| CDK synthesis assertions (encryption, TLS-only, IAM scoping, PITR, IAM-auth API) | `make test-infra` | CI |
| Offline benchmark against ground-truth incidents | `make bench` | CI |

`make check` runs all of them. The standards behind these gates, and why they matter
more (not less) when Claude Code writes the code, are in
[docs/ENGINEERING_STANDARDS.md](docs/ENGINEERING_STANDARDS.md). A guided tour of the
Python data structures and idioms used, with the file each one lives in, is in
[docs/PYTHON_PATTERNS.md](docs/PYTHON_PATTERNS.md).

## Layout

```
src/incident_agent/
  domain/          enums, pydantic boundary models, frozen dataclass value objects, agent state
  core/            retry/backoff decorators, token bucket + bounded gather, context managers
  ingestion/       section-aware overlapping chunking, streaming log readers, ingest pipeline
  retrieval/       BM25 inverted index, NumPy vector index, query planner + RRF fusion
  llm/             provider Protocols, Claude-on-Bedrock + Titan adapters, router + cost ledger
  tools/           domain-governed tool servers (logs, tickets, knowledge) + MCP exposure
  reasoning/       context budgeting, structured reasoning, citation validation
  orchestration/   typed state graph with per-node audit snapshots; the triage agent
  observability/   CloudWatch Embedded Metric Format emitter
  storage/         DynamoDB repositories, S3 index snapshot
  analytics/       pandas MTTR / SLA / feedback-quality analysis (offline)
  evaluation/      ground-truth benchmark, feedback -> eval-case loop
  handlers/        Lambda entry points: triage, ingest, api
infra/             CDK app (Python) + synthesis tests
data/samples/      runbooks, incidents, logs, service catalog, ground truth
tests/             unit/ and integration/ (moto)
```

## Deploying

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Short version: enable model access for
Claude Haiku/Sonnet/Opus 5.5 and Titan Text Embeddings v2 in Bedrock, then
`make deploy STAGE=dev`.
