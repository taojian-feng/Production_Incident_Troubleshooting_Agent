# Engineering standards

This codebase is written largely with Claude Code. That makes standards *more*
important, not less: an AI assistant produces plausible code quickly, so the things
that catch "plausible but wrong" (types, tests, linters, explicit data models) have to
be automatic, strict and non-negotiable. This document is the contract. `CLAUDE.md`
points Claude Code at it, and `make check` enforces it.

## 1. How AI-written code gets to `main`

1. **The human owns the design.** Architecture, data models and module boundaries are
   decided by an engineer (see `docs/ARCHITECTURE.md`); Claude implements within them.
2. **Claude works inside the gates.** `CLAUDE.md` tells it to run `make check` before
   proposing a change and to never weaken a gate (no blanket `# type: ignore`, no
   lowering the coverage floor, no deleting a failing test).
3. **Machines check what machines are good at.** Formatting, lint, `mypy --strict`,
   tests, CDK assertions and the benchmark run in pre-commit and CI. A reviewer never
   spends attention on style.
4. **Humans review what humans are good at.** Reviewers focus on: is the data model
   right, is the failure mode safe, is the IAM scope minimal, does the test actually
   pin the behaviour. The PR template prompts for exactly these.
5. **Every suppression is justified inline.** `# noqa: CODE - reason` and
   `# type: ignore[code]` with a specific code. `mypy` is configured to reject a bare
   `type: ignore` and to flag unused ones.

## 2. Typing

* `mypy --strict` over `src/` **and** `tests/`. Tests are code; untyped tests hide
  mismatches between what we test and what we ship.
* Modern syntax only (enforced by ruff `UP`): `X | None`, `list[str]`, PEP 695 generics
  (`def retry[**P, R](...)`), `Self`.
* `from __future__ import annotations` everywhere; imports used only for annotations
  live under `if TYPE_CHECKING:` (ruff `TC`), which keeps Lambda cold starts lean.
* **Protocols over base classes** for seams (`LLMProvider`, `EmbeddingProvider`,
  `ToolServer`, `IncidentRepository`, `ReportStore`). Implementations do not inherit
  anything; mypy checks they match structurally. This is how the agent stays
  vendor-neutral and how tests swap in fakes without mocking frameworks.
* `ParamSpec` keeps decorators signature-preserving (`core/retry.py`), so a decorated
  function is still fully type-checked at every call site.
* `Any` is banned in signatures (ruff `ANN401`) except at the edge of untyped third-party
  APIs, each with an inline justification.

## 3. Choosing a data structure for a model

| Use | When | Why | Examples |
| --- | --- | --- | --- |
| **pydantic `BaseModel`** (`frozen=True, extra="forbid"`) | Data crossing a trust boundary: events, API bodies, LLM output, DB items, reports | Parsing + validation with good errors, JSON schema generation (we hand it to the LLM as the output contract), immutability after validation | `Alert`, `HistoricalIncident`, `LLMTriageOutput`, `TriageReport`, `Feedback` |
| **`@dataclass(frozen=True, slots=True)`** | Internal value objects on hot paths | No validation overhead, ~half the memory (`slots`), hashable and safe to share across concurrent branches (`frozen`) | `LogEvent`, `Chunk`, `SearchHit`, `TokenUsage`, `ModelSpec`, `BackoffPolicy` |
| **`@dataclass(slots=True)`** (mutable) | State that is *meant* to change, owned by one place | Explicit, typed, cheap; copy semantics documented | `AgentState`, `CostLedger`, `ToolRegistry` |
| **`StrEnum`** | Closed vocabularies | Exhaustive, serialises as plain strings | `Severity`, `StepKind`, `ModelTier` |
| **`typing.Protocol`** | Dependency seams | Structural typing, no inheritance coupling | see §2 |

Rules of thumb:

* Validate **once**, at the boundary. Inside the core, trust the types.
* Prefer immutable. When a frozen object needs to change, derive a new one
  (`model_copy(update=...)`, `dataclasses.replace`).
* Collections inside frozen objects are themselves immutable (`tuple`,
  `MappingProxyType`) so "frozen" is not merely shallow (`Chunk.metadata`).
* Mutable defaults always go through `default_factory`; ruff `B006`/`RUF009` enforce it.

## 4. Errors and resilience

* One exception root (`IncidentAgentError`) so callers can catch our failures without
  swallowing programming errors.
* Transient vs permanent is decided **in one function** (`core.errors.is_transient`) using
  typed SDK exceptions and AWS error codes, never by matching message strings.
* Retries: exponential backoff with full jitter, bounded attempts, one layer only (SDK
  retries are disabled where we wrap a client). `asyncio.CancelledError` is never retried.
* Degrade, don't crash: a failing tool domain yields "no evidence from that domain",
  is counted in metrics, and the report is still produced.
* FinOps guardrail: `CostLedger` raises `BudgetExceededError` when a single triage run
  exceeds its dollar budget.

## 5. Async

* `asyncio` for I/O fan-out (tool domains, embeddings); every concurrent fan-out is
  bounded (`gather_bounded`) and rate-limited (`AsyncTokenBucket`).
* Blocking SDKs (boto3) run in `asyncio.to_thread`; ruff's `ASYNC` rules catch blocking
  calls inside coroutines.
* Concurrent branches never share mutable state: each gets a deep-copied fork
  (`AgentState.fork`). See `tests/unit/test_domain.py::TestStateCopySemantics` for the
  aliasing bug this prevents.

## 6. Determinism around a probabilistic model

* The model only sees evidence we selected and numbered (`reasoning/context.py`).
* Output is constrained by a JSON schema generated from the pydantic model, then
  re-validated by pydantic; one repair attempt, then fail loudly.
* Every step is checked against the evidence by plain code (`reasoning/citations.py`).
  Unverified steps mark the report `needs_review`; the model is never trusted to grade
  itself.
* Model choice is a routing table, not an ad-hoc string (`llm/router.py`).

## 7. Testing

* **Unit tests** for pure logic, with deterministic fakes injected through Protocols.
* **Integration tests** against moto-mocked DynamoDB, S3 and CloudWatch Logs exercise
  the real boto3 code paths (they already caught one real bug: binary S3 objects were
  decoded before their type was checked).
* **Infrastructure tests** assert security invariants on the synthesised template.
* **Benchmark** against ground-truth incidents tracks retrieval recall and grounding.
* Tests name behaviour, not implementation (`test_hallucinated_step_flags_report`).
* Warnings are errors in pytest; a deprecation is fixed when it appears.

## 8. Lint configuration rationale

Selected ruff families and why they earn their place in an AI-assisted codebase:

| Family | Catches |
| --- | --- |
| `B` bugbear | mutable defaults, loop-variable capture in closures, `except` misuse |
| `S` bandit | hard-coded secrets, unsafe deserialisation, `assert` in prod code |
| `ASYNC` | blocking I/O inside coroutines |
| `DTZ` | naive datetimes (a classic incident-timeline bug) |
| `PERF` | accidental quadratic patterns and needless copies |
| `ANN` + `D` | missing annotations and public docstrings |
| `TC` | runtime imports only needed for typing |
| `RUF` | mutable class defaults, ambiguous unicode, unused `noqa` |
