# Working in this repository (for Claude Code)

Read `docs/ENGINEERING_STANDARDS.md` before changing code. The essentials:

## Before you say a change is done

Run `make check` (ruff, ruff format, `mypy --strict` on src and tests, pytest with the
85% coverage floor, CDK synthesis tests). Report failures with their output; do not
claim success without a green run.

## Never

* Weaken a gate: no lowering `fail_under`, no new ignores in `pyproject.toml`, no bare
  `# type: ignore` or `# noqa` (always a specific code plus a reason).
* Delete, skip or loosen a failing test to get green.
* Use `Any` in a signature, a mutable default, a naive `datetime`, or a blocking call
  inside `async def`.
* Call a cloud SDK from domain/reasoning code; go through a Protocol in `llm/`, `tools/`
  or `storage/`.
* Broaden IAM in `infra/` to `*` resources.

## Always

* New boundary data -> a frozen pydantic model in `domain/models.py`.
  New internal value -> `@dataclass(frozen=True, slots=True)` in `domain/values.py`.
  (Decision table: `docs/ENGINEERING_STANDARDS.md` section 3.)
* New external dependency -> a `Protocol`, an AWS implementation, and an in-memory or
  fake implementation used by tests.
* New behaviour -> a test named after the behaviour; AWS adapters get a moto test.
* Keep the triage output grounded: any new step type must go through
  `reasoning/citations.py`.

## Commands

```bash
make install   # uv sync
make fmt       # format + safe fixes
make check     # all gates
make demo      # offline end-to-end run
make bench     # ground-truth benchmark
```
