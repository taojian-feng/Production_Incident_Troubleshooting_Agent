## What and why

## Reviewer focus
<!-- Machines already checked style, types and tests. Spend review time here: -->
- [ ] Data model: right type for the job (pydantic at boundaries, frozen dataclass inside)?
- [ ] Failure mode: what happens when the dependency is slow, throttled or down?
- [ ] Grounding: can any new output reach an engineer without passing citation validation?
- [ ] IAM / security: is every new permission scoped to specific resources?
- [ ] Tests: does a test fail if this behaviour regresses?

## Checks
- [ ] `make check` is green locally
