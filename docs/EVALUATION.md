# Evaluation

Two suites measure the agent on labelled synthetic data. Both live in `data/eval/`.

```bash
make eval-offline   # free: proves the harness works, using a scripted model
make eval-live      # real models: costs about $0.35, needs AWS credentials
```

Live runs need the Claude API key in Secrets Manager and Bedrock access for Titan. They stop
starting new cases at `--max-usd`. Scorecards are written to `build/eval/`. Offline scores say
nothing about model quality, so CI only checks that the harness runs.

## Incident suite (`cases.json`)

| Category | What it tests |
| --- | --- |
| `standard` | The agent finds the right precedent and the verified fix |
| `misleading` | The alert's wording points at the wrong cause, or nothing needs fixing |
| `missing_knowledge` | No runbook or resolution exists, so the report must not invent one |
| `adversarial` | Logs or alerts contain instructions aimed at the model |

A case passes only if the expected precedent was retrieved, the expected actions are named, no
forbidden action is recommended, and a judge model answers yes to all four questions: correct
diagnosis, safe actions, honest about gaps, and actionable.

Passing does not depend on a report's status. The scorecard reports the share of reports sent to
human review, because too many false alarms erode trust as well.

## Grounding suite (`grounding_cases.json`)

Labelled claim and source pairs test the grounding check directly. The scorecard reports how
many unsupported claims it caught and how many supported claims it wrongly flagged.

## Adding cases from feedback

```bash
uv run incident-agent export-feedback --reports-table <Reports> --feedback-table <Feedback>
```

This turns thumbs-down feedback with a correction into candidate cases in
`build/eval/candidates.json`. A person reviews them before adding them to `cases.json`.

## Limits

The suites are small and hand-labelled, and the judge is itself a model. Treat results as a
regression signal for changes, not as proof of production accuracy.
