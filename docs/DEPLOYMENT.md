# Deployment

## Prerequisites

* AWS account and credentials in your shell (`aws sts get-caller-identity` works).
* `uv`, Node.js 22 (for the CDK CLI via `npx`), Python 3.12+.
* Bedrock model access enabled in the target region for:
  `anthropic.claude-haiku-5-5`, `anthropic.claude-sonnet-5-5`, `anthropic.claude-opus-5-5`
  and `amazon.titan-embed-text-v2:0`.
* CDK bootstrapped once per account/region: `npx aws-cdk@2 bootstrap`.

## Deploy

```bash
uv sync --all-extras --group dev --group infra
make check                 # every gate green first
make deploy STAGE=dev      # builds build/lambda (py3.13, arm64) and runs cdk deploy
```

Outputs: `ApiUrl`, `KnowledgeBucket`, `IncidentsTable`, `NotificationTopic`.

## Use a Claude API key instead of Bedrock

For accounts without Claude access on Bedrock, store the key in Secrets Manager and name
the secret at deploy time. Triage then calls the Claude API; embeddings stay on Titan.
Incident text goes to Anthropic's API instead of staying inside AWS.

```bash
read -rs KEY && aws secretsmanager create-secret --name incident-agent/anthropic-api-key \
  --secret-string "$KEY" --query Name --output text; unset KEY
make deploy STAGE=dev CDK_ARGS="-c anthropic_key_secret=incident-agent/anthropic-api-key"
```

## Load knowledge

```bash
aws s3 cp data/samples/knowledge/ s3://<KnowledgeBucket>/knowledge/ --recursive
# Incidents: one JSON object per file under knowledge/ (also upserted into DynamoDB)
python - <<'PY'
import json, pathlib
for i in json.loads(pathlib.Path("data/samples/incidents.json").read_text()):
    pathlib.Path(f"build/{i['incident_id']}.json").write_text(json.dumps(i))
PY
aws s3 cp build/ s3://<KnowledgeBucket>/knowledge/ --recursive --exclude "*" --include "INC*.json"
```

Each upload triggers the ingest Lambda, which re-indexes and writes `index/` snapshots.
Running functions check the snapshot on every invocation and reload it when it changes,
so new knowledge is used on the next triage without a restart.

## Try it with a synthetic service

`data/synthetic/orders-api` holds made-up runbooks and past incidents for a service that
does not exist. The script loads them, creates the `/app/orders-api` log group and the
`ia-orders-api-5xx` alarm, and then simulates a cache failover that trips the alarm:

```bash
uv run python scripts/synthetic_orders_incident.py load-knowledge
uv run python scripts/synthetic_orders_incident.py setup
uv run python scripts/synthetic_orders_incident.py fire
```

## Connect an alarm

Name the alarm with the `ia-` prefix and put routing metadata in its description:

```bash
aws cloudwatch put-metric-alarm --alarm-name ia-payments-api-5xx \
  --alarm-description $'service=payments-api\nseverity=SEV1\nlog_group=/app/payments-api' \
  ...
```

## Configuration

| Variable / context | Default | Meaning |
| --- | --- | --- |
| `IA_MODEL_LIGHT` / `_STANDARD` / `_HEAVY` | Haiku / Sonnet / Opus 5.5 | Override model ids per tier (e.g. a cross-region inference profile id) |
| `IA_MAX_USD_PER_RUN` | `0.75` | Per-triage spend cap (raises `BudgetExceededError`) |
| `IA_BEDROCK_RPS` | `2.0` | Client-side request rate per Lambda instance |
| `IA_EMBEDDING_DIM` | `512` | Titan v2 output dimensions |
| `-c alarm_name_prefix=` | `ia-` | Which alarms trigger triage |
| `-c log_group_prefix=` | `/app/` | Log groups the triage role may read |
| `-c ingest_reserved_concurrency=` | `1` | Use `0` where the account's Lambda limit is too low to reserve; then upload knowledge one file at a time |
| `-c ui_origins=` | `http://localhost:5173` | CORS origins for the analyst UI |

## Before production: verify

* **Bedrock IAM for the Messages-API endpoint.** Claude is called through the Anthropic
  SDK's Bedrock client (`AnthropicBedrockMantle`), which authorises
  `bedrock-mantle:CreateInference` on `project/*`. Titan embeddings use `bedrock-runtime`
  and `bedrock:InvokeModel`. The stack grants both.
* **Model ids.** If the region requires cross-region inference profiles, set the
  `IA_MODEL_*` overrides.
* **Prices** in `llm/bedrock.py` `default_bedrock_models` are list prices used for the
  cost ledger; replace them with your Bedrock contract rates.
