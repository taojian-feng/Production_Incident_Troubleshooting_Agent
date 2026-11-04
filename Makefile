# One entry point for humans, CI and Claude Code alike: `make check` must pass before every push.
PY ?= uv run
LAMBDA_DIR := build/lambda

.PHONY: install fmt lint typecheck test test-infra check package synth deploy demo bench eval-offline eval-live clean

install:            ## Create .venv with runtime + dev + infra dependencies
	uv sync --all-extras --group dev --group infra

fmt:                ## Auto-format and apply safe lint fixes
	$(PY) ruff format .
	$(PY) ruff check . --fix

lint:               ## Lint + formatting check (no changes)
	$(PY) ruff check .
	$(PY) ruff format --check .

typecheck:          ## mypy --strict on src and tests
	$(PY) mypy src tests

test:               ## Unit + moto integration tests with coverage gate
	$(PY) pytest --cov --cov-report=term-missing

test-infra:         ## CDK synthesis assertions
	$(PY) pytest infra/tests -q

check: lint typecheck test test-infra  ## Everything CI runs

package:            ## Build the Lambda bundle for python3.13 / arm64 (no pandas, no dev deps)
	rm -rf build/lambda build/dist
	uv build --wheel --out-dir build/dist .
	uv pip install --quiet --target $(LAMBDA_DIR) \
	  --python-platform aarch64-manylinux2014 --python-version 3.13 --only-binary :all: \
	  build/dist/*.whl

synth: package      ## Synthesise CloudFormation
	cd infra && npx --yes aws-cdk@2 synth

deploy: package     ## Deploy to the account in your AWS profile (STAGE=dev|prod)
	cd infra && npx --yes aws-cdk@2 deploy -c env=$${STAGE:-dev} $(CDK_ARGS)

demo:               ## Run the agent offline on the sample data
	$(PY) incident-agent triage --service payments-api --severity SEV1 \
	  --title "payments-api 504 rate above 3%" \
	  --description "p99 latency 4.2s, ERR_GATEWAY_TIMEOUT climbing after deploy"

bench:              ## Ground-truth benchmark on the sample data
	$(PY) incident-agent benchmark

eval-offline:       ## Prove the evaluation harness works, with a scripted model (free)
	$(PY) incident-agent eval

eval-live:          ## Score real models on labelled incidents. Costs about $1. Needs AWS credentials
	$(PY) incident-agent eval --live

clean:
	rm -rf build dist .mypy_cache .ruff_cache .pytest_cache .coverage infra/cdk.out
