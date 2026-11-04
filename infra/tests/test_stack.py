"""Synthesis assertions: security and wiring invariants of the stack."""

from __future__ import annotations

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

from stacks.incident_agent_stack import IncidentAgentStack


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Template:
    asset = tmp_path_factory.mktemp("lambda")
    (asset / "placeholder.txt").write_text("asset")
    app = cdk.App(context={"lambda_asset_path": str(asset)})
    stack = IncidentAgentStack(
        app, "Test", stage="test", env=cdk.Environment(account="111111111111", region="us-east-1")
    )
    return Template.from_stack(stack)


def test_three_python_lambdas_on_arm(template: Template) -> None:
    functions = template.find_resources("AWS::Lambda::Function")
    ours = {
        f["Properties"]["Handler"]
        for f in functions.values()
        if str(f["Properties"].get("Handler", "")).startswith("incident_agent.")
    }
    assert ours == {
        "incident_agent.handlers.triage.handler",
        "incident_agent.handlers.ingest.handler",
        "incident_agent.handlers.api.handler",
    }
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {
            "Handler": "incident_agent.handlers.triage.handler",
            "Runtime": "python3.13",
            "Architectures": ["arm64"],
            "TracingConfig": {"Mode": "Active"},
        },
    )
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {"Handler": "incident_agent.handlers.ingest.handler", "ReservedConcurrentExecutions": 1},
    )


def test_bucket_is_private_encrypted_tls_only(template: Template) -> None:
    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            },
            "VersioningConfiguration": {"Status": "Enabled"},
        },
    )
    template.has_resource_properties(
        "AWS::S3::BucketPolicy",
        {
            "PolicyDocument": {
                "Statement": Match.array_with(
                    [
                        Match.object_like(
                            {
                                "Effect": "Deny",
                                "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                            }
                        )
                    ]
                )
            }
        },
    )


def test_tables_have_pitr_and_service_index(template: Template) -> None:
    template.resource_count_is("AWS::DynamoDB::GlobalTable", 3)
    template.has_resource_properties(
        "AWS::DynamoDB::GlobalTable",
        {"GlobalSecondaryIndexes": [Match.object_like({"IndexName": "by_service"})]},
    )
    tables = template.find_resources("AWS::DynamoDB::GlobalTable").values()
    for table in tables:
        replica = table["Properties"]["Replicas"][0]
        assert replica["PointInTimeRecoverySpecification"]["PointInTimeRecoveryEnabled"] is True


def test_alarm_rule_targets_triage(template: Template) -> None:
    template.has_resource_properties(
        "AWS::Events::Rule",
        {
            "EventPattern": {
                "source": ["aws.cloudwatch"],
                "detail-type": ["CloudWatch Alarm State Change"],
                "detail": {"alarmName": [{"prefix": "ia-"}], "state": {"value": ["ALARM"]}},
            }
        },
    )


def test_api_routes_require_iam(template: Template) -> None:
    routes = template.find_resources("AWS::ApiGatewayV2::Route")
    assert {r["Properties"]["RouteKey"] for r in routes.values()} == {
        "POST /alerts",
        "GET /reports/{report_id}",
        "POST /reports/{report_id}/feedback",
    }
    assert {r["Properties"]["AuthorizationType"] for r in routes.values()} == {"AWS_IAM"}


def test_bedrock_access_is_scoped_to_named_models(template: Template) -> None:
    policies = template.find_resources("AWS::IAM::Policy")
    statements = [
        s
        for p in policies.values()
        for s in p["Properties"]["PolicyDocument"]["Statement"]
        if "bedrock:InvokeModel"
        in (s["Action"] if isinstance(s["Action"], list) else [s["Action"]])
    ]
    assert statements, "no Bedrock statement found"
    for statement in statements:
        resources = statement["Resource"]
        assert "*" not in resources
        assert any("claude-opus-5-5" in str(r) for r in resources)


def test_claude_mantle_endpoint_is_authorised_on_projects_only(template: Template) -> None:
    policies = template.find_resources("AWS::IAM::Policy")
    statements = [
        s
        for p in policies.values()
        for s in p["Properties"]["PolicyDocument"]["Statement"]
        if s["Action"] == "bedrock-mantle:CreateInference"
    ]
    assert len(statements) >= 1, "no Mantle inference statement found"
    for statement in statements:
        assert statement["Resource"] != "*"
        assert ":project/*" in str(statement["Resource"])


def test_key_secret_grants_both_functions_read_and_skips_mantle(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    asset = tmp_path_factory.mktemp("lambda-key")
    (asset / "placeholder.txt").write_text("asset")
    app = cdk.App(
        context={"lambda_asset_path": str(asset), "anthropic_key_secret": "incident-agent/key"}
    )
    stack = IncidentAgentStack(
        app,
        "KeyTest",
        stage="test",
        env=cdk.Environment(account="111111111111", region="us-east-1"),
    )
    template = Template.from_stack(stack)
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {
            "Handler": "incident_agent.handlers.triage.handler",
            "Environment": {
                "Variables": Match.object_like({"IA_ANTHROPIC_KEY_SECRET": "incident-agent/key"})
            },
        },
    )
    statements = [
        s
        for p in template.find_resources("AWS::IAM::Policy").values()
        for s in p["Properties"]["PolicyDocument"]["Statement"]
    ]
    secret_reads = [s for s in statements if "secretsmanager:GetSecretValue" in s["Action"]]
    assert len(secret_reads) == 2
    assert all("incident-agent/key" in str(s["Resource"]) for s in secret_reads)
    assert not [s for s in statements if s["Action"] == "bedrock-mantle:CreateInference"]


def test_ingest_reservation_can_be_skipped_for_low_limit_accounts(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    asset = tmp_path_factory.mktemp("lambda-limit")
    (asset / "placeholder.txt").write_text("asset")
    app = cdk.App(context={"lambda_asset_path": str(asset), "ingest_reserved_concurrency": "0"})
    stack = IncidentAgentStack(
        app,
        "LimitTest",
        stage="test",
        env=cdk.Environment(account="111111111111", region="us-east-1"),
    )
    functions = Template.from_stack(stack).find_resources("AWS::Lambda::Function")
    ingest = next(
        f
        for f in functions.values()
        if f["Properties"].get("Handler") == "incident_agent.handlers.ingest.handler"
    )
    assert "ReservedConcurrentExecutions" not in ingest["Properties"]


def test_budget_alert_is_created_only_when_configured(
    template: Template, tmp_path_factory: pytest.TempPathFactory
) -> None:
    template.resource_count_is("AWS::Budgets::Budget", 0)

    asset = tmp_path_factory.mktemp("lambda-budget")
    (asset / "placeholder.txt").write_text("asset")
    app = cdk.App(
        context={
            "lambda_asset_path": str(asset),
            "budget_email": "oncall@example.com",
            "monthly_budget_usd": "20",
        }
    )
    stack = IncidentAgentStack(
        app,
        "BudgetTest",
        stage="test",
        env=cdk.Environment(account="111111111111", region="us-east-1"),
    )
    budget = next(iter(Template.from_stack(stack).find_resources("AWS::Budgets::Budget").values()))
    assert budget["Properties"]["Budget"]["BudgetLimit"] == {"Amount": 20, "Unit": "USD"}
    alerts = budget["Properties"]["NotificationsWithSubscribers"]
    assert [
        (a["Notification"]["NotificationType"], a["Notification"]["Threshold"]) for a in alerts
    ] == [
        ("ACTUAL", 80),
        ("ACTUAL", 100),
        ("FORECASTED", 100),
    ]
    assert alerts[0]["Subscribers"] == [
        {"SubscriptionType": "EMAIL", "Address": "oncall@example.com"}
    ]
