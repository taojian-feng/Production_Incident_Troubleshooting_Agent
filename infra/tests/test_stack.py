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
