"""CDK entry point: ``cdk deploy -c env=dev``."""

from __future__ import annotations

import os

import aws_cdk as cdk

from stacks.incident_agent_stack import IncidentAgentStack

app = cdk.App()
stage = app.node.try_get_context("env") or "dev"
IncidentAgentStack(
    app,
    f"IncidentAgent-{stage}",
    stage=stage,
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    ),
)
cdk.Tags.of(app).add("project", "incident-agent")
cdk.Tags.of(app).add("stage", stage)
app.synth()
