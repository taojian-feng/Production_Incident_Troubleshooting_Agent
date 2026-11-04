"""AWS infrastructure for the incident troubleshooting agent.

::

  CloudWatch alarm ──(EventBridge: state=ALARM)──> triage λ ──> DynamoDB reports ──> SNS on-call
                                                     │  ├─ CloudWatch Logs (logs domain)
  Analyst UI ──(HTTP API, IAM auth)──> api λ ──async─┘  ├─ DynamoDB incidents (tickets domain)
                                                        ├─ S3 index snapshot (knowledge domain)
  S3 knowledge/ ──(ObjectCreated)──> ingest λ ──────────┘  └─ Bedrock (Claude + Titan embeddings)

Least privilege throughout: each function gets only the tables, prefixes and
model ARNs it needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_apigatewayv2_authorizers as authorizers
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_lambda_event_sources as sources
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sqs as sqs

if TYPE_CHECKING:
    from constructs import Construct

METRICS_NAMESPACE = "IncidentAgent"
CLAUDE_MODELS = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5")
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"


class IncidentAgentStack(Stack):
    """Everything needed to run the agent in one account/region."""

    def __init__(self, scope: Construct, construct_id: str, *, stage: str, **kwargs: Any) -> None:
        super().__init__(scope, construct_id, **kwargs)
        prod = stage == "prod"
        removal = RemovalPolicy.RETAIN if prod else RemovalPolicy.DESTROY
        asset_path: str = self.node.try_get_context("lambda_asset_path") or "../build/lambda"
        alarm_prefix: str = self.node.try_get_context("alarm_name_prefix") or "ia-"
        log_prefix: str = self.node.try_get_context("log_group_prefix") or "/app/"
        # Name of a Secrets Manager secret holding a Claude API key; unset means Bedrock.
        key_secret: str | None = self.node.try_get_context("anthropic_key_secret")

        # ------------------------------------------------------------------ storage
        bucket = s3.Bucket(
            self,
            "Knowledge",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=removal,
            auto_delete_objects=not prod,
        )

        def table(name: str, pk: str, sk: str | None = None) -> dynamodb.TableV2:
            return dynamodb.TableV2(
                self,
                name,
                partition_key=dynamodb.Attribute(name=pk, type=dynamodb.AttributeType.STRING),
                sort_key=dynamodb.Attribute(name=sk, type=dynamodb.AttributeType.STRING)
                if sk
                else None,
                billing=dynamodb.Billing.on_demand(),
                point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                    point_in_time_recovery_enabled=True
                ),
                removal_policy=removal,
            )

        incidents = table("Incidents", "incident_id")
        incidents.add_global_secondary_index(
            index_name="by_service",
            partition_key=dynamodb.Attribute(name="service", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="opened_at", type=dynamodb.AttributeType.STRING),
        )
        reports = table("Reports", "report_id")
        feedback = table("Feedback", "report_id", "created_at")

        topic = sns.Topic(self, "TriageNotifications", display_name="Incident triage reports")
        topic.add_to_resource_policy(
            iam.PolicyStatement(
                effect=iam.Effect.DENY,
                principals=[iam.AnyPrincipal()],
                actions=["sns:Publish"],
                resources=[topic.topic_arn],
                conditions={"Bool": {"aws:SecureTransport": "false"}},
            )
        )

        # ------------------------------------------------------------------ functions
        code = lambda_.Code.from_asset(asset_path)
        env = {
            "IA_KNOWLEDGE_BUCKET": bucket.bucket_name,
            "IA_INCIDENTS_TABLE": incidents.table_name,
            "IA_REPORTS_TABLE": reports.table_name,
            "IA_FEEDBACK_TABLE": feedback.table_name,
            "IA_NOTIFY_TOPIC_ARN": topic.topic_arn,
            "IA_LOG_GROUP_PREFIX": log_prefix,
            "IA_MAX_USD_PER_RUN": "0.75",
            "POWERTOOLS_SERVICE_NAME": "incident-agent",
            **({"IA_ANTHROPIC_KEY_SECRET": key_secret} if key_secret else {}),
        }

        def function(
            name: str, handler: str, *, timeout: Duration, memory: int, **extra: Any
        ) -> lambda_.Function:
            return lambda_.Function(
                self,
                name,
                runtime=lambda_.Runtime.PYTHON_3_13,
                architecture=lambda_.Architecture.ARM_64,
                code=code,
                handler=handler,
                timeout=timeout,
                memory_size=memory,
                environment=env,
                tracing=lambda_.Tracing.ACTIVE,
                log_group=logs.LogGroup(
                    self,
                    f"{name}Logs",
                    retention=logs.RetentionDays.ONE_MONTH,
                    removal_policy=removal,
                ),
                **extra,
            )

        dlq = sqs.Queue(
            self,
            "TriageDLQ",
            retention_period=Duration.days(14),
            enforce_ssl=True,
            encryption=sqs.QueueEncryption.SQS_MANAGED,
        )
        triage = function(
            "Triage",
            "incident_agent.handlers.triage.handler",
            timeout=Duration.minutes(5),
            memory=1769,  # = 1 full vCPU: NumPy search + concurrent I/O
            dead_letter_queue=dlq,
            retry_attempts=1,
        )
        ingest = function(
            "Ingest",
            "incident_agent.handlers.ingest.handler",
            timeout=Duration.minutes(5),
            memory=1769,
            # Serialises index snapshot writes. Without it, concurrent uploads can lose entries.
            reserved_concurrent_executions=self._ingest_reservation(),
        )
        api_fn = function(
            "Api",
            "incident_agent.handlers.api.handler",
            timeout=Duration.seconds(29),
            memory=512,
        )
        api_fn.add_environment("IA_TRIAGE_FUNCTION", triage.function_name)

        # ------------------------------------------------------------------ permissions
        self._allow_models(triage, ingest, key_secret)
        self._add_budget_alert(stage)
        for fn in (triage, ingest):
            bucket.grant_read_write(fn, "index/*")
        bucket.grant_read(ingest, "knowledge/*")
        incidents.grant_read_data(triage)
        incidents.grant_read_write_data(ingest)
        reports.grant_write_data(triage)
        reports.grant_read_data(api_fn)
        feedback.grant_read_write_data(api_fn)
        topic.grant_publish(triage)
        triage.grant_invoke(api_fn)
        triage.add_to_role_policy(
            iam.PolicyStatement(
                actions=["logs:FilterLogEvents"],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:{log_prefix}*",
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:{log_prefix}*:*",
                ],
            )
        )

        # ------------------------------------------------------------------ triggers
        events.Rule(
            self,
            "AlarmToTriage",
            description="Triage every agent-managed CloudWatch alarm that enters ALARM",
            event_pattern=events.EventPattern(
                source=["aws.cloudwatch"],
                detail_type=["CloudWatch Alarm State Change"],
                detail={
                    "alarmName": [{"prefix": alarm_prefix}],
                    "state": {"value": ["ALARM"]},
                },
            ),
            targets=[targets.LambdaFunction(triage, dead_letter_queue=dlq, retry_attempts=2)],
        )
        ingest.add_event_source(
            sources.S3EventSource(
                bucket,
                events=[s3.EventType.OBJECT_CREATED],
                filters=[s3.NotificationKeyFilter(prefix="knowledge/")],
            )
        )

        http_api = apigw.HttpApi(
            self,
            "AnalystApi",
            default_authorizer=authorizers.HttpIamAuthorizer(),
            cors_preflight=apigw.CorsPreflightOptions(
                allow_methods=[apigw.CorsHttpMethod.GET, apigw.CorsHttpMethod.POST],
                allow_origins=self.node.try_get_context("ui_origins") or ["http://localhost:5173"],
                allow_headers=[
                    "authorization",
                    "content-type",
                    "x-amz-date",
                    "x-amz-security-token",
                ],
            ),
        )
        integration = integrations.HttpLambdaIntegration("ApiIntegration", api_fn)
        for method, path in (
            (apigw.HttpMethod.POST, "/alerts"),
            (apigw.HttpMethod.GET, "/reports/{report_id}"),
            (apigw.HttpMethod.POST, "/reports/{report_id}/feedback"),
        ):
            http_api.add_routes(path=path, methods=[method], integration=integration)

        # ------------------------------------------------------------------ observability
        def search(metric: str, stat: str, label: str) -> cloudwatch.MathExpression:
            return cloudwatch.MathExpression(
                expression=(
                    f"SEARCH('{{{METRICS_NAMESPACE},Service}} MetricName=\"{metric}\"', "
                    f"'{stat}', 300)"
                ),
                label=label,
                period=Duration.minutes(5),
            )

        dashboard = cloudwatch.Dashboard(
            self, "Dashboard", dashboard_name=f"incident-agent-{stage}"
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Triage latency p90 (ms)",
                left=[search("TriageLatency", "p90", "p90 by service")],
            ),
            cloudwatch.GraphWidget(
                title="LLM spend (micro-USD) and tokens",
                left=[search("CostMicroUSD", "Sum", "cost")],
                right=[search("InputTokens", "Sum", "in"), search("OutputTokens", "Sum", "out")],
            ),
            cloudwatch.GraphWidget(
                title="Grounding: reports needing human review",
                left=[search("NeedsReview", "Sum", "needs review")],
                right=[search("UnverifiedSteps", "Sum", "unverified steps")],
            ),
            cloudwatch.GraphWidget(
                title="Tool-call success vs failure",
                left=[
                    search("ToolCallSuccess", "Sum", "ok"),
                    search("ToolCallFailure", "Sum", "fail"),
                ],
            ),
        )
        cloudwatch.Alarm(
            self,
            "TriageErrors",
            metric=triage.metric_errors(period=Duration.minutes(5)),
            threshold=1,
            evaluation_periods=1,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="Triage Lambda failed; alerts are not being triaged.",
        )
        cloudwatch.Alarm(
            self,
            "TriageDLQNotEmpty",
            metric=dlq.metric_approximate_number_of_messages_visible(period=Duration.minutes(5)),
            threshold=1,
            evaluation_periods=1,
            alarm_description="Alerts failed triage and landed in the DLQ.",
        )

        CfnOutput(self, "ApiUrl", value=http_api.api_endpoint)
        CfnOutput(self, "KnowledgeBucket", value=bucket.bucket_name)
        CfnOutput(self, "IncidentsTable", value=incidents.table_name)
        CfnOutput(self, "NotificationTopic", value=topic.topic_arn)

    def _ingest_reservation(self) -> int | None:
        """Default 1. New accounts have a Lambda limit of 10, which rejects any reservation."""
        configured = self.node.try_get_context("ingest_reserved_concurrency")
        # CLI context values arrive as strings; 0 skips the reservation.
        return (1 if configured is None else int(configured)) or None

    def _add_budget_alert(self, stage: str) -> None:
        """Email on account-wide AWS spend. Both context values are needed to create it."""
        email: str | None = self.node.try_get_context("budget_email")
        limit = self.node.try_get_context("monthly_budget_usd")
        if not email or not limit:
            return
        alerts = (("ACTUAL", 80), ("ACTUAL", 100), ("FORECASTED", 100))
        budgets.CfnBudget(
            self,
            "MonthlyBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name=f"incident-agent-{stage}",
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(amount=float(limit), unit="USD"),
            ),
            notifications_with_subscribers=[
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        notification_type=kind,
                        comparison_operator="GREATER_THAN",
                        threshold=percent,
                        threshold_type="PERCENTAGE",
                    ),
                    subscribers=[
                        budgets.CfnBudget.SubscriberProperty(
                            subscription_type="EMAIL", address=email
                        )
                    ],
                )
                for kind, percent in alerts
            ],
        )

    def _allow_models(
        self, triage: lambda_.Function, ingest: lambda_.Function, key_secret: str | None
    ) -> None:
        """Grant model access: both functions build the Claude client, only triage calls it."""
        model_arns = [
            *(
                f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.{m}"
                for m in CLAUDE_MODELS
            ),
            f"arn:aws:bedrock:{self.region}::foundation-model/{EMBEDDING_MODEL}",
            f"arn:aws:bedrock:*:{self.account}:inference-profile/*",
        ]
        invoke = iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            resources=model_arns,
        )
        triage.add_to_role_policy(invoke)
        ingest.add_to_role_policy(invoke)
        if key_secret:
            secret = secretsmanager.Secret.from_secret_name_v2(self, "ClaudeApiKey", key_secret)
            secret.grant_read(triage)
            secret.grant_read(ingest)
            return
        # The Mantle (Messages API) endpoint authorises its own action on Bedrock projects.
        triage.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock-mantle:CreateInference"],
                resources=[f"arn:aws:bedrock-mantle:{self.region}:{self.account}:project/*"],
            )
        )
