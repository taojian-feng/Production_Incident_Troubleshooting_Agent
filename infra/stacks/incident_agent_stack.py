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

from pathlib import Path
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
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_lambda_event_sources as sources
from aws_cdk import aws_logs as logs
from aws_cdk import aws_logs_destinations as destinations
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3deploy
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sqs as sqs

if TYPE_CHECKING:
    from collections.abc import Callable

    from constructs import Construct

UI_DIR = Path(__file__).resolve().parents[2] / "ui"
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
        )
        api_fn = function(
            "Api",
            "incident_agent.handlers.api.handler",
            timeout=Duration.seconds(29),
            memory=512,
        )
        api_fn.add_environment("IA_TRIAGE_FUNCTION", triage.function_name)

        # ------------------------------------------------------------------ MCP servers
        # One function per domain, each with a role holding only that domain's access.
        logs_mcp = self._add_mcp_server(function, triage, "Logs", "logs", 512)
        tickets_mcp = self._add_mcp_server(function, triage, "Tickets", "tickets", 512)
        knowledge_mcp = self._add_mcp_server(function, triage, "Knowledge", "knowledge", 1024)

        # ------------------------------------------------------------------ permissions
        self._allow_models(triage, (ingest, knowledge_mcp), key_secret)
        self._add_budget_alert(stage)
        bucket.grant_read_write(ingest, "index/*")
        bucket.grant_read(knowledge_mcp, "index/*")
        bucket.grant_read(ingest, "knowledge/*")
        bucket.grant_put(api_fn, "knowledge/*")
        self._add_log_sync(function, bucket, log_prefix)
        incidents.grant_read_data(tickets_mcp)
        incidents.grant_read_write_data(ingest)
        reports.grant_write_data(triage)
        reports.grant_read_data(api_fn)
        feedback.grant_read_write_data(api_fn)
        topic.grant_publish(triage)
        triage.grant_invoke(api_fn)
        logs_mcp.add_to_role_policy(
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

        http_api = self._add_api(api_fn)
        self._add_observability(stage, triage, dlq)

        self._add_ui(http_api, api_fn, stage)
        CfnOutput(self, "ApiUrl", value=http_api.api_endpoint)
        CfnOutput(self, "KnowledgeBucket", value=bucket.bucket_name)
        CfnOutput(self, "IncidentsTable", value=incidents.table_name)
        CfnOutput(self, "NotificationTopic", value=topic.topic_arn)

    def _add_ui(self, http_api: apigw.HttpApi, api_fn: lambda_.Function, stage: str) -> None:
        """Analyst UI: sign-in, a private static site, and token-authenticated API routes.

        CloudFront serves the page and the API from one origin, so no CORS is needed.
        """
        pool = cognito.UserPool(
            self,
            "Analysts",
            self_sign_up_enabled=False,  # an administrator creates each account
            sign_in_aliases=cognito.SignInAliases(email=True),
            mfa=cognito.Mfa.OPTIONAL,
            mfa_second_factor=cognito.MfaSecondFactor(otp=True, sms=False),
            removal_policy=RemovalPolicy.RETAIN if stage == "prod" else RemovalPolicy.DESTROY,
        )
        domain = pool.add_domain(
            "SignIn",
            cognito_domain=cognito.CognitoDomainOptions(
                domain_prefix=f"incident-agent-{stage}-{self.account}"
            ),
        )
        site = s3.Bucket(
            self,
            "UiSite",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        distribution = self._ui_distribution(site, http_api, domain.base_url())
        page_url = f"https://{distribution.distribution_domain_name}/"
        client = pool.add_client(
            "AnalystUi",
            generate_secret=False,
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL],
                callback_urls=[page_url],
                logout_urls=[page_url],
            ),
        )
        authorizer = authorizers.HttpUserPoolAuthorizer(
            "AnalystToken", pool, user_pool_clients=[client]
        )
        integration = integrations.HttpLambdaIntegration("UiIntegration", api_fn)
        for method, path in (
            (apigw.HttpMethod.GET, "/api/reports"),
            (apigw.HttpMethod.GET, "/api/reports/{report_id}"),
            (apigw.HttpMethod.POST, "/api/reports/{report_id}/feedback"),
        ):
            http_api.add_routes(
                path=path, methods=[method], integration=integration, authorizer=authorizer
            )
        config = self.to_json_string(
            {
                "clientId": client.user_pool_client_id,
                "authDomain": domain.base_url(),
                "redirectUri": page_url,
            }
        )
        s3deploy.BucketDeployment(
            self,
            "UiFiles",
            destination_bucket=site,
            distribution=distribution,
            sources=[
                s3deploy.Source.asset(str(UI_DIR)),
                s3deploy.Source.data("config.js", f"window.IA_CONFIG={config};"),
            ],
        )
        admin_email: str | None = self.node.try_get_context("ui_admin_email")
        if admin_email:
            # Cognito emails this address a temporary password.
            cognito.CfnUserPoolUser(
                self,
                "FirstAnalyst",
                user_pool_id=pool.user_pool_id,
                username=admin_email,
                desired_delivery_mediums=["EMAIL"],
                user_attributes=[
                    cognito.CfnUserPoolUser.AttributeTypeProperty(name="email", value=admin_email),
                    cognito.CfnUserPoolUser.AttributeTypeProperty(
                        name="email_verified", value="true"
                    ),
                ],
            )
        CfnOutput(self, "AnalystUiUrl", value=page_url)

    def _ui_distribution(
        self, site: s3.Bucket, http_api: apigw.HttpApi, sign_in_url: str
    ) -> cloudfront.Distribution:
        headers = cloudfront.ResponseHeadersPolicy(
            self,
            "UiHeaders",
            security_headers_behavior=cloudfront.ResponseSecurityHeadersBehavior(
                content_security_policy=cloudfront.ResponseHeadersContentSecurityPolicy(
                    content_security_policy=(
                        "default-src 'none'; script-src 'self'; style-src 'self'; "
                        f"connect-src 'self' {sign_in_url}; base-uri 'none'; "
                        f"form-action 'self'; frame-ancestors 'none'"
                    ),
                    override=True,
                ),
                content_type_options=cloudfront.ResponseHeadersContentTypeOptions(override=True),
                frame_options=cloudfront.ResponseHeadersFrameOptions(
                    frame_option=cloudfront.HeadersFrameOption.DENY, override=True
                ),
                referrer_policy=cloudfront.ResponseHeadersReferrerPolicy(
                    referrer_policy=cloudfront.HeadersReferrerPolicy.NO_REFERRER, override=True
                ),
                strict_transport_security=cloudfront.ResponseHeadersStrictTransportSecurity(
                    access_control_max_age=Duration.days(365), override=True
                ),
            ),
        )
        return cloudfront.Distribution(
            self,
            "UiCdn",
            default_root_object="index.html",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(site),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                response_headers_policy=headers,
            ),
            additional_behaviors={
                "/api/*": cloudfront.BehaviorOptions(
                    origin=origins.HttpOrigin(
                        f"{http_api.api_id}.execute-api.{self.region}.{self.url_suffix}"
                    ),
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                ),
            },
        )

    def _add_mcp_server(
        self,
        function: Callable[..., lambda_.Function],
        triage: lambda_.Function,
        name: str,
        domain: str,
        memory: int,
    ) -> lambda_.Function:
        server = function(
            f"{name}Mcp",
            "incident_agent.handlers.mcp_lambda.handler",
            timeout=Duration.seconds(30),
            memory=memory,
        )
        server.add_environment("IA_MCP_DOMAIN", domain)
        url = server.add_function_url(auth_type=lambda_.FunctionUrlAuthType.AWS_IAM)
        # Function URLs need both permissions.
        url.grant_invoke_url(triage)
        server.grant_invoke(triage)
        endpoint = f"{url.url}mcp"  # the SDK serves MCP at /mcp; url.url ends with a slash
        triage.add_environment(f"IA_MCP_{domain.upper()}_URL", endpoint)
        CfnOutput(self, f"{name}McpUrl", value=endpoint)
        return server

    def _add_api(self, api_fn: lambda_.Function) -> apigw.HttpApi:
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
            (apigw.HttpMethod.POST, "/tickets/jira"),
            (apigw.HttpMethod.POST, "/tickets/servicenow"),
        ):
            http_api.add_routes(path=path, methods=[method], integration=integration)
        return http_api

    def _add_observability(self, stage: str, triage: lambda_.Function, dlq: sqs.Queue) -> None:
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

    def _add_log_sync(
        self, function: Callable[..., lambda_.Function], bucket: s3.Bucket, log_prefix: str
    ) -> None:
        """Index first-seen error patterns from the log groups named in context."""
        configured: str = self.node.try_get_context("log_sync_groups") or ""
        groups = [group.strip() for group in configured.split(",") if group.strip()]
        if not groups:
            return
        sync = function(
            "LogSync",
            "incident_agent.handlers.log_sync.handler",
            timeout=Duration.seconds(30),
            memory=256,
        )
        bucket.grant_put(sync, "knowledge/log-*")
        for group in groups:
            if not group.startswith(log_prefix):
                msg = f"{group} is outside the configured log group prefix {log_prefix}"
                raise ValueError(msg)
            logs.SubscriptionFilter(
                self,
                f"LogSync{group.replace('/', '').replace('-', '')}",
                log_group=logs.LogGroup.from_log_group_name(self, f"Group{group}", group),
                destination=destinations.LambdaDestination(sync),
                filter_pattern=logs.FilterPattern.any_term("ERROR", "Exception", "FATAL"),
            )

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
        self,
        triage: lambda_.Function,
        embedding_users: tuple[lambda_.Function, ...],
        key_secret: str | None,
    ) -> None:
        """Embeddings for functions that search or index; Claude for triage only."""
        embeddings = iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[f"arn:aws:bedrock:{self.region}::foundation-model/{EMBEDDING_MODEL}"],
        )
        for fn in embedding_users:
            fn.add_to_role_policy(embeddings)
        if key_secret:
            secretsmanager.Secret.from_secret_name_v2(self, "ClaudeApiKey", key_secret).grant_read(
                triage
            )
            return
        triage.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=[
                    *(
                        f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.{m}"
                        for m in CLAUDE_MODELS
                    ),
                    f"arn:aws:bedrock:*:{self.account}:inference-profile/*",
                ],
            )
        )
        # The Mantle (Messages API) endpoint authorises its own action on Bedrock projects.
        triage.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock-mantle:CreateInference"],
                resources=[f"arn:aws:bedrock-mantle:{self.region}:{self.account}:project/*"],
            )
        )
