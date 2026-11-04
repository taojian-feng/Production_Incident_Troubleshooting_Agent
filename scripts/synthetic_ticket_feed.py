"""Send synthetic Jira and ServiceNow webhooks to the deployed API.

Run with AWS credentials in the environment. Each ticket is searchable as soon as the
ingest function has indexed it, usually within seconds.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

STACK = "IncidentAgent-dev"
TICKETS = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "tickets"
FEEDS = (
    ("/tickets/jira", "jira-issue-resolved.json"),
    ("/tickets/servicenow", "servicenow-incident-open.json"),
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Post each synthetic payload to its webhook route."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    session = boto3.Session()
    stack = session.client("cloudformation").describe_stacks(StackName=STACK)["Stacks"][0]
    api = next(o["OutputValue"] for o in stack["Outputs"] if o["OutputKey"] == "ApiUrl")
    region = session.region_name
    for path, filename in FEEDS:
        body = (TICKETS / filename).read_bytes()
        request = AWSRequest(
            method="POST",
            url=f"{api}{path}",
            data=body,
            headers={"content-type": "application/json"},
        )
        SigV4Auth(session.get_credentials(), "execute-api", region).add_auth(request)
        http = urllib.request.Request(  # noqa: S310 - the URL is this stack's https API
            request.url, data=body, headers=dict(request.headers), method="POST"
        )
        with urllib.request.urlopen(http, timeout=30) as response:  # noqa: S310 - same URL
            logger.info("%s -> %s %s", path, response.status, response.read().decode())


if __name__ == "__main__":
    main()
