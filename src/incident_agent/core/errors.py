"""Exception hierarchy and transient-error classification.

One base class (``IncidentAgentError``) so callers can catch "our" failures
without swallowing programming errors like ``TypeError``.
"""

from __future__ import annotations

from botocore.exceptions import ClientError, ConnectionError, ReadTimeoutError

# Throttling / capacity codes that AWS services return and that are safe to retry.
TRANSIENT_AWS_CODES: frozenset[str] = frozenset(
    {
        "ThrottlingException",
        "Throttling",
        "TooManyRequestsException",
        "RequestLimitExceeded",
        "ProvisionedThroughputExceededException",
        "ServiceUnavailableException",
        "ServiceUnavailable",
        "InternalServerException",
        "ModelNotReadyException",
        "LimitExceededException",
    }
)


class IncidentAgentError(Exception):
    """Base class for all errors raised by this package."""


class TransientError(IncidentAgentError):
    """A failure that is expected to succeed on retry (429, 5xx, timeouts)."""


class ToolAccessDeniedError(IncidentAgentError):
    """An agent role tried to call a tool domain it is not granted."""


class ToolCallError(IncidentAgentError):
    """A remote tool server could not be reached or reported a failure."""


class BudgetExceededError(IncidentAgentError):
    """A triage run exceeded its token/cost budget (FinOps guardrail)."""


class LLMOutputError(IncidentAgentError):
    """The model returned output that does not match the required schema."""


def is_transient(exc: BaseException) -> bool:
    """Classify an exception as retryable.

    Handles our own ``TransientError``, botocore client/network errors and the
    Anthropic SDK's typed errors (rate limit, 5xx, connection) without
    string-matching messages.
    """
    if isinstance(exc, TransientError | ConnectionError | ReadTimeoutError | TimeoutError):
        return True
    if isinstance(exc, ClientError):
        error = exc.response.get("Error", {})
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        return error.get("Code", "") in TRANSIENT_AWS_CODES or status == 429 or status >= 500
    try:  # anthropic is a runtime dependency, but keep this module importable without it
        import anthropic  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return False
    if isinstance(exc, anthropic.RateLimitError | anthropic.APIConnectionError):
        return True
    return isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500
