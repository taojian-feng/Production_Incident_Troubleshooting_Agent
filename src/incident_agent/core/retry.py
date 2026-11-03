"""Retry decorators with exponential backoff and full jitter.

Why our own instead of the SDKs' built-in retries? We want one policy, one
set of metrics and one place that decides what is "transient" across Bedrock,
CloudWatch Logs and DynamoDB. SDK-level retries are switched off where we wrap
a client (``max_retries=0``) so the two layers do not multiply.

Full jitter (``uniform(0, min(cap, base * 2**n))``) is the AWS-recommended
variant: it spreads retries of many concurrent callers instead of having them
hammer the throttled endpoint in lock-step.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from incident_agent.core.errors import is_transient

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    """Immutable retry policy. Frozen, so a shared default can't be mutated."""

    max_attempts: int = 5
    base_delay_s: float = 0.25
    max_delay_s: float = 8.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            msg = "max_attempts must be >= 1"
            raise ValueError(msg)
        if self.base_delay_s < 0 or self.max_delay_s < self.base_delay_s:
            msg = "require 0 <= base_delay_s <= max_delay_s"
            raise ValueError(msg)

    def delay(self, attempt: int, rng: random.Random | None = None) -> float:
        """Full-jitter delay before retry number ``attempt`` (1-based)."""
        ceiling = min(self.max_delay_s, self.base_delay_s * self.multiplier ** (attempt - 1))
        return (rng or _RNG).uniform(0, ceiling)


_RNG = random.Random()  # noqa: S311 - jitter, not cryptography
DEFAULT_POLICY = BackoffPolicy()


def retry[**P, R](
    *,
    policy: BackoffPolicy = DEFAULT_POLICY,
    should_retry: Callable[[BaseException], bool] = is_transient,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry a synchronous function on transient errors.

    ``ParamSpec`` (``**P``) preserves the wrapped function's exact signature for
    type checkers, so decorated functions are still fully typed at call sites.

    Example:
        >>> @retry(policy=BackoffPolicy(max_attempts=3))
        ... def fetch() -> str: ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == policy.max_attempts or not should_retry(exc):
                        raise
                    delay = policy.delay(attempt)
                    logger.warning(
                        "retrying %s after %s (attempt %d/%d, sleeping %.2fs)",
                        func.__qualname__,
                        type(exc).__name__,
                        attempt,
                        policy.max_attempts,
                        delay,
                    )
                    sleep(delay)
            raise AssertionError("unreachable")  # pragma: no cover

        return wrapper

    return decorator


def async_retry[**P, R](
    *,
    policy: BackoffPolicy = DEFAULT_POLICY,
    should_retry: Callable[[BaseException], bool] = is_transient,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Async twin of :func:`retry`; sleeps with ``asyncio.sleep`` (non-blocking).

    ``asyncio.CancelledError`` is a ``BaseException``, not an ``Exception``, so it
    is never caught here: cancelling a triage run stops retries immediately.
    """

    def decorator(
        func: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    if attempt == policy.max_attempts or not should_retry(exc):
                        raise
                    delay = policy.delay(attempt)
                    logger.warning(
                        "retrying %s after %s (attempt %d/%d, sleeping %.2fs)",
                        func.__qualname__,
                        type(exc).__name__,
                        attempt,
                        policy.max_attempts,
                        delay,
                    )
                    await sleep(delay)
            raise AssertionError("unreachable")  # pragma: no cover

        return wrapper

    return decorator
