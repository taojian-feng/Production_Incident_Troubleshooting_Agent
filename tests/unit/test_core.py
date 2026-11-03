from __future__ import annotations

import asyncio
import random
from functools import partial
from typing import Any

import anthropic
import httpx2 as httpx
import pytest
from botocore.exceptions import ClientError

from incident_agent.core.errors import TransientError, is_transient
from incident_agent.core.rate_limit import AsyncTokenBucket, gather_bounded
from incident_agent.core.retry import BackoffPolicy, async_retry, retry
from incident_agent.core.timing import Stopwatch, timed

FAST = BackoffPolicy(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0)


def _client_error(code: str, status: int = 400) -> ClientError:
    response: Any = {
        "Error": {"Code": code, "Message": "x"},
        "ResponseMetadata": {"HTTPStatusCode": status},
    }
    return ClientError(response, "Op")


class TestTransientClassification:
    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (_client_error("ThrottlingException"), True),
            (_client_error("Whatever", 503), True),
            (_client_error("ValidationException"), False),
            (TransientError(), True),
            (TimeoutError(), True),
            (ValueError(), False),
        ],
    )
    def test_is_transient(self, exc: BaseException, expected: bool) -> None:
        assert is_transient(exc) is expected

    def test_anthropic_sdk_errors(self) -> None:
        request = httpx.Request("POST", "https://example.invalid")
        rate_limited = anthropic.RateLimitError(
            "slow down", response=httpx.Response(429, request=request), body=None
        )
        bad_request = anthropic.BadRequestError(
            "bad", response=httpx.Response(400, request=request), body=None
        )
        assert is_transient(rate_limited)
        assert not is_transient(bad_request)


class TestBackoffPolicy:
    def test_delay_is_capped_full_jitter(self) -> None:
        policy = BackoffPolicy(base_delay_s=1.0, max_delay_s=4.0)
        rng = random.Random(0)
        delays = [policy.delay(n, rng) for n in range(1, 10)]
        assert all(0 <= d <= 4.0 for d in delays)

    @pytest.mark.parametrize("kwargs", [{"max_attempts": 0}, {"base_delay_s": 5, "max_delay_s": 1}])
    def test_invalid(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            BackoffPolicy(**kwargs)  # type: ignore[arg-type]


class TestRetry:
    def test_retries_transient_then_succeeds(self) -> None:
        calls: list[int] = []
        sleeps: list[float] = []

        @retry(policy=FAST, sleep=sleeps.append)
        def flaky(x: int) -> int:
            calls.append(x)
            if len(calls) < 3:
                raise _client_error("ThrottlingException")
            return x * 2

        assert flaky(21) == 42
        assert len(calls) == 3
        assert len(sleeps) == 2
        assert flaky.__name__ == "flaky"  # functools.wraps preserved metadata

    def test_does_not_retry_permanent_errors(self) -> None:
        calls = 0

        @retry(policy=FAST, sleep=lambda _: None)
        def broken() -> None:
            nonlocal calls
            calls += 1
            raise _client_error("AccessDeniedException")

        with pytest.raises(ClientError):
            broken()
        assert calls == 1

    def test_gives_up_after_max_attempts(self) -> None:
        @retry(policy=FAST, sleep=lambda _: None)
        def always() -> None:
            raise TransientError("still down")

        with pytest.raises(TransientError):
            always()

    async def test_async_retry(self) -> None:
        attempts = 0

        async def no_sleep(_: float) -> None:
            return None

        @async_retry(policy=FAST, sleep=no_sleep)
        async def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TransientError
            return "ok"

        assert await flaky() == "ok"
        assert attempts == 2

    async def test_async_retry_does_not_swallow_cancellation(self) -> None:
        @async_retry(policy=FAST)
        async def cancelled() -> None:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await cancelled()


class TestRateLimit:
    async def test_gather_bounded_limits_concurrency_and_keeps_order(self) -> None:
        in_flight = 0
        peak = 0

        async def work(i: int) -> int:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.001 * (5 - i % 5))
            in_flight -= 1
            return i

        results = await gather_bounded([partial(work, i) for i in range(20)], limit=3)
        assert results == list(range(20))
        assert peak == 3

    async def test_token_bucket_throttles(self) -> None:
        bucket = AsyncTokenBucket(rate=200.0, capacity=2)
        loop = asyncio.get_running_loop()
        start = loop.time()
        for _ in range(6):
            async with bucket:
                pass
        # 2 burst tokens free, 4 more at 200/s >= ~20ms
        assert loop.time() - start >= 0.015

    async def test_token_bucket_validation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            AsyncTokenBucket(rate=0)
        with pytest.raises(ValueError, match="capacity"):
            await AsyncTokenBucket(rate=1, capacity=1).acquire(2)

    async def test_gather_bounded_validation(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            await gather_bounded([], limit=0)


class TestTiming:
    def test_stopwatch_records_failure_and_reraises(self) -> None:
        ticks = iter([1.0, 1.25])
        with pytest.raises(RuntimeError), Stopwatch(clock=lambda: next(ticks)) as watch:
            raise RuntimeError
        assert watch.failed
        assert watch.elapsed_ms == pytest.approx(250.0)

    def test_timed_reports_even_on_error(self) -> None:
        seen: list[tuple[str, bool]] = []
        with pytest.raises(KeyError), timed("lookup", lambda n, _ms, ok: seen.append((n, ok))):
            raise KeyError
        with timed("ok", lambda n, _ms, ok: seen.append((n, ok))):
            pass
        assert seen == [("lookup", False), ("ok", True)]
