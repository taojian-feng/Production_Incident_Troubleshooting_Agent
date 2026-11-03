"""Client-side rate limiting for concurrent LLM / AWS calls.

Two complementary controls:

* :class:`AsyncTokenBucket` caps *throughput* (requests per second), matching
  Bedrock's per-model RPM quotas so we throttle ourselves before AWS does.
* :func:`gather_bounded` caps *concurrency* (in-flight requests) with a
  semaphore while still using ``asyncio.gather`` to run work concurrently.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable
    from types import TracebackType


class AsyncTokenBucket:
    """Token bucket: ``rate`` tokens/second refill, burst up to ``capacity``.

    Usable as an async context manager::

        async with bucket:
            await client.call()
    """

    def __init__(
        self,
        rate: float,
        capacity: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate <= 0:
            msg = "rate must be positive"
            raise ValueError(msg)
        self._rate = rate
        self._capacity = capacity if capacity is not None else rate
        self._tokens = self._capacity
        self._clock = clock
        self._updated = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
        self._updated = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """Wait until ``tokens`` are available, then consume them.

        The lock serialises waiters so they are served FIFO and the bucket's
        arithmetic is never interleaved across ``await`` points.
        """
        if tokens > self._capacity:
            msg = f"cannot acquire {tokens} tokens from a bucket of capacity {self._capacity}"
            raise ValueError(msg)
        async with self._lock:
            self._refill()
            while self._tokens < tokens:
                await asyncio.sleep((tokens - self._tokens) / self._rate)
                self._refill()
            self._tokens -= tokens

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


async def gather_bounded[T](
    factories: Iterable[Callable[[], Awaitable[T]]],
    *,
    limit: int,
    limiter: AsyncTokenBucket | None = None,
) -> list[T]:
    """Run awaitables concurrently with at most ``limit`` in flight.

    Takes zero-arg *factories* rather than coroutine objects so that nothing
    starts (and no coroutine is left un-awaited) until a slot is free. Results
    are returned in input order, like ``asyncio.gather``. The first failure
    propagates; callers that want partial results wrap their factories.
    """
    if limit < 1:
        msg = "limit must be >= 1"
        raise ValueError(msg)
    semaphore = asyncio.Semaphore(limit)

    async def run(factory: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            if limiter is not None:
                await limiter.acquire()
            return await factory()

    return list(await asyncio.gather(*(run(f) for f in factories)))
