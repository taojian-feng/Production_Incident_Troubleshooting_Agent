"""Context managers for timing and resource cleanup.

Shows both styles on purpose:

* :class:`Stopwatch` - class-based (``__enter__``/``__exit__``), useful when the
  object carries state the caller reads afterwards (``elapsed_ms``).
* :func:`timed` - generator-based ``@contextmanager``, the concise choice when
  you only need setup/teardown around a block.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import TracebackType

logger = logging.getLogger(__name__)


class Stopwatch:
    """Measures wall time of a ``with`` block; records whether it raised."""

    __slots__ = ("_clock", "_start", "elapsed_ms", "failed")

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._start = 0.0
        self.elapsed_ms = 0.0
        self.failed = False

    def __enter__(self) -> Self:
        self._start = self._clock()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_ms = (self._clock() - self._start) * 1000
        self.failed = exc_type is not None
        # Returning None (falsy) re-raises any exception: we measure, never swallow.


@contextmanager
def timed(name: str, sink: Callable[[str, float, bool], None] | None = None) -> Iterator[None]:
    """Time a block and report ``(name, ms, ok)`` to ``sink`` - even on error.

    The ``try/finally`` guarantees the metric is emitted when the block raises,
    which is exactly when you most want the data point.
    """
    start = time.perf_counter()
    ok = False
    try:
        yield
        ok = True
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        (sink or _log_sink)(name, elapsed, ok)


def _log_sink(name: str, elapsed_ms: float, ok: bool) -> None:
    logger.info("timed %s %.1fms ok=%s", name, elapsed_ms, ok)
