"""A cached value that is rebuilt when its source changes.

Warm Lambda environments keep expensive objects between invocations. This keeps
that benefit while still noticing new data: each read makes one cheap version
check and rebuilds only when the version differs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RefreshingValue[T]:
    """Rebuilds ``build()`` when ``version()`` changes.

    The first build must succeed. After that a failed check or rebuild logs the
    error and returns the previous value, because triage during an incident is
    better served by slightly stale knowledge than by no triage at all.
    """

    version: Callable[[], str | None]
    build: Callable[[], T]
    _value: T | None = field(default=None, init=False)
    _version: str | None = field(default=None, init=False)

    def get(self) -> T:
        """Return the value, rebuilding first if the source has changed."""
        if self._value is None:
            # Read the version first: a change during the build triggers another rebuild.
            current = self.version()
            self._value = self.build()
            self._version = current
            return self._value
        try:
            current = self.version()
            if current != self._version:
                self._value = self.build()
                self._version = current
        except Exception:
            logger.exception("refresh failed; keeping the previous value")
        return self._value
