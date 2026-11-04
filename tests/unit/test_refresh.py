"""RefreshingValue: rebuild on change, keep serving when a refresh fails."""

from __future__ import annotations

from incident_agent.core.refresh import RefreshingValue


class Source:
    def __init__(self) -> None:
        self.version: str | None = "v1"
        self.builds = 0
        self.fail_version = False
        self.fail_build = False

    def read_version(self) -> str | None:
        if self.fail_version:
            raise OSError("version check failed")
        return self.version

    def build(self) -> str:
        if self.fail_build:
            raise ValueError("build failed")
        self.builds += 1
        return f"value-{self.builds}"

    def cache(self) -> RefreshingValue[str]:
        return RefreshingValue(version=self.read_version, build=self.build)


def test_builds_once_while_the_version_is_unchanged() -> None:
    source = Source()
    cache = source.cache()
    assert [cache.get(), cache.get()] == ["value-1", "value-1"]
    assert source.builds == 1


def test_rebuilds_when_the_version_changes() -> None:
    source = Source()
    cache = source.cache()
    cache.get()
    source.version = "v2"
    assert cache.get() == "value-2"
    assert cache.get() == "value-2"


def test_rebuilds_when_the_source_first_appears() -> None:
    source = Source()
    source.version = None
    cache = source.cache()
    cache.get()
    source.version = "v1"
    assert cache.get() == "value-2"


def test_failed_version_check_keeps_the_previous_value() -> None:
    source = Source()
    cache = source.cache()
    cache.get()
    source.fail_version = True
    assert cache.get() == "value-1"


def test_failed_rebuild_keeps_the_previous_value_and_retries_next_time() -> None:
    source = Source()
    cache = source.cache()
    cache.get()
    source.version = "v2"
    source.fail_build = True
    assert cache.get() == "value-1"
    source.fail_build = False
    assert cache.get() == "value-2"
