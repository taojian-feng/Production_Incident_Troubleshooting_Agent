"""Closed vocabularies used across the system.

``StrEnum`` members compare equal to their string values, so they serialise
cleanly to JSON / DynamoDB while still giving exhaustiveness checks in mypy
(``match`` statements over an enum are checked with ``assert_never``).
"""

from enum import StrEnum


class Severity(StrEnum):
    """Incident severity, SEV1 being customer-facing and revenue-impacting."""

    SEV1 = "SEV1"
    SEV2 = "SEV2"
    SEV3 = "SEV3"
    SEV4 = "SEV4"

    @property
    def is_critical(self) -> bool:
        """SEV1/SEV2 are paged and get the high-capability reasoning model."""
        return self in {Severity.SEV1, Severity.SEV2}


class EvidenceSource(StrEnum):
    """Which domain-governed tool server produced a piece of evidence."""

    LOGS = "logs"
    TICKETS = "tickets"
    KNOWLEDGE = "knowledge"


class RetrievalMode(StrEnum):
    """How the hybrid retriever should search for a given alert."""

    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


class StepKind(StrEnum):
    """The fixed, auditable order of a triage report (design section 3)."""

    SUMMARY = "summary"
    PRECEDENT = "precedent"
    RECOMMENDATION = "recommendation"


class ReportStatus(StrEnum):
    """Whether every conclusion in a report was tied to verified evidence."""

    VALIDATED = "validated"
    NEEDS_REVIEW = "needs_review"


class TaskType(StrEnum):
    """LLM task categories used by the FinOps model router."""

    CLASSIFY = "classify"
    EXTRACT = "extract"
    SUMMARIZE = "summarize"
    REASON = "reason"


class ModelTier(StrEnum):
    """Cost/capability tiers; the router maps tasks onto these."""

    LIGHT = "light"
    STANDARD = "standard"
    HEAVY = "heavy"
