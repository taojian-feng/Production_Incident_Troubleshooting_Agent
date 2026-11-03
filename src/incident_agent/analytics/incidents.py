"""Incident and FinOps analytics with pandas (offline / notebook / batch job).

Rules this module follows (and the tests enforce):

* No Python ``for`` loops over rows - every transformation is vectorised
  (column arithmetic, ``np.where``, ``groupby().agg``).
* Joins are explicit about cardinality (``validate=``) and surface key
  mismatches (``indicator=True``) instead of silently dropping rows.
* Missing data is handled deliberately: unresolved incidents are excluded from
  MTTR, not filled with zero; unknown owners are labelled, not dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

    from incident_agent.domain.models import Feedback, HistoricalIncident, TriageReport

UNOWNED = "unowned"


def incidents_frame(incidents: Iterable[HistoricalIncident]) -> pd.DataFrame:
    """Typed frame from validated models (one row per incident, deduplicated)."""
    frame = pd.DataFrame.from_records(
        [
            i.model_dump(include={"incident_id", "service", "severity", "opened_at", "resolved_at"})
            for i in incidents
        ]
    )
    if frame.empty:
        return frame
    frame["severity"] = frame["severity"].astype("string")
    frame["opened_at"] = pd.to_datetime(frame["opened_at"], utc=True)
    frame["resolved_at"] = pd.to_datetime(frame["resolved_at"], utc=True)
    # Same ticket synced twice (e.g. ServiceNow + Jira mirror): keep the latest view.
    return frame.sort_values("opened_at").drop_duplicates("incident_id", keep="last")


def services_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the service catalog: one row per service."""
    return raw.drop_duplicates(subset="service", keep="first").reset_index(drop=True)


def mttr_by_service(incidents: pd.DataFrame, services: pd.DataFrame) -> pd.DataFrame:
    """Median/p90 time-to-resolve and SLA breach rate per service and team.

    Returns one row per service with columns
    ``team, incidents, resolved, mttr_median_min, mttr_p90_min, sla_breach_rate, sev1_share``.
    """
    merged = incidents.merge(
        services_frame(services),
        on="service",
        how="left",
        validate="many_to_one",  # raises if the catalog still has duplicate services
        indicator=True,
    )
    merged["team"] = merged["team"].fillna(UNOWNED)
    merged["sla_minutes"] = merged["sla_minutes"].fillna(merged["sla_minutes"].median())

    # Vectorised: whole-column datetime subtraction, NaN where unresolved.
    merged["ttr_min"] = (merged["resolved_at"] - merged["opened_at"]).dt.total_seconds() / 60
    merged["breached"] = np.where(
        merged["ttr_min"].isna(), np.nan, (merged["ttr_min"] > merged["sla_minutes"]).astype(float)
    )
    merged["is_sev1"] = (merged["severity"] == "SEV1").astype(int)

    summary = (
        merged.groupby(["service", "team"], as_index=False)
        .agg(
            incidents=("incident_id", "nunique"),
            resolved=("ttr_min", "count"),  # count() ignores NaN = unresolved
            mttr_median_min=("ttr_min", "median"),
            mttr_p90_min=("ttr_min", lambda s: s.quantile(0.9)),
            sla_breach_rate=("breached", "mean"),
            sev1_share=("is_sev1", "mean"),
        )
        .sort_values(["mttr_median_min", "service"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )
    return summary.round(2)


def unmatched_services(incidents: pd.DataFrame, services: pd.DataFrame) -> list[str]:
    """Services that appear in incidents but not in the catalog (data quality check)."""
    merged = (
        incidents[["service"]]
        .drop_duplicates()
        .merge(services_frame(services)[["service"]], on="service", how="left", indicator=True)
    )
    return sorted(merged.loc[merged["_merge"] == "left_only", "service"].tolist())


def feedback_quality(reports: Iterable[TriageReport], feedback: Iterable[Feedback]) -> pd.DataFrame:
    """Thumbs-up rate, cost and validation rate per retrieval mode.

    Reports without feedback are kept (left join) so coverage is visible.
    """
    reports_df = pd.DataFrame.from_records(
        [
            {
                "report_id": r.report_id,
                "retrieval_mode": r.retrieval_mode.value,
                "validated": r.status == "validated",
                "usd": r.cost.usd,
            }
            for r in reports
        ]
    )
    feedback_df = pd.DataFrame.from_records(
        [{"report_id": f.report_id, "up": f.rating == "up"} for f in feedback],
        columns=["report_id", "up"],
    )
    merged = reports_df.merge(feedback_df, on="report_id", how="left", validate="one_to_many")
    merged["has_feedback"] = merged["up"].notna()
    merged["up"] = merged["up"].astype("float")  # NaN stays NaN; mean() skips it
    return (
        merged.groupby("retrieval_mode", as_index=False)
        .agg(
            reports=("report_id", "nunique"),
            feedback=("has_feedback", "sum"),
            thumbs_up_rate=("up", "mean"),
            validated_rate=("validated", "mean"),
            avg_usd=("usd", "mean"),
        )
        .round(4)
    )
