from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import CFG

log = logging.getLogger(__name__)

MIN_HEADLINES_FOR_ALERT = 5


def check_and_send_alerts(risk_summary: pd.DataFrame) -> list[str]:
    """
    Check for high-risk entities above alert threshold.

    Email functionality removed.
    Alerts are logged only.

    Returns:
        list[str]: alerted entity names
    """
    if risk_summary.empty:
        log.info("Risk summary empty.")
        return []

    required_cols = {
        "risk_tier",
        "composite_risk_score",
        "total_headlines",
    }

    if not required_cols.issubset(risk_summary.columns):
        log.warning("Risk summary missing required columns.")
        return []

    high_risk = risk_summary[
        (risk_summary["risk_tier"] == "HIGH")
        & (
            risk_summary["composite_risk_score"]
            >= CFG.risk.alert_threshold
        )
        & (
            risk_summary["total_headlines"]
            >= MIN_HEADLINES_FOR_ALERT
        )
    ]

    if high_risk.empty:
        log.info("No alert-level entities found.")
        return []

    alerted = high_risk["entity"].tolist()

    lines = ["HIGH RISK ENTITIES DETECTED:"]

    for _, row in high_risk.iterrows():
        lines.append(
            f"- {row['entity']}: "
            f"score={row['composite_risk_score']:.1f}/100 | "
            f"negative={row['negative_ratio'] * 100:.1f}% | "
            f"anomaly_days={row['anomaly_days']} | "
            f"headlines={row['total_headlines']}"
        )

    body = "\n".join(lines)
    log.warning("\n%s", body)

    return alerted