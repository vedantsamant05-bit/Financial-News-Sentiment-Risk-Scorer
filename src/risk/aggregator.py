from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import CFG

log = logging.getLogger(__name__)
_R = CFG.risk


def compute_rolling_zscore(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["entity", "date"])

    chunks = []

    for entity, group in work.groupby("entity", sort=False):
        group = group.copy()

        # Skip entities with too little history
        if len(group["date"].dt.date.unique()) < 3:
            continue

        if "time_weight" in group.columns:
            group["weighted_sentiment"] = (
                group["sentiment_score"] * group["time_weight"]
            )
        else:
            group["weighted_sentiment"] = group["sentiment_score"]

        daily = (
            group.groupby("date", sort=True)["weighted_sentiment"]
            .mean()
            .reset_index()
            .rename(columns={"weighted_sentiment": "daily_avg_sentiment"})
        )

        daily["rolling_mean"] = (
            daily["daily_avg_sentiment"]
            .rolling(_R.rolling_window, min_periods=2)
            .mean()
            .shift(1)
        )

        daily["rolling_std"] = (
            daily["daily_avg_sentiment"]
            .rolling(_R.rolling_window, min_periods=2)
            .std()
            .shift(1)
        )

        std_safe = daily["rolling_std"].replace(0.0, np.nan)

        daily["z_score"] = (
            daily["daily_avg_sentiment"] - daily["rolling_mean"]
        ) / std_safe

        daily["entity"] = entity
        daily["anomaly_flag"] = (
            daily["z_score"].abs() > _R.zscore_threshold
        )

        chunks.append(daily)

    if not chunks:
        return pd.DataFrame(
            columns=[
                "date",
                "daily_avg_sentiment",
                "rolling_mean",
                "rolling_std",
                "z_score",
                "entity",
                "anomaly_flag",
            ]
        )

    return pd.concat(chunks, ignore_index=True)


def compute_entity_risk_summary(
    scored_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
) -> pd.DataFrame:
    if scored_df.empty:
        return pd.DataFrame()

    rows = []

    for entity, headlines in scored_df.groupby("entity", sort=False):
        n = len(headlines)

        neg_ratio = (
            (headlines["predicted_label"] == "negative").sum()
            / max(n, 1)
        )

        if "time_weight" in headlines.columns:
            weighted_sent = (
                headlines["sentiment_score"] * headlines["time_weight"]
            )
            avg_sent = float(weighted_sent.mean())
        else:
            avg_sent = float(headlines["sentiment_score"].mean())

        ent_temp = temporal_df[temporal_df["entity"] == entity]

        anomaly_days = (
            int(ent_temp["anomaly_flag"].sum())
            if not ent_temp.empty
            else 0
        )

        total_days = max(len(ent_temp), 1)
        anomaly_ratio = anomaly_days / total_days

        # Only negative sentiment increases risk
        sentiment_risk = max(0.0, -avg_sent)

        risk_score = (
            neg_ratio * _R.weight_neg_ratio
            + sentiment_risk * _R.weight_sentiment
            + anomaly_ratio * _R.weight_anomaly
        )

        # Confidence penalty
        confidence_factor = 0.2 + 0.8 * min(n / 25, 1.0)
        risk_score *= confidence_factor

        # Extra penalty for tiny sample size
        if n < 5:
            risk_score *= 0.5

        risk_score = min(risk_score, 100)

        if risk_score >= 60:
            tier = "HIGH"
        elif risk_score >= 30:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        rows.append(
            {
                "entity": str(entity),
                "total_headlines": n,
                "negative_ratio": round(neg_ratio, 3),
                "avg_sentiment_score": round(avg_sent, 3),
                "anomaly_days": anomaly_days,
                "composite_risk_score": round(risk_score, 2),
                "risk_tier": tier,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("composite_risk_score", ascending=False)
        .reset_index(drop=True)
    )