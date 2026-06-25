from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROLLING_WINDOW = 7
ZSCORE_THRESHOLD = 1.5
WEIGHT_NEG_RATIO = 35.0
WEIGHT_SENTIMENT = 25.0
WEIGHT_ANOMALY = 40.0


def compute_rolling_zscore(
    df: pd.DataFrame,
    *,
    entity_col: str = "entity",
    score_col: str = "sentiment_score",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    For each entity, compute rolling z-score over daily average sentiment.
    """
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work = work.sort_values([entity_col, date_col])

    chunks: list[pd.DataFrame] = []

    for entity, group in work.groupby(entity_col, sort=False):
        daily = (
            group.groupby(date_col, sort=True)[score_col]
            .mean()
            .reset_index()
            .rename(columns={score_col: "daily_avg_sentiment"})
        )

        daily["rolling_mean"] = (
            daily["daily_avg_sentiment"]
            .rolling(ROLLING_WINDOW, min_periods=2)
            .mean()
        )

        daily["rolling_std"] = (
            daily["daily_avg_sentiment"]
            .rolling(ROLLING_WINDOW, min_periods=2)
            .std()
        )

        std_safe = daily["rolling_std"].replace(0.0, np.nan)

        daily["z_score"] = (
            (daily["daily_avg_sentiment"] - daily["rolling_mean"])
            / std_safe
        )

        daily["z_score"] = daily["z_score"].fillna(0)
        daily["entity"] = entity

        daily["anomaly_flag"] = (
            daily["z_score"].abs() > ZSCORE_THRESHOLD
        )

        chunks.append(daily)

    return pd.concat(chunks, ignore_index=True)


def compute_entity_risk_summary(
    scored_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    for entity, headlines in scored_df.groupby("entity", sort=False):
        n = len(headlines)

        neg_ratio = (
            (headlines["predicted_label"] == "negative").sum()
            / max(n, 1)
        )

        headlines = headlines.copy()
        headlines["date"] = pd.to_datetime(headlines["date"])

        days_old = (pd.Timestamp.today() - headlines["date"]).dt.days
        headlines["recency_weight"] = np.exp(-days_old / 30)

        weighted_sentiment = (
            headlines["sentiment_score"] * headlines["recency_weight"]
        ).sum() / headlines["recency_weight"].sum()

        avg_sent = float(weighted_sentiment)

        entity_temporal = temporal_df[
            temporal_df["entity"] == entity
        ]

        anomaly_days = int(entity_temporal["anomaly_flag"].sum())
        total_days = max(len(entity_temporal), 1)

        recency_factor = float(headlines["recency_weight"].mean())

        risk_score = (
            neg_ratio * WEIGHT_NEG_RATIO
            + ((-avg_sent + 1.0) / 2.0) * WEIGHT_SENTIMENT
            + (anomaly_days / total_days) * WEIGHT_ANOMALY
        ) * (0.7 + 0.3 * recency_factor)

        if risk_score > 30:
            tier = "HIGH"
        elif risk_score > 15:
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