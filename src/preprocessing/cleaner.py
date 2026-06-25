from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

log = logging.getLogger(__name__)

VALID_LABELS = {"positive", "negative", "neutral", ""}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw ingested headlines.

    Steps:
    - Strip whitespace
    - Validate labels
    - Parse datetime
    - Drop invalid rows
    - Remove duplicates
    """
    df = df.copy()

    df["headline"] = df["headline"].astype(str).str.strip()
    df["entity"] = df["entity"].astype(str).str.strip()

    df["true_label"] = (
        df["true_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .where(lambda s: s.isin(VALID_LABELS), other="")
    )

    # IMPORTANT: keep full datetime for real-time analytics
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date"])
    df = df[df["headline"].str.len() >= 15]
    df = df[df["entity"].notna()]
    df = df[df["entity"] != ""]
    df = df[df["entity"] != "nan"]

    df = df.drop_duplicates(subset=["headline", "entity", "date"])

    return df.sort_values("date").reset_index(drop=True)


def apply_time_decay(
    df: pd.DataFrame,
    half_life_days: int = 3,
) -> pd.DataFrame:
    """
    Apply exponential time decay.

    Headlines today: weight = 1.0
    Headlines 3 days old: weight ≈ 0.5
    Headlines 6 days old: weight ≈ 0.25
    """
    df = df.copy()

    if df.empty:
        df["time_weight"] = []
        return df

    today = datetime.now().date()
    df["published_date"] = pd.to_datetime(df["date"]).dt.date

    def decay_weight(pub_date) -> float:
        age_days = max((today - pub_date).days, 0)
        return 0.5 ** (age_days / half_life_days)

    df["time_weight"] = df["published_date"].apply(decay_weight)

    log.info(
        "Time decay applied | mean weight: %.3f | min: %.3f",
        df["time_weight"].mean(),
        df["time_weight"].min(),
    )

    return df