from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import RAW_DATA_PATH

logger = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame) -> None:
    required = {
        "date",
        "entity",
        "headline",
        "true_label",
        "source",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["headline"])
    df = df.drop_duplicates()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load raw headlines CSV, validate columns, return cleaned DataFrame."""
    csv_path = path or RAW_DATA_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Run data/load_real_data.py first."
        )

    df = pd.read_csv(
        csv_path,
        parse_dates=["date"],
        dtype={
            "headline": str,
            "entity": str,
            "source": str,
        },
    )

    validate_schema(df)
    df = clean_data(df)

    logger.info("Loaded %d headlines from %s", len(df), csv_path)
    return df


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows → %s", len(df), path)