from __future__ import annotations

"""
CSV-based storage backend.

Replaces PostgreSQL while preserving DB-like interfaces.
Used by:
- orchestrator
- API
- dashboard

Features:
- Atomic writes (prevents corruption)
- Persistent dedup cache
- Append-only article storage
"""

import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import CFG

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
NEWS_CSV = CFG.paths.raw_csv
SCORED_CSV = CFG.paths.scored_csv
TEMPORAL_CSV = CFG.paths.temporal_csv
SUMMARY_CSV = CFG.paths.summary_csv
SOURCE_SUMMARY_CSV = CFG.paths.source_summary_csv
DEDUP_JSON = CFG.paths.data_dir / ".seen_hashes.json"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")

    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            df.to_csv(fh, index=False)
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(tmp_path, path)
        log.debug("Saved %s", path.name)

    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _headline_hash(
    headline: str,
    entity: str = "",
    date: str = "",
    source: str = "",
) -> str:
    payload = (
        f"{headline.strip().lower()}|"
        f"{entity.strip().lower()}|"
        f"{str(date).strip()}|"
        f"{source.strip().lower()}"
    )
    return hashlib.md5(payload.encode()).hexdigest()


# ─────────────────────────────────────────────
# DEDUP CACHE
# ─────────────────────────────────────────────
def load_seen_hashes() -> set[str]:
    if not DEDUP_JSON.exists():
        return set()

    try:
        with DEDUP_JSON.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if isinstance(data, list):
            return set(data)

        log.warning("Dedup cache malformed. Resetting.")
        return set()

    except Exception as e:
        log.warning("Failed loading dedup cache: %s", e)
        return set()


def save_seen_hashes(seen: set[str]) -> None:
    DEDUP_JSON.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=DEDUP_JSON.parent,
        suffix=".tmp"
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(sorted(seen), fh)
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(tmp_path, DEDUP_JSON)

    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─────────────────────────────────────────────
# WRITE OPERATIONS
# ─────────────────────────────────────────────
def upsert_articles(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    df = df.copy()

    df["headline_hash"] = df.apply(
        lambda row: _headline_hash(
            row["headline"],
            row.get("entity", ""),
            row.get("date", ""),
            row.get("source", ""),
        ),
        axis=1,
    )

    existing = _read_csv_safe(NEWS_CSV)

    if not existing.empty:
        if "headline_hash" not in existing.columns:
            existing["headline_hash"] = existing.apply(
                lambda row: _headline_hash(
                    row["headline"],
                    row.get("entity", ""),
                    row.get("date", ""),
                    row.get("source", ""),
                ),
                axis=1,
            )

        known = set(existing["headline_hash"])
    else:
        known = set()

    new_rows = df[~df["headline_hash"].isin(known)]

    if new_rows.empty:
        log.info("No new articles.")
        return 0

    combined = (
        pd.concat([existing, new_rows], ignore_index=True)
        if not existing.empty
        else new_rows
    )

    _atomic_write(combined, NEWS_CSV)

    log.info("Added %d new articles", len(new_rows))
    return len(new_rows)


def write_sentiment_scores(df: pd.DataFrame) -> None:
    if df.empty:
        return

    df = df.copy()

    if "headline_hash" not in df.columns:
        df["headline_hash"] = df.apply(
            lambda row: _headline_hash(
                row["headline"],
                row.get("entity", ""),
                row.get("date", ""),
                row.get("source", ""),
            ),
            axis=1,
        )

    existing = _read_csv_safe(SCORED_CSV)

    if not existing.empty:
        if "headline_hash" not in existing.columns:
            existing["headline_hash"] = existing.apply(
                lambda row: _headline_hash(
                    row["headline"],
                    row.get("entity", ""),
                    row.get("date", ""),
                    row.get("source", ""),
                ),
                axis=1,
            )

        existing = existing[
            ~existing["headline_hash"].isin(df["headline_hash"])
        ]

        combined = pd.concat(
            [existing, df],
            ignore_index=True
        )
    else:
        combined = df

    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(
            combined["date"],
            errors="coerce"
        )
        combined = combined.sort_values("date")

    combined = combined.reset_index(drop=True)

    _atomic_write(combined, SCORED_CSV)

    log.info(
        "Stored %d total scored headlines (%d new)",
        len(combined),
        len(df),
    )


def write_temporal_risk(df: pd.DataFrame) -> None:
    if df.empty:
        return

    _atomic_write(df, TEMPORAL_CSV)


def write_risk_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return

    _atomic_write(df, SUMMARY_CSV)


def write_source_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return

    _atomic_write(df, SOURCE_SUMMARY_CSV)


# ─────────────────────────────────────────────
# READ OPERATIONS
# ─────────────────────────────────────────────
def read_latest_risk_summary() -> pd.DataFrame:
    return _read_csv_safe(SUMMARY_CSV)


def read_scored_headlines(entity: str | None = None) -> pd.DataFrame:
    df = _read_csv_safe(SCORED_CSV, parse_dates=["date"])

    if entity:
        df = df[df["entity"] == entity]

    return df


def read_temporal_risk(entity: str | None = None) -> pd.DataFrame:
    df = _read_csv_safe(TEMPORAL_CSV, parse_dates=["date"])

    if entity:
        df = df[df["entity"] == entity]

    return df


def read_raw_articles(entity: str | None = None) -> pd.DataFrame:
    df = _read_csv_safe(NEWS_CSV, parse_dates=["date"])

    if entity:
        df = df[df["entity"] == entity]

    return df


def read_source_summary() -> pd.DataFrame:
    return _read_csv_safe(SOURCE_SUMMARY_CSV)


def storage_health() -> dict:
    return {
        "news_articles": len(_read_csv_safe(NEWS_CSV)),
        "scored_headlines": len(_read_csv_safe(SCORED_CSV)),
        "temporal_risk": len(_read_csv_safe(TEMPORAL_CSV)),
        "risk_summary": len(_read_csv_safe(SUMMARY_CSV)),
        "source_summary": len(_read_csv_safe(SOURCE_SUMMARY_CSV)),
        "dedup_cache": len(load_seen_hashes()),
    }