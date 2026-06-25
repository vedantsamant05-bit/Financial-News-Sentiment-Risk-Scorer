from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything else reads os.environ
load_dotenv()

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────
# Safe environment parsing helpers
# ─────────────────────────────────────────────────────────────
def _get_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[WARN] Invalid integer for {key}: {value}. Using {default}.")
        return default


def _get_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[WARN] Invalid float for {key}: {value}. Using {default}.")
        return default


# ─────────────────────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class ModelConfig:
    base_model: str = "Blackdaemonium/finbert-finetuned-risk"
    max_length: int = 128
    batch_size: int = 32

    @property
    def active_model(self) -> str:
        return self.base_model


# ─────────────────────────────────────────────────────────────
# RISK CONFIG
# ─────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class RiskConfig:
    rolling_window: int = 7
    zscore_threshold: float = 2.0
    alert_threshold: float = field(
        default_factory=lambda: _get_float("RISK_ALERT_THRESHOLD", 35.0)
    )

    weight_neg_ratio: float = 40.0
    weight_sentiment: float = 35.0
    weight_anomaly: float = 25.0


# ─────────────────────────────────────────────────────────────
# INGESTION CONFIG
# ─────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class IngestionConfig:
    refresh_interval_minutes: int = field(
        default_factory=lambda: _get_int(
            "REFRESH_INTERVAL_MINUTES",
            60,
        )
    )

    rate_limit_seconds: float = 1.5
    max_articles_per_entity: int = 20

    newsapi_key: str | None = field(
        default_factory=lambda: os.environ.get("NEWSAPI_KEY")
    )


# ─────────────────────────────────────────────────────────────
# PATH CONFIG
# ─────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class PathConfig:
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    models_dir: Path = field(default_factory=lambda: ROOT / "models")
    logs_dir: Path = field(default_factory=lambda: ROOT / "logs")

    raw_csv: Path = field(
        default_factory=lambda: ROOT / "data" / "financial_news.csv"
    )

    scored_csv: Path = field(
        default_factory=lambda: ROOT / "data" / "scored_headlines.csv"
    )

    temporal_csv: Path = field(
        default_factory=lambda: ROOT / "data" / "temporal_risk.csv"
    )

    summary_csv: Path = field(
        default_factory=lambda: ROOT / "data" / "risk_summary.csv"
    )

    source_summary_csv: Path = field(
        default_factory=lambda: ROOT / "data" / "source_summary.csv"
    )


# ─────────────────────────────────────────────────────────────
# APP CONFIG
# ─────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    env: str = field(
        default_factory=lambda: os.environ.get("ENV", "development")
    )


CFG = AppConfig()

# Ensure important directories exist
CFG.paths.data_dir.mkdir(parents=True, exist_ok=True)
CFG.paths.models_dir.mkdir(parents=True, exist_ok=True)
CFG.paths.logs_dir.mkdir(parents=True, exist_ok=True)