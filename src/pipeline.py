"""
End-to-end pipeline.

Run from project root:
    python -m src.pipeline
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from .config import (
    DATA_DIR,
    RISK_SUMMARY_PATH,
    SCORED_DATA_PATH,
    TEMPORAL_RISK_PATH,
)
from .data_loader import load_raw, save
from .entity_extractor import enrich_with_entities
from .risk_aggregator import (
    compute_entity_risk_summary,
    compute_rolling_zscore,
)
from .sentiment_model import FinBERTScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("pipeline")


def run(raw_path: Path | None = None) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    t0 = time.perf_counter()

    # 1. Load data
    logger.info("=== 1 / 5 Loading data ===")
    df = load_raw(raw_path)

    # 2. Entity enrichment
    logger.info("=== 2 / 5 Entity enrichment ===")
    df = enrich_with_entities(df)

    # 3. FinBERT scoring
    logger.info("=== 3 / 5 FinBERT scoring ===")
    scorer = FinBERTScorer()
    scored_df = scorer.score_dataframe(df)

    # NEW: Remove PhraseBank rows from live analytics
    live_df = scored_df[scored_df["source"] != "phrasebank"].copy()

    logger.info(
        "Live headlines for risk analytics: %d (excluded %d PhraseBank rows)",
        len(live_df),
        len(scored_df) - len(live_df),
    )

    # 4. Temporal aggregation (ONLY live news)
    logger.info("=== 4 / 5 Temporal risk aggregation ===")
    temporal_df = compute_rolling_zscore(live_df)

    # 5. Risk summary (ONLY live news)
    logger.info("=== 5 / 5 Entity risk summary ===")
    risk_summary = compute_entity_risk_summary(
        live_df,
        temporal_df,
    )

    print("\n=== TOP RISK ENTITIES ===")
    print(risk_summary.to_string(index=False))

    # Source summary (ONLY live news)
    source_summary = (
        live_df.groupby("source")
        .agg(
            headlines=("headline", "count"),
            avg_sentiment=("sentiment_score", "mean"),
        )
        .reset_index()
    )

    # Save outputs
    save(scored_df, SCORED_DATA_PATH)   # save full dataset
    save(temporal_df, TEMPORAL_RISK_PATH)
    save(risk_summary, RISK_SUMMARY_PATH)
    save(source_summary, DATA_DIR / "source_summary.csv")

    elapsed = time.perf_counter() - t0
    logger.info("Pipeline complete in %.2f seconds", elapsed)

    return scored_df, temporal_df, risk_summary


if __name__ == "__main__":
    run()