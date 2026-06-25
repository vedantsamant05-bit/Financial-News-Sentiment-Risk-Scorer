from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import CFG
from src.ingestion.scraper import run_ingestion
from src.model.inference import FinBERTScorer
from src.preprocessing.cleaner import apply_time_decay, clean
from src.risk.aggregator import (
    compute_entity_risk_summary,
    compute_rolling_zscore,
)
from src.risk.alerts import check_and_send_alerts
from src.storage.csv_store import (
    load_seen_hashes,
    read_scored_headlines,
    save_seen_hashes,
    upsert_articles,
    write_risk_summary,
    write_sentiment_scores,
    write_temporal_risk,
    write_source_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("orchestrator")

_scorer: FinBERTScorer | None = None


def get_scorer() -> FinBERTScorer:
    global _scorer
    if _scorer is None:
        _scorer = FinBERTScorer()
    return _scorer


def _headline_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def run_pipeline() -> None:
    try:
        log.info("══ Pipeline starting ══════════════════════════════════")
        t0 = time.perf_counter()

        seen = load_seen_hashes()

        # STEP 1 — Ingestion
        log.info("── Step 1: Ingestion")
        raw_rows = run_ingestion(seen=seen)

        if raw_rows:
            raw_df = pd.DataFrame(raw_rows)

            # STEP 2 — Cleaning
            log.info("── Step 2: Cleaning")
            clean_df = clean(raw_df)
            clean_df = apply_time_decay(clean_df)

            if clean_df.empty:
                log.warning("No valid headlines after cleaning.")
            else:
                # Existing scored hashes
                existing_scored = read_scored_headlines()
                existing_hashes = set()

                if not existing_scored.empty and "headline" in existing_scored.columns:
                    existing_hashes = set(
                        existing_scored["headline"]
                        .astype(str)
                        .apply(_headline_hash)
                    )

                clean_df = clean_df.copy()
                clean_df["headline_hash"] = clean_df["headline"].apply(_headline_hash)

                # Only keep truly new rows
                new_df = clean_df[
                    ~clean_df["headline_hash"].isin(existing_hashes)
                ].copy()

                # STEP 3 — Persist articles
                log.info("── Step 3: Persisting articles")
                added_count = upsert_articles(clean_df)

                if added_count == 0 or new_df.empty:
                    log.info("No truly new articles to score.")
                else:
                    # STEP 4 — FinBERT inference
                    log.info("── Step 4: FinBERT inference")
                    scored_new = get_scorer().score_dataframe(new_df)
                    write_sentiment_scores(scored_new)
        else:
            log.warning("No new headlines — using historical data only.")

        # STEP 5 — Load history
        log.info("── Step 5: Loading history")
        full_scored = read_scored_headlines()

        if full_scored.empty:
            log.warning("No scored headlines available.")
            return

        full_scored = apply_time_decay(full_scored)

        # STEP 6 — Temporal risk
        log.info("── Step 6: Temporal risk aggregation")
        temporal_df = compute_rolling_zscore(full_scored)
        write_temporal_risk(temporal_df)

        # STEP 7 — Risk summary
        log.info("── Step 7: Risk summary")
        risk_summary = compute_entity_risk_summary(
            full_scored,
            temporal_df,
        )
        write_risk_summary(risk_summary)

        # STEP 7.5 — Source summary
        log.info("── Step 7.5: Source summary")
        source_summary = (
            full_scored.groupby("source")
            .size()
            .reset_index(name="headlines")
            .sort_values("headlines", ascending=False)
        )
        write_source_summary(source_summary)

        # STEP 8 — Alerts
        log.info("── Step 8: Alert check")
        alerted = check_and_send_alerts(risk_summary)

        if alerted:
            log.warning("Alerted entities: %s", alerted)

        save_seen_hashes(seen)

        elapsed = time.perf_counter() - t0
        log.info(
            "══ Pipeline complete in %.1fs ══════════════════════════",
            elapsed,
        )

        print(risk_summary.to_string(index=False))

    except Exception:
        log.exception("Pipeline failed")


def run_scheduled() -> None:
    from apscheduler.schedulers.blocking import BlockingScheduler

    interval = CFG.ingestion.refresh_interval_minutes
    log.info("Scheduler starting — refresh every %d minutes", interval)

    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_pipeline,
        trigger="interval",
        minutes=interval,
        id="finrisk_pipeline",
        max_instances=1,
        coalesce=True,
    )

    run_pipeline()

    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scheduler stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run pipeline continuously",
    )

    args = parser.parse_args()

    if args.schedule:
        run_scheduled()
    else:
        run_pipeline()