"""
api/routes.py
FastAPI router definitions for the FinRisk REST API.

Run the API with:
    uvicorn api.routes:app --reload
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from src.config import RISK_SUMMARY_PATH, SCORED_DATA_PATH, TEMPORAL_RISK_PATH

app = FastAPI(
    title="FinRisk API",
    description="REST API for the Financial Sentiment Risk Scorer",
    version="1.0.0",
)


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"{label} not found. Run the pipeline first.",
        )
    return pd.read_csv(path)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health() -> dict:
    return {"status": "ok"}


@app.get("/risk/summary", summary="Entity risk summary")
def risk_summary(
    tier: str | None = Query(None, description="Filter by tier: HIGH | MEDIUM | LOW"),
    limit: int = Query(50, ge=1, le=500),
) -> JSONResponse:
    """Return the latest entity risk summary, optionally filtered by tier."""
    df = _load_csv(RISK_SUMMARY_PATH, "risk_summary.csv")
    if tier:
        df = df[df["risk_tier"].str.upper() == tier.upper()]
    return JSONResponse(content=df.head(limit).to_dict(orient="records"))


@app.get("/risk/temporal", summary="Temporal z-score data")
def temporal_risk(
    entity: str | None = Query(None, description="Filter by entity name"),
    limit: int = Query(200, ge=1, le=2000),
) -> JSONResponse:
    """Return rolling z-score timeseries, optionally filtered by entity."""
    df = _load_csv(TEMPORAL_RISK_PATH, "temporal_risk.csv")
    if entity:
        df = df[df["entity"].str.lower() == entity.lower()]
    return JSONResponse(content=df.head(limit).to_dict(orient="records"))


@app.get("/headlines", summary="Scored headlines")
def headlines(
    source: str | None = Query(None, description="Filter by source"),
    label: str | None = Query(None, description="Filter by predicted_label"),
    limit: int = Query(100, ge=1, le=1000),
) -> JSONResponse:
    """Return scored headlines with sentiment probabilities."""
    df = _load_csv(SCORED_DATA_PATH, "scored_headlines.csv")
    if source:
        df = df[df["source"].str.lower() == source.lower()]
    if label:
        df = df[df["predicted_label"].str.lower() == label.lower()]
    return JSONResponse(content=df.head(limit).to_dict(orient="records"))
