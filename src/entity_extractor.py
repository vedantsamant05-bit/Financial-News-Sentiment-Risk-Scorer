from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_KNOWN: frozenset[str] = frozenset({
    "Goldman Sachs",
    "HDFC Bank",
    "Reliance Industries",
    "Adani Group",
    "Infosys",
    "Tata Motors",
    "Yes Bank",
    "Paytm",
    "Zomato",
    "ICICI Bank",
})


def _match(text: str) -> str:
    """Return first known entity found in text, else Unknown."""
    lower = text.lower()

    return next(
        (entity for entity in _KNOWN if entity.lower() in lower),
        "Unknown",
    )


def enrich_with_entities(
    df: pd.DataFrame,
    text_col: str = "headline",
) -> pd.DataFrame:
    """
    Infer entity from headlines if entity column is absent.
    """
    if text_col not in df.columns:
        raise ValueError(f"Missing column: {text_col}")

    out = df.copy()

    if "entity" not in out.columns:
        logger.info("Entity column missing. Extracting entities.")
        out["entity"] = out[text_col].map(_match)

    return out