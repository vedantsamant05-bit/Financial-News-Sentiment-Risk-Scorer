"""
config.py: Centralized configuration settings for the financial risk scorer.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "financial_news.csv"
SCORED_DATA_PATH = DATA_DIR / "scored_headlines.csv"
TEMPORAL_RISK_PATH = DATA_DIR / "temporal_risk.csv"
RISK_SUMMARY_PATH = DATA_DIR / "risk_summary.csv"

# Model Parameters
SENTIMENT_MODEL_NAME = "ProsusAI/finbert"
FINE_TUNED_MODEL_PATH = BASE_DIR / "models" / "finbert-finetuned"
SPACY_MODEL_NAME = "en_core_web_sm"

# Risk Thresholds
DEFAULT_RISK_THRESHOLD = 0.5
TIME_WINDOW_DAYS = 7
