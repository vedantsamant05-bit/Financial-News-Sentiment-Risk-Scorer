from src.data_loader import load_raw
from src.sentiment_model import FinBERTScorer
from src.risk_aggregator import (
    compute_rolling_zscore,
    compute_entity_risk_summary,
)

print("Loading data...")
df = load_raw().head(1000)   # use 200 first, not 5000

print("Running FinBERT...")
scorer = FinBERTScorer()
scored_df = scorer.score_dataframe(df)

print("Computing temporal risk...")
temporal_df = compute_rolling_zscore(scored_df)

print("Computing summary...")
summary_df = compute_entity_risk_summary(scored_df, temporal_df)

print("\n=== TOP RISK ENTITIES ===")
print(summary_df.head())