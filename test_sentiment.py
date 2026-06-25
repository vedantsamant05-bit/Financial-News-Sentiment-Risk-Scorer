from src.data_loader import load_raw
from src.sentiment_model import FinBERTScorer

df = load_raw().head(5)

scorer = FinBERTScorer()
scored_df = scorer.score_dataframe(df)

print(
    scored_df[
        [
            "headline",
            "predicted_label",
            "confidence",
            "sentiment_score",
        ]
    ]
)