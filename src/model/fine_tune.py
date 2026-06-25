"""
src/model/fine_tune.py
Re-exports the fine-tuning routines from src/fine_tune.py
so that the structured package layout (src/model/) works transparently.
"""
from __future__ import annotations

# The canonical implementation lives in src/fine_tune.py.
# We simply re-export so callers can do:
#   from src.model.fine_tune import fine_tune, evaluate_saved_model
from src.fine_tune import (  # noqa: F401
    fine_tune,
    evaluate_saved_model,
    SentimentDataset,
    load_labelled_data,
    compute_metrics,
)
