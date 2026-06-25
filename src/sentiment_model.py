from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import SENTIMENT_MODEL_NAME , FINE_TUNED_MODEL_PATH

logger = logging.getLogger(__name__)

MAX_LENGTH = 128
BATCH_SIZE = 32


class FinBERTScorer:
    """
    FinBERT batch inference scorer.
    """

    def __init__(self, device: str | None = None) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading FinBERT on %s...", self._device)

        model_path = (
            str(FINE_TUNED_MODEL_PATH)
            if FINE_TUNED_MODEL_PATH.exists()
            else SENTIMENT_MODEL_NAME
        )

        logger.info("Loading model from: %s", model_path)

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        self._model.to(self._device)
        self._model.eval()

    def _encode(self, batch: list[str]) -> dict[str, torch.Tensor]:
        encoded = self._tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        return {k: v.to(self._device) for k, v in encoded.items()}

    def _batch_probs(self, headlines: list[str]) -> np.ndarray:
        all_probs = []

        for i in tqdm(range(0, len(headlines), BATCH_SIZE), desc="FinBERT inference"):
            batch = headlines[i:i + BATCH_SIZE]
            inputs = self._encode(batch)

            with torch.inference_mode():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            all_probs.append(probs)

        return np.vstack(all_probs)

    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "headline",
    ) -> pd.DataFrame:
        headlines = df[text_col].tolist()
        probs = self._batch_probs(headlines)

        out = df.copy()

        out["prob_positive"] = probs[:, 0]
        out["prob_negative"] = probs[:, 1]
        out["prob_neutral"] = probs[:, 2]

        labels = ["positive", "negative", "neutral"]

        predicted_indices = np.argmax(probs, axis=1)

        out["predicted_label"] = [
            labels[idx] for idx in predicted_indices
        ]

        out["confidence"] = np.max(probs, axis=1)

        out["sentiment_score"] = (
            out["prob_positive"] - out["prob_negative"]
        )

        return out