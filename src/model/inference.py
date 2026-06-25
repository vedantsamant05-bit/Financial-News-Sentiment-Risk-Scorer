from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import CFG

log = logging.getLogger(__name__)


class FinBERTScorer:
    def __init__(self, device: str | None = None) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model_path = CFG.model.active_model
        model_source = "huggingface"

        log.info(
            "Loading %s FinBERT from %s on %s",
            model_source,
            model_path,
            self._device,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        self._model.to(self._device)
        self._model.eval()

        self._id2label = {
            int(k): v.lower()
            for k, v in self._model.config.id2label.items()
        }

    def _encode(self, batch: list[str]) -> dict[str, torch.Tensor]:
        enc = self._tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=CFG.model.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self._device) for k, v in enc.items()}

    def _batch_probs(self, headlines: list[str]) -> np.ndarray:
        if not headlines:
            return np.empty((0, 3))

        all_probs = []

        for i in tqdm(
            range(0, len(headlines), CFG.model.batch_size),
            desc="Scoring",
        ):
            inputs = self._encode(
                headlines[i: i + CFG.model.batch_size]
            )

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
        if df.empty:
            return df.copy()

        probs = self._batch_probs(df[text_col].tolist())
        out = df.copy()

        label_to_index = {
            label: idx for idx, label in self._id2label.items()
        }

        pos_idx = label_to_index["positive"]
        neg_idx = label_to_index["negative"]
        neu_idx = label_to_index["neutral"]

        out["prob_positive"] = probs[:, pos_idx]
        out["prob_negative"] = probs[:, neg_idx]
        out["prob_neutral"] = probs[:, neu_idx]

        out["predicted_label"] = [
            self._id2label[int(np.argmax(p))]
            for p in probs
        ]

        out["sentiment_score"] = (
            out["prob_positive"] - out["prob_negative"]
        )

        return out