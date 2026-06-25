"""
src/fine_tune.py

Fine-tunes ProsusAI/finbert on Financial PhraseBank labels from your pipeline CSV.
Only rows with a real true_label (positive/negative/neutral) are used — the
unlabelled live-scraped rows are automatically ignored.

Usage:
    python src/fine_tune.py
    python src/fine_tune.py --epochs 5 --batch-size 16
    python src/fine_tune.py --eval-only   # just evaluate the saved model
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .config import DATA_DIR, SENTIMENT_MODEL_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_MODEL   = "ProsusAI/finbert"
MODEL_OUT    = Path(__file__).resolve().parents[1] / "models" / "finbert-finetuned"
LABEL2ID     = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL     = {v: k for k, v in LABEL2ID.items()}
VALID_LABELS = set(LABEL2ID.keys())


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
        labels: list[int],
    ) -> None:
        self.encodings = encodings
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_labelled_data() -> tuple[list[str], list[int]]:
    """
    Reads data/financial_news.csv and returns only rows that have a
    real expert label (true_label in positive/negative/neutral).
    Live-scraped rows with empty true_label are silently dropped.
    """
    from .config import RAW_DATA_PATH
    csv_path = RAW_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run data/load_real_data.py first."
        )

    df = pd.read_csv(csv_path)
    df = df[df["true_label"].isin(VALID_LABELS)].copy()

    if len(df) < 50:
        raise ValueError(
            f"Only {len(df)} labelled rows found — need at least 50 to fine-tune. "
            "Make sure Financial PhraseBank was downloaded (run load_real_data.py "
            "without --skip-kaggle)."
        )

    texts  = df["headline"].astype(str).str.strip().tolist()
    labels = df["true_label"].map(LABEL2ID).tolist()

    log.info(
        "Loaded %d labelled rows | %s",
        len(df),
        df["true_label"].value_counts().to_dict(),
    )
    return texts, labels


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    accuracy = (preds == labels).mean()
    return {
        "macro_f1":    round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "accuracy":    round(float(accuracy), 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# FINE-TUNE
# ──────────────────────────────────────────────────────────────────────────────

def fine_tune(
    epochs:     int   = 4,
    batch_size: int   = 4,
    lr:         float = 2e-5,
    max_length: int   = 128,
    test_size:  float = 0.15,
    seed:       int   = 42,
) -> None:
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    texts, labels = load_labelled_data()

    # Stratified split so all 3 classes appear in both train and val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    log.info("Train: %d  |  Val: %d", len(train_texts), len(val_texts))

    # 2. Tokenise
    log.info("Tokenising…")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_enc = tokenizer(
        train_texts,
        padding=True, truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    val_enc = tokenizer(
        val_texts,
        padding=True, truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    train_dataset = SentimentDataset(train_enc, train_labels)
    val_dataset   = SentimentDataset(val_enc,   val_labels)

    # 3. Load base model
    log.info("Loading base model: %s", BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # 4. Training arguments
    # fp16 only on CUDA — CPU training stays in fp32
    use_fp16 = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=str(MODEL_OUT / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        logging_steps=50,
        logging_dir=str(MODEL_OUT / "logs"),
        report_to="none",           # disable wandb/tensorboard unless you want them
        seed=seed,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 6. Train
    log.info("Starting fine-tuning on %s…", "GPU" if use_fp16 else "CPU")
    trainer.train()

    # 7. Final evaluation
    log.info("Final evaluation on validation set…")
    results = trainer.evaluate()
    log.info("Val results: %s", results)

    preds_output = trainer.predict(val_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    print("\n── Classification Report (Validation Set) ───────────────")
    print(classification_report(
        val_labels, preds,
        target_names=["positive", "negative", "neutral"],
        zero_division=0,
    ))
    print("─────────────────────────────────────────────────────────\n")

    # 8. Save fine-tuned model + tokenizer to models/finbert-finetuned/
    log.info("Saving model → %s", MODEL_OUT)
    trainer.save_model(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))

    # Save label mapping alongside model for reference
    with (MODEL_OUT / "label_map.json").open("w") as fh:
        import json
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, fh, indent=2)

    log.info("Fine-tuning complete.")
    print(f"\n  Model saved to: {MODEL_OUT}")
    print("  Next step: python src/pipeline.py")
    print("  (pipeline will auto-detect and use the fine-tuned model)\n")


# ──────────────────────────────────────────────────────────────────────────────
# EVAL ONLY  — run against saved model without re-training
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_saved_model(max_length: int = 128) -> None:
    if not MODEL_OUT.exists():
        raise FileNotFoundError(
            f"No saved model at {MODEL_OUT}. Run fine_tune() first."
        )

    texts, labels = load_labelled_data()
    _, val_texts, _, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_OUT))
    model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_OUT))
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    enc = tokenizer(
        val_texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        logits = model(**enc).logits.cpu().numpy()

    preds = np.argmax(logits, axis=-1)

    print("\n── Saved Model Evaluation ────────────────────────────────")
    print(classification_report(
        val_labels, preds,
        target_names=["positive", "negative", "neutral"],
        zero_division=0,
    ))
    macro_f1 = f1_score(val_labels, preds, average="macro", zero_division=0)
    print(f"  Macro F1: {macro_f1:.4f}")
    print("─────────────────────────────────────────────────────────\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT on PhraseBank labels")
    parser.add_argument("--epochs",     type=int,   default=4,    help="Training epochs (default 3)")
    parser.add_argument("--batch-size", type=int,   default=4,   help="Batch size per device (default 4)")
    parser.add_argument("--lr",         type=float, default=2e-5, help="Learning rate (default 2e-5)")
    parser.add_argument("--eval-only",  action="store_true",      help="Evaluate saved model, skip training")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_saved_model()
    else:
        fine_tune(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )