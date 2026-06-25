"""
data/load_real_data.py

Combined data loader:
  - Financial PhraseBank via Kaggle API  (labelled, for F1 evaluation)
  - Google News RSS + NewsAPI            (live Indian entity headlines, unlabelled)

Outputs data/financial_news.csv in the exact schema the pipeline expects.
Replaces data/generate_dataset.py entirely.

Usage:
    python data/load_real_data.py                          # PhraseBank + Google RSS only
    python data/load_real_data.py --newsapi-key YOUR_KEY   # adds NewsAPI live feed
    python data/load_real_data.py --skip-kaggle            # live scrape only (re-run safe)
    python data/load_real_data.py --newsapi-key KEY --append  # daily refresh mode
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEDUP_CACHE = DATA_DIR / ".seen_hashes.json"
OUT_CSV     = DATA_DIR / "financial_news.csv"

# ──────────────────────────────────────────────────────────────────────────────
# INDIAN ENTITY CONFIGURATION
# Each entry: display_name → (yahoo_ticker_or_None, google_news_search_query)
# ──────────────────────────────────────────────────────────────────────────────
INDIAN_ENTITIES: dict[str, tuple[str | None, str]] = {
    "HDFC Bank":            ("HDB",  "HDFC Bank India earnings NPA"),
    "ICICI Bank":           ("IBN",  "ICICI Bank India results quarterly"),
    "Infosys":              ("INFY", "Infosys IT India revenue guidance"),
    "Tata Motors":          ("TTM",  "Tata Motors JLR quarterly results"),
    "Reliance Industries":  (None,   "Reliance Industries RIL Jio earnings"),
    "Yes Bank":             (None,   "Yes Bank India RBI NPA"),
    "Paytm":                (None,   "Paytm One97 Communications India"),
    "Zomato":               (None,   "Zomato India food delivery quarterly"),
    "Adani Group":          (None,   "Adani Group India Gautam Adani"),
    "Wipro":                ("WIT",  "Wipro IT India quarterly revenue"),
}

# PhraseBank label variants across Kaggle CSV versions
PHRASEBANK_LABEL_MAP: dict[str, str] = {
    "positive": "positive",
    "negative": "negative",
    "neutral":  "neutral",
    "2": "positive",
    "1": "neutral",
    "0": "negative",
}

VALID_LABELS = {"positive", "negative", "neutral"}


# ──────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION CACHE
# ──────────────────────────────────────────────────────────────────────────────

def _load_seen() -> set[str]:
    if DEDUP_CACHE.exists():
        with DEDUP_CACHE.open() as fh:
            return set(json.load(fh))
    return set()


def _save_seen(seen: set[str]) -> None:
    with DEDUP_CACHE.open("w") as fh:
        json.dump(sorted(seen), fh)


def _hash(text: str) -> str:
    text = text.strip().lower()

    # remove numbers (18%, 500 crore etc.)
    text = re.sub(r"\d+(\.\d+)?", "NUM", text)

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # collapse spaces
    text = re.sub(r"\s+", " ", text)

    return hashlib.md5(text.encode()).hexdigest()

# ──────────────────────────────────────────────────────────────────────────────
# SHARED HTTP HELPER
# ──────────────────────────────────────────────────────────────────────────────

def _fetch(url: str, timeout: int = 14) -> bytes | None:
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/rss+xml,application/xml,text/xml,*/*",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except HTTPError as e:
        log.warning("HTTP %s → %s", e.code, url)
    except URLError as e:
        log.warning("URLError → %s: %s", url, e.reason)
    except Exception as e:
        log.warning("Fetch failed → %s: %s", url, e)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# RSS PARSER  (stdlib only — no feedparser)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_rss(raw: bytes) -> list[dict[str, str]]:
    from xml.etree import ElementTree as ET
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        log.warning("RSS parse error: %s", e)
        return []

    items = []
    for item in root.iter("item"):
        title   = item.find("title")
        pubdate = item.find("pubDate")
        if title is None or not (title.text or "").strip():
            continue
        items.append({
            "title":   (title.text or "").strip(),
            "pubDate": (pubdate.text or "").strip() if pubdate is not None else "",
        })
    return items


def _parse_date(raw: str) -> str:
    if not raw:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cleaned = re.sub(r"\s+[A-Z]{2,4}$", "", raw.strip())
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove ' - Source Name' suffixes injected by Google News
    text = re.sub(r"\s[-–]\s[A-Z][A-Za-z &]{2,35}$", "", text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE A  — FINANCIAL PHRASEBANK  (Kaggle, labelled)
# ──────────────────────────────────────────────────────────────────────────────

def _kaggle_download() -> None:
    """Download Financial PhraseBank via Kaggle CLI. Skips if already present."""
    flag = DATA_DIR / ".phrasebank_downloaded"
    if flag.exists():
        log.info("PhraseBank already downloaded — skipping.")
        return

    log.info("Downloading Financial PhraseBank from Kaggle…")
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", "ankurzing/sentiment-analysis-for-financial-news",
            "-p", str(DATA_DIR),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle CLI failed:\n{result.stderr}\n\n"
            "Make sure kaggle is installed (`pip install kaggle`) and "
            "~/.kaggle/kaggle.json exists."
        )
    flag.touch()
    log.info("PhraseBank downloaded.")


def _find_phrasebank_csv() -> Path:
    """
    The Kaggle zip unpacks to different filenames across versions.
    Try every known candidate before giving up.
    """
    candidates = [
        "all-data.csv",
        "data.csv",
        "Sentences_AllAgree.txt",
        "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
        "Sentences_75Agree.txt",
    ]
    for name in candidates:
        p = DATA_DIR / name
        if p.exists():
            return p
    # Fallback: any CSV/TXT with a recognisable name
    for p in sorted(DATA_DIR.glob("*.csv")) + sorted(DATA_DIR.glob("*.txt")):
        if any(k in p.stem.lower() for k in ("data", "sentence", "phrasebank", "agree")):
            return p
    raise FileNotFoundError(
        f"Cannot find PhraseBank file in {DATA_DIR}. "
        f"Contents: {[f.name for f in DATA_DIR.iterdir()]}"
    )


def load_phrasebank() -> pd.DataFrame:
    """
    Returns rows from Financial PhraseBank with real expert labels.
    Synthetic dates are assigned (evenly spaced 2020-2023) because
    PhraseBank has no timestamps — this is expected and documented.
    """
    _kaggle_download()
    path = _find_phrasebank_csv()
    log.info("Reading PhraseBank from %s", path.name)

    if path.suffix == ".txt":
        rows = []
        with path.open(encoding="latin-1") as fh:
            for line in fh:
                line = line.strip()
                if "@" not in line:
                    continue
                text, label = line.rsplit("@", 1)
                rows.append({"Sentence": text.strip(), "Sentiment": label.strip()})
        raw = pd.DataFrame(rows)
    else:
        raw = None
        for sep in (",", ";", "\t"):
            try:
                candidate = pd.read_csv(
                    path,
                    sep=sep,
                    header=None,
                    names=["Sentiment", "Sentence"],
                    encoding="latin-1",
                )
                if len(candidate.columns) >= 2 and len(candidate) > 100:
                    raw = candidate
                    break
            except Exception:
                continue
        if raw is None:
            raise ValueError(f"Could not parse {path} as CSV.")

    raw = raw.dropna(subset=["Sentence"]).copy()
    raw["Sentence"]  = raw["Sentence"].astype(str).str.strip()
    raw["Sentiment"] = (
        raw["Sentiment"].astype(str).str.strip().str.lower()
        .map(PHRASEBANK_LABEL_MAP)
    )
    raw = raw.dropna(subset=["Sentiment"])

    # Spread evenly across 2020-01-01 → 2023-12-31 for temporal plausibility
    from datetime import date, timedelta
    start_d = date(2020, 1, 1)
    span    = (date(2023, 12, 31) - start_d).days
    n       = len(raw)
    raw["date"] = [
        (start_d + timedelta(days=int(i * span / max(n - 1, 1)))).isoformat()
        for i in range(n)
    ]
    # PhraseBank has no entity column — we infer from text
    raw["entity"] = raw["Sentence"].apply(_infer_entity)
    # Rows with no matching entity are still kept under "General Market"
    raw["entity"] = raw["entity"].fillna("General Market")

    log.info(
        "PhraseBank: %d rows | %s",
        len(raw),
        raw["Sentiment"].value_counts().to_dict(),
    )
    return pd.DataFrame({
        "date":       raw["date"],
        "entity":     raw["entity"],
        "headline":   raw["Sentence"],
        "true_label": raw["Sentiment"],
        "source":     "phrasebank",
    })


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE B  — GOOGLE NEWS RSS  (live, unlabelled, Indian focus)
# ──────────────────────────────────────────────────────────────────────────────

def scrape_google_news(
    entity: str,
    query: str,
    seen: set[str],
    rate_limit: float = 1.5,
) -> list[dict]:
    encoded = quote_plus(query)
    # ceid=IN:en biases results toward Indian publications (Economic Times,
    # Business Standard, Mint, MoneyControl etc.)
    url = (
        f"https://news.google.com/rss/search"
        f"?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
    )
    log.info("Google News  %-26s '%s'", entity, query)
    raw = _fetch(url)
    time.sleep(rate_limit)

    if raw is None:
        return []

    rows = []
    for item in _parse_rss(raw):
        raw_title = item["title"]

        parts = raw_title.rsplit(" - ", 1)

        if len(parts) == 2:
            headline = _clean(parts[0])
            source_name = parts[1].strip()
        else:
            headline = _clean(raw_title)
            source_name = "Google News"
        if len(headline) < 20:
            continue
        h = _hash(headline)
        if h in seen:
            continue
        seen.add(h)
        rows.append({
            "date":       _parse_date(item["pubDate"]),
            "entity":     entity,
            "headline":   headline,
            "true_label": "",           # FinBERT predicts at inference time
            "source":   source_name,
        })

    log.info("  → %d new headlines", len(rows))
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE C  — NEWSAPI  (optional, structured, 100 req/day free)
# ──────────────────────────────────────────────────────────────────────────────

def scrape_newsapi(
    entity: str,
    query: str,
    api_key: str,
    seen: set[str],
    page_size: int = 8,
    rate_limit: float = 1.2,
) -> list[dict]:
    # Narrow to financial context to reduce noise
    financial_query = quote_plus(
    f'"{query}" (earnings OR revenue OR profit OR loss OR stock OR market OR regulation OR compliance OR investigation OR RBI OR SEBI)'
    )
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={financial_query}"
        f"&language=en"
        f"&sortBy=publishedAt"
        f"&pageSize={page_size}"
        f"&apiKey={api_key}"
    )
    log.info("NewsAPI      %-26s '%s'", entity, query)
    raw = _fetch(url)
    time.sleep(rate_limit)

    if raw is None:
        return []

    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        log.warning("NewsAPI JSON error: %s", e)
        return []

    if data.get("status") != "ok":
        log.warning("NewsAPI: %s", data.get("message", "unknown error"))
        return []

    rows = []
    for article in data.get("articles", []):
        title = article.get("title") or ""
        description = article.get("description") or ""
        source_name = article.get("source", {}).get("name", "Unknown")
        headline = _clean(f"{title}. {description}")
        
        if not headline or headline.lower() == "[removed]" or len(headline) < 20:
            continue
        h = _hash(headline)
        if h in seen:
            continue
        seen.add(h)
        rows.append({
            "date":       _parse_date(article.get("publishedAt", "")),
            "entity":     entity,
            "headline":   headline,
            "true_label": "",
            "source": source_name    ,
        })

    log.info("  → %d new headlines", len(rows))
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# ENTITY INFERENCE  (for PhraseBank rows which have no entity column)
# ──────────────────────────────────────────────────────────────────────────────

ENTITY_KEYWORDS: dict[str, list[str]] = {
    "HDFC Bank": [
        "hdfc bank",
        "hdfc",
        "housing development finance",
    ],

    "ICICI Bank": [
        "icici bank",
        "icici",
    ],

    "Yes Bank": [
        "yes bank",
        "yesbank",
    ],

    "Infosys": [
        "infosys",
        "infy",
    ],

    "Tata Motors": [
        "tata motors",
        "tata motor",
        "jaguar",
        "land rover",
        "jlr",
    ],

    "Reliance Industries": [
        "reliance industries",
        "reliance",
        "ril",
        "jio",
    ],

    "Paytm": [
        "paytm",
        "one97",
        "one97 communications",
    ],

    "Zomato": [
        "zomato",
    ],

    "Adani Group": [
        "adani",
        "gautam adani",
    ],

    "Wipro": [
        "wipro",
    ],

    # Global names that appear in PhraseBank
    "Goldman Sachs": [
        "goldman",
        "goldman sachs",
    ],

    "JPMorgan Chase": [
        "jpmorgan",
        "jp morgan",
        "jpm",
    ],

    "Morgan Stanley": [
        "morgan stanley",
    ],

    "Apple": [
        "apple inc",
        "apple",
        "aapl",
        "iphone",
    ],

    "Microsoft": [
        "microsoft",
        "msft",
        "azure",
    ],

    "Amazon": [
        "amazon",
        "aws",
    ],

    "Tesla": [
        "tesla",
        "elon musk",
    ],

    "Nvidia": [
        "nvidia",
        "nvda",
        "gpu",
    ],
}


def _infer_entity(text: str) -> str | None:
    lower = text.lower()

    for entity, keywords in ENTITY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lower:
                return entity

    return None

# ──────────────────────────────────────────────────────────────────────────────
# LIVE SCRAPER ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

def scrape_live(
    newsapi_key: str | None = None,
    seen: set[str] | None = None,
) -> list[dict]:
    if seen is None:
        seen = _load_seen()

    all_rows: list[dict] = []
    total = len(INDIAN_ENTITIES)

    for idx, (entity, (_, query)) in enumerate(INDIAN_ENTITIES.items(), 1):
        log.info("[%d/%d] %s", idx, total, entity)

        all_rows.extend(scrape_google_news(entity, query, seen))

        if newsapi_key:
            all_rows.extend(scrape_newsapi(entity, query, newsapi_key, seen))

    _save_seen(seen)
    return all_rows


# ──────────────────────────────────────────────────────────────────────────────
# FINAL MERGE + CLEAN + SAVE
# ──────────────────────────────────────────────────────────────────────────────

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["headline"]   = df["headline"].astype(str).str.strip()
    df["entity"]     = df["entity"].astype(str).str.strip()
    df["true_label"] = (
        df["true_label"].astype(str).str.strip().str.lower()
        .where(lambda s: s.isin(VALID_LABELS | {""}), other="")
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])
    df = df[df["headline"].str.len() >= 15]
    df = df[df["entity"].notna() & (df["entity"] != "") & (df["entity"] != "nan")]
    df = df.drop_duplicates(subset=["headline", "entity", "date"])
    return df.sort_values("date").reset_index(drop=True)


def build(
    skip_kaggle:  bool = False,
    newsapi_key:  str | None = None,
    append:       bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    # ── PhraseBank (labelled ground truth) ────────────────────────────────────
    if not skip_kaggle:
        try:
            phrasebank_df = load_phrasebank()
            phrasebank_path = DATA_DIR / "phrasebank_eval.csv"
            phrasebank_df.to_csv(phrasebank_path, index=False)
            log.info("Saved PhraseBank evaluation dataset → %s", phrasebank_path)
            frames.append(phrasebank_df)
        except Exception as e:
            log.error("PhraseBank load failed: %s", e)
            log.error("Continuing with live data only.")

    # ── Live Indian headlines ─────────────────────────────────────────────────
    seen = _load_seen()
    live_rows = scrape_live(newsapi_key=newsapi_key, seen=seen)
    if live_rows:
        frames.append(pd.DataFrame(live_rows))

    if not frames:
        raise RuntimeError(
            "No data loaded from either source. "
            "Check your kaggle.json and internet connection."
        )

    new_data = _clean_df(pd.concat(frames, ignore_index=True))

    # ── Append mode: merge with existing CSV ──────────────────────────────────
    if append and OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        combined = _clean_df(pd.concat([existing, new_data], ignore_index=True))
    else:
        combined = new_data

    combined.to_csv(OUT_CSV, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    labelled   = (combined["true_label"].isin(VALID_LABELS)).sum()
    unlabelled = (combined["true_label"] == "").sum()

    print("\n── Dataset Build Summary ─────────────────────────────────")
    print(f"  Total rows      : {len(combined):,}")
    print(f"  Labelled rows   : {labelled:,}  ← used for F1 evaluation")
    print(f"  Unlabelled rows : {unlabelled:,}  ← FinBERT predicts label")
    print(f"  Unique entities : {combined['entity'].nunique()}")
    print(f"  Date range      : {combined['date'].min()} → {combined['date'].max()}")
    print(f"  Sources         :")
    for src, cnt in combined["source"].value_counts().items():
        print(f"    {src:<20} {cnt:>5}")
    print(f"  Label breakdown :")
    for lbl, cnt in combined["true_label"].value_counts().items():
        tag = lbl if lbl else "(unlabelled)"
        print(f"    {tag:<20} {cnt:>5}")
    print(f"\n  Saved → {OUT_CSV}")
    print("  Next step: python src/pipeline.py")
    print("─────────────────────────────────────────────────────────\n")

    return combined


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load Financial PhraseBank + scrape live Indian financial news."
    )
    parser.add_argument(
        "--newsapi-key",
        type=str,
        default=None,
        help="NewsAPI.org key (optional). Free tier at newsapi.org.",
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Skip PhraseBank download (use when re-running for live refresh only).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new headlines to existing CSV instead of overwriting.",
    )
    args = parser.parse_args()

    build(
        skip_kaggle=args.skip_kaggle,
        newsapi_key=args.newsapi_key,
        append=args.append,
    )
