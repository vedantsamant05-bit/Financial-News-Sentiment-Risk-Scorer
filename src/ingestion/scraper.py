from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import CFG

log = logging.getLogger(__name__)

MAX_AGE_DAYS = 7
BATCH_STATE_FILE = CFG.paths.data_dir / "batch_state.json"

ALL_ENTITIES: dict[str, str] = {
    "HDFC Bank": "HDFC Bank latest news",
    "ICICI Bank": "ICICI Bank latest news",
    "Infosys": "Infosys latest news",
    "Tata Motors": "Tata Motors latest news",
    "Reliance Industries": "Reliance Industries latest news",
    "Yes Bank": "Yes Bank latest news",
    "Paytm": "Paytm latest news",
    "Zomato": "Zomato latest news",
    "Adani Group": "Adani Group latest news",
    "Wipro": "Wipro latest news",
}

ENTITY_BATCHES = [
    [
        "HDFC Bank",
        "ICICI Bank",
        "Infosys",
        "Wipro",
        "Tata Motors",
    ],
    [
        "Reliance Industries",
        "Paytm",
        "Zomato",
        "Yes Bank",
        "Adani Group",
    ],
]


def load_batch_index() -> int:
    if BATCH_STATE_FILE.exists():
        try:
            with open(BATCH_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("batch_index", 0)
        except Exception:
            return 0
    return 0


def save_batch_index(index: int) -> None:
    with open(BATCH_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"batch_index": index}, f)


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
        log.warning("Fetch failed %s: %s", url, e)
        return None
    except (URLError, Exception) as e:
        log.warning("Fetch failed %s: %s", url, e)
        return None


def _parse_rss(raw: bytes) -> list[dict]:
    from xml.etree import ElementTree as ET

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []

    items = []

    for item in root.iter("item"):
        title = item.find("title")
        pubdate = item.find("pubDate")
        link = item.find("link")

        if title is None or not (title.text or "").strip():
            continue

        items.append(
            {
                "title": (title.text or "").strip(),
                "pubDate": (pubdate.text or "").strip() if pubdate is not None else "",
                "link": (link.text or "").strip() if link is not None else "",
            }
        )

    return items


def _parse_date(raw: str) -> datetime | None:
    if not raw:
        return None

    cleaned = re.sub(r"\s+[A-Z]{2,4}$", "", raw.strip())

    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(cleaned, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    return None


def _is_recent(dt: datetime, max_age_days: int = MAX_AGE_DAYS) -> bool:
    today = datetime.now(timezone.utc)
    age = (today - dt).days
    return 0 <= age <= max_age_days


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s[-–]\s[A-Za-z0-9.&\- ]{2,50}$", "", text)
    return text


def scrape_google_news(entity: str, query: str, seen: set[str]) -> list[dict]:
    url = (
        f"https://news.google.com/rss/search"
        f"?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    )

    raw = _fetch(url)
    time.sleep(CFG.ingestion.rate_limit_seconds)

    if raw is None:
        return []

    rows = []

    for item in _parse_rss(raw):
        headline = _clean(item["title"])

        if len(headline) < 20:
            continue

        article_dt = _parse_date(item["pubDate"])
        if article_dt is None or not _is_recent(article_dt):
            continue

        h = hashlib.md5(
            f"{headline}|{entity}|{article_dt.strftime('%Y-%m-%d')}".lower().encode()
        ).hexdigest()

        if h in seen:
            continue

        seen.add(h)

        rows.append(
            {
                "date": article_dt.strftime("%Y-%m-%d"),
                "entity": entity,
                "headline": headline,
                "true_label": "",
                "source": "google_news",
                "url": item["link"],
            }
        )

    rows = rows[: CFG.ingestion.max_articles_per_entity]

    log.info("Google News %-26s → %d headlines", entity, len(rows))
    return rows


def run_ingestion(seen: set[str] | None = None) -> list[dict]:
    if seen is None:
        seen = set()

    batch_index = load_batch_index()
    batch = ENTITY_BATCHES[batch_index]

    log.info("Running batch %d/%d", batch_index + 1, len(ENTITY_BATCHES))

    all_rows = []

    for entity in batch:
        query = ALL_ENTITIES[entity]
        all_rows.extend(scrape_google_news(entity, query, seen))

    next_batch = (batch_index + 1) % len(ENTITY_BATCHES)
    save_batch_index(next_batch)

    log.info("Ingestion complete: %d new headlines", len(all_rows))
    return all_rows