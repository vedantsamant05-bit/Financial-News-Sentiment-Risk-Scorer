from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

SEED = 42
random.seed(SEED)

# -------------------------------------------------------------------
# Each entity has a "story arc" — phases of sentiment over time
# This makes the temporal z-score detection actually meaningful
# -------------------------------------------------------------------

@dataclass
class EntityProfile:
    name: str
    # List of (duration_days, sentiment_bias) tuples
    # bias: 1.0 = mostly positive, -1.0 = mostly negative, 0 = neutral
    arc: list[tuple[int, float]]
    sector: str


ENTITIES: list[EntityProfile] = [
    EntityProfile("Goldman Sachs", [
        (180, 0.4), (90, -0.8), (180, 0.2), (55, 0.6),   # strong → crisis → recovery
    ], "banking"),
    EntityProfile("HDFC Bank", [
        (365, 0.6), (90, -0.3), (90, 0.5),                # consistently strong, minor dip
    ], "banking"),
    EntityProfile("Yes Bank", [
        (60, 0.2), (120, -0.9), (120, -0.7), (60, 0.1),  # collapse arc (mirrors real event)
    ], "banking"),
    EntityProfile("Adani Group", [
        (200, 0.5), (60, -0.95), (60, -0.6), (180, 0.1), # short-seller attack arc
    ], "conglomerate"),
    EntityProfile("Infosys", [
        (300, 0.5), (60, -0.4), (145, 0.4),               # whistleblower → recovery
    ], "tech"),
    EntityProfile("Reliance Industries", [
        (365, 0.3), (100, 0.7), (100, 0.2),               # steady with rights issue boom
    ], "conglomerate"),
    EntityProfile("Tata Motors", [
        (120, -0.3), (120, 0.6), (100, 0.4), (125, 0.1), # JLR turnaround arc
    ], "auto"),
    EntityProfile("Paytm", [
        (180, 0.1), (90, -0.7), (90, -0.5), (145, -0.2), # IPO hype → RBI action
    ], "fintech"),
    EntityProfile("Zomato", [
        (90, 0.4), (90, -0.3), (180, 0.2), (145, -0.1),
    ], "consumer"),
    EntityProfile("ICICI Bank", [
        (200, 0.5), (80, -0.2), (285, 0.6),               # fraud case → strong recovery
    ], "banking"),
]

# -------------------------------------------------------------------
# Richer templates — varied structure, domain vocabulary
# -------------------------------------------------------------------

# Format keys: {entity}, {sector}, {num}, {pct}, {amount}
POSITIVE: list[str] = [
    "{entity} Q{num} net profit surges {pct}% YoY, NIM expansion drives beat",
    "Analysts upgrade {entity} to BUY; price target raised to ₹{amount}",
    "{entity} gross NPA falls to decade-low {pct}%, provisioning coverage strengthens",
    "{entity} secures ₹{amount} crore order book, revenue visibility improves",
    "RBI clears {entity} for new product licence; {sector} peers rally in sympathy",
    "{entity} FCF yield hits {pct}%, buyback programme announced",
    "{entity} rights issue oversubscribed {num}x; promoter stake rises",
    "CRISIL upgrades {entity} outlook to Stable; cost of funds drops {pct}bps",
    "{entity} board approves ₹{amount} crore capex for FY26; execution risk seen low",
    "{entity} wins landmark ₹{amount} crore government contract in {sector}",
    "Foreign institutional investors accumulate {entity}; net inflow ₹{amount} crore in week",
    "{entity} EBITDA margin expands {pct}bps on operating leverage, ahead of street",
    "{entity} completes ₹{amount} crore bond redemption ahead of schedule",
    "{entity} tier-1 capital ratio at {pct}%, well above regulatory minimum",
]

NEGATIVE: list[str] = [
    "{entity} NPA ratio deteriorates to {pct}%, provisioning cost spikes sharply",
    "ED raids {entity} premises over alleged ₹{amount} crore forex violation",
    "{entity} defaults on ₹{amount} crore commercial paper, triggers rating review",
    "SEBI issues show-cause notice to {entity} board over disclosure lapses",
    "{entity} CFO exits abruptly; auditor flags going-concern risk in {sector} unit",
    "Credit Suisse downgrades {entity} to SELL citing write-off cycle ahead",
    "Whistleblower letter alleges round-tripping at {entity}, ₹{amount} crore at risk",
    "{entity} Q{num} PAT misses by {pct}%; management withdraws FY guidance",
    "Moody's places {entity} on review for downgrade; leverage ratio breaches covenant",
    "{entity} exposure to stressed {sector} assets rises to ₹{amount} crore",
    "RBI imposes ₹{amount} crore penalty on {entity} for KYC norm violations",
    "{entity} market cap erodes ₹{amount} crore as short-seller report goes viral",
    "IL&FS contagion: {entity} marks down ₹{amount} crore of structured debt",
    "{entity} restatement reveals ₹{amount} crore revenue recognition error",
]

NEUTRAL: list[str] = [
    "{entity} board meet on {num} Jan to consider Q{num} results",
    "{entity} files draft red herring prospectus for ₹{amount} crore QIP",
    "NCLT admits insolvency petition against minor {entity} subsidiary",
    "{entity} appoints ex-RBI official as independent director",
    "{entity} annual report discloses ₹{amount} crore contingent liability",
    "Shareholding pattern shows {pct}% FII holding in {entity}, flat QoQ",
    "{entity} management to address analysts at {sector} investor day",
    "{entity} receives RBI approval for new branch licences in tier-{num} cities",
    "Credit rating of {entity}'s ₹{amount} crore NCD programme affirmed AA",
    "{entity} completes routine RBI inspection; no material observations reported",
    "NSE adds {entity} to F&O ban list due to high OI concentration",
    "{entity} to raise ₹{amount} crore via infrastructure bonds in H{num} FY26",
]


def _fill(template: str, entity: str, sector: str) -> str:
    return template.format(
        entity=entity,
        sector=sector,
        num=random.randint(1, 4),
        pct=round(random.uniform(2.5, 38.0), 1),
        amount=random.choice([
            250, 500, 750, 1200, 2000, 3500, 5000, 8000, 12000, 20000
        ]),
    )


def _bias_to_weights(bias: float) -> list[float]:
    """Convert a bias in [-1, 1] to [pos, neg, neutral] sampling weights."""
    pos   = max(0.05, 0.35 + bias * 0.35)
    neg   = max(0.05, 0.35 - bias * 0.35)
    neu   = max(0.10, 1.0 - pos - neg)
    total = pos + neg + neu
    return [pos / total, neg / total, neu / total]


def _generate_for_entity(
    profile: EntityProfile,
    start: datetime,
) -> list[dict]:
    rows: list[dict] = []
    current_date = start

    for duration, bias in profile.arc:
        weights = _bias_to_weights(bias)
        for _ in range(duration):
            # Random number of headlines per day
            n_today = min(random.randint(0, 2), 3)
            for _ in range(n_today):
                sentiment = random.choices(
                    ["positive", "negative", "neutral"],
                    weights=weights,
                    k=1,
                )[0]
                match sentiment:
                    case "positive": pool = POSITIVE
                    case "negative": pool = NEGATIVE
                    case _:          pool = NEUTRAL

                headline = _fill(
                    random.choice(pool),
                    entity=profile.name,
                    sector=profile.sector,
                )
                rows.append({
                    "date":       current_date.strftime("%Y-%m-%d"),
                    "entity":     profile.name,
                    "sector":     profile.sector,
                    "headline":   headline,
                    "true_label": sentiment,
                    "arc_bias":   round(bias, 2),   # useful for debugging/analysis
                })
            current_date += timedelta(days=1)

    return rows


def generate(out_path: Path | None = None) -> Path:
    out_path = out_path or Path(__file__).parent / "financial_news.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = datetime(2022, 1, 1)
    all_rows: list[dict] = []
    for profile in ENTITIES:
        all_rows.extend(_generate_for_entity(profile, start))

    all_rows.sort(key=lambda r: (r["date"], r["entity"]))

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    counts: dict[str, int] = {}
    for r in all_rows:
        counts[r["true_label"]] = counts.get(r["true_label"], 0) + 1
    print(f"Saved {len(all_rows)} rows → {out_path}")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    return out_path


if __name__ == "__main__":
    generate()