"""
Financial News Sentiment Risk Scorer — Redesigned Dashboard
Python 3.11 | FinBERT + Rolling Z-Score Anomaly Detection

Launch:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
#from scripts.orchestrator import run_pipeline
from config.settings import CFG

SCORED_DATA_PATH = CFG.paths.scored_csv
TEMPORAL_RISK_PATH = CFG.paths.temporal_csv
RISK_SUMMARY_PATH = CFG.paths.summary_csv
DATA_DIR = CFG.paths.data_dir

ROLLING_WINDOW = 7
ZSCORE_THRESHOLD = 1.5

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinRisk Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("# 🚨 NEW REDESIGNED DASHBOARD LOADED 🚨")

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ─────────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0B1120;
    color: #E2E8F0;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 1400px;
}

/* ── Alert Banner ──────────────────────────────────────────────────────────── */
.alert-banner {
    background: linear-gradient(135deg, #1a0a0a 0%, #2d0f0f 100%);
    border: 1px solid #ef4444;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.15);
}
.alert-banner .alert-text {
    font-size: 0.92rem;
    color: #fca5a5;
    font-weight: 500;
    letter-spacing: 0.01em;
}
.alert-banner .alert-badge {
    background: #ef4444;
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    white-space: nowrap;
}

/* ── Hero ──────────────────────────────────────────────────────────────────── */
.hero-container {
    padding: 2.5rem 0 2rem 0;
    border-bottom: 1px solid #1e2d45;
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #22d3ee;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #e2e8f0 30%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.7rem;
}
.hero-subtitle {
    font-size: 1rem;
    color: #64748b;
    font-weight: 400;
    max-width: 640px;
    line-height: 1.6;
}

/* ── KPI Cards ─────────────────────────────────────────────────────────────── */
.kpi-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.cyan::before  { background: linear-gradient(90deg, #22d3ee, #0891b2); }
.kpi-card.green::before { background: linear-gradient(90deg, #22c55e, #16a34a); }
.kpi-card.red::before   { background: linear-gradient(90deg, #ef4444, #dc2626); }
.kpi-card.amber::before { background: linear-gradient(90deg, #f59e0b, #d97706); }

.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.5rem;
}
.kpi-icon {
    font-size: 1.1rem;
    margin-bottom: 0.4rem;
    display: block;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.35rem;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-card.cyan  .kpi-value { color: #22d3ee; }
.kpi-card.green .kpi-value { color: #22c55e; }
.kpi-card.red   .kpi-value { color: #ef4444; }
.kpi-card.amber .kpi-value { color: #f59e0b; }
.kpi-sub {
    font-size: 0.78rem;
    color: #475569;
    font-weight: 400;
}

/* ── Section Headers ───────────────────────────────────────────────────────── */
.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: 0.01em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22d3ee;
    display: inline-block;
}

/* ── Chart Cards ───────────────────────────────────────────────────────────── */
.chart-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    height: 100%;
}

/* ── Entity Selector ───────────────────────────────────────────────────────── */
.entity-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.3rem;
}
.entity-sector-badge {
    display: inline-block;
    background: #1e2d45;
    border: 1px solid #22d3ee33;
    color: #22d3ee;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* ── AI Insight Card ───────────────────────────────────────────────────────── */
.insight-card {
    background: linear-gradient(135deg, #0f1e35 0%, #0d1929 100%);
    border: 1px solid #1e3a5f;
    border-left: 3px solid #22d3ee;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
}
.insight-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #22d3ee;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.insight-label::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #22d3ee;
    box-shadow: 0 0 6px #22d3ee;
}
.insight-text {
    font-size: 0.97rem;
    color: #cbd5e1;
    line-height: 1.65;
    font-weight: 400;
}
.insight-text strong { color: #e2e8f0; font-weight: 600; }

/* ── Mini Metric Cards (Entity tab) ────────────────────────────────────────── */
.mini-metric {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.mini-metric-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.4rem;
}
.mini-metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
}
.mini-metric-value.red   { color: #ef4444; }
.mini-metric-value.amber { color: #f59e0b; }
.mini-metric-value.cyan  { color: #22d3ee; }
.mini-metric-value.green { color: #22c55e; }

/* ── Risk Tier Badge ───────────────────────────────────────────────────────── */
.tier-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
}
.tier-badge.HIGH   { background: #2d0f0f; color: #ef4444; border: 1px solid #ef444455; }
.tier-badge.MEDIUM { background: #1f1500; color: #f59e0b; border: 1px solid #f59e0b55; }
.tier-badge.LOW    { background: #0a1f0e; color: #22c55e; border: 1px solid #22c55e55; }

/* ── Tab Styling ───────────────────────────────────────────────────────────── */
div[data-testid="stTabs"] button {
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    padding: 0.6rem 1.2rem;
    color: #475569;
    border-bottom: 2px solid transparent;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #22d3ee;
    border-bottom: 2px solid #22d3ee;
}

/* ── Selectbox ─────────────────────────────────────────────────────────────── */
div[data-testid="stSelectbox"] > div {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 10px;
}

/* ── Dividers ──────────────────────────────────────────────────────────────── */
hr { border-color: #1e2d45; margin: 1.5rem 0; }

/* ── Expander ──────────────────────────────────────────────────────────────── */
details {
    background: #111827;
    border: 1px solid #1e2d45 !important;
    border-radius: 10px;
}

/* ── Dataframe ─────────────────────────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid #1e2d45;
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ──────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    title_font=dict(family="Inter, sans-serif", color="#e2e8f0", size=14),
    margin=dict(l=0, r=0, t=40, b=0),
)

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral":  "#22d3ee",
}

TIER_COLORS = {
    "HIGH":   "#ef4444",
    "MEDIUM": "#f59e0b",
    "LOW":    "#22c55e",
}

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_summary_path = DATA_DIR / "source_summary.csv"

    missing = [
        p for p in (
            SCORED_DATA_PATH,
            TEMPORAL_RISK_PATH,
            RISK_SUMMARY_PATH,
            source_summary_path,
        )
        if not p.exists()
    ]

    if missing:
        st.error(
            "Required dashboard CSV files are missing.\n\n"
            + "\n".join(str(p) for p in missing)
            + "\n\nRun orchestrator locally before deployment."
        )
        st.stop()


    return (
        pd.read_csv(SCORED_DATA_PATH, parse_dates=["date"]),
        pd.read_csv(TEMPORAL_RISK_PATH, parse_dates=["date"]),
        pd.read_csv(RISK_SUMMARY_PATH),
        pd.read_csv(source_summary_path),
    )


with st.spinner("Loading intelligence data…"):
    scored_df, temporal_df, risk_summary, source_summary = _load()


# Safety validation before dashboard metrics/charts
required_cols = ["risk_tier", "composite_risk_score"]

if risk_summary.empty or not all(
    col in risk_summary.columns for col in required_cols
):
    st.warning("Risk summary unavailable. Run orchestrator first.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# DERIVED METRICS
# ──────────────────────────────────────────────────────────────────────────────
high_risk_entities  = risk_summary[risk_summary["risk_tier"] == "HIGH"]
if risk_summary.empty:
    st.warning("No risk data yet. Run orchestrator first.")
    st.stop()

high_risk_entities = (
    risk_summary[risk_summary["risk_tier"] == "HIGH"]
    if "risk_tier" in risk_summary.columns
    else pd.DataFrame()
)

top_risk_entity = risk_summary.iloc[0]["entity"]
top_risk_score = risk_summary.iloc[0]["composite_risk_score"]
top_risk_tier = risk_summary.iloc[0]["risk_tier"]

total_anomalies = (
    int(temporal_df["anomaly_flag"].sum())
    if "anomaly_flag" in temporal_df.columns
    else 0
)   

avg_sentiment = (
    scored_df["sentiment_score"].mean()
    if "sentiment_score" in scored_df.columns
    else 0
)

sent_direction = (
    "Positive" if avg_sentiment > 0.05
    else "Negative" if avg_sentiment < -0.05
    else "Neutral"
)

# ──────────────────────────────────────────────────────────────────────────────
# ALERT BANNER
# ──────────────────────────────────────────────────────────────────────────────
if len(high_risk_entities) > 0:
    names = ", ".join(high_risk_entities["entity"].tolist())
    st.markdown(f"""
    <div class="alert-banner">
        <span class="alert-badge">⚠ Live Alert</span>
        <span class="alert-text">
            <strong>{len(high_risk_entities)} {'entity' if len(high_risk_entities)==1 else 'entities'}</strong>
            {'is' if len(high_risk_entities)==1 else 'are'} currently classified as
            <strong style="color:#ef4444">HIGH RISK</strong> —
            {names}.
            Highest risk entity: <strong>{top_risk_entity}</strong>
            (score {top_risk_score:.1f}/100).
        </span>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_overview, tab_entity, tab_analytics = st.tabs([
    "  📊  Overview  ",
    "  🏦  Entity Analysis  ",
    "  🔬  Analytics & Data  ",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:

    # Hero
    st.markdown("""
    <div class="hero-container">
        <div class="hero-eyebrow">Powered by FinBERT · Anomaly Detection · Real-time Scoring</div>
        <div class="hero-title">AI-Powered Financial<br>Risk Intelligence</div>
        <div class="hero-subtitle">
            Entity-level sentiment surveillance across financial news.
            FinBERT extracts directional signals; rolling z-score analysis
            surfaces entities with statistically abnormal risk trajectories.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ──
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card cyan">
            <span class="kpi-icon">📰</span>
            <div class="kpi-label">Headlines Analysed</div>
            <div class="kpi-value">{len(scored_df):,}</div>
            <div class="kpi-sub">across {scored_df['entity'].nunique()} entities</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card red">
            <span class="kpi-icon">🚨</span>
            <div class="kpi-label">High-Risk Entities</div>
            <div class="kpi-value">{len(high_risk_entities)}</div>
            <div class="kpi-sub">require immediate review</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card amber">
            <span class="kpi-icon">⚡</span>
            <div class="kpi-label">Anomaly Spikes</div>
            <div class="kpi-value">{total_anomalies:,}</div>
            <div class="kpi-sub">|z-score| &gt; {ZSCORE_THRESHOLD}</div>
        </div>""", unsafe_allow_html=True)

    sent_color = "green" if avg_sentiment > 0.05 else "red" if avg_sentiment < -0.05 else "cyan"
    with c4:
        st.markdown(f"""
        <div class="kpi-card {sent_color}">
            <span class="kpi-icon">📈</span>
            <div class="kpi-label">Market Sentiment</div>
            <div class="kpi-value">{avg_sentiment:+.3f}</div>
            <div class="kpi-sub">{sent_direction} bias across corpus</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    st.subheader("Top News Sources")

    fig_sources = px.bar(
    source_summary.sort_values("headlines", ascending=False).head(10),
    x="source",
    y="headlines",
    color="headlines",
    color_continuous_scale="RdYlGn",
    )

    fig_sources.update_layout(
        xaxis_title="Source",
        yaxis_title="Headline Count",
        xaxis_tickangle=-45,
    )

    st.plotly_chart(fig_sources, width="stretch")

    # ── Two Main Charts ──
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-header"><span class="dot"></span>Entity Risk Ranking</div>', unsafe_allow_html=True)
        sorted_summary = risk_summary.sort_values("composite_risk_score")
        fig_rank = go.Figure()
        fig_rank.add_trace(go.Bar(
            x=sorted_summary["composite_risk_score"],
            y=sorted_summary["entity"],
            orientation="h",
            marker=dict(
                color=sorted_summary["risk_tier"].map(TIER_COLORS),
                line=dict(width=0),
            ),
            text=sorted_summary["composite_risk_score"].round(1),
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Risk Score: %{x:.1f}<br>"
                "<extra></extra>"
            ),
        ))
        fig_rank.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            xaxis_title="Composite Risk Score",
            xaxis=dict(
                gridcolor="#1e2d45",
                linecolor="#1e2d45",
                tickcolor="#475569",
                range=[0, 100],
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_rank, width="stretch", config={"displayModeBar": False})

    with col_right:
        st.markdown('<div class="section-header"><span class="dot"></span>Risk Tier Distribution</div>', unsafe_allow_html=True)
        tier_counts = risk_summary["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        fig_donut = go.Figure(go.Pie(
            labels=tier_counts["tier"],
            values=tier_counts["count"],
            hole=0.62,
            marker=dict(
                colors=[TIER_COLORS.get(t, "#475569") for t in tier_counts["tier"]],
                line=dict(color="#0B1120", width=3),
            ),
            textinfo="label+percent",
            textfont=dict(color="#e2e8f0", size=12),
            hovertemplate="<b>%{label}</b><br>%{value} entities<extra></extra>",
        ))
        fig_donut.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            annotations=[dict(
                text=f"<b>{len(risk_summary)}</b><br><span style='font-size:10px'>entities</span>",
                x=0.5, y=0.5,
                font=dict(size=18, color="#e2e8f0"),
                showarrow=False,
            )],
            showlegend=True,
            legend=dict(
                bgcolor="rgba(17,24,39,0.8)",
                bordercolor="#1e2d45",
                borderwidth=1,
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.05,
            ),
        )
        st.plotly_chart(fig_donut, width="stretch", config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ENTITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_entity:

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Entity selector
    entity_list = risk_summary.sort_values("composite_risk_score", ascending=False)["entity"].tolist()
    selected = st.selectbox(
        "Select an entity to analyse",
        entity_list,
        index=0,
        key="entity_selector",
    )

    # Pull entity data
    ent_row     = risk_summary[risk_summary["entity"] == selected].iloc[0]
    ent_scored  = scored_df[scored_df["entity"] == selected]
    ent_temporal = temporal_df[temporal_df["entity"] == selected].copy()
    ent_temporal["date"] = pd.to_datetime(ent_temporal["date"])

    tier        = ent_row["risk_tier"]
    neg_ratio   = ent_row["negative_ratio"]
    risk_score  = ent_row["composite_risk_score"]
    anomaly_days = ent_row["anomaly_days"]
    avg_sent_ent = ent_row["avg_sentiment_score"]
    total_hl    = ent_row["total_headlines"]
    sector      = ent_scored["sector"].iloc[0] if "sector" in ent_scored.columns else "Financial"

    # Entity header row
    h_col, badge_col = st.columns([5, 1])
    with h_col:
        st.markdown(f"""
        <div class="entity-header">{selected}</div>
        <span class="entity-sector-badge">{sector}</span>
        """, unsafe_allow_html=True)
    with badge_col:
        st.markdown(f"""
        <div style='text-align:right; padding-top:0.5rem'>
            <span class="tier-badge {tier}">{tier} RISK</span>
        </div>
        """, unsafe_allow_html=True)

    # ── AI Insight Card ──
    # Generate a concise natural-language insight from the numbers
    sent_word  = "positive" if avg_sent_ent > 0.05 else "negative" if avg_sent_ent < -0.05 else "neutral"
    trend_desc = (
        "experiencing significant negative pressure"
        if tier == "HIGH" and neg_ratio > 0.4
        else "showing signs of sentiment recovery"
        if tier == "MEDIUM"
        else "maintaining a relatively stable outlook"
    )
    anomaly_severity = (
        "a critical number of" if anomaly_days > 20
        else "several" if anomaly_days > 10
        else "a few"
    )
    insight_text = (
        f"<strong>{selected}</strong> is currently classified as "
        f"<strong style='color:{TIER_COLORS[tier]}'>{tier} RISK</strong> "
        f"with a composite score of <strong>{risk_score:.1f}/100</strong>. "
        f"Analysis of {total_hl:,} headlines reveals "
        f"<strong>{neg_ratio*100:.1f}%</strong> negative sentiment — "
        f"the entity is {trend_desc}. "
        f"The rolling z-score detector flagged {anomaly_severity} "
        f"<strong>{anomaly_days} anomaly day{'s' if anomaly_days != 1 else ''}</strong> "
        f"over the observation period, indicating "
        f"{'statistically unusual volatility in news tone' if anomaly_days > 10 else 'broadly stable news trajectory'}."
    )

    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-label">AI-Generated Risk Insight</div>
        <div class="insight-text">{insight_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Temporal Chart ──
    st.markdown('<div class="section-header"><span class="dot"></span>Sentiment Trajectory</div>', unsafe_allow_html=True)

    fig_temp = go.Figure()

    # Fill area under the line
    fig_temp.add_trace(go.Scatter(
        x=ent_temporal["date"],
        y=ent_temporal["daily_avg_sentiment"],
        fill="tozeroy",
        fillcolor="rgba(34,211,238,0.05)",
        line=dict(color="#22d3ee", width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig_temp.add_trace(go.Scatter(
        x=ent_temporal["date"],
        y=ent_temporal["daily_avg_sentiment"],
        mode="lines",
        name="Daily Sentiment",
        line=dict(color="#22d3ee", width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Sentiment: %{y:.3f}<extra></extra>",
    ))
    fig_temp.add_trace(go.Scatter(
        x=ent_temporal["date"],
        y=ent_temporal["rolling_mean"],
        mode="lines",
        name=f"{ROLLING_WINDOW}d Rolling Mean",
        line=dict(color="#475569", width=1.5, dash="dot"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Rolling Mean: %{y:.3f}<extra></extra>",
    ))

    # Upper/lower sigma bands
    upper = ent_temporal["rolling_mean"] + (ZSCORE_THRESHOLD * ent_temporal["rolling_std"].fillna(0))
    lower = ent_temporal["rolling_mean"] - (ZSCORE_THRESHOLD * ent_temporal["rolling_std"].fillna(0))
    fig_temp.add_trace(go.Scatter(
        x=pd.concat([ent_temporal["date"], ent_temporal["date"][::-1]]),
        y=pd.concat([upper, lower[::-1]]),
        fill="toself",
        fillcolor="rgba(71,85,105,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
        name=f"±{ZSCORE_THRESHOLD}σ Band",
    ))

    # Anomaly markers with glow effect
    anomalies = ent_temporal[ent_temporal["anomaly_flag"]]
    if not anomalies.empty:
        fig_temp.add_trace(go.Scatter(
            x=anomalies["date"],
            y=anomalies["daily_avg_sentiment"],
            mode="markers",
            name=f"Anomaly (|z|>{ZSCORE_THRESHOLD})",
            marker=dict(
                color="#ef4444",
                size=10,
                symbol="circle",
                line=dict(color="#ef4444", width=2),
            ),
            hovertemplate=(
                "<b>ANOMALY</b><br>"
                "%{x|%d %b %Y}<br>"
                "Sentiment: %{y:.3f}<extra></extra>"
            ),
        ))

    fig_temp.add_hline(y=0, line_dash="dot", line_color="#1e2d45", line_width=1)
    fig_temp.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        yaxis_title="Sentiment Score",
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(17,24,39,0.8)",
            bordercolor="#1e2d45",
            borderwidth=1,
            orientation="h",
            y=1.12,
            x=0,
        ),
    )
    st.plotly_chart(fig_temp, width="stretch", config={"displayModeBar": False})

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── 3 Mini Metric Cards ──
    m1, m2, m3 = st.columns(3)
    neg_color = "red" if neg_ratio > 0.4 else "amber" if neg_ratio > 0.25 else "green"
    score_color = "red" if risk_score > 40 else "amber" if risk_score > 20 else "green"
    anom_color  = "red" if anomaly_days > 20 else "amber" if anomaly_days > 10 else "cyan"

    with m1:
        st.markdown(f"""
        <div class="mini-metric">
            <div class="mini-metric-label">Negative Sentiment Ratio</div>
            <div class="mini-metric-value {neg_color}">{neg_ratio*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="mini-metric">
            <div class="mini-metric-label">Composite Risk Score</div>
            <div class="mini-metric-value {score_color}">{risk_score:.1f}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="mini-metric">
            <div class="mini-metric-label">Anomaly Days Detected</div>
            <div class="mini-metric-value {anom_color}">{anomaly_days}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Sentiment breakdown bar for this entity ──
    st.markdown('<div class="section-header"><span class="dot"></span>Sentiment Breakdown</div>', unsafe_allow_html=True)
    ent_sent_counts = (
        ent_scored["predicted_label"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "label", "predicted_label": "count",
                         "count": "count"})
    )
    # Normalise column names across pandas versions
    ent_sent_counts.columns = ["label", "count"]

    fig_breakdown = go.Figure(go.Bar(
        x=ent_sent_counts["label"],
        y=ent_sent_counts["count"],
        marker_color=[SENTIMENT_COLORS.get(l, "#475569") for l in ent_sent_counts["label"]],
        text=ent_sent_counts["count"],
        textposition="outside",
        textfont=dict(color="#94a3b8"),
        hovertemplate="<b>%{x}</b><br>Headlines: %{y}<extra></extra>",
    ))
    fig_breakdown.update_layout(
        **PLOTLY_LAYOUT,
        height=220,
        showlegend=False,
        yaxis_title="Headline Count",
    )
    st.plotly_chart(fig_breakdown, width="stretch", config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS & DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Anomaly Timeline ──
    with st.expander("📈  Anomaly Timeline & Event Tracking", expanded=True):
        st.markdown("""
        <div style='font-size:0.85rem; color:#64748b; margin-bottom:1rem; line-height:1.5'>
            Visualize the sentiment trajectory over time for selected entities. 
            The line shows daily average sentiment, and <span style='color:#ef4444; font-weight:600;'>red markers</span> 
            represent statistical anomalies (where absolute rolling z-score > {}). 
            This highlights the exact onset of a crisis and the subsequent recovery trajectory.
        </div>
        """.format(ZSCORE_THRESHOLD), unsafe_allow_html=True)

        all_entities = sorted(risk_summary["entity"].unique().tolist())
        # Default select the top 3 high-risk entities
        default_selected = risk_summary.head(3)["entity"].tolist()

        selected_entities = st.multiselect(
            "Select entities to plot:",
            options=all_entities,
            default=default_selected,
            key="anomaly_timeline_select",
        )

        if selected_entities:
            fig_timeline = go.Figure()

            # Elegant color scheme for multiple lines
            colors = ["#22d3ee", "#a78bfa", "#f472b6", "#34d399", "#fbbf24", "#38bdf8", "#818cf8"]

            for idx, entity in enumerate(selected_entities):
                color = colors[idx % len(colors)]
                ent_df = temporal_df[temporal_df["entity"] == entity].copy()
                ent_df["date"] = pd.to_datetime(ent_df["date"])
                ent_df = ent_df.sort_values("date")

                # Daily average sentiment line
                fig_timeline.add_trace(go.Scatter(
                    x=ent_df["date"],
                    y=ent_df["daily_avg_sentiment"],
                    mode="lines",
                    name=entity,
                    line=dict(color=color, width=2.5),
                    hovertemplate=f"<b>{entity}</b><br>Date: %{{x|%d %b %Y}}<br>Sentiment: %{{y:.3f}}<extra></extra>",
                ))

                # Highlight anomalies
                anoms = ent_df[ent_df["anomaly_flag"]]
                if not anoms.empty:
                    fig_timeline.add_trace(go.Scatter(
                        x=anoms["date"],
                        y=anoms["daily_avg_sentiment"],
                        mode="markers",
                        name=f"{entity} Anomaly",
                        marker=dict(
                            color="#ef4444",
                            size=10,
                            symbol="circle",
                            line=dict(color="#ffffff", width=1.5)
                        ),
                        customdata=anoms["z_score"],
                        hovertemplate=f"<b>🚨 Anomaly: {entity}</b><br>Date: %{{x|%d %b %Y}}<br>Sentiment: %{{y:.3f}}<br>Z-Score: %{{customdata:.2f}}<extra></extra>",
                        showlegend=False,
                    ))

            fig_timeline.add_hline(y=0, line_dash="dot", line_color="#1e2d45", line_width=1)
            fig_timeline.update_layout(
                **PLOTLY_LAYOUT,
                height=380,
                xaxis=dict(
                    gridcolor="#1e2d45",
                    linecolor="#1e2d45",
                    tickcolor="#475569",
                ),
                yaxis=dict(
                    gridcolor="#1e2d45",
                    linecolor="#1e2d45",
                    tickcolor="#475569",
                    title="Daily Avg Sentiment Score",
                ),
                hovermode="closest",
                legend=dict(
                    bgcolor="rgba(17,24,39,0.8)",
                    bordercolor="#1e2d45",
                    borderwidth=1,
                    orientation="h",
                    x=0.5,
                    xanchor="center",
                    y=-0.15,
                ),
            )
            st.plotly_chart(fig_timeline, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Please select at least one entity to display the anomaly timeline.")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Risk Leaderboard & Sparkline Cards ──
    with st.expander("🏆  Risk Leaderboard & Sparkline Cards", expanded=True):
        st.markdown("""
        <div style='font-size:0.82rem; color:#475569; margin-bottom:1rem; line-height:1.6'>
            Real-time risk status and historical sentiment trends across all tracked entities.
        </div>
        """, unsafe_allow_html=True)

        col_sort, _ = st.columns([2, 3])
        with col_sort:
            sort_order = st.radio(
                "Sort Leaderboard:",
                options=["Highest Risk First", "Lowest Risk First", "Alphabetical (A-Z)"],
                horizontal=True,
                key="leaderboard_sort_order",
            )

        # Sort data
        sorted_summary = risk_summary.copy()
        if sort_order == "Highest Risk First":
            sorted_summary = sorted_summary.sort_values("composite_risk_score", ascending=False)
        elif sort_order == "Lowest Risk First":
            sorted_summary = sorted_summary.sort_values("composite_risk_score", ascending=True)
        else:
            sorted_summary = sorted_summary.sort_values("entity", ascending=True)

        # Render in a grid of 3 columns
        num_cols = 3
        cols = st.columns(num_cols)

        for idx, row in enumerate(sorted_summary.itertuples()):
            col_idx = idx % num_cols
            with cols[col_idx]:
                entity = row.entity
                tier   = row.risk_tier
                score  = row.composite_risk_score

                # Compute trend arrow from last 5 data points
                ent_hist = (
                    temporal_df[temporal_df["entity"] == entity]
                    .sort_values("date")
                    ["daily_avg_sentiment"]
                    .tail(5)
                )
                if len(ent_hist) >= 2:
                    slope = ent_hist.iloc[-1] - ent_hist.iloc[0]
                    if slope > 0.02:
                        trend_icon  = "▲"
                        trend_color = "#22c55e"
                        trend_label = "Rising"
                    elif slope < -0.02:
                        trend_icon  = "▼"
                        trend_color = "#ef4444"
                        trend_label = "Falling"
                    else:
                        trend_icon  = "→"
                        trend_color = "#94a3b8"
                        trend_label = "Stable"
                else:
                    trend_icon, trend_color, trend_label = "→", "#94a3b8", "Stable"

                st.markdown(f"""
                <div style="
                    background: #111827;
                    border: 1px solid #1e2d45;
                    border-left: 3px solid {TIER_COLORS.get(tier, '#22d3ee')};
                    border-radius: 12px;
                    padding: 1rem 1.2rem;
                    margin-top: 0.5rem;
                ">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.55rem">
                        <span style="font-weight:700;font-size:0.9rem;color:#e2e8f0;
                                     overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                                     max-width:130px">{entity}</span>
                        <span class="tier-badge {tier}" style="font-size:0.6rem;padding:2px 7px">{tier}</span>
                    </div>
                    <div style="display:flex;align-items:baseline;gap:6px;margin-bottom:0.45rem">
                        <span style="font-size:1.5rem;font-family:'JetBrains Mono',monospace;
                                     font-weight:800;color:{TIER_COLORS.get(tier,'#e2e8f0')}">{score:.1f}</span>
                        <span style="font-size:0.68rem;color:#475569;text-transform:uppercase;
                                     letter-spacing:0.05em">/ 100</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:5px">
                        <span style="font-size:0.85rem;color:{trend_color};font-weight:700">{trend_icon}</span>
                        <span style="font-size:0.72rem;color:{trend_color}">{trend_label} risk</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Dedicated entity trend chart ──
        st.markdown("""
        <div style='height:1.2rem'></div>
        <div class='section-header'><span class='dot'></span>Entity Sentiment Trend — Detail View</div>
        """, unsafe_allow_html=True)

        trend_entities = sorted_summary["entity"].tolist()
        trend_sel = st.selectbox(
            "Select entity to inspect:",
            options=trend_entities,
            index=0,
            key="leaderboard_trend_select",
        )

        t_hist = (
            temporal_df[temporal_df["entity"] == trend_sel]
            .copy()
        )
        t_hist["date"] = pd.to_datetime(t_hist["date"])
        t_hist = t_hist.sort_values("date")
        # Smooth with 5-pt rolling mean for clarity
        t_hist["smooth"] = t_hist["daily_avg_sentiment"].rolling(5, min_periods=1).mean()

        fig_trend_card = go.Figure()
        fig_trend_card.add_trace(go.Scatter(
            x=t_hist["date"],
            y=t_hist["daily_avg_sentiment"],
            mode="lines",
            name="Daily",
            line=dict(color="#1e2d45", width=1),
            hovertemplate="%{x|%d %b}<br>Sentiment: %{y:.3f}<extra></extra>",
        ))
        fig_trend_card.add_trace(go.Scatter(
            x=t_hist["date"],
            y=t_hist["smooth"],
            mode="lines",
            name="5-day avg",
            line=dict(color=TIER_COLORS.get(
                risk_summary[risk_summary["entity"] == trend_sel]["risk_tier"].iloc[0]
                if len(risk_summary[risk_summary["entity"] == trend_sel]) else "LOW",
                "#22d3ee"
            ), width=2.5),
            hovertemplate="%{x|%d %b}<br>Smoothed: %{y:.3f}<extra></extra>",
        ))
        anoms_t = t_hist[t_hist["anomaly_flag"]]
        if not anoms_t.empty:
            fig_trend_card.add_trace(go.Scatter(
                x=anoms_t["date"],
                y=anoms_t["daily_avg_sentiment"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#ef4444", size=8, symbol="circle"),
                hovertemplate="Anomaly<br>%{x|%d %b}<br>Sentiment: %{y:.3f}<extra></extra>",
            ))
        fig_trend_card.add_hline(y=0, line_dash="dot", line_color="#1e2d45", line_width=1)
        fig_trend_card.update_layout(
            **PLOTLY_LAYOUT,
            height=260,
            yaxis_title="Sentiment Score",
            hovermode="x unified",
            legend=dict(
                bgcolor="rgba(17,24,39,0.8)",
                bordercolor="#1e2d45",
                borderwidth=1,
                orientation="h",
                x=0, y=1.12,
            ),
        )
        st.plotly_chart(fig_trend_card, width="stretch", config={"displayModeBar": False})

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── All-entity sentiment distribution ──
    with st.expander("📈  Sentiment Distribution — All Entities", expanded=False):
        sent_counts = (
            scored_df.groupby(["entity", "predicted_label"], sort=False)
            .size()
            .reset_index(name="count")
        )
        fig_all_sent = px.bar(
            sent_counts,
            x="entity", y="count",
            color="predicted_label",
            color_discrete_map=SENTIMENT_COLORS,
            barmode="stack",
        )
        fig_all_sent.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            xaxis_tickangle=-30,
            legend_title="Sentiment",
        )
        st.plotly_chart(fig_all_sent, width="stretch", config={"displayModeBar": False})

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Headline Drill-Down ──
    with st.expander("🔍  Headline Drill-Down — Raw Data", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            ent_filter = st.multiselect(
                "Filter by entity",
                options=sorted(scored_df["entity"].unique().tolist()),
                default=[],
                placeholder="All entities",
                key="hl_entity_filter",
            )
        with col_f2:
            sent_filter = st.multiselect(
                "Filter by sentiment",
                ["positive", "negative", "neutral"],
                default=["negative"],
                key="hl_sent_filter",
            )

        view = scored_df.copy()
        if ent_filter:
            view = view[view["entity"].isin(ent_filter)]
        if sent_filter:
            view = view[view["predicted_label"].isin(sent_filter)]

        display_cols = [
            "date", "entity", "headline", "predicted_label",
            "sentiment_score", "prob_positive", "prob_negative", "prob_neutral",
        ]
        available_cols = [c for c in display_cols if c in view.columns]

        st.markdown(
            f"<div style='font-size:0.8rem;color:#475569;margin-bottom:0.6rem'>"
            f"Showing {len(view):,} of {len(scored_df):,} headlines</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            view[available_cols]
            .sort_values("sentiment_score")
            .reset_index(drop=True),
            height=380,
        )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Risk Summary Table ──
    with st.expander("📋  Full Entity Risk Summary", expanded=False):
        def _style_tier(val: str) -> str:
            match val:
                case "HIGH":   return "background-color:#2d0f0f;color:#ef4444;font-weight:600"
                case "MEDIUM": return "background-color:#1f1500;color:#f59e0b;font-weight:600"
                case "LOW":    return "background-color:#0a1f0e;color:#22c55e;font-weight:600"
                case _:        return ""

        st.dataframe(
            risk_summary.style.map(_style_tier, subset=["risk_tier"]),
        )

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='
    border-top: 1px solid #1e2d45;
    margin-top: 3rem;
    padding-top: 1.2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.75rem;
    color: #334155;
'>
    <span>FinRisk Intelligence · FinBERT + Rolling Z-Score Anomaly Detection</span>
    <span style="font-family:'JetBrains Mono',monospace">Python 3.11 · Streamlit · Plotly</span>
</div>
""", unsafe_allow_html=True)