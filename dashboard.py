"""Spotify AI Review Discovery Engine — Executive Dashboard."""

from __future__ import annotations

import glob
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify AI Review Discovery Engine",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

OUTPUT_DIR = Path(__file__).parent / "output"

# ── Design tokens ─────────────────────────────────────────────────────────────
GREEN       = "#1DB954"
GREEN_DARK  = "#158A3E"
GREEN_LIGHT = "#E8F8EE"
RED         = "#E53935"
ORANGE      = "#F57C00"
AMBER       = "#F9A825"
BLUE        = "#1565C0"
PURPLE      = "#6A1B9A"

PRIORITY_COLOR = {"critical": RED, "high": ORANGE, "medium": AMBER, "low": GREEN}
SENTIMENT_COLOR = {"positive": GREEN, "negative": RED, "neutral": "#78909C", "mixed": AMBER}
PRIORITY_ORDER  = ["critical", "high", "medium", "low"]

CHART_COLORS = [GREEN, "#26C6DA", "#AB47BC", "#EF5350", "#FFA726", "#42A5F5", "#66BB6A", "#EC407A"]

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Design tokens ── */
:root {
    --bg:            #0E0E10;
    --surface:       #181818;
    --surface-raised:#1D1D1F;
    --border:        #2A2A2E;
    --border-hover:  #3D3D42;
    --text-primary:  #FFFFFF;
    --text-secondary:#B3B3B3;
    --text-tertiary: #7A7A80;
    --accent:        #1DB954;
    --accent-dim:    rgba(29,185,84,0.16);
    --radius-lg: 14px;
    --radius-md: 10px;
    --radius-sm: 8px;
    --space-1: 0.5rem;
    --space-2: 0.75rem;
    --space-3: 1.25rem;
    --space-4: 1.75rem;
    --space-5: 2.5rem;
    --space-6: 3.5rem;
    --shadow-card: 0 1px 3px rgba(0,0,0,0.32);
    --transition: all 0.16s ease;
}

/* ── Base ── */
[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stHeader"]           { background: transparent; }

/* Sidebar removed in favor of an inline filter bar — hide the sidebar panel
   and its collapse/expand toggle so no empty chrome remains. */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* Remove default padding top */
.block-container { padding-top: 1.5rem !important; max-width: 1240px; }

/* ── Typography ── */
h1, h2, h3, h4 { font-family: "Inter", "Helvetica Neue", sans-serif; }
p, span, div { -webkit-font-smoothing: antialiased; }

/* ── Section headers (H2 — subsections nested within a report area) ── */
.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin: var(--space-5) 0 var(--space-2) 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--accent);
    display: inline-block;
}
/* First section header inside a freshly-opened area shouldn't add extra
   top air on top of the area's own opening margin. */
div[class*='st-key-rs_body_'] > div:first-child .section-header,
div[class*='st-key-aiq_body_'] .section-header { margin-top: var(--space-2); }

/* ── H1 — major report headline (Executive Summary, AI Product Intelligence) ── */
.section-header-h1 {
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0 0 0.3rem 0;
}
.section-subtitle {
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.5;
    margin: 0 0 var(--space-3) 0;
    max-width: 760px;
}

/* ── Divider between major report chapters — replaces bare <br> spacers ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border) 15%, var(--border) 85%, transparent);
    margin: var(--space-6) 0;
    border: none;
}

/* ── Generic card surface — shared visual language for every card type ── */
.kpi-card-premium, .stat-card,
div[class*="st-key-cc_"], div[class*="st-key-cluster_detail"],
div[class*="st-key-ds_card_"], .aiq-insight-card, .opp-item, .narrative-card {
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-card);
    transition: var(--transition);
}

/* ── Chart card ── (st.container(key="cc_*") wrapper — see chart_card()) */
div[class*="st-key-cc_"] {
    background: var(--surface);
    padding: var(--space-3);
    margin-bottom: var(--space-3);
}

/* ── Insight cards ── */
/* st.container(key="cluster_detail") wrapper — border-left color set dynamically per priority */
div[class*="st-key-cluster_detail"] {
    background: var(--surface);
    padding: var(--space-4);
    margin-bottom: var(--space-3);
    border-left: 4px solid var(--accent);
}
.cluster-theme {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
}
.cluster-meta {
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-bottom: 0.75rem;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: 4px;
}
.badge-critical { background: rgba(229,57,53,0.18);  color: #FF8A80; }
.badge-high     { background: rgba(245,124,0,0.18);  color: #FFB74D; }
.badge-medium   { background: rgba(249,168,37,0.18); color: #FFD54F; }
.badge-low      { background: rgba(29,185,84,0.18);  color: #4ADE80; }

/* ── Review quote ── */
.review-quote {
    background: var(--surface-raised);
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #D8D8DC;
    margin: 0.5rem 0;
    line-height: 1.55;
}
.review-meta {
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}

/* ── Opportunity item ── */
.opp-item {
    background: var(--surface);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem 0.85rem 1.25rem;
    margin: 0.4rem 0;
    border-left: 4px solid var(--accent);
}
.opp-meta { font-size: 0.72rem; color: var(--text-secondary); margin-bottom: 0.2rem; }
.opp-text { font-size: 0.88rem; color: var(--text-primary); font-weight: 500; }

/* ── AI Product Intelligence — question cards ── */
.aiq-subhead {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; color: var(--text-secondary); margin: var(--space-3) 0 0.6rem 0;
    display: flex; align-items: center; gap: 6px;
}
.aiq-insight-card {
    background: var(--surface); border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 1rem 1.1rem; margin-bottom: 0.65rem;
}
.aiq-insight-theme { font-size: 0.85rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.45rem; }
.aiq-insight-problem { font-size: 0.87rem; color: var(--text-primary); line-height: 1.55; margin: 0 0 0.5rem 0; }
.aiq-insight-cause  { font-size: 0.8rem;  color: var(--text-secondary); line-height: 1.5;  margin: 0 0 0.5rem 0; }
.aiq-insight-users  { font-size: 0.76rem; color: var(--text-tertiary); margin: 0; }
.aiq-chip {
    display: inline-block; font-size: 0.72rem; padding: 3px 10px; border-radius: 99px;
    background: #212121; border: 1px solid var(--border); color: var(--text-secondary); margin: 0 6px 6px 0;
}
.aiq-metric { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-sm);
              padding: 0.65rem 0.8rem; text-align: center; }
.aiq-metric-val { font-size: 1.05rem; font-weight: 700; color: var(--text-primary); }
.aiq-metric-lbl { font-size: 0.68rem; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }

/* Accordion row buttons — one per strategic question. Chevron is a generated
   ::after glyph (not baked into the label) so it can be rotated with a
   transition instead of swapping ▸/▾ characters. */
div[class*='st-key-aiq_row_'] { margin-top: var(--space-1); }
div[class*='st-key-aiq_row_'] button {
    width: 100%; text-align: left; justify-content: flex-start;
    display: flex !important; align-items: center;
    background: #141414 !important; border: 1px solid var(--border) !important;
    color: #D8D8DC !important; font-size: 0.9rem !important; font-weight: 600 !important;
    padding: 0.75rem 1.1rem !important; border-radius: var(--radius-md) !important;
}
/* Streamlit wraps the button label in its own internal flex div (default
   justify-content: center) — the rule above only reaches the <button> itself,
   so the label still visually centers unless this inner wrapper is overridden too. */
div[class*='st-key-aiq_row_'] button > div { justify-content: flex-start !important; }
div[class*='st-key-aiq_row_'] button::after {
    content: "\\25B8"; margin-left: auto; padding-left: 0.75rem;
    color: var(--text-tertiary); transition: transform 0.16s ease;
}
div[class*='st-key-aiq_row_'] button:hover { border-color: var(--accent) !important; color: #FFFFFF !important; }
div[class*='st-key-aiq_body_'] {
    padding: var(--space-2) 0.1rem var(--space-1) 0.1rem;
    animation: aiq-reveal 0.15s ease;
}
@keyframes aiq-reveal { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }

/* ── Report sections (top-level collapsible groups: Research Context, Visual
   Evidence, Cluster Explorer, Supporting Evidence, Other Signals) ── */
div[class*='st-key-rs_row_'] { margin-top: var(--space-4); }
div[class*='st-key-rs_row_'] button {
    width: 100%; text-align: left; justify-content: flex-start;
    display: flex !important; align-items: center;
    background: #141414 !important; border: 1px solid var(--border) !important;
    border-left: 4px solid var(--accent) !important;
    color: var(--text-primary) !important; font-size: 1.02rem !important; font-weight: 700 !important;
    padding: 0.95rem 1.25rem !important; border-radius: var(--radius-md) !important;
}
/* Same inner-wrapper override as the aiq_row_ rule above — Streamlit's own
   button-label div defaults to justify-content: center regardless of the
   outer button's flex-start. */
div[class*='st-key-rs_row_'] button > div { justify-content: flex-start !important; }
div[class*='st-key-rs_row_'] button::after {
    content: "\\25B8"; margin-left: auto; padding-left: 0.75rem;
    color: var(--text-tertiary); font-size: 0.85rem; transition: transform 0.16s ease;
}
div[class*='st-key-rs_row_'] button:hover { border-color: var(--accent) !important; background: var(--surface-raised) !important; }
.report-section-header {
    font-size: 1.02rem; font-weight: 700; color: var(--text-primary);
    background: #141414; border: 1px solid var(--border); border-left: 4px solid var(--accent);
    border-radius: var(--radius-md); padding: 0.95rem 1.25rem; margin-top: var(--space-4);
}
/* Supporting-research eyebrow tag + subtitle shown under every report-section
   button — visible whether the section is collapsed or open, so the reader
   knows what's inside before clicking. */
.rs-caption { margin: 0.55rem 0.1rem 0 0.1rem; }
.rs-tag {
    display: inline-block; font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; color: var(--text-tertiary); background: var(--surface-raised);
    border: 1px solid var(--border); border-radius: 99px; padding: 1px 9px; margin-right: 8px;
}
.rs-caption-text { font-size: 0.82rem; color: var(--text-secondary); }

/* ── Streamlit metric override ── */
[data-testid="stMetric"] { background: transparent !important; }

/* ── Generic button polish (accordion buttons above have higher specificity
   and are unaffected) ── */
.stButton > button {
    border-radius: var(--radius-md);
    transition: var(--transition);
}
.stButton > button:not([kind="primary"]) {
    border: 1px solid var(--border);
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--accent);
    color: var(--text-primary);
}

/* ── Secondary stat card (Research Context) ── */
.stat-card {
    background: var(--surface);
    padding: var(--space-2) var(--space-3);
    text-align: center;
    height: 100%;
    display: flex; flex-direction: column; justify-content: center;
    min-height: 96px;
}
.stat-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.1;
    margin: 0.2rem 0;
}
.stat-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
}
div[class*="st-key-stat_row"] div[data-testid="stHorizontalBlock"] { align-items: stretch; }
div[class*="st-key-stat_row"] div[data-testid="stColumn"] { display: flex; }
div[class*="st-key-stat_row"] div[data-testid="stColumn"] > div { width: 100%; }
div[class*="st-key-stat_row"] div[data-testid="stVerticalBlock"] { height: 100%; }

/* ── Narrative summary ── */
.narrative-card {
    background: var(--surface);
    padding: var(--space-3) var(--space-4);
    border-left: 4px solid var(--accent);
    font-size: 0.95rem;
    color: #D8D8DC;
    line-height: 1.65;
    margin-bottom: var(--space-3);
}

/* ── Hero v2 (landing) ── */
.hero-v2 { padding: 1.75rem 0 0.5rem 0; text-align: center; }
.hero-v2-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--text-primary);
    font-family: "Inter", "Helvetica Neue", sans-serif;
    letter-spacing: -0.02em;
    margin: 0 0 0.6rem 0;
}
.hero-v2-title span { color: var(--accent); }
.hero-v2-tag {
    font-size: 1rem;
    color: var(--text-secondary);
    font-weight: 400;
    max-width: 680px;
    width: 100%;
    margin: 0 auto !important;
    line-height: 1.55;
    text-align: center !important;
}
/* Higher-specificity guard: Streamlit applies its own default alignment/width
   to <p> tags inside stMarkdownContainer, which otherwise beats the plain
   .hero-v2-tag class rule above and left-shifts this specific paragraph. */
[data-testid="stMarkdownContainer"] p.hero-v2-tag {
    margin-left: auto !important;
    margin-right: auto !important;
    text-align: center !important;
}

/* ── Landing summary strip (Total Reviews / Sources) — compact, single line,
   sits above the Review Sources cards so the CTA stays visible without
   scrolling. ── */
.landing-stats { text-align: center; margin: 0.75rem 0 0.4rem 0; }
.landing-stat-chip {
    display: inline-block; font-size: 0.8rem; font-weight: 600;
    color: var(--text-secondary); background: var(--surface);
    border: 1px solid var(--border); border-radius: 99px;
    padding: 4px 16px; margin: 0 6px;
}
.landing-stat-chip b { color: var(--text-primary); font-weight: 800; }
/* The "Review Sources" heading follows the stat strip directly (no
   collapsible accordion in between), so it doesn't need the standard
   section-header's large space-5 top margin meant for separating chapters
   further down the page. */
div[class*="st-key-review_sources_header"] .section-header { margin-top: var(--space-2); }

/* ── Review Sources grid ── */
/* Scoped to the dataset_grid container only, so other st.columns() layouts
   elsewhere in the app (KPI rows, chart columns) are unaffected. */
div[class*="st-key-dataset_grid"] div[data-testid="stHorizontalBlock"] {
    align-items: stretch;                 /* Flexbox: stretch every column to row height */
}
div[class*="st-key-dataset_grid"] div[data-testid="stColumn"] {
    display: flex;
    height: 100%;
}
div[class*="st-key-dataset_grid"] div[data-testid="stColumn"] > div {
    width: 100%;
}
div[class*="st-key-dataset_grid"] div[data-testid="stVerticalBlock"] {
    height: 100%;                         /* cascade the stretch down to the card itself */
}

/* ── Review Source cards — informational only, no longer interactive ── */
div[class*="st-key-ds_card_"] {
    background: var(--surface);
    padding: 0.75rem 0.85rem 0.65rem 0.85rem;
    text-align: center;
    display: flex;
    flex-direction: column;
    min-height: 128px;                    /* fixed minimum so short-copy cards stay uniform */
    height: 100%;
    cursor: default;
}
div[class*="st-key-ds_card_"]:hover {
    border-color: var(--border-hover);
    box-shadow: 0 4px 14px rgba(0,0,0,0.4);
    transform: translateY(-2px);
}
.dataset-name   { font-size: 0.8rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.1rem; }
.dataset-count  { font-size: 1.25rem; font-weight: 800; color: var(--accent); line-height: 1.1; }
.dataset-sub    { font-size: 0.6rem; color: var(--text-tertiary); text-transform: uppercase;
                   letter-spacing: 0.06em; margin-bottom: 0.35rem; }
.dataset-desc   { font-size: 0.68rem; color: var(--text-secondary); line-height: 1.35; }

/* ── Premium KPI cards (Executive Summary) ── */
div[class*="st-key-kpi_row"] div[data-testid="stHorizontalBlock"] { align-items: stretch; }
div[class*="st-key-kpi_row"] div[data-testid="stColumn"] { display: flex; }
div[class*="st-key-kpi_row"] div[data-testid="stColumn"] > div { width: 100%; }
div[class*="st-key-kpi_row"] div[data-testid="stVerticalBlock"] { height: 100%; }
.kpi-card-premium {
    background: linear-gradient(180deg, var(--surface) 0%, #141414 100%);
    padding: var(--space-2) var(--space-2);
    text-align: center;
    position: relative;
    overflow: hidden;
    height: 100%;
    display: flex; flex-direction: column; justify-content: center;
    min-height: 92px;
}
.kpi-card-premium::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent), #26C6DA);
}
.kpi-icon-premium  { font-size: 1rem; margin-bottom: 0.2rem; }
.kpi-value-premium { font-size: 1.5rem; font-weight: 800; color: var(--text-primary); line-height: 1.1; margin: 0.1rem 0; }
.kpi-label-premium {
    font-size: 0.64rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.07em; color: var(--text-secondary);
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .block-container { padding-left: 0.9rem !important; padding-right: 0.9rem !important; }
    .hero-v2 { padding-top: 0.75rem; }
    .hero-v2-title { font-size: 1.65rem; }
    .hero-v2-tag { font-size: 0.88rem; padding: 0 0.25rem; }
    .section-header-h1 { font-size: 1.25rem; }
    .kpi-value-premium { font-size: 1.25rem; }
    .kpi-icon-premium { font-size: 0.9rem; }
    div[class*='st-key-rs_row_'] button,
    div[class*='st-key-aiq_row_'] button { font-size: 0.85rem !important; padding: 0.75rem 0.9rem !important; }
    .section-divider { margin: var(--space-4) 0; }
}
</style>
""", unsafe_allow_html=True)


# ── Data loaders (cached) ─────────────────────────────────────────────────────

@st.cache_data
def load_summary() -> dict:
    p = OUTPUT_DIR / "analytics_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

@st.cache_data
def load_clusters() -> list[dict]:
    p = OUTPUT_DIR / "review_clusters.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

@st.cache_data
def load_insights() -> list[dict]:
    p = OUTPUT_DIR / "cluster_insights.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

@st.cache_data
def load_master() -> pd.DataFrame:
    p = OUTPUT_DIR / "master_reviews.csv"
    return pd.read_csv(p, index_col=0, encoding="utf-8") if p.exists() else pd.DataFrame()

@st.cache_data
def load_ai() -> pd.DataFrame:
    files = sorted(glob.glob(str(OUTPUT_DIR / "ai_analysis" / "batch_*.json")))
    if not files:
        return pd.DataFrame()
    return pd.concat(
        [pd.DataFrame(json.loads(open(f, encoding="utf-8").read())) for f in files],
        ignore_index=True,
    )


def _flat(series: pd.Series) -> list[str]:
    out: list[str] = []
    for cell in series.dropna():
        if isinstance(cell, list):
            out.extend(str(v) for v in cell if v)
        elif isinstance(cell, str) and cell:
            out.append(cell)
    return out


def _chart_layout(fig: go.Figure, h: int = 380) -> go.Figure:
    fig.update_layout(
        height=h,
        paper_bgcolor="#181818",
        plot_bgcolor="#181818",
        font=dict(family="Inter, Helvetica Neue, sans-serif", size=12, color="#D8D8DC"),
        margin=dict(t=40, l=8, r=8, b=8),
        title_font=dict(size=13, color="#FFFFFF", family="Inter, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color="#D8D8DC")),
        hoverlabel=dict(bgcolor="#212121", bordercolor="#2A2A2E",
                        font=dict(color="#FFFFFF", family="Inter, sans-serif")),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2A2A2E", zeroline=False, color="#D8D8DC")
    fig.update_yaxes(showgrid=True, gridcolor="#2A2A2E", zeroline=False, color="#D8D8DC")
    return fig


def _badge(priority: str) -> str:
    cls = f"badge badge-{priority}"
    return f"<span class='{cls}'>{priority}</span>"


PRIORITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def _build_question_cards(question_defs: list[dict], insight_map: dict[int, dict]) -> list[dict]:
    """Aggregate one or more cluster insights into a single strategic-question card.

    Reads only from cluster_insights.json (via insight_map) — deliberately does not
    join against review_clusters.json, since that file was regenerated with different
    cluster numbering after the spotify_community fix and no longer aligns with the
    cluster_ids baked into the (still-stale, intentionally unregenerated) insights file.
    """
    cards = []
    for q in question_defs:
        contributing = [insight_map[cid] for cid in q["cluster_ids"] if cid in insight_map]
        if not contributing:
            continue
        worst = max(contributing, key=lambda c: PRIORITY_RANK.get(c.get("priority", "low"), 1))
        confidences = [c.get("confidence", 0.5) for c in contributing]
        cards.append({
            "title": q["title"],
            "priority": worst.get("priority", "low"),
            "confidence": sum(confidences) / len(confidences),
            "insights": contributing,
            "total_excerpts": sum(len(c.get("evidence", [])) for c in contributing),
        })
    return cards


def _section(title: str) -> None:
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)


def _section_h1(title: str, subtitle: str = "") -> None:
    """Major report headline (Executive Summary, AI Product Intelligence) —
    heavier weight than _section() so the hierarchy between a chapter title
    and its internal subsections is unambiguous at a glance."""
    st.markdown(f"<div class='section-header-h1'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p class='section-subtitle'>{subtitle}</p>", unsafe_allow_html=True)


def _divider() -> None:
    """Intentional chapter break between major report sections — replaces
    bare <br> spacers with a visible, deliberate transition."""
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)


_chart_card_seq = 0

def chart_card():
    """A real Streamlit container (not a raw HTML string) so chart content
    actually nests inside the white card — st.markdown open/close tags across
    separate calls never wrap anything, since each call renders its own
    isolated DOM fragment."""
    global _chart_card_seq
    _chart_card_seq += 1
    return st.container(key=f"cc_{_chart_card_seq}")


def render_report_section(
    key: str, icon: str, title: str, subtitle: str = "",
    default_expanded: bool = False, collapsible: bool = True, accent: str = GREEN,
) -> bool:
    """Top-level report section header. Same button + session_state accordion
    pattern as the AI Product Intelligence question cards, but each section
    toggles independently (not mutually exclusive) and can optionally be
    locked open via collapsible=False.

    The "Supporting Research" tag + subtitle render underneath the button
    regardless of expanded state, so a reader can tell what a section holds
    before ever clicking it."""
    if not collapsible:
        st.markdown(
            f"<div class='report-section-header' style='border-left-color:{accent}'>"
            f"{icon}&nbsp;&nbsp;{title}</div>",
            unsafe_allow_html=True,
        )
        if subtitle:
            st.markdown(
                f"<div class='rs-caption'><span class='rs-caption-text'>{subtitle}</span></div>",
                unsafe_allow_html=True,
            )
        return True

    state_key = f"rs_{key}_expanded"
    st.session_state.setdefault(state_key, default_expanded)
    is_expanded = st.session_state[state_key]

    # One combined style block per row: custom accent color (if any) and the
    # chevron rotation + "stays lit while open" treatment for the active state.
    # The attribute selector is doubled (repeated verbatim) purely to win
    # specificity over the generic rs_row_ base rule deterministically —
    # both rules are equally specific single-attribute selectors otherwise,
    # which makes the override a same-specificity tie decided only by
    # stylesheet insertion order (fragile against Streamlit's rerender timing).
    sel = f"div[class*='st-key-rs_row_{key}'][class*='st-key-rs_row_{key}']"
    rules = [f"{sel} button {{ border-left-color: {accent} !important; }}"] \
        if accent != GREEN else []
    if is_expanded:
        rules.append(f"{sel} button {{ background: #1D1D1F !important; border-color: {accent} !important; }}")
        rules.append(f"{sel} button::after {{ transform: rotate(90deg); }}")
    if rules:
        st.markdown(f"<style>{''.join(rules)}</style>", unsafe_allow_html=True)

    with st.container(key=f"rs_row_{key}"):
        if st.button(f"{icon}  {title}", key=f"rs_btn_{key}", use_container_width=True):
            st.session_state[state_key] = not is_expanded
            st.rerun()

    if subtitle:
        st.markdown(
            f"<div class='rs-caption'><span class='rs-tag'>Supporting Research</span>"
            f"<span class='rs-caption-text'>{subtitle}</span></div>",
            unsafe_allow_html=True,
        )

    return is_expanded


# ── Load data ─────────────────────────────────────────────────────────────────
summary  = load_summary()
clusters = load_clusters()
insights = load_insights()
master   = load_master()
ai_df    = load_ai()

insight_map: dict[int, dict] = {ins["cluster_id"]: ins for ins in insights}
cluster_map: dict[int, dict] = {c["cluster_id"]: c  for c in clusters}

desc   = summary.get("descriptive_statistics", {})
cl_sum = summary.get("clustering", {})


# ── Filter inputs (widgets render inline, below, once the report is visible) ──
all_sources = sorted(master["source"].dropna().unique()) if not master.empty else []


# ═══════════════════════════════════════════════════════════════════════════════
# LANDING EXPERIENCE — Hero, Review Sources overview, loading sequence, Executive Summary.
# The dashboard never runs the AI pipeline live — it loads the latest outputs already
# produced by the offline review-analysis workflow (see main.py / src/cluster_analyser.py).
# Review Sources below are informational only; there is no per-source selection, since
# every section always reflects the full combined corpus in output/.
# ═══════════════════════════════════════════════════════════════════════════════

st.session_state.setdefault("pipeline_done", False)
st.session_state.setdefault("pipeline_started", False)

# These four are used again, unchanged, by the untouched Executive Overview
# section below — keep the exact same formulas so downstream code is unaffected.
total_reviews  = summary.get("total_reviews", len(master))
num_clusters   = cl_sum.get("num_clusters", len(clusters))
num_sources    = len(desc.get("reviews_per_source", {})) or len(all_sources)
high_pri_opps  = sum(
    len(ins.get("product_opportunities", []))
    for ins in insights
    if ins.get("priority") in ("critical", "high")
)

# No dataset selector — every section always reflects the full combined corpus
# loaded from output/. sel_sources / master_f / ai_f are kept as names (rather
# than inlining `master`/`ai_df` everywhere below) purely so the many existing
# per-section references further down don't need to change.
sel_sources = all_sources
master_f = master.copy()
ai_f     = ai_df.copy()

# Fixed "active dataset" description — the app only ever analyzes the combined
# corpus now, so this is a constant rather than a selection result.
active_ds = {"name": "Combined", "count": total_reviews}

src_data = desc.get("reviews_per_source", {})
SOURCE_CARDS = [
    {"name": "Reddit", "count": src_data.get("reddit", 0),
     "desc": "Long-form community discussions, complaints and feature debates."},
    {"name": "Play Store", "count": src_data.get("google_play", 0),
     "desc": "Android user reviews and star ratings from Google Play."},
    {"name": "Spotify Community", "count": src_data.get("spotify_community", 0),
     "desc": "Community threads collected — feature ideas and complaints from Spotify's community forum."},
    {"name": "App Store", "count": src_data.get("app_store", 0),
     "desc": "iOS user reviews from the US and India App Store."},
]

LOADING_STAGES = [
    "Loading Review Dataset",
    "Loading AI Analysis Results",
    "Loading Cluster Insights",
    "Preparing Product Intelligence",
    "Preparing Dashboard",
]

WORKFLOW_NOTE = (
    "Insights shown in this dashboard are generated from the latest processed review "
    "dataset using the offline AI review analysis workflow."
)

st.markdown(
    """
    <div class="hero-v2">
        <div class="hero-v2-title">Spotify Review <span>Intelligence</span></div>
        <p class="hero-v2-tag">
            AI-powered analysis of Spotify user feedback across Play Store, Reddit,
            App Store and Spotify Community.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.pipeline_done:
    st.markdown(
        f"<div class='landing-stats'>"
        f"<span class='landing-stat-chip'><b>{total_reviews:,}</b> Total Reviews</span>"
        f"<span class='landing-stat-chip'><b>{num_sources}</b> Sources</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.container(key="review_sources_header"):
        _section("Review Sources")

    with st.container(key="dataset_grid"):
        ds_cols = st.columns(len(SOURCE_CARDS))
        for col, ds in zip(ds_cols, SOURCE_CARDS):
            card_key = f"ds_card_{ds['name'].replace(' ', '_')}"
            with col:
                with st.container(key=card_key):
                    st.markdown(
                        f"<div class='dataset-name'>{ds['name']}</div>"
                        f"<div class='dataset-count'>{ds['count']:,}</div>"
                        f"<div class='dataset-sub'>reviews</div>"
                        f"<div class='dataset-desc'>{ds['desc']}</div>",
                        unsafe_allow_html=True,
                    )

    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

    cta_l, cta_mid, cta_r = st.columns([1, 1.4, 1])
    with cta_mid:
        launch_clicked = st.button(
            "🚀  Launch Review Intelligence", type="primary", use_container_width=True,
            key="launch_review_btn",
        )
        st.markdown(
            f"<p style='text-align:center;font-size:0.75rem;color:#9CA3AF;margin-top:0.4rem'>"
            f"Loads the latest processed outputs &middot; {total_reviews:,} reviews across "
            f"{num_sources} sources</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center;font-size:0.72rem;color:#7A7A80;margin-top:0.6rem;"
            f"line-height:1.5'>{WORKFLOW_NOTE}</p>",
            unsafe_allow_html=True,
        )

    if launch_clicked:
        st.session_state.pipeline_started = True

    if st.session_state.pipeline_started:
        with st.status("Loading review intelligence…", expanded=True) as status:
            bar = st.progress(0, text="Starting…")
            n_stages = len(LOADING_STAGES)
            for i, stage in enumerate(LOADING_STAGES, start=1):
                row = st.empty()
                row.markdown(f"⏳ {stage}")
                time.sleep(0.4)
                row.markdown(f"✅ {stage}")
                bar.progress(i / n_stages, text=stage)
            status.update(label="Ready — dashboard loaded", state="complete", expanded=False)

        st.session_state.pipeline_done = True
        st.session_state.pipeline_started = False
        st.rerun()

    st.stop()

else:
    # ── Executive Summary — premium KPI cards ──────────────────────────────
    hdr_l, hdr_r = st.columns([4, 1])
    with hdr_l:
        _section_h1(
            "Executive Summary",
            f"Loaded from the latest processed outputs in <code>output/</code> &middot; "
            f"{total_reviews:,} reviews across {num_sources} sources. {WORKFLOW_NOTE}",
        )
    with hdr_r:
        if st.button("↺ Back to Review Sources", key="reset_pipeline_btn", use_container_width=True):
            st.session_state.pipeline_done = False
            st.rerun()

    with st.container(key="kpi_row"):
        kp1, kp2, kp3, kp4 = st.columns(4)
        for col, icon, val, lbl in [
            (kp1, "🎧", f"{len(master_f):,}", "Reviews Analyzed"),
            (kp2, "🧩", str(num_clusters),    "Clusters Identified"),
            (kp3, "🌐", str(len(sel_sources)),"Sources"),
            (kp4, "🚀", str(high_pri_opps),   "High Priority Opportunities"),
        ]:
            col.markdown(
                f"<div class='kpi-card-premium'>"
                f"<div class='kpi-icon-premium'>{icon}</div>"
                f"<div class='kpi-value-premium'>{val}</div>"
                f"<div class='kpi-label-premium'>{lbl}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    _divider()


# ═══════════════════════════════════════════════════════════════════════════════
# AI PRODUCT INTELLIGENCE — six strategic product questions, each answered by a
# reusable "question card" component. Cluster-to-question mapping is internal;
# only theme names and AI-generated content are ever rendered.
# ═══════════════════════════════════════════════════════════════════════════════
_section_h1(
    "AI Product Intelligence",
    "Executive research briefs synthesized from AI cluster analysis — diagnosis, "
    "evidence, and recommended action for each strategic product question.",
)

QUESTION_DEFINITIONS = [
    {"title": "Why do users feel Spotify's recommendations and discovery have stalled?",
     "cluster_ids": [9, 3]},
    {"title": "How does Spotify's discovery experience hold up against competitors?",
     "cluster_ids": [8, 1]},
    {"title": "Should Spotify invest in community-driven and artist-facing discovery tools?",
     "cluster_ids": [10, 5]},
    {"title": "Are Spotify's personalized mix formats deep enough to drive lasting engagement?",
     "cluster_ids": [0, 2, 11]},
    {"title": "Where are technical reliability issues undermining the listening experience?",
     "cluster_ids": [7, 4]},
    {"title": "Is the free-tier experience undermining premium conversion and retention?",
     "cluster_ids": [6]},
]

if not insight_map:
    st.info("Run `python main.py --cluster-analysis` to generate AI insights for this section.")
else:
    question_cards = _build_question_cards(QUESTION_DEFINITIONS, insight_map)
    st.session_state.setdefault("aiq_expanded", None)

    # Highlight whichever row is currently expanded — same dynamic-style-override
    # pattern used for ds_card selection and cluster_detail's priority border.
    # Selector is doubled to deterministically out-specificity the generic
    # aiq_row_ base rule instead of relying on stylesheet order (see the same
    # technique + rationale in render_report_section). Skipped entirely when
    # nothing is expanded (all questions start collapsed).
    if st.session_state.aiq_expanded is not None:
        _aiq_sel = f"div[class*='st-key-aiq_row_{st.session_state.aiq_expanded}'][class*='st-key-aiq_row_{st.session_state.aiq_expanded}']"
        st.markdown(
            f"<style>{_aiq_sel} button "
            f"{{ border-color: {GREEN} !important; background: #181818 !important; color: #FFFFFF !important; }}"
            f"{_aiq_sel} button::after "
            f"{{ transform: rotate(90deg); }}</style>",
            unsafe_allow_html=True,
        )

    for i, qc in enumerate(question_cards):
        is_expanded = st.session_state.aiq_expanded == i
        n_insights = len(qc["insights"])
        label = (
            f"{qc['title']}   ·   {qc['priority'].upper()}   ·   "
            f"{qc['confidence']:.0%} confidence   ·   "
            f"{n_insights} AI insight{'s' if n_insights != 1 else ''}"
        )
        with st.container(key=f"aiq_row_{i}"):
            if st.button(label, key=f"aiq_btn_{i}", use_container_width=True):
                # Toggle: clicking the open question collapses it; clicking any
                # other question expands it and implicitly collapses the rest,
                # since only one index is ever tracked.
                st.session_state.aiq_expanded = None if is_expanded else i
                st.rerun()

        if is_expanded:
            with st.container(key=f"aiq_body_{i}"):
                # ── 1. What We Learned ───────────────────────────────────────
                st.markdown("<div class='aiq-subhead'>\U0001F50D What We Learned</div>",
                            unsafe_allow_html=True)
                for ins in qc["insights"]:
                    p_color = PRIORITY_COLOR.get(ins.get("priority", "low"), GREEN)
                    st.markdown(
                        f"<div class='aiq-insight-card' style='border-left:4px solid {p_color}'>"
                        f"<div class='aiq-insight-theme'>{ins.get('theme', '—')} "
                        f"{_badge(ins.get('priority', 'low'))}"
                        f"<span style='font-size:0.7rem;color:#6B7280;font-weight:400;margin-left:6px'>"
                        f"{ins.get('confidence', 0):.0%} confidence</span></div>"
                        f"<p class='aiq-insight-problem'>{ins.get('problem_statement', '—')}</p>"
                        f"<p class='aiq-insight-cause'><b style='color:#D8D8DC'>Root cause: </b>"
                        f"{ins.get('root_cause_hypothesis', '—')}</p>"
                        f"<p class='aiq-insight-users'>Affected: {ins.get('affected_users', '—')}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # ── 2. How We Know ───────────────────────────────────────────
                st.markdown("<div class='aiq-subhead'>\U0001F4CA How We Know</div>",
                            unsafe_allow_html=True)

                chips = "".join(
                    f"<span class='aiq-chip'>{ins.get('theme', '—')}: "
                    f"{ins.get('priority', '—')} priority · {ins.get('impact', '—')} impact · "
                    f"{ins.get('effort', '—')} effort</span>"
                    for ins in qc["insights"]
                )
                st.markdown(chips, unsafe_allow_html=True)

                st.markdown(
                    "<p style='font-size:0.78rem;font-weight:700;color:#9CA3AF;margin:0.8rem 0 0.35rem'>"
                    "Affected user segments</p>",
                    unsafe_allow_html=True,
                )
                for ins in qc["insights"]:
                    st.markdown(
                        f"<p style='font-size:0.82rem;color:#D8D8DC;line-height:1.55;margin:0 0 0.4rem'>"
                        f"&bull; <b>{ins.get('theme', '—')}</b> — {ins.get('affected_users', '—')}</p>",
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    "<p style='font-size:0.78rem;font-weight:700;color:#9CA3AF;margin:0.9rem 0 0.35rem'>"
                    "Representative reviews</p>",
                    unsafe_allow_html=True,
                )
                quote_budget = 4
                for ins in qc["insights"]:
                    if quote_budget <= 0:
                        break
                    for quote in ins.get("evidence", [])[:2]:
                        if quote_budget <= 0:
                            break
                        snippet = quote if len(quote) <= 320 else quote[:320] + "…"
                        st.markdown(f"<div class='review-quote'>{snippet}</div>", unsafe_allow_html=True)
                        quote_budget -= 1

                mm1, mm2, mm3 = st.columns(3)
                for col, val, lbl in [
                    (mm1, n_insights,          "Contributing AI insights"),
                    (mm2, qc["total_excerpts"], "Supporting review excerpts"),
                    (mm3, f"{qc['confidence']:.0%}", "Avg. AI confidence"),
                ]:
                    col.markdown(
                        f"<div class='aiq-metric'><div class='aiq-metric-val'>{val}</div>"
                        f"<div class='aiq-metric-lbl'>{lbl}</div></div>",
                        unsafe_allow_html=True,
                    )

                # ── 3. What Spotify Should Build ─────────────────────────────
                st.markdown("<div class='aiq-subhead'>\U0001F4A1 What Spotify Should Build</div>",
                            unsafe_allow_html=True)
                for ins in qc["insights"]:
                    opps = ins.get("product_opportunities", [])
                    if not opps:
                        continue
                    st.markdown(
                        f"<p style='font-size:0.76rem;color:#6B7280;margin:0.6rem 0 0.3rem'>"
                        f"From: {ins.get('theme', '—')}</p>",
                        unsafe_allow_html=True,
                    )
                    for o in opps:
                        st.markdown(
                            f"<div class='opp-item'><div class='opp-text'>{o}</div></div>",
                            unsafe_allow_html=True,
                        )

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    _divider()


# ── Filters — Cluster/Priority only. Dataset/source selection now lives
# solely on the landing page (sel_sources, computed above); these two remain
# scoped to the supporting-research sections below (Research Context, Visual
# Evidence, Cluster Explorer, Other Signals). The AI Product Intelligence
# briefs always reflect the full corpus. ──────────────────────────────────────
cluster_options = sorted(c["cluster_id"] for c in clusters)
priority_options = ["critical", "high", "medium", "low"]

# No Cluster/Priority filter UI — Cluster Explorer, Recommended Product
# Opportunities, Other Signals, and Supporting Evidence always consider every
# cluster and priority. (Opportunities has its own Priority/Impact/Effort
# sub-filter further down for narrowing the opportunity list specifically.)
sel_clusters   = cluster_options
sel_priorities = priority_options

clusters_f = [c for c in clusters if c["cluster_id"] in sel_clusters]
insights_f = [insight_map[cid] for cid in sel_clusters if cid in insight_map
              and insight_map[cid].get("priority", "low") in sel_priorities]

# Clusters whose theme is operational/monetization rather than music-discovery
# itself (app stability, free-tier/ads, widget) — surfaced separately in
# "Other Signals" instead of diluting the discovery narrative below.
OTHER_SIGNAL_CLUSTER_IDS = {4, 6, 7}
clusters_f_discovery  = [c for c in clusters_f if c["cluster_id"] not in OTHER_SIGNAL_CLUSTER_IDS]
insights_f_discovery  = [i for i in insights_f if i["cluster_id"] not in OTHER_SIGNAL_CLUSTER_IDS]
insights_f_other      = [i for i in insights_f if i["cluster_id"] in OTHER_SIGNAL_CLUSTER_IDS]


# ═══════════════════════════════════════════════════════════════════════════════
# 📚 RESEARCH CONTEXT — dataset overview, source breakdown, review volume,
# rating distribution. Collapsed by default: background for the findings above,
# not the headline.
# ═══════════════════════════════════════════════════════════════════════════════
if render_report_section(
    "research_context", "📚", "Research Context",
    subtitle="Dataset composition, review volume, and baseline statistics behind this analysis.",
    default_expanded=False,
):
    with st.container(key="rs_body_research_context"):
        _section("Dataset Overview")

        has_ratings   = not master_f.empty and master_f["rating"].notna().any()
        avg_rating    = master_f["rating"].mean()   if has_ratings else None
        median_rating = master_f["rating"].median() if has_ratings else None
        avg_len       = (
            master_f["review_text"].dropna().astype(str).str.split().str.len().mean()
            if not master_f.empty else 0
        )
        top_theme     = max(insights, key=lambda i: cluster_map.get(i["cluster_id"], {}).get("size", 0))["theme"] \
                        if insights else "—"

        rating_sentence = (
            f"users rate Spotify's music discovery experience an average of <b>{avg_rating:.2f} / 5</b> "
            f"({median_rating:.0f} median), skewing toward critical feedback. "
            if has_ratings else
            "star ratings aren't collected for this source — insights below are drawn from review text only. "
        )

        st.markdown(
            f"<div class='narrative-card'>"
            f"Across <b>{len(master_f):,} reviews</b> from <b>{active_ds['name']}</b> "
            f"({len(sel_sources)} source{'s' if len(sel_sources) != 1 else ''}), {rating_sentence}"
            f"This sits within a combined corpus of <b>{total_reviews:,} reviews</b> across {num_sources} sources, "
            f"which the AI pipeline distilled into <b>{num_clusters} thematic clusters</b>, surfacing "
            f"<b>{high_pri_opps} high-priority product opportunities</b> — with &ldquo;{top_theme}&rdquo; standing "
            f"out as the largest single theme by volume."
            f"</div>",
            unsafe_allow_html=True,
        )

        with st.container(key="stat_row"):
            s1, s2, s3, s4 = st.columns(4)
            for col, val, lbl in [
                (s1, f"{avg_rating:.2f} ★" if has_ratings else "N/A", "Average Rating"),
                (s2, f"{avg_len:.0f} words",     "Avg. Review Length"),
                (s3, f"{len(master_f):,}",       "Reviews in Selection"),
                (s4, f"{len(sel_sources)}",      "Sources in Selection"),
            ]:
                col.markdown(
                    f"<div class='stat-card'><div class='stat-label'>{lbl}</div>"
                    f"<div class='stat-value'>{val}</div></div>",
                    unsafe_allow_html=True,
                )

        # Source Breakdown & Review Volume
        _section("Source Breakdown & Review Volume")

        src_counts = master_f["source"].value_counts() if not master_f.empty else pd.Series(dtype=int)
        if not src_counts.empty:
            df_src = src_counts.reset_index()
            df_src.columns = ["Source", "Count"]

            col_src1, col_src2 = st.columns([1.3, 1])

            with col_src1:
                fig = px.pie(
                    df_src, names="Source", values="Count",
                    title=f"Reviews by Source — {active_ds['name']}",
                    color_discrete_sequence=CHART_COLORS,
                    hole=0.52,
                )
                fig.update_traces(textposition="outside", textinfo="percent+label",
                                  marker=dict(line=dict(color="#181818", width=2)))
                fig = _chart_layout(fig, 380)
                fig.update_layout(showlegend=False)
                with chart_card():
                    st.plotly_chart(fig, use_container_width=True)

            with col_src2:
                df_src_tbl = df_src.sort_values("Count", ascending=False).copy()
                df_src_tbl["Share"] = (df_src_tbl["Count"] / df_src_tbl["Count"].sum() * 100).round(1).astype(str) + "%"
                with chart_card():
                    st.markdown("<p style='font-size:0.85rem;font-weight:700;color:#FFFFFF;margin-bottom:0.75rem'>Review Volume by Source</p>",
                                unsafe_allow_html=True)
                    st.dataframe(df_src_tbl, hide_index=True, use_container_width=True, height=300)

        # Rating Distribution
        _section("Rating Distribution")

        if has_ratings:
            rating_dist = master_f["rating"].dropna().astype(int).value_counts().sort_index().to_dict()
            df_rat = pd.DataFrame(
                [{"Star Rating": f"{k} ★", "Reviews": v} for k, v in rating_dist.items()]
            )
            STAR_COLORS = [RED, ORANGE, AMBER, "#66BB6A", GREEN]
            fig = px.bar(
                df_rat, x="Star Rating", y="Reviews",
                title=f"Rating Distribution — {active_ds['name']}",
                color="Star Rating",
                color_discrete_sequence=STAR_COLORS,
                text="Reviews",
            )
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig = _chart_layout(fig, 380)
            fig.update_layout(showlegend=False,
                              xaxis_title="", yaxis_title="Number of Reviews")
            with chart_card():
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                f"No star ratings are available for **{active_ds['name']}** — this source doesn't "
                f"collect numeric ratings."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 📊 VISUAL EVIDENCE — every supporting chart behind the AI Product Intelligence
# briefs above (cluster overview, pain points, feature requests, discovery
# feature mentions, recommendation components, sentiment).
# ═══════════════════════════════════════════════════════════════════════════════
if render_report_section(
    "visual_evidence", "📊", "Visual Evidence",
    subtitle="Charts and analytics supporting the AI Product Intelligence findings above.",
    default_expanded=False,
):
    with st.container(key="rs_body_visual_evidence"):
        # Cluster priority treemap — bird's-eye view of where the volume & risk sits
        _section("Cluster Priority Overview")
        st.caption("🌐 Reflects the complete combined dataset — cluster structure isn't recomputed per source.")
        if insights:
            df_tree = pd.DataFrame([
                {
                    "Theme": ins.get("theme", f"C{ins['cluster_id']}"),
                    "Size":  cluster_map.get(ins["cluster_id"], {}).get("size", 10),
                    "Priority": ins.get("priority", "medium").capitalize(),
                }
                for ins in insights
            ])
            fig = px.treemap(
                df_tree, path=["Priority", "Theme"], values="Size",
                title="Clusters by Priority & Size",
                color="Priority",
                color_discrete_map={
                    "Critical": RED, "High": ORANGE, "Medium": AMBER, "Low": GREEN,
                },
            )
            fig.update_traces(textinfo="label+value", marker_line_width=1,
                              marker_line_color="#181818")
            fig = _chart_layout(fig, 420)
            fig.update_layout(margin=dict(t=40, l=4, r=4, b=4))
            with chart_card():
                st.plotly_chart(fig, use_container_width=True)

        def _no_ai_data(what: str) -> None:
            if ai_df.empty:
                st.info(f"Run `python main.py --analyse` to generate AI analysis data for {what}.")
            else:
                st.info(f"No AI-analyzed reviews are available for **{active_ds['name']}** to show {what} yet.")

        _section("Top Pain Points")
        if not ai_f.empty:
            pain_counter = Counter(_flat(ai_f["pain_points"]))
            if pain_counter:
                df_pp = pd.DataFrame(pain_counter.most_common(12), columns=["Pain Point", "Count"])
                df_pp = df_pp.sort_values("Count")
                fig = px.bar(
                    df_pp, x="Count", y="Pain Point", orientation="h",
                    title=f"Top Pain Points — {active_ds['name']}",
                    color="Count",
                    color_continuous_scale=[[0, "#FFCDD2"], [1, RED]],
                )
                fig.update_traces(marker_line_width=0)
                fig = _chart_layout(fig, 430)
                fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Mentions")
                with chart_card():
                    st.plotly_chart(fig, use_container_width=True)
        else:
            _no_ai_data("pain points")

        _section("Top Feature Requests")
        if not ai_f.empty:
            feat_counter = Counter(_flat(ai_f["feature_requests"]))
            if feat_counter:
                df_fr = pd.DataFrame(feat_counter.most_common(12), columns=["Feature Request", "Count"])
                df_fr = df_fr.sort_values("Count")
                fig = px.bar(
                    df_fr, x="Count", y="Feature Request", orientation="h",
                    title=f"Top Feature Requests — {active_ds['name']}",
                    color="Count",
                    color_continuous_scale=[[0, GREEN_LIGHT], [1, GREEN_DARK]],
                )
                fig.update_traces(marker_line_width=0)
                fig = _chart_layout(fig, 430)
                fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Mentions")
                with chart_card():
                    st.plotly_chart(fig, use_container_width=True)
        else:
            _no_ai_data("feature requests")

        _section("Music Discovery Insights")

        DISCOVERY_FEATURES = [
            "Discover Weekly", "Smart Shuffle", "AI DJ",
            "Daily Mix", "Radio", "Release Radar", "Autoplay", "Blend",
        ]

        col_d1, col_d2, col_d3 = st.columns([1.2, 1, 1])

        with col_d1:
            if not ai_f.empty and "mentioned_features" in ai_f.columns:
                feat_counter = Counter(_flat(ai_f["mentioned_features"]))
                disc_data = [(f, feat_counter.get(f, 0)) for f in DISCOVERY_FEATURES]
                disc_data.sort(key=lambda x: x[1])
                df_disc = pd.DataFrame(disc_data, columns=["Feature", "Mentions"])
                fig = px.bar(
                    df_disc, x="Mentions", y="Feature", orientation="h",
                    title=f"Feature Mentions — {active_ds['name']}",
                    color="Mentions",
                    color_continuous_scale=[[0, GREEN_LIGHT], [1, GREEN]],
                )
                fig.update_traces(marker_line_width=0)
                fig = _chart_layout(fig, 340)
                fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Reviews Mentioning")
                with chart_card():
                    st.plotly_chart(fig, use_container_width=True)
            else:
                rec_kw = summary.get("recommendation_keywords", [])[:10]
                if rec_kw:
                    df_kw = pd.DataFrame(rec_kw).rename(columns={"keyword": "Keyword", "reviews_containing": "Reviews"})
                    df_kw = df_kw.sort_values("Reviews")
                    fig = px.bar(df_kw, x="Reviews", y="Keyword", orientation="h",
                                 title="Recommendation Keyword Mentions (combined corpus)",
                                 color_discrete_sequence=[GREEN])
                    fig = _chart_layout(fig, 340)
                    with chart_card():
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    _no_ai_data("feature mentions")

        with col_d2:
            if not ai_f.empty and "recommendation_component" in ai_f.columns:
                disc_reviews = ai_f[ai_f.get("music_discovery_related", pd.Series(dtype=bool)) == True] \
                    if "music_discovery_related" in ai_f.columns else ai_f
                comp = (
                    disc_reviews["recommendation_component"]
                    .replace("None", pd.NA).dropna()
                    .value_counts().head(7).reset_index()
                )
                comp.columns = ["Component", "Count"]
                fig = px.pie(
                    comp, names="Component", values="Count",
                    title=f"Recommendation Components — {active_ds['name']}",
                    color_discrete_sequence=CHART_COLORS,
                    hole=0.45,
                )
                fig.update_traces(textposition="outside", textinfo="percent+label",
                                  marker=dict(line=dict(color="#181818", width=2)))
                fig = _chart_layout(fig, 340)
                fig.update_layout(showlegend=False)
                with chart_card():
                    st.plotly_chart(fig, use_container_width=True)
            else:
                _no_ai_data("recommendation components")

        with col_d3:
            if not ai_f.empty and "sentiment" in ai_f.columns:
                sent_counts = ai_f["sentiment"].value_counts().reset_index()
                sent_counts.columns = ["Sentiment", "Count"]
                fig = px.pie(
                    sent_counts, names="Sentiment", values="Count",
                    title=f"Overall Sentiment — {active_ds['name']}",
                    color="Sentiment",
                    color_discrete_map=SENTIMENT_COLOR,
                    hole=0.45,
                )
                fig.update_traces(textposition="outside", textinfo="percent+label",
                                  marker=dict(line=dict(color="#181818", width=2)))
                fig = _chart_layout(fig, 340)
                fig.update_layout(showlegend=False)
                with chart_card():
                    st.plotly_chart(fig, use_container_width=True)
            else:
                _no_ai_data("sentiment")


# ═══════════════════════════════════════════════════════════════════════════════
# 🧩 CLUSTER EXPLORER — interactive drill-down into individual clusters.
# Scoped to discovery-related clusters; app-stability/ads/widget clusters live
# in "Other Signals" below instead.
# ═══════════════════════════════════════════════════════════════════════════════
if render_report_section(
    "cluster_explorer", "🧩", "Cluster Explorer",
    subtitle="Explore individual review clusters, their themes, priority, and supporting evidence.",
    default_expanded=False,
):
    with st.container(key="rs_body_cluster_explorer"):
        if clusters_f_discovery:
            # Scatter: avg review length vs cluster size, coloured by priority
            scatter_rows = []
            for c in clusters_f_discovery:
                ins = insight_map.get(c["cluster_id"], {})
                scatter_rows.append({
                    "Theme":       ins.get("theme", f"Cluster {c['cluster_id']}"),
                    "Size":        c["size"],
                    "Avg Length":  c["avg_review_length_words"],
                    "Priority":    ins.get("priority", "medium").capitalize(),
                    "Source":      c["dominant_source"],
                    "Keywords":    ", ".join(c["top_keywords"][:4]),
                })
            df_sc = pd.DataFrame(scatter_rows)

            fig_sc = px.scatter(
                df_sc, x="Avg Length", y="Size",
                size="Size", color="Priority",
                hover_name="Theme",
                hover_data={"Keywords": True, "Source": True, "Size": True, "Avg Length": True},
                title="Cluster Map — Volume vs. Review Depth",
                size_max=55,
                color_discrete_map={
                    "Critical": RED, "High": ORANGE, "Medium": AMBER, "Low": GREEN,
                },
            )
            fig_sc = _chart_layout(fig_sc, 380)
            fig_sc.update_layout(
                xaxis_title="Average Review Length (words)",
                yaxis_title="Number of Reviews",
            )
            with chart_card():
                st.plotly_chart(fig_sc, use_container_width=True)

            # Cluster selector
            st.markdown("<br>", unsafe_allow_html=True)
            selected_cid = st.selectbox(
                "Select a cluster to explore",
                options=[c["cluster_id"] for c in sorted(clusters_f_discovery, key=lambda x: -x["size"])],
                format_func=lambda cid: f"Cluster {cid} — {insight_map.get(cid, {}).get('theme', '?')} ({cluster_map.get(cid, {}).get('size', 0)} reviews)",
            )

            if selected_cid is not None:
                c   = cluster_map[selected_cid]
                ins = insight_map.get(selected_cid, {})
                priority = ins.get("priority", "low")
                p_color  = PRIORITY_COLOR.get(priority, GREEN)

                # Dynamic border-left color per priority — scoped to this keyed container
                # since it changes with the selected cluster on every rerun.
                st.markdown(
                    f"<style>.st-key-cluster_detail {{ border-left-color: {p_color} !important; }}</style>",
                    unsafe_allow_html=True,
                )

                with st.container(key="cluster_detail"):
                    # Header row
                    hc1, hc2 = st.columns([3, 1])
                    with hc1:
                        st.markdown(
                            f"<div class='cluster-theme'>{ins.get('theme', f'Cluster {selected_cid}')}</div>"
                            f"<div class='cluster-meta'>"
                            f"{c['size']} reviews &nbsp;·&nbsp; Dominant source: {c['dominant_source']} &nbsp;·&nbsp; "
                            f"Avg length: {c['avg_review_length_words']:.0f} words"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with hc2:
                        st.markdown(
                            f"<div style='text-align:right;padding-top:0.25rem'>"
                            f"{_badge(priority)}"
                            f"<br><span style='font-size:0.72rem;color:#9CA3AF;margin-top:4px;display:block'>"
                            f"Impact: {ins.get('impact','—')} &nbsp; Effort: {ins.get('effort','—')}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown("<hr style='border:none;border-top:1px solid #2A2A2E;margin:0.75rem 0'>",
                                unsafe_allow_html=True)

                    # Details
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        st.markdown("**Problem Statement**")
                        st.markdown(
                            f"<p style='font-size:0.88rem;color:#D8D8DC;line-height:1.55'>"
                            f"{ins.get('problem_statement','—')}</p>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("**Root Cause Hypothesis**")
                        st.markdown(
                            f"<p style='font-size:0.85rem;color:#B3B3B3;line-height:1.5'>"
                            f"{ins.get('root_cause_hypothesis','—')}</p>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("**Affected Users**")
                        st.markdown(
                            f"<p style='font-size:0.85rem;color:#B3B3B3'>{ins.get('affected_users','—')}</p>",
                            unsafe_allow_html=True,
                        )

                    with dc2:
                        opps = ins.get("product_opportunities", [])
                        if opps:
                            st.markdown("**Product Opportunities**")
                            for o in opps:
                                st.markdown(
                                    f"<div style='background:rgba(29,185,84,0.14);border-left:3px solid {GREEN};"
                                    f"border-radius:0 6px 6px 0;padding:6px 10px;margin:4px 0;"
                                    f"font-size:0.83rem;color:#4ADE80'>{o}</div>",
                                    unsafe_allow_html=True,
                                )
                        feats = ins.get("recommended_features", [])
                        if feats:
                            st.markdown("**Recommended Features**")
                            for f_ in feats:
                                st.markdown(
                                    f"<div style='background:rgba(66,165,245,0.14);border-left:3px solid {BLUE};"
                                    f"border-radius:0 6px 6px 0;padding:6px 10px;margin:4px 0;"
                                    f"font-size:0.83rem;color:#64B5F6'>{f_}</div>",
                                    unsafe_allow_html=True,
                                )

                    # Success metrics
                    metrics = ins.get("success_metrics", [])
                    if metrics:
                        st.markdown("**Success Metrics**")
                        m_cols = st.columns(min(len(metrics), 3))
                        for i, m in enumerate(metrics[:3]):
                            m_cols[i].markdown(
                                f"<div style='background:#212121;border:1px solid #2A2A2E;border-radius:8px;"
                                f"padding:10px 12px;font-size:0.8rem;color:#D8D8DC;line-height:1.4'>{m}</div>",
                                unsafe_allow_html=True,
                            )

                    # Representative reviews
                    st.markdown(
                        "<p style='font-size:0.85rem;font-weight:700;color:#FFFFFF;margin:1rem 0 0.4rem'>"
                        "Representative Reviews</p>",
                        unsafe_allow_html=True,
                    )
                    for rev in c["representative_reviews"][:3]:
                        st.markdown(
                            f"<div class='review-quote'>{rev[:450]}{'…' if len(rev)>450 else ''}</div>",
                            unsafe_allow_html=True,
                        )


# ═══════════════════════════════════════════════════════════════════════════════
# 💡 RECOMMENDED PRODUCT OPPORTUNITIES — the executive action section. Always
# expanded. Scoped to discovery-related clusters; non-discovery findings are
# broken out in "Other Signals" below.
# ═══════════════════════════════════════════════════════════════════════════════
render_report_section(
    "product_opportunities", "💡", "Recommended Product Opportunities",
    subtitle="Prioritized, discovery-focused product opportunities and roadmap.",
    default_expanded=True, collapsible=False,
)

if insights_f_discovery:
    opp_rows = []
    for ins in insights_f_discovery:
        for opp in ins.get("product_opportunities", []):
            opp_rows.append({
                "Cluster":    ins["cluster_id"],
                "Theme":      ins.get("theme", "—"),
                "Opportunity": opp,
                "Priority":   ins.get("priority", "low"),
                "Impact":     ins.get("impact", "—"),
                "Effort":     ins.get("effort", "—"),
            })

    if opp_rows:
        df_opps = pd.DataFrame(opp_rows)

        # Inline sub-filters — wrapped in a form so changes only take effect on
        # "Apply Filters", same reasoning as the Cluster/Priority filter bar above.
        with st.form(key="opp_filter_form", border=False):
            of1, of2, of3, of_btn = st.columns([1, 1, 1, 0.7])
            with of1:
                pf = st.multiselect("Filter by Priority", PRIORITY_ORDER, default=PRIORITY_ORDER, key="opp_p")
            with of2:
                if_ = st.multiselect("Filter by Impact", ["high", "medium", "low"],
                                      default=["high", "medium", "low"], key="opp_i")
            with of3:
                ef = st.multiselect("Filter by Effort", ["low", "medium", "high"],
                                     default=["low", "medium", "high"], key="opp_e")
            with of_btn:
                st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
                st.form_submit_button("Apply Filters", use_container_width=True)

        df_f = df_opps[
            df_opps["Priority"].isin(pf) &
            df_opps["Impact"].isin(if_) &
            df_opps["Effort"].isin(ef)
        ]

        st.markdown(f"<p style='font-size:0.8rem;color:#9CA3AF'>{len(df_f)} opportunities</p>",
                    unsafe_allow_html=True)

        for _, row in df_f.iterrows():
            p_color = PRIORITY_COLOR.get(row["Priority"], GREEN)
            st.markdown(
                f"<div class='opp-item' style='border-left-color:{p_color}'>"
                f"<div class='opp-meta'>"
                f"Cluster {row['Cluster']} · {row['Theme']} &nbsp;·&nbsp; "
                f"<span style='color:{p_color};font-weight:700'>{row['Priority'].upper()}</span>"
                f" &nbsp;·&nbsp; Impact: {row['Impact']} &nbsp;·&nbsp; Effort: {row['Effort']}"
                f"</div>"
                f"<div class='opp-text'>{row['Opportunity']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
else:
    st.info("No insights match the current filters.")


_section("Product Roadmap — Impact vs. Effort")

if insights_f_discovery:
    LEVEL = {"high": 3, "medium": 2, "low": 1, "critical": 4}
    rm_rows = [
        {
            "Theme":     ins.get("theme", f"C{ins['cluster_id']}"),
            "Cluster":   ins["cluster_id"],
            "Priority":  ins.get("priority", "medium").capitalize(),
            "Impact_n":  LEVEL.get(ins.get("impact",  "medium"), 2),
            "Effort_n":  LEVEL.get(ins.get("effort",  "medium"), 2),
            "Size":      cluster_map.get(ins["cluster_id"], {}).get("size", 10),
            "Confidence": ins.get("confidence", 0.5),
        }
        for ins in insights_f_discovery
    ]
    df_rm = pd.DataFrame(rm_rows)

    fig_rm = px.scatter(
        df_rm, x="Effort_n", y="Impact_n",
        size="Size", color="Priority",
        hover_name="Theme",
        hover_data={"Size": True, "Confidence": True, "Effort_n": False, "Impact_n": False},
        title="Impact vs. Effort Matrix  (bubble size = cluster size)",
        size_max=55,
        color_discrete_map={
            "Critical": RED, "High": ORANGE, "Medium": AMBER, "Low": GREEN,
        },
    )
    fig_rm.add_hline(y=2.5, line_dash="dot", line_color="#4B5563", line_width=1.5)
    fig_rm.add_vline(x=2.5, line_dash="dot", line_color="#4B5563", line_width=1.5)

    for x, y, label, clr in [
        (1.2, 3.65, "Quick Wins",      GREEN),
        (3.5, 3.65, "Strategic Bets",  BLUE),
        (1.2, 1.35, "Fill-ins",        "#9CA3AF"),
        (3.5, 1.35, "Thankless Tasks", "#9CA3AF"),
    ]:
        fig_rm.add_annotation(
            x=x, y=y, text=f"<b>{label}</b>", showarrow=False,
            font=dict(size=11, color=clr),
            bgcolor="rgba(24,24,24,0.85)", borderpad=4,
        )

    for _, row in df_rm.iterrows():
        short = (row["Theme"][:24] + "…") if len(row["Theme"]) > 24 else row["Theme"]
        fig_rm.add_annotation(
            x=row["Effort_n"], y=row["Impact_n"],
            text=short, showarrow=False,
            font=dict(size=8.5, color="#D8D8DC"),
            yshift=20, bgcolor="rgba(24,24,24,0.75)", borderpad=2,
        )

    fig_rm = _chart_layout(fig_rm, 520)
    fig_rm.update_layout(
        xaxis=dict(tickvals=[1,2,3], ticktext=["Low","Medium","High"], title="Effort", range=[0.4,3.9]),
        yaxis=dict(tickvals=[1,2,3], ticktext=["Low","Medium","High"], title="Impact", range=[0.4,3.9]),
        legend=dict(title="Priority", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    with chart_card():
        st.plotly_chart(fig_rm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 📝 SUPPORTING EVIDENCE — searchable review explorer (search, filters,
# representative reviews). Unfiltered by discovery/other-signal split — this is
# the raw evidence archive, not narrative.
# ═══════════════════════════════════════════════════════════════════════════════
if render_report_section(
    "supporting_evidence", "📝", "Supporting Evidence",
    subtitle="Search and filter the underlying review corpus.",
    default_expanded=False,
):
    with st.container(key="rs_body_supporting_evidence"):
        st.caption(f"Searching within **{active_ds['name']}** · {len(master_f):,} reviews in the current selection.")
        search = st.text_input(
            "Search",
            placeholder="Filter by keyword — e.g. shuffle, AI DJ, recommendations …",
            label_visibility="collapsed",
        )

        # Real per-review evidence — driven by the dataset filter (master_f) and
        # the Cluster filter above, rather than a fixed per-cluster excerpt list.
        ev_df = master_f[master_f["cluster_id"].isin(sel_clusters)] if sel_clusters else master_f.iloc[0:0]
        if search:
            ev_df = ev_df[ev_df["review_text"].str.contains(search, case=False, na=False, regex=False)]

        st.markdown(
            f"<p style='font-size:0.8rem;color:#9CA3AF;margin-bottom:0.5rem'>"
            f"Showing {min(len(ev_df), 50)} of {len(ev_df)} reviews</p>",
            unsafe_allow_html=True,
        )

        SOURCE_BADGE_COLOR = {
            "reddit":            ("#FF7A50", "rgba(255,69,0,0.16)"),
            "google_play":       ("#4ADE80", "rgba(29,185,84,0.16)"),
            "app_store":         ("#64B5F6", "rgba(66,165,245,0.16)"),
            "spotify_community": ("#C084FC", "rgba(171,71,188,0.18)"),
        }

        for _, rev in ev_df.head(50).iterrows():
            text = str(rev["review_text"])
            theme = insight_map.get(rev["cluster_id"], {}).get("theme", f"Cluster {rev['cluster_id']}")
            if search:
                idx = text.lower().find(search.lower())
                if idx >= 0:
                    pre   = text[max(0, idx-100):idx]
                    match = text[idx:idx+len(search)]
                    post  = text[idx+len(search):idx+len(search)+100]
                    snippet = (
                        f"…{pre}"
                        f"<mark style='background:rgba(29,185,84,0.30);color:#6EE7B7;border-radius:2px;"
                        f"padding:0 2px'>{match}</mark>"
                        f"{post}…"
                    )
                else:
                    snippet = text[:350] + ("…" if len(text) > 350 else "")
            else:
                snippet = text[:350] + ("…" if len(text) > 350 else "")

            src_color, src_bg = SOURCE_BADGE_COLOR.get(rev["source"], ("#B3B3B3", "#212121"))

            st.markdown(
                f"<div style='background:#181818;border-radius:10px;padding:1rem 1.25rem;"
                f"margin:0.5rem 0;box-shadow:0 1px 3px rgba(0,0,0,0.3);border:1px solid #2A2A2E;"
                f"border-left:3px solid {GREEN}'>"
                f"<div class='review-meta'>"
                f"<span style='background:{src_bg};color:{src_color};padding:2px 8px;"
                f"border-radius:99px;font-size:0.7rem;font-weight:600;margin-right:6px'>{rev['source']}</span>"
                f"<span style='color:#9CA3AF'>Cluster {rev['cluster_id']} · {theme}</span>"
                f"</div>"
                f"<div style='font-size:0.87rem;color:#FFFFFF;line-height:1.6'>{snippet}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ⚠️ OTHER SIGNALS — clusters/findings not directly about music discovery
# (app stability, free-tier/ads, widget) kept separate from the discovery
# narrative above. Reuses the same insight-card and opportunity-item styling.
# ═══════════════════════════════════════════════════════════════════════════════
if render_report_section(
    "other_signals", "⚠️", "Other Signals",
    subtitle="Findings outside the core music-discovery narrative — app stability, monetization, and widget feedback.",
    default_expanded=False, accent=AMBER,
):
    with st.container(key="rs_body_other_signals"):
        if insights_f_other:
            for ins in insights_f_other:
                p_color = PRIORITY_COLOR.get(ins.get("priority", "low"), GREEN)
                c = cluster_map.get(ins["cluster_id"], {})
                st.markdown(
                    f"<div class='aiq-insight-card' style='border-left:4px solid {p_color}'>"
                    f"<div class='aiq-insight-theme'>{ins.get('theme', '—')} "
                    f"{_badge(ins.get('priority', 'low'))}"
                    f"<span style='font-size:0.7rem;color:#6B7280;font-weight:400;margin-left:6px'>"
                    f"{c.get('size', 0)} reviews &middot; {ins.get('confidence', 0):.0%} confidence</span></div>"
                    f"<p class='aiq-insight-problem'>{ins.get('problem_statement', '—')}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                for o in ins.get("product_opportunities", []):
                    st.markdown(
                        f"<div class='opp-item'><div class='opp-text'>{o}</div></div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No out-of-scope findings match the current filters.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='text-align:center;margin-top:3rem;padding:1.5rem;border-top:1px solid #2A2A2E;"
    f"color:#7A7A80;font-size:0.78rem'>"
    f"Spotify AI Review Discovery Engine &nbsp;·&nbsp; "
    f"Data from <code>output/</code> &nbsp;·&nbsp; No live API calls"
    f"</div>",
    unsafe_allow_html=True,
)
