"""Reusable Streamlit UI primitives and theme."""

from __future__ import annotations

from typing import Any, Optional, Sequence

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore


def _html_esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def inject_custom_css() -> None:
    if st is None:
        return
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'DM Sans', sans-serif;
    }
    .block-container { padding-top: 1.2rem; max-width: 1400px; }
    .dss-header {
        box-sizing: border-box;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 55%, #f1f5f9 100%);
        color: #0f172a;
        padding: 1.35rem 1.5rem 1.35rem 1.35rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(15, 23, 42, 0.1);
        border-left: 6px solid #2563eb;
        box-shadow:
            0 1px 3px rgba(15, 23, 42, 0.06),
            0 8px 24px rgba(15, 23, 42, 0.06);
    }
    .dss-header h1 {
        margin: 0;
        font-size: clamp(1.45rem, 2.5vw, 1.9rem);
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #0f172a;
        line-height: 1.25;
    }

    /* Executive Overview — top KPI strip (four cards) */
    .exec-kpi-card {
      box-sizing: border-box;
      min-height: 7.5rem;
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
      background: linear-gradient(158deg, #ffffff 0%, #f9fafb 40%, #eef2f6 100%);
      border: 1px solid rgba(15, 23, 42, 0.09);
      border-radius: 14px;
      padding: 1.05rem 1.3rem 1.2rem;
      margin: 0 0 0.5rem;
      border-left: 4px solid #1d4ed8;
      box-shadow:
        0 1px 2px rgba(15, 23, 42, 0.05),
        0 6px 18px rgba(15, 23, 42, 0.07);
      transition: transform 0.18s ease, box-shadow 0.18s ease, border-left-color 0.18s ease;
    }
    .exec-kpi-card:hover {
      transform: translateY(-2px);
      box-shadow:
        0 4px 12px rgba(15, 23, 42, 0.07),
        0 14px 32px rgba(29, 78, 216, 0.12);
      border-left-color: #3b82f6;
    }
    .exec-kpi-label {
      display: block;
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #64748b;
      margin-bottom: 0.52rem;
      line-height: 1.35;
    }
    .exec-kpi-value {
      display: block;
      font-size: clamp(1.42rem, 2.8vw, 1.78rem);
      font-weight: 700;
      color: #0f172a;
      font-variant-numeric: tabular-nums;
      letter-spacing: -0.038em;
      line-height: 1.18;
      margin-bottom: auto;
    }
    .exec-kpi-hint {
      display: block;
      font-size: 0.74rem;
      font-weight: 500;
      color: #64748b;
      line-height: 1.4;
      margin-top: 0.55rem;
      max-width: 100%;
    }

    /* Executive — full-width risk band bar */
    .risk-band-bar {
      box-sizing: border-box;
      width: 100%;
      margin: 0.4rem 0 1rem;
      padding: 1rem 1.3rem;
      border-radius: 14px;
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-left-width: 5px;
      box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.85),
        0 2px 10px rgba(15, 23, 42, 0.05);
    }
    .risk-band-bar-text {
      display: inline-block;
      font-size: 1.06rem;
      letter-spacing: -0.022em;
    }
    .risk-band-prefix {
      font-weight: 600;
      opacity: 0.88;
      margin-right: 0.4rem;
    }
    .risk-band-bar--low {
      border-left-color: #22c55e;
      background: linear-gradient(95deg, rgba(34, 197, 94, 0.11) 0%, #ffffff 52%, #f9fafb 100%);
    }
    .risk-band-bar--moderate {
      border-left-color: #ca8a04;
      background: linear-gradient(95deg, rgba(202, 138, 4, 0.12) 0%, #ffffff 52%, #f9fafb 100%);
    }
    .risk-band-bar--high {
      border-left-color: #ea580c;
      background: linear-gradient(95deg, rgba(234, 88, 12, 0.12) 0%, #fffbeb 52%, #ffffff 100%);
    }
    .risk-band-bar--critical {
      border-left-color: #dc2626;
      background: linear-gradient(95deg, rgba(220, 38, 38, 0.12) 0%, #fef2f2 50%, #ffffff 100%);
    }
    .exec-maritime-title,
    .dss-section-title {
      margin: 0 0 1.05rem;
    }
    .exec-maritime-title-text,
    .dss-section-title-text {
      display: inline-block;
      font-size: 1.3rem;
      font-weight: 700;
      color: #0f172a;
      letter-spacing: -0.025em;
      padding-bottom: 0.42rem;
      border-bottom: 3px solid #2563eb;
      box-decoration-break: clone;
    }
    .dss-section-title {
      margin-top: 1.35rem;
      margin-bottom: 0.95rem;
    }
    .dss-section-title--compact {
      margin-top: 0.65rem;
      margin-bottom: 0.65rem;
    }

    .maritime-kpi-card {
      box-sizing: border-box;
      min-height: 5.75rem;
      background: linear-gradient(145deg, #ffffff 0%, #f8fafc 55%, #f1f5f9 100%);
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 12px;
      padding: 1.05rem 1.25rem;
      margin: 0 0 0.5rem;
      border-left: 4px solid #1e40af;
      box-shadow:
        0 1px 2px rgba(15, 23, 42, 0.04),
        0 6px 16px rgba(15, 23, 42, 0.07);
      transition: transform 0.18s ease, box-shadow 0.18s ease, border-left-color 0.18s ease;
    }
    .maritime-kpi-card:hover {
      transform: translateY(-2px);
      box-shadow:
        0 4px 8px rgba(15, 23, 42, 0.06),
        0 14px 28px rgba(30, 64, 175, 0.13);
      border-left-color: #3b82f6;
    }
    .maritime-kpi-card--alert {
      border-left-color: #b45309;
      background: linear-gradient(145deg, #fffbeb 0%, #ffffff 65%, #f8fafc 100%);
    }
    .maritime-kpi-card--alert:hover {
      border-left-color: #f59e0b;
      box-shadow:
        0 4px 8px rgba(15, 23, 42, 0.06),
        0 14px 28px rgba(180, 83, 9, 0.14);
    }
    .maritime-kpi-label {
      display: block;
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: #64748b;
      margin-bottom: 0.42rem;
      line-height: 1.35;
      max-width: 100%;
    }
    .maritime-kpi-value {
      display: block;
      font-size: clamp(1.35rem, 2.8vw, 1.72rem);
      font-weight: 700;
      color: #0f172a;
      font-variant-numeric: tabular-nums;
      letter-spacing: -0.035em;
      line-height: 1.1;
    }

    /* Alert-board row severity tiles (Executive maritime) */
    .maritime-severity-chip {
      box-sizing: border-box;
      text-align: center;
      padding: 0.85rem 0.65rem;
      border-radius: 10px;
      border: 1px solid rgba(15, 23, 42, 0.08);
      background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
      box-shadow: 0 1px 4px rgba(15, 23, 42, 0.05);
    }
    .maritime-severity-chip .msc-label {
      display: block;
      font-size: 0.72rem;
      font-weight: 600;
      color: #64748b;
      margin-bottom: 0.38rem;
    }
    .maritime-severity-chip .msc-value {
      font-size: 1.45rem;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }
    .maritime-severity-chip--critical { border-top: 3px solid #dc2626; }
    .maritime-severity-chip--high { border-top: 3px solid #ea580c; }
    .maritime-severity-chip--moderate { border-top: 3px solid #ca8a04; }
    .maritime-severity-chip--resolved { border-top: 3px solid #22c55e; }

    /* AI Decision Recommendations (expanders whose body includes .ai-reco-card-body) */
    .ai-recos-heading-block {
      margin-top: 1.75rem;
      margin-bottom: 0.85rem;
    }

    .stApp div[data-testid="stExpander"]:has(div.ai-reco-card-body),
    .stApp details[data-testid="stExpander"]:has(div.ai-reco-card-body) {
      border: 1px solid rgba(15, 23, 42, 0.1);
      border-radius: 14px;
      overflow: hidden;
      margin-bottom: 0.7rem;
      box-shadow:
        0 1px 3px rgba(15, 23, 42, 0.06),
        0 6px 18px rgba(15, 23, 42, 0.055);
      background: linear-gradient(178deg, #fdfefe 0%, #ffffff 48%, #f8fafc 100%);
      transition: box-shadow 0.18s ease, transform 0.18s ease;
    }
    .stApp div[data-testid="stExpander"]:has(div.ai-reco-card-body):hover,
    .stApp details[data-testid="stExpander"]:has(div.ai-reco-card-body):hover {
      box-shadow:
        0 3px 10px rgba(15, 23, 42, 0.07),
        0 12px 28px rgba(30, 58, 138, 0.09);
    }

    .stApp div[data-testid="stExpander"]:has(div.ai-reco-card-body) summary,
    .stApp details[data-testid="stExpander"]:has(div.ai-reco-card-body) summary {
      padding: 0.95rem 1rem 0.95rem 1.15rem;
      font-weight: 600;
      font-size: 0.93rem;
      letter-spacing: -0.015em;
      color: #334155;
      background: linear-gradient(90deg, rgba(37, 99, 235, 0.06) 0%, rgba(248, 250, 252, 0.4) 58%, transparent 100%);
      border-bottom: 1px solid rgba(15, 23, 42, 0.06);
    }

    .ai-reco-card-body {
      padding: 0.85rem 1.15rem 1.05rem;
      margin: 0;
    }
    .ai-reco-row {
      display: grid;
      grid-template-columns: minmax(5.75rem, 7.75rem) 1fr;
      gap: 0.65rem 1.1rem;
      align-items: start;
      padding: 0.62rem 0;
      border-bottom: 1px solid #f1f5f9;
    }
    .ai-reco-row:first-child { padding-top: 0.15rem; }
    .ai-reco-row:last-child { border-bottom: none; padding-bottom: 0.15rem; }
    .ai-reco-k {
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: #64748b;
      line-height: 1.45;
      padding-top: 0.12rem;
    }
    .ai-reco-v {
      font-size: 0.92rem;
      color: #0f172a;
      line-height: 1.5;
    }
    .ai-reco-pill {
      display: inline-block;
      padding: 0.2rem 0.55rem;
      border-radius: 7px;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.04em;
    }
    .ai-reco-pill--p0 {
      background: linear-gradient(180deg, #fef2f2 0%, #fee2e2 100%);
      color: #b91c1c;
      border: 1px solid rgba(220, 38, 38, 0.2);
    }
    .ai-reco-pill--p1 {
      background: linear-gradient(180deg, #fffbeb 0%, #fef3c7 100%);
      color: #b45309;
      border: 1px solid rgba(245, 158, 11, 0.28);
    }
    .ai-reco-pill--p2 {
      background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
      color: #475569;
      border: 1px solid rgba(100, 116, 139, 0.25);
    }

    @media (max-width: 520px) {
      .ai-reco-row {
        grid-template-columns: 1fr;
        gap: 0.2rem 0;
        padding: 0.75rem 0;
      }
      .ai-reco-k { padding-top: 0; }
    }

    @media (prefers-reduced-motion: reduce) {
      .exec-kpi-card { transition: none; }
      .exec-kpi-card:hover { transform: none; }
      .maritime-kpi-card { transition: none; }
      .maritime-kpi-card:hover { transform: none; }
      .stApp div[data-testid="stExpander"]:has(div.ai-reco-card-body),
      .stApp details[data-testid="stExpander"]:has(div.ai-reco-card-body) { transition: none; }
      .stApp div[data-testid="stExpander"]:has(div.ai-reco-card-body):hover,
      .stApp details[data-testid="stExpander"]:has(div.ai-reco-card-body):hover { transform: none; }
    }
    .risk-low { color: #16a34a; font-weight: 700; }
    .risk-mod { color: #ca8a04; font-weight: 700; }
    .risk-high { color: #ea580c; font-weight: 700; }
    .risk-crit { color: #dc2626; font-weight: 700; }

    /* Global polish — tabs, sidebar, expanders (non–AI-reco), alerts, data tables */
    .stTabs [data-baseweb="tab-list"] {
      gap: 0.35rem;
      background-color: #f1f5f9;
      padding: 0.42rem;
      border-radius: 14px;
      border: 1px solid #e2e8f0;
    }
    div[data-testid="stSidebar"] h5 {
      font-size: 0.74rem !important;
      letter-spacing: 0.07em !important;
      text-transform: uppercase !important;
      color: #94a3b8 !important;
      font-weight: 700 !important;
      margin: 1rem 0 0.45rem !important;
    }
    .stApp div[data-testid="stExpander"]:not(:has(div.ai-reco-card-body)),
    .stApp details[data-testid="stExpander"]:not(:has(div.ai-reco-card-body)) {
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      overflow: hidden;
      margin-bottom: 0.5rem;
      background: linear-gradient(178deg, #ffffff 0%, #fafafa 55%, #f8fafc 100%);
    }
    [data-testid="stDataFrame"],
    [data-testid="stStyledDataFrame"] {
      border: 1px solid #e2e8f0;
      border-radius: 11px;
      overflow: hidden;
      box-shadow: 0 1px 5px rgba(15,23,42,0.04);
    }
    div[data-testid="stAlert"] > div[data-baseweb="notification"],
    div[data-testid="stAlert"] article {
      border-radius: 14px !important;
      border: 1px solid rgba(15,23,42,0.07) !important;
    }
    [data-testid="stSidebar"] hr {
      border-color: rgba(148,163,184,0.35);
      margin: 1rem 0;
    }
    div[data-testid="stMarkdownContainer"] p {
      color: #334155;
      line-height: 1.55;
    }
    div[data-testid="stMarkdownContainer"] h3 {
      color: #0f172a;
      font-weight: 700;
      letter-spacing: -0.03em;
    }

    .alert-banner {
      box-sizing: border-box;
      padding: 1.05rem 1.25rem 1.15rem;
      border-radius: 14px;
      margin: 0.85rem 0 1rem;
      border: 1px solid rgba(14, 165, 233, 0.2);
      border-left: 5px solid #0284c7;
      background: linear-gradient(
        92deg,
        rgba(224, 242, 254, 0.97) 0%,
        rgba(255, 255, 255, 0.99) 50%,
        #f8fafc 100%
      );
      color: #334155;
      font-size: 0.93rem;
      line-height: 1.56;
      box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.95),
        0 2px 14px rgba(14, 116, 179, 0.09),
        0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .alert-banner p {
      margin: 0;
    }
    .alert-banner-plain {
      color: inherit;
      font-weight: 400;
    }
    .alert-banner--driver .alert-banner-lead {
      display: inline;
      font-weight: 700;
      text-transform: uppercase;
      font-size: 0.66rem;
      letter-spacing: 0.08em;
      color: #0369a1;
    }
    .alert-banner--driver .alert-banner-em {
      color: #0f172a;
      font-weight: 700;
    }
    .alert-banner--driver .alert-banner-tail {
      color: #475569;
    }

    /* Multiselect (Base Web tags) — align with DSS palette */
    .stApp [data-testid="stMultiSelect"] [data-baseweb="select"] > div:first-child {
      border-radius: 12px !important;
      border-color: rgba(15, 23, 42, 0.1) !important;
      background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
      box-shadow: 0 1px 4px rgba(15, 23, 42, 0.05) !important;
      min-height: 2.75rem;
    }
    .stApp [data-baseweb="tag"] {
      border-radius: 8px !important;
      border: none !important;
      background: linear-gradient(180deg, #1d4ed8 0%, #1e3a8a 100%) !important;
      color: #f8fafc !important;
      font-weight: 600 !important;
      font-size: 0.8rem !important;
    }
    .stApp [data-baseweb="tag"] [aria-label="Delete"] {
      color: #e0e7ff !important;
    }

    .method-card {
        background: linear-gradient(178deg, #ffffff 0%, #fafbfc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.05rem 1.2rem;
        margin: 0.55rem 0;
        box-shadow: 0 1px 6px rgba(15, 23, 42, 0.045);
    }
    div[data-testid="stSidebar"] { background: #0f172a; color: #e2e8f0; }
    div[data-testid="stSidebar"] * { color: #e2e8f0; }

    /* Sidebar — stacked page nav & action buttons */
    div[data-testid="stSidebar"] div[data-testid="element-container"] button {
      width: 100% !important;
      border-radius: 11px !important;
      min-height: 2.75rem !important;
      padding: 0.5rem 0.82rem !important;
      justify-content: flex-start !important;
      text-align: left !important;
      font-weight: 600 !important;
      font-size: 0.84rem !important;
      line-height: 1.38 !important;
      white-space: normal !important;
      height: auto !important;
      margin-bottom: 0.4rem !important;
    }
    div[data-testid="stSidebar"] div[data-testid="element-container"] button[kind="secondary"] {
      background-color: rgba(255, 255, 255, 0.07) !important;
      border: 1px solid rgba(255, 255, 255, 0.22) !important;
      color: #e8edf5 !important;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stSidebar"] div[data-testid="element-container"] button[kind="secondary"]:hover {
      border-color: rgba(56, 189, 248, 0.55) !important;
      background-color: rgba(255, 255, 255, 0.11) !important;
      color: #f8fafc !important;
    }
    div[data-testid="stSidebar"] div[data-testid="element-container"] button[kind="primary"] {
      background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%) !important;
      border: 1px solid rgba(191, 219, 254, 0.45) !important;
      box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.12),
        0 4px 14px rgba(37, 99, 235, 0.35);
    }

    .risk-legend { font-size: 0.85rem; color: #334155; margin: 0.5rem 0; }
</style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str = "") -> None:
    if st is None:
        return
    _ = subtitle
    st.markdown(
        f'<div class="dss-header"><h1>{_html_esc(title)}</h1></div>',
        unsafe_allow_html=True,
    )


def render_section_title(text: str) -> None:
    if st is None:
        return
    st.markdown(f"### {text}")


def render_page_section_title(text: str, *, compact: bool = False) -> None:
    """Accent underline subsection — matches Executive maritime block titles."""
    if st is None:
        return
    mod = " dss-section-title--compact" if compact else ""
    st.markdown(
        f'<div class="dss-section-title{mod}"><span class="dss-section-title-text">{_html_esc(text)}</span></div>',
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, hint: str = "") -> None:
    if st is None:
        return
    hint_blk = (
        f'<span class="exec-kpi-hint">{_html_esc(hint)}</span>' if (hint or "").strip() else ""
    )
    st.markdown(
        f'<div class="exec-kpi-card"><span class="exec-kpi-label">{_html_esc(label)}</span>'
        f'<span class="exec-kpi-value">{_html_esc(value)}</span>{hint_blk}</div>',
        unsafe_allow_html=True,
    )


def _risk_keyword(level: str) -> str:
    first = (level or "").strip().split()[0] if level else ""
    return first


def risk_class_style(level: str) -> str:
    raw = _risk_keyword(level)
    if not raw:
        return "risk-mod"
    key = raw[0].upper() + raw[1:].lower() if len(raw) > 1 else raw.upper()
    m = {
        "Low": "risk-low",
        "Moderate": "risk-mod",
        "High": "risk-high",
        "Critical": "risk-crit",
    }
    return m.get(key, "risk-mod")


def _risk_band_tier_css(label: str) -> str:
    w = _risk_keyword(label).lower()
    return {"low": "low", "moderate": "moderate", "high": "high", "critical": "critical"}.get(w, "moderate")


def render_risk_badge(level: str) -> None:
    if st is None:
        return
    tier = _risk_band_tier_css(level)
    css_txt = risk_class_style(level)
    esc = _html_esc(level)
    st.markdown(
        f'<div class="risk-band-bar risk-band-bar--{tier}">'
        f'<span class="risk-band-bar-text {css_txt}">'
        f'<span class="risk-band-prefix">Risk band:</span> {esc}'
        "</span></div>",
        unsafe_allow_html=True,
    )


def render_alert_banner(text: str) -> None:
    if st is None:
        return
    st.markdown(
        f'<div class="alert-banner"><p class="alert-banner-plain">{_html_esc(text)}</p></div>',
        unsafe_allow_html=True,
    )


def render_exec_driver_banner(driver_name: str, detail_excerpt: str) -> None:
    """Executive insight strip: top risk driver + forecast excerpt (HTML escaped)."""
    if st is None:
        return
    st.markdown(
        '<div class="alert-banner alert-banner--driver">'
        '<span class="alert-banner-lead">Top risk driver</span>'
        '<span class="alert-banner-colon">: </span>'
        f'<strong class="alert-banner-em">{_html_esc(driver_name)}</strong>'
        f'<span class="alert-banner-tail">. {_html_esc(detail_excerpt)}</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_recommendation_card(title: str, body: str, source: str = "") -> None:
    if st is None:
        return
    src = f'<div style="font-size:0.75rem;color:#64748b">Source: {source}</div>' if source else ""
    st.markdown(
        f'<div class="method-card"><strong>{title}</strong><p style="margin:0.5rem 0">{body}</p>{src}</div>',
        unsafe_allow_html=True,
    )


def render_methodology_card(title: str, body: str) -> None:
    if st is None:
        return
    st.markdown(
        f'<div class="method-card"><strong>{title}</strong><p style="margin:0.45rem 0">{body}</p></div>',
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    if st is None:
        return


def _select_sidebar_page(session_key_: str, label: str) -> None:
    st.session_state[session_key_] = label


def render_sidebar_page_nav(pages: Sequence[tuple[str, str]], session_key: str = "dss_nav_page") -> str:
    """
    Stacked sidebar navigation as full-width buttons (replaces radio list).
    """
    if st is None:
        return ""
    lst = list(pages)
    if not lst:
        return ""
    labels = [p[0] for p in lst]
    st.session_state.setdefault(session_key, labels[0])
    if st.session_state[session_key] not in labels:
        st.session_state[session_key] = labels[0]
    st.markdown("##### Navigation")
    for label, slug in lst:
        st.button(
            label,
            key=f"nav_btn_{slug}",
            use_container_width=True,
            type="primary" if st.session_state[session_key] == label else "secondary",
            on_click=_select_sidebar_page,
            args=(session_key, label),
        )
    return str(st.session_state[session_key])
