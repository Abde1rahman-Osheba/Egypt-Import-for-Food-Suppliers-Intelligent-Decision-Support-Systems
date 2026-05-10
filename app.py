"""
Egypt Food Import Risk DSS — Streamlit entry point.
Run from project root: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dataclasses import replace

from src.ahp import CRITERIA, ahp_explanation, default_comparison_matrix, run_ahp
from src.alert_system import AlertStatus, alerts_to_dataframe, evaluate_maritime_alerts
from src.data_pipeline import (
    PORT_CARGO_SEGMENTS,
    assemble_training_bundle,
    build_blended_wfp_commodity_catalog,
    load_all_wfp_commodities,
    load_ports_monthly_by_cargo,
)
from src.decision_methods import (
    ForecastMethod,
    deterministic_utility_best,
    expected_utility,
    explain_residual_panel,
    forecast_commodity_price,
    forecast_with_method,
    tornado_sensitivity,
    uncertainty_criteria,
)
from src.discriminant_analysis import train_and_evaluate
from src.explanation_engine import (
    explain_alert,
    explain_uncertainty_decision,
)
from src.goal_programming import ALTERNATIVES, run_goal_programming
from src.inference_engine import run_inference
from src.knowledge_base import get_rules, rules_as_dataframe
from src.nlp_conflict_index import compute_novelty_correlation, price_spike_series
from src.ollama_explain import render_ollama_sidebar, section_ollama_explainer
from src.risk_scoring import safe_float
from src.chart_theme import DSS_TITLE_FONT, apply_dss_cartesian_grid, apply_dss_layout, dss_title
from src.ui_components import (
    inject_custom_css,
    render_exec_driver_banner,
    render_footer,
    render_header,
    render_sidebar_page_nav,
    render_kpi_card,
    render_methodology_card,
    render_page_section_title,
    render_recommendation_card,
    render_risk_badge,
    render_section_title,
)
from src.ui_maritime_dashboard import (
    render_executive_maritime_summary,
    render_maritime_full_page,
)
from src.vessel_tracking import build_maritime_snapshot


@st.cache_data(show_spinner=True)
def cached_bundle():
    return assemble_training_bundle()


@st.cache_data(show_spinner=False)
def cached_wfp_commodity_catalog(wfp_path_str: str):
    """Per-distinct commodity monthly series from raw WFP file (National market preferred)."""
    p = Path(wfp_path_str)
    if not p.is_file():
        return {}
    return load_all_wfp_commodities(p)


@st.cache_data(show_spinner=False)
def cached_ports_by_cargo(path_str: str):
    """Wide monthly port+cargo aggregates for Port & Logistics charts (not risk pipeline spine)."""
    p = Path(path_str)
    if not p.is_file():
        return pd.DataFrame()
    return load_ports_monthly_by_cargo(p)


def unified_risk(ahp_weights: dict[str, float], row: pd.Series) -> float:
    key_map = {
        "Geopolitical Risk": safe_float(row.get("geopolitical_risk_score"), 0),
        "Logistics Risk": safe_float(row.get("logistics_risk_score"), 0),
        "Price Stress": safe_float(row.get("price_stress_score"), 0),
        "Supplier Dependency": safe_float(row.get("supplier_dependency_proxy"), 55),
        "Strategic Stock Level": safe_float(row.get("strategic_stock_proxy"), 50),
        "Alternative Supplier Readiness": safe_float(row.get("alt_supplier_proxy"), 50),
    }
    inv_stock = 100 - key_map["Strategic Stock Level"]
    inv_alt = 100 - key_map["Alternative Supplier Readiness"]
    key_map["Strategic Stock Level"] = inv_stock
    key_map["Alternative Supplier Readiness"] = inv_alt
    acc = 0.0
    for k, w in ahp_weights.items():
        acc += w * key_map.get(k, 50)
    return float(np.clip(acc, 0, 100))


def _last_features(bundle) -> pd.Series:
    if bundle.risk_features.empty:
        return pd.Series(dtype=float)
    row = bundle.risk_features.iloc[-1].copy()
    row["supplier_dependency_proxy"] = safe_float(row.get("geopolitical_risk_score"), 50) * 0.8 + 20
    row["strategic_stock_proxy"] = 100 - safe_float(row.get("price_stress_score"), 50) * 0.6
    row["alt_supplier_proxy"] = 100 - safe_float(row.get("logistics_risk_score"), 50) * 0.7
    return row


def lda_class_caption(class_id: str) -> str:
    if str(class_id) == "1":
        return "Elevated — High/Critical band (label 1)"
    if str(class_id) == "0":
        return "Baseline — Low/Moderate band (label 0)"
    return "Discriminant output"


def page_executive(bundle, forecast_res, lda_res, ahp_res, context):
    render_header("Egypt Food Import Intelligence DSS")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Unified risk", f"{context['unified_risk']:.1f}", "AHP-weighted composite")
    with col2:
        render_kpi_card("Forecast trend", forecast_res.trend.upper(), "Blended wheat series")
    with col3:
        render_kpi_card("LDA class", str(lda_res.last_prediction), lda_class_caption(lda_res.last_prediction))
    with col4:
        render_kpi_card("AHP consistency", f"CR={ahp_res.cr:.3f}", "Weight reliability")

    alert_level = (
        "Critical"
        if context["unified_risk"] >= 75
        else "High"
        if context["unified_risk"] >= 55
        else "Moderate"
        if context["unified_risk"] >= 35
        else "Low"
    )
    render_risk_badge(alert_level + " alert band")

    snap = build_maritime_snapshot(context)
    render_executive_maritime_summary(snap, context, bundle, forecast_res, lda_res, ahp_res)

    top_driver = max(
        [
            ("Geopolitical", context["geo"]),
            ("Logistics", context["log"]),
            ("Price stress", context["pss"]),
        ],
        key=lambda x: x[1],
    )[0]
    render_exec_driver_banner(
        top_driver,
        f"{forecast_res.warning_explanation[:220]}...",
    )

    inf = run_inference(context)
    if inf:
        render_recommendation_card(
            "Primary recommendation",
            explain_alert(inf[0], context),
            "Knowledge base + inference engine",
        )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=context["unified_risk"],
            title={"text": "Final risk score", "font": DSS_TITLE_FONT},
            number={
                "font": dict(family="DM Sans, Arial, sans-serif", size=36, color="#0f172a"),
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#cbd5e1",
                    "tickfont": dict(size=11, color="#64748b"),
                },
                "bar": {"color": "#1e3a8a"},
                "bgcolor": "rgba(241,245,249,0.85)",
                "borderwidth": 1,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [0, 35], "color": "#bbf7d0"},
                    {"range": [35, 55], "color": "#fef08a"},
                    {"range": [55, 75], "color": "#fdba74"},
                    {"range": [75, 100], "color": "#fecaca"},
                ],
            },
        )
    )
    apply_dss_layout(fig, height=340, margin=dict(l=28, r=28, t=52, b=34))

    c1, c2 = st.columns((1, 1))
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pick = st.multiselect(
            "Factors in bar chart",
            ["Geopolitical", "Logistics", "Price stress"],
            default=["Geopolitical", "Logistics", "Price stress"],
            key="exec_bar_factors",
        )
        cmap = {"Geopolitical": context["geo"], "Logistics": context["log"], "Price stress": context["pss"]}
        if pick:
            contrib = pd.DataFrame({"factor": pick, "score": [cmap[k] for k in pick]})
        else:
            contrib = pd.DataFrame({"factor": [], "score": []})
        if not contrib.empty:
            fig_h = max(320, 140 + len(contrib) * 52)
            blues = ["#1e3a8a", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"]
            bar_colors = [blues[i % len(blues)] for i in range(len(contrib))]
            fig2 = go.Figure(
                go.Bar(
                    x=contrib["score"],
                    y=contrib["factor"],
                    orientation="h",
                    marker=dict(
                        color=bar_colors,
                        line=dict(width=1.2, color="rgba(15,23,42,0.13)"),
                    ),
                )
            )
            fig2.update_layout(
                title=dss_title("Risk contribution snapshot"),
                xaxis=dict(
                    title="Score",
                    range=[0, 100],
                    showgrid=True,
                    gridcolor="#e2e8f0",
                    zeroline=False,
                ),
                yaxis=dict(showgrid=False, automargin=True),
                bargap=0.32,
            )
            apply_dss_layout(fig2, height=fig_h, margin=dict(l=8, r=28, t=56, b=40))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Select at least one factor for the bar chart.")

    section_ollama_explainer(
        "executive",
        "Executive Overview",
        [
            f"Unified risk score (AHP-weighted): {context['unified_risk']:.1f}",
            f"Forecast trend (blended wheat): {forecast_res.trend}",
            f"LDA binary class: {lda_res.last_prediction} ({lda_class_caption(lda_res.last_prediction)})",
            f"Top driver: {top_driver}",
            f"AHP consistency ratio: {ahp_res.cr:.3f}",
            f"Geopolitical / logistics / price stress: {context['geo']:.1f}, {context['log']:.1f}, {context['pss']:.1f}",
        ],
        extra_context=forecast_res.warning_explanation[:400],
    )


def page_prices(bundle, forecast_res):
    render_header("Commodity Price Intelligence")

    raw_cat = cached_wfp_commodity_catalog(str(ROOT / "wfp_food_prices_egy.csv"))
    if not raw_cat and bundle.wheat_prices.empty:
        st.warning("WFP Egypt CSV missing or unusable. Demo mode unavailable for price charts.")
        return

    if not raw_cat:
        raw_cat = {"Wheat flour (blended pipeline)": bundle.wheat_prices.copy()}
    catalog, catalog_note = build_blended_wfp_commodity_catalog(raw_cat, bundle)

    names = sorted(catalog.keys(), key=str.lower)
    default_i = 0
    for i, n in enumerate(names):
        if "wheat" in n.lower():
            default_i = i
            break

    c0, c1 = st.columns(2)
    with c0:
        product = st.selectbox(
            "Commodity (from WFP Egypt file)",
            names,
            index=default_i,
            key="cpi_product",
            help="Each row is monthly mean price for that commodity. Wheat flour often includes a FAO/WFP blend in the pipeline; "
            "other commodities use WFP retail columns only.",
        )
    method_labels: list[tuple[str, ForecastMethod]] = [
        ("Naive (repeat last price)", "naive"),
        ("Simple moving average", "sma"),
        ("Weighted moving average (recent months weigh more)", "wma"),
        ("Exponential smoothing", "exponential_smoothing"),
    ]
    with c1:
        pick_lab = st.selectbox(
            "Forecast model",
            [x[0] for x in method_labels],
            index=3,
            key="cpi_method",
        )
    method_map = dict(method_labels)
    method: ForecastMethod = method_map[pick_lab]

    df_sel = catalog[product].copy()
    fc = forecast_with_method(df_sel, method, commodity_name=product)

    blend_bits = bundle.price_blend_note
    if catalog_note.strip():
        blend_bits = f"{blend_bits} {catalog_note}"
    if "wheat" in product.lower():
        st.caption(blend_bits)
    else:
        st.caption(
            f"{blend_bits} "
            "**Non-wheat:** national-average retail from **WFP** when allowed; blended with **FAO** when Item names "
            "match (see `fao_commodities_egy.csv`)."
        )

    if fc.history_actual.empty or len(fc.history_actual) < 2:
        st.warning("Not enough USD price points for this commodity.")
        return

    max_win = len(fc.history_actual)
    win = st.slider(
        "Trailing months on chart",
        min_value=min(6, max_win),
        max_value=max_win,
        value=min(60, max_win),
        key="price_chart_window",
    )
    act_usd = fc.history_actual.iloc[-win:].reset_index(drop=True)
    fit_usd = fc.history_fitted.iloc[-win:].reset_index(drop=True)
    idx = np.arange(len(act_usd))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=act_usd,
            name="Actual USD/kg",
            mode="lines",
            line=dict(color="#2563eb"),
            yaxis="y",
        )
    )
    if fit_usd.notna().any():
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=fit_usd,
                name="Model baseline USD",
                line=dict(dash="dash", color="#64748b"),
                yaxis="y",
            )
        )

    last_x = len(act_usd) - 1 if len(act_usd) else 0
    fig.add_trace(
        go.Scatter(
            x=[last_x + 1],
            y=[fc.next_price],
            name="Next forecast USD",
            mode="markers",
            marker=dict(size=14, color="crimson"),
            yaxis="y",
        )
    )

    fig.update_layout(
        title=dss_title(f"{product} — USD vs model baseline — last {win} months + next"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(title="USD / kg", side="left", showgrid=True),
    )
    apply_dss_layout(fig, height=460, margin=dict(t=58, b=36))
    apply_dss_cartesian_grid(fig)
    st.plotly_chart(fig, use_container_width=True)

    err_s = (act_usd.astype(float).values - fit_usd.astype(float).values)
    err_series = pd.Series(err_s, index=act_usd.index)
    fig2 = go.Figure(go.Bar(x=idx, y=err_series, marker_color="#64748b", name="Residual USD"))
    fig2.update_layout(title=dss_title("Forecast error vs model baseline (USD, selected window)"))
    apply_dss_layout(fig2, height=280, margin=dict(t=52))
    apply_dss_cartesian_grid(fig2)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(explain_residual_panel(act_usd.reset_index(drop=True), fit_usd.reset_index(drop=True), err_series.reset_index(drop=True), win, "USD"))

    badge = "warning" if fc.trend == "up" else "stable"
    st.metric("Forecast direction", fc.trend.upper(), help=badge)
    st.write(fc.confidence_note)
    st.success(fc.warning_explanation)

    section_ollama_explainer(
        "prices",
        "Commodity Price Intelligence",
        [
            f"Product: {product}",
            f"Forecast model: {fc.method_label}",
            f"Forecast trend: {fc.trend}",
            f"Next-period USD: {fc.next_price:.4f}" if not pd.isna(fc.next_price) else "Forecast n/a",
            (bundle.price_blend_note[:120] + " " + catalog_note[:120]).strip(),
            f"Residual MAE (full series): {fc.residual_mae}" if fc.residual_mae else "Residual n/a",
        ],
        extra_context=(fc.confidence_note or "")[:500],
    )


def page_ports(bundle):
    render_header("Port & Logistics Intelligence")
    port_csv = ROOT / "Daily_Port_Activity_Data_and_Trade_Estimates.csv"
    p_cargo = cached_ports_by_cargo(str(port_csv))
    fallback = bundle.port_monthly.copy()

    can_segment = (
        not p_cargo.empty
        and "portcalls_dry_bulk" in p_cargo.columns
        and "import_dry_bulk" in p_cargo.columns
        and "export_container" in p_cargo.columns
    )
    p = p_cargo if can_segment else fallback
    if p.empty:
        st.warning(
            "Port activity file not loaded or filter empty. Ensure Daily_Port_Activity CSV is present (first load can take several minutes)."
        )
        return
    if not can_segment:
        st.info(
            "Cargo-type columns (`*_dry_bulk`, `*_container`, …) were not loaded from the port file — "
            "showing **all-vessel aggregates** only."
        )

    commodity_labels = [seg[0] for seg in PORT_CARGO_SEGMENTS]
    if can_segment:
        commodity = st.selectbox(
            "Cargo / commodity type (port file segments)",
            commodity_labels,
            index=0,
            key="ports_cargo_seg",
            help="Each option maps to aligned `portcalls_*`, `import_*`, `export_*` columns in the UNCTAD-style daily port CSV.",
        )
    else:
        commodity = commodity_labels[0]
        st.selectbox(
            "Cargo / commodity type (port file segments)",
            [commodity],
            disabled=True,
            key="ports_cargo_seg_disabled",
            help="Segment columns unavailable; using aggregate totals only.",
        )

    seg_map = {seg[0]: seg[1:] for seg in PORT_CARGO_SEGMENTS}
    p_call, imp_col, exp_col = seg_map[commodity]

    iso_opts = sorted(p["ISO3"].dropna().astype(str).unique())
    default_iso = [x for x in ("EGY", "UKR", "RUS") if x in iso_opts]
    if not default_iso:
        default_iso = iso_opts[:3] if len(iso_opts) > 3 else iso_opts
    chosen = st.multiselect("ISO3 regions to plot", iso_opts, default=default_iso, key="ports_iso")
    if not chosen:
        st.warning("Select at least one ISO3 code.")
        return
    metric = st.radio("Primary metric", ["portcalls", "import", "export"], horizontal=True, key="ports_metric")
    metric_to_col = {"portcalls": p_call, "import": imp_col, "export": exp_col}

    fig = go.Figure()
    colors = ["#0ea5e9", "#f97316", "#22c55e", "#a855f7", "#64748b"]
    for i, iso in enumerate(chosen):
        sub = p[p["ISO3"] == iso].sort_values("month")
        mc = metric_to_col[metric]
        if mc not in sub.columns:
            yv = sub["portcalls"] if "portcalls" in sub.columns else sub.iloc[:, 2]
        else:
            yv = sub[mc]
        fig.add_trace(
            go.Scatter(
                x=sub["month"],
                y=yv,
                name=f"{iso} — {metric}",
                mode="lines",
                line=dict(color=colors[i % len(colors)]),
            )
        )
    fig.update_layout(
        title=dss_title(f"Port logistics — {commodity} — {metric} by ISO3"),
        legend=dict(orientation="h"),
    )
    apply_dss_layout(fig, height=460, margin=dict(t=56))
    apply_dss_cartesian_grid(fig)
    st.plotly_chart(fig, use_container_width=True)

    section_ollama_explainer(
        "ports",
        "Port & Logistics Intelligence",
        [
            f"Cargo segment: {commodity}",
            f"Selected ISO3: {', '.join(chosen)}",
            f"Plot columns: `{p_call}`, `{imp_col}`, `{exp_col}` for calls / import / export",
            f"Rows in monthly table: {len(p)}",
            "Note: fused **logistics_risk_score** in the DSS still uses all-vessel port aggregates.",
        ],
    )


def page_geo(bundle):
    render_header("Geopolitical Intelligence")
    g = bundle.gdelt_yearly
    if g.empty:
        st.warning("GDELT file missing or empty.")
        return
    all_c = sorted(g["CountryCode"].dropna().astype(str).unique())
    default_c = [x for x in ["EG", "UA", "RS"] if x in all_c]
    sel = st.multiselect("Country codes (GDELT)", all_c, default=default_c or all_c[:5], key="geo_countries")
    if not sel:
        st.warning("Select at least one country.")
        return
    focus = g[g["CountryCode"].isin(sel)].copy()
    fig = go.Figure()
    for code in focus["CountryCode"].unique():
        sub = focus[focus["CountryCode"] == code]
        fig.add_trace(
            go.Scatter(
                x=sub["year"],
                y=sub["conflict_intensity"],
                name=str(code),
                mode="lines+markers",
            )
        )
    fig.update_layout(title=dss_title("Conflict intensity index (interactive selection)"))
    apply_dss_layout(fig, height=440)
    apply_dss_cartesian_grid(fig)
    st.plotly_chart(fig, use_container_width=True)

    section_ollama_explainer(
        "geo",
        "Geopolitical Intelligence",
        [f"Countries plotted: {', '.join(sel)}", f"Years covered: {int(g['year'].min())}–{int(g['year'].max())}"],
    )


def page_fao_nlp_shipping(bundle):
    """Cross-source price blend, shipping stress, and bilingual news NLP vs price spikes."""
    render_header("Cross-source signals: prices, shipping & NLP")

    raw_cat = cached_wfp_commodity_catalog(str(ROOT / "wfp_food_prices_egy.csv"))
    if not raw_cat and bundle.wheat_prices.empty:
        st.warning("WFP Egypt CSV missing. This page needs monthly retail prices.")
        return
    if not raw_cat:
        raw_cat = {"Wheat flour (blended pipeline)": bundle.wheat_prices.copy()}
    catalog, catalog_note = build_blended_wfp_commodity_catalog(raw_cat, bundle)

    names = sorted(catalog.keys(), key=str.lower)
    wheat_idx = next((i for i, n in enumerate(names) if "wheat" in n.lower()), 0)
    product = st.selectbox(
        "Commodity (price track + NLP spike proxy)",
        names,
        index=wheat_idx,
        key="novelty_commodity",
        help=(
            "WFP Egypt monthly series; FAO overlays when rows match FAOSTAT Item names (`fao_commodities_egy.csv`) "
            "or when wheat merges with `fao_wheat_egy.csv` on the pipeline spine."
        ),
    )

    w = catalog[product].copy()

    if w.empty or "price_usd" not in w.columns:
        st.warning("No USD price series for this commodity.")
        return

    if "wheat" in product.lower():
        st.info(bundle.price_blend_note)
    else:
        st.caption(
            "**Non-wheat:** WFP USD retail; FAOSTAT items in `fao_commodities_egy.csv` are auto-matched by name overlap. "
            + catalog_note
        )

    spike_legend = "Price spike proxy (std. |Δ| USD)"

    _y_usd_wfp = pd.to_numeric(w["price_usd_wfp"], errors="coerce") if "price_usd_wfp" in w.columns else pd.Series(dtype=float)
    _y_usd_fao = pd.to_numeric(w["price_usd_fao"], errors="coerce") if "price_usd_fao" in w.columns else pd.Series(dtype=float)
    _y_blend = pd.to_numeric(w["price_usd"], errors="coerce")
    month_x = pd.to_datetime(w["month"], errors="coerce")

    _wfp_vis = bool(_y_usd_wfp.notna().any())
    _fao_vis = bool(_y_usd_fao.notna().any())

    if not w.empty and _wfp_vis and _fao_vis:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=month_x,
                y=_y_usd_wfp,
                name="WFP retail proxy (EGP→USD)",
                mode="lines",
                line=dict(color="#1e40af", width=2.5),
                connectgaps=False,
                hovertemplate="%{x|%Y-%m}<br>WFP %{y:.4f} USD/kg<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=month_x,
                y=_y_usd_fao,
                name="FAO / policy-relevant series",
                mode="lines",
                line=dict(color="#0e7490", width=2.5, dash="dot"),
                connectgaps=False,
                hovertemplate="%{x|%Y-%m}<br>FAO %{y:.4f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=month_x,
                y=_y_blend,
                name="Blended modeling price (priority: both → 45/55; else single source)",
                mode="lines",
                line=dict(color="#dc2626", width=3.2),
                connectgaps=False,
                hovertemplate="%{x|%Y-%m}<br>Blend %{y:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dss_title(f"{product} — WFP × FAO decomposition + blend"),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
            hovermode="x unified",
        )
        apply_dss_layout(fig, height=410, margin=dict(t=72))
        apply_dss_cartesian_grid(fig)
        st.plotly_chart(fig, use_container_width=True)
    elif not w.empty and _wfp_vis:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=month_x,
                y=_y_blend,
                name=f"{product} — USD (WFP retail)",
                mode="lines",
                line=dict(color="#2563eb", width=2.5),
                hovertemplate="%{x|%Y-%m}<br>%{y:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dss_title(f"{product} — WFP only (no FAO overlap matched for this commodity)"),
            hovermode="x unified",
        )
        apply_dss_layout(fig, height=360, margin=dict(t=54))
        apply_dss_cartesian_grid(fig)
        st.plotly_chart(fig, use_container_width=True)
    elif not w.empty:
        fig = go.Figure(
            go.Scatter(
                x=month_x,
                y=_y_blend,
                name=f"{product} — modeling USD",
                mode="lines",
                line=dict(color="#64748b", width=2.5),
            )
        )
        fig.update_layout(title=dss_title(f"{product} — price track"))
        apply_dss_layout(fig, height=360, margin=dict(t=54))
        apply_dss_cartesian_grid(fig)
        st.plotly_chart(fig, use_container_width=True)

    render_page_section_title("Shipping tracker + port-derived stress")
    sh = bundle.shipping_monthly
    if sh.empty:
        st.warning("No shipping layer — add shipping_metrics.csv or ensure port activity loads.")
    else:
        fig2 = go.Figure()
        if "shipping_stress_score_composite" in sh.columns:
            fig2.add_trace(
                go.Scatter(
                    x=sh["month"],
                    y=sh["shipping_stress_score_composite"],
                    name="Composite route stress",
                    mode="lines",
                    line=dict(color="#b91c1c"),
                )
            )
        elif "shipping_stress_score" in sh.columns:
            fig2.add_trace(
                go.Scatter(x=sh["month"], y=sh["shipping_stress_score"], name="Port-derived stress", mode="lines")
            )
        if "shipping_congestion" in sh.columns:
            fig2.add_trace(
                go.Scatter(x=sh["month"], y=sh["shipping_congestion"], name="Tracker congestion index", mode="lines", yaxis="y2")
            )
            fig2.update_layout(yaxis2=dict(overlaying="y", side="right", title="Congestion"), height=400)
        else:
            fig2.update_layout(height=400)
        fig2.update_layout(title=dss_title("Logistics / shipping early-warning layer"))
        apply_dss_layout(fig2, height=400, margin=dict(t=54))
        st.plotly_chart(fig2, use_container_width=True)

    render_page_section_title("Bilingual NLP conflict & sentiment index")
    nm = bundle.news_monthly

    novelty_live = compute_novelty_correlation(w, nm)
    if bundle.novelty.used_synthetic_news:
        novelty_live = replace(
            novelty_live,
            note="Demo headlines synthesized from price volatility pattern.",
        )

    if bundle.novelty.used_synthetic_news:
        st.warning(
            "Headlines were **synthesized** from price-spike patterns because news_headlines_bilingual.csv was missing. "
            "Replace with real Arabic/English feeds for production novelty evidence."
        )
    if not nm.empty and not w.empty:
        merged = w[["month", "price_usd"]].merge(nm, on="month", how="inner")
        merged["price_spike_z"] = price_spike_series(merged["price_usd"]).values
        fig3 = go.Figure()
        fig3.add_trace(
            go.Bar(x=merged["month"], y=merged["nlp_conflict_index"], name="NLP conflict keyword density (monthly avg)", marker_color="#7c3aed")
        )
        fig3.add_trace(
            go.Scatter(
                x=merged["month"],
                y=merged["price_spike_z"],
                name=spike_legend,
                mode="lines+markers",
                yaxis="y2",
            )
        )
        fig3.update_layout(
            title=dss_title(f"Novelty: bilingual news conflict intensity vs {product} price volatility"),
            yaxis2=dict(title="Spike z", overlaying="y", side="right"),
            height=440,
            legend=dict(orientation="h"),
        )
        apply_dss_layout(fig3, height=440, margin=dict(t=56))
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = go.Figure(
            go.Scatter(
                x=merged["nlp_conflict_index"],
                y=merged["price_spike_z"],
                mode="markers",
                text=merged["month"].dt.strftime("%Y-%m"),
                marker=dict(size=10, color=merged["news_sentiment_avg"], colorscale="RdYlGn", showscale=True),
                name="months",
            )
        )
        fig4.update_layout(
            title=dss_title("Scatter: NLP conflict index vs price spike (color = avg sentiment)"),
            xaxis_title="Conflict NLP index",
            yaxis_title="Price spike z",
            height=400,
        )
        apply_dss_layout(fig4, height=400, margin=dict(t=54))
        apply_dss_cartesian_grid(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    nvis = novelty_live
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "ρ (conflict NLP, spike)",
            f"{nvis.pearson_conflict_vs_spike:.3f}" if nvis.pearson_conflict_vs_spike is not None else "n/a",
        )
    with c2:
        st.metric(
            "ρ (sentiment, spike)",
            f"{nvis.pearson_sentiment_vs_spike:.3f}" if nvis.pearson_sentiment_vs_spike is not None else "n/a",
        )
    with c3:
        st.metric("Overlapping headline months", f"{nvis.headline_months}")

    with st.expander("Sample bilingual headlines (training corpus)"):
        st.dataframe(bundle.news_headlines.head(40), use_container_width=True)

    ollama_note = (
        f"Commodity: **{product}**. Correlations use spike proxy from this commodity’s **price_usd** vs monthly NLP."
        if not bundle.novelty.used_synthetic_news
        else novelty_live.note
    )
    section_ollama_explainer(
        "novelty",
        "Cross-source signals: prices, shipping & NLP",
        [
            (catalog_note[:200] + f" Product: {product}").strip(),
            f"ρ conflict NLP vs spike: {nvis.pearson_conflict_vs_spike}",
            f"ρ sentiment vs spike: {nvis.pearson_sentiment_vs_spike}",
            f"Synthetic news flag: {bundle.novelty.used_synthetic_news}",
            f"Overlapping headline months: {nvis.headline_months}",
        ],
        extra_context=ollama_note or "",
    )


def page_warning(bundle, ahp_res, context):
    render_header("Unified Risk Early Warning")
    st.session_state.setdefault("sens_geo", 1.0)
    st.session_state.setdefault("sens_log", 1.0)
    st.session_state.setdefault("sens_price", 1.0)

    c1, c2, c3 = st.columns(3)
    with c1:
        sg = st.slider("Geopolitical weight multiplier", 0.5, 1.5, st.session_state["sens_geo"], 0.05)
    with c2:
        sl = st.slider("Logistics weight multiplier", 0.5, 1.5, st.session_state["sens_log"], 0.05)
    with c3:
        sp = st.slider("Price stress weight multiplier", 0.5, 1.5, st.session_state["sens_price"], 0.05)
    st.session_state["sens_geo"] = sg
    st.session_state["sens_log"] = sl
    st.session_state["sens_price"] = sp

    row = _last_features(bundle)
    w = ahp_res.weights.copy()
    w["Geopolitical Risk"] *= sg
    w["Logistics Risk"] *= sl
    w["Price Stress"] *= sp
    s = sum(w.values()) or 1.0
    w = {k: v / s for k, v in w.items()}
    adj_score = unified_risk(w, row)

    base = context["unified_risk"]
    st.metric("Adjusted unified risk", f"{adj_score:.1f}", f"{adj_score - base:+.1f} vs baseline")
    swings = {
        "Geopolitical × weight": (base * 0.9, base * 1.15),
        "Logistics × weight": (base * 0.95, base * 1.1),
        "Price stress × weight": (base * 0.92, base * 1.12),
        "Wheat shock (+10%)": (base, min(100, base + 8)),
        "Port drop stress": (base, min(100, base + 5)),
    }
    tor = tornado_sensitivity(base, swings)
    fig = go.Figure(
        go.Bar(
            x=tor["spread"],
            y=tor["factor"],
            orientation="h",
            marker_color="#b91c1c",
        )
    )
    fig.update_layout(title=dss_title("Tornado sensitivity (illustrative swings)"))
    apply_dss_layout(fig, height=360)
    apply_dss_cartesian_grid(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(tor, use_container_width=True)
    if not tor.empty:
        top = tor.iloc[0]["factor"]
        render_methodology_card(
            "Sensitivity narrative",
            f"The final disruption risk is most influenced by {top} in this illustrative sweep, "
            "holding other DSS layers fixed.",
        )

    section_ollama_explainer(
        "warning",
        "Unified Risk Early Warning",
        [
            f"Baseline unified risk: {context['unified_risk']:.1f}",
            f"Adjusted risk (sliders): {adj_score:.1f}",
            f"Sliders: geo×{sg:.2f}, logistics×{sl:.2f}, price×{sp:.2f}",
            f"Tornado top factor: {tor.iloc[0]['factor']}" if not tor.empty else "No tornado",
        ],
    )


def render_decision_analysis_inner(suffix: str = "standalone") -> None:
    """Certainty, risk (EU), and uncertainty criteria — used on standalone page and Scenario tab."""
    sk = suffix
    render_page_section_title("Decision under certainty", compact=True)
    utils = {
        "Maintain partners": 62,
        "Early procurement": 71,
        "Diversify origin": 74,
    }
    best_u, val = deterministic_utility_best(utils)
    st.success(f"Best under certainty (deterministic utility): **{best_u}** (U={val:.1f})")

    render_page_section_title("Decision under risk (expected utility)", compact=True)
    scenarios = ["Normal market", "Moderate disruption", "Severe disruption", "Critical disruption"]
    pdef = [0.55, 0.25, 0.12, 0.08]
    probs = {s: p for s, p in zip(scenarios, pdef)}
    pay = {
        "Maintain partners": dict(zip(scenarios, [82, 65, 45, 30])),
        "Early procurement": dict(zip(scenarios, [70, 72, 68, 58])),
        "Diversify origin": dict(zip(scenarios, [74, 76, 72, 60])),
    }
    st.write("Payoff matrix (utility 0–100)")
    pay_df = pd.DataFrame(pay).T
    st.dataframe(pay_df, use_container_width=True)

    st.write("Scenario probabilities")
    c = st.columns(4)
    adj = {}
    for i, s in enumerate(scenarios):
        adj[s] = c[i].slider(s, 0.0, 1.0, probs[s], 0.01, key=f"dec_{sk}_scen_{i}")
    tot = sum(adj.values()) or 1.0
    adj = {k: v / tot for k, v in adj.items()}
    eus = expected_utility(pay, adj)
    st.dataframe(pd.Series(eus, name="Expected utility").to_frame(), use_container_width=True)
    best_r = max(eus.items(), key=lambda x: x[1])[0]
    st.info(f"Best under risk: **{best_r}**")

    render_page_section_title("Decision under uncertainty", compact=True)
    hurw = st.slider("Hurwicz optimism α", 0.0, 1.0, 0.45, key=f"hurwicz_alpha_{sk}")
    ures = uncertainty_criteria(pay, hurwicz_alpha=hurw)
    st.write(
        pd.DataFrame(
            {
                "Criterion": ["Maximax", "Maximin", "Laplace", "Minimax regret", "Hurwicz"],
                "Choice": [
                    ures["maximax"],
                    ures["maximin"],
                    ures["laplace"],
                    ures["minimax_regret"],
                    ures["hurwicz"],
                ],
            }
        ),
        use_container_width=True,
    )
    st.write("Regret matrix")
    st.dataframe(
        pd.DataFrame(
            ures["regret_matrix"],
            index=ures["alternatives"],
            columns=ures["state_labels"],
        ),
        use_container_width=True,
    )
    st.caption(
        explain_uncertainty_decision("Maximin", ures["maximin"])
        + " Compare with Laplace/Hurwicz for optimism assumptions."
    )
    section_ollama_explainer(
        f"decision_{suffix}",
        "Decision Analysis (certainty / risk / uncertainty)",
        [
            f"Best under certainty: {best_u} (U={val:.1f})",
            f"Best under risk (EU): {best_r}",
            f"Maximin pick: {ures['maximin']}; Hurwicz α={hurw}",
        ],
    )


def page_scenario(bundle, context, ahp_weights: dict):
    render_header("Scenario Simulator")
    tab_w, tab_d = st.tabs(["What-if scenarios", "Decision Analysis"])
    with tab_w:
        st.session_state.setdefault("stock", 48.0)
        st.session_state["stock"] = st.slider(
            "Strategic stock proxy", 0.0, 100.0, float(st.session_state["stock"])
        )
        row = _last_features(bundle)
        row["strategic_stock_proxy"] = st.session_state["stock"]
        ur = unified_risk(ahp_weights, row)
        st.metric("Adjusted unified risk (scenario)", f"{ur:.1f}")
        section_ollama_explainer(
            "scenario_whatif",
            "Scenario — strategic stock what-if",
            [
                f"Strategic stock proxy: {st.session_state['stock']:.1f}",
                f"Unified risk under AHP weights: {ur:.1f}",
            ],
        )
    with tab_d:
        st.caption("Structured comparison of importer alternatives under certainty, risk, and uncertainty.")
        render_decision_analysis_inner("scenario_tab")


def page_goal_prog():
    render_header("Goal Programming Optimizer")
    df, expl, wts = run_goal_programming()
    render_page_section_title("Goal weights")
    st.json(wts)
    st.dataframe(df, use_container_width=True)
    st.success(expl)
    render_methodology_card("Interpretation", expl)
    best = df.iloc[0]["alternative"] if not df.empty else "n/a"
    section_ollama_explainer(
        "goal_prog",
        "Goal Programming Optimizer",
        [f"Recommended alternative (lowest penalty): {best}", expl[:300]],
    )


def page_ahp(ahp_res, context):
    render_header("AHP Weighting & Sensitivity Analysis")
    render_page_section_title("Pairwise comparison matrix", compact=True)
    st.caption("Default expert judgments")
    st.dataframe(pd.DataFrame(default_comparison_matrix(), index=CRITERIA, columns=CRITERIA))
    render_page_section_title("AHP priority weights")
    st.json(ahp_res.weights)
    st.metric("Consistency ratio", f"{ahp_res.cr:.3f}")
    if not ahp_res.consistent:
        st.warning("Consistency ratio exceeds 0.10 — refine comparisons.")
    st.info(ahp_explanation(ahp_res))

    mode = st.radio("Weighting mode", ["AHP-derived", "Manual sliders"])
    if mode == "Manual sliders":
        manual = {}
        cols = st.columns(3)
        for i, name in enumerate(CRITERIA):
            manual[name] = cols[i % 3].slider(name, 0.0, 1.0, float(ahp_res.weights.get(name, 1 / 6)), 0.02)
        s = sum(manual.values()) or 1.0
        manual = {k: v / s for k, v in manual.items()}
        row = _last_features(st.session_state["bundle"])
        st.metric("Risk using manual weights", f"{unified_risk(manual, row):.1f}")

    section_ollama_explainer(
        "ahp",
        "AHP Weighting & Sensitivity",
        [
            f"Consistency ratio: {ahp_res.cr:.3f}",
            f"AHP consistent: {ahp_res.consistent}",
            f"Top criterion weight: {max(ahp_res.weights.items(), key=lambda x: x[1])[0]}",
        ],
        extra_context=str(ahp_res.weights),
    )


def page_recommendations(context):
    render_header("Recommendations Center")
    snap = build_maritime_snapshot(context)
    malerts = evaluate_maritime_alerts(snap["vessels"], snap["ports"], context)
    render_page_section_title("Operational maritime alerts (demo)")
    act = [a for a in malerts if a.status != AlertStatus.RESOLVED]
    if act:
        st.dataframe(alerts_to_dataframe(act), use_container_width=True)
    else:
        st.caption("No active maritime alerts.")
    section_ollama_explainer(
        "recommendations",
        "Recommendations Center",
        ["General guidance for using DSS outputs for Egyptian wheat import decisions."],
    )


def page_port_vessel_intelligence(context):
    render_header("Port & Vessel Intelligence")
    snap = build_maritime_snapshot(context)
    render_maritime_full_page(snap, context, key_prefix="pv_page")
    section_ollama_explainer(
        "port_vessel",
        "Port & Vessel Intelligence",
        [
            "Full-screen maritime control: Scattergeo map, selectors, alerts, optional Ollama Q&A.",
            "Synthetic AIS/demo CSV mode — no paid APIs required.",
        ],
    )


def page_knowledge():
    render_header("Knowledge Base & Inference Engine")
    render_page_section_title("Rules (table)")
    st.dataframe(rules_as_dataframe(), use_container_width=True, hide_index=True)

    section_ollama_explainer(
        "knowledge",
        "Knowledge Base & Inference Engine",
        [
            f"Rule count: {len(get_rules())}",
            "Rules link WFP, GDELT, ports, NLP, and AHP signals to suggested actions.",
        ],
    )


PAGES = [
    ("Executive Overview", "overview"),
    ("Commodity Price Intelligence", "prices"),
    ("Port & Logistics Intelligence", "ports"),
    ("Geopolitical Intelligence", "geo"),
    ("Cross-source signals: prices, shipping & NLP", "novelty"),
    ("Unified Risk Early Warning", "warning"),
    ("Port & Vessel Intelligence", "pvessel"),
    ("Scenario Simulator", "scenario"),
    ("Goal Programming Optimizer", "gp"),
    ("AHP Weighting & Sensitivity Analysis", "ahp"),
    ("Recommendations Center", "rec"),
    ("Knowledge Base & Inference Engine", "kb"),
]


def main():
    st.set_page_config(page_title="Egypt Import Risk DSS", layout="wide", initial_sidebar_state="expanded")
    inject_custom_css()

    with st.sidebar:
        choice = render_sidebar_page_nav(PAGES)
        render_ollama_sidebar()
    page_map = dict(PAGES)
    if choice not in page_map:
        choice = PAGES[0][0]
        st.session_state["dss_nav_page"] = choice
    key = page_map[choice]

    with st.spinner("Loading and harmonizing datasets (port CSV may be large on first run)..."):
        bundle = cached_bundle()
    st.session_state["bundle"] = bundle

    if bundle.risk_features.empty:
        st.error(
            "Risk feature table is empty. Check that wfp_food_prices_egy.csv is present and contains Wheat flour."
        )

    row = _last_features(bundle)
    ahp_res = run_ahp()
    forecast_res = forecast_commodity_price(bundle.wheat_prices)
    lda_res = train_and_evaluate(bundle.risk_features)

    context = {
        "price_stress_score": safe_float(row.get("price_stress_score"), 0),
        "geopolitical_risk_score": safe_float(row.get("geopolitical_risk_score"), 0),
        "logistics_risk_score": safe_float(row.get("logistics_risk_score"), 0),
        "port_activity_drop": safe_float(row.get("port_activity_drop"), 0),
        "forecast_trend": forecast_res.trend,
        "strategic_stock_score": safe_float(row.get("strategic_stock_proxy"), 50),
        "unified_risk": unified_risk(ahp_res.weights, row),
        "unified_risk_score": unified_risk(ahp_res.weights, row),
        "ahp_cr": ahp_res.cr,
        "geo": safe_float(row.get("geopolitical_risk_score"), 0),
        "log": safe_float(row.get("logistics_risk_score"), 0),
        "pss": safe_float(row.get("price_stress_score"), 0),
    }

    if key == "overview":
        page_executive(bundle, forecast_res, lda_res, ahp_res, context)
    elif key == "prices":
        page_prices(bundle, forecast_res)
    elif key == "ports":
        page_ports(bundle)
    elif key == "geo":
        page_geo(bundle)
    elif key == "novelty":
        page_fao_nlp_shipping(bundle)
    elif key == "warning":
        page_warning(bundle, ahp_res, context)
    elif key == "pvessel":
        page_port_vessel_intelligence(context)
    elif key == "scenario":
        page_scenario(bundle, context, ahp_res.weights)
    elif key == "gp":
        page_goal_prog()
    elif key == "ahp":
        page_ahp(ahp_res, context)
    elif key == "rec":
        page_recommendations(context)
    elif key == "kb":
        page_knowledge()

    render_footer()


if __name__ == "__main__":
    main()
