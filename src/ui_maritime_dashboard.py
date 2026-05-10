"""Streamlit UI blocks for ministry-grade port/vessel intelligence."""

from __future__ import annotations

import html
import math
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.alert_system import AlertStatus, alerts_to_dataframe, evaluate_maritime_alerts
from src.chart_theme import apply_dss_cartesian_grid, apply_dss_layout, dss_title
from src.maritime_viz import build_maritime_intelligence_figure
from src.ollama_advisor import advisor_route_explanation
from src.ollama_explain import init_ollama_session_state
from src.port_intelligence import port_by_id
from src.recommendations import build_ministry_recommendations
from src.ui_components import render_page_section_title
from src.vessel_tracking import vessel_by_id


def _finalize_px_figure(fig: Any, *, height: int, grid: bool = True) -> None:
    apply_dss_layout(fig, height=height, margin=dict(t=54, b=42))
    if grid:
        try:
            apply_dss_cartesian_grid(fig)
        except Exception:
            pass


_PORT_DETAIL_FIELDS: list[tuple[str, str]] = [
    ("Port ID", "port_id"),
    ("Name", "name"),
    ("Country", "country"),
    ("City / region", "city_region"),
    ("Latitude (°N)", "lat"),
    ("Longitude (°E)", "lon"),
    ("Port type", "port_type"),
    ("Main goods imported", "main_goods_imported"),
    ("Main goods exported", "main_goods_exported"),
    ("Inbound ships", "inbound_ships"),
    ("Outbound ships", "outbound_ships"),
    ("Current activity level", "current_activity_level"),
    ("Congestion level", "congestion_level"),
    ("Logistics risk score", "logistics_risk_score"),
    ("Nearby conflict risk score", "nearby_conflict_risk_score"),
    ("Overall risk score", "overall_risk_score"),
    ("Overall risk level", "overall_risk_level"),
    ("Explanation", "explanation"),
]

_VESSEL_DETAIL_FIELDS: list[tuple[str, str]] = [
    ("Ship ID", "ship_id"),
    ("Name", "ship_name"),
    ("Type", "ship_type"),
    ("Origin port", "origin_port_id"),
    ("Destination port", "dest_port_id"),
    ("Current latitude", "current_lat"),
    ("Current longitude", "current_lon"),
    ("Cargo", "cargo_type"),
    ("Cargo (metric tons)", "cargo_quantity_mt"),
    ("Departure (UTC)", "departure_utc"),
    ("ETA (UTC)", "eta_utc"),
    ("Journey progress %", "journey_progress_pct"),
    ("Elapsed hours (approx.)", "elapsed_hours_approx"),
    ("Remaining hours (approx.)", "remaining_hours_approx"),
    ("Total journey hours (approx.)", "total_journey_hours_approx"),
    ("Speed (kn)", "speed_kn"),
    ("Route risk score", "route_risk_score"),
    ("Route risk level", "route_risk_level"),
    ("Near risk zone", "near_risk_zone"),
    ("Nearby zones", "near_zone_names"),
    ("Delay probability", "delay_probability"),
    ("Recommendation", "recommendation"),
    ("Alternative route note", "alt_route_explanation"),
    ("Alt. route latitudes", "alt_route_lats"),
    ("Alt. route longitudes", "alt_route_lons"),
    ("Route comparison (original vs alt.)", "route_compare"),
    ("Assessment text", "explainer"),
]


def _display_scalar(v: Any, col: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, dict):
        parts: list[str] = []
        for k, val in v.items():
            parts.append(f"{k}: {_display_scalar(val, str(k))}")
        return "; ".join(parts) if parts else "—"
    if isinstance(v, list):
        return ", ".join(_display_scalar(x, "") for x in v) if v else "—"
    try:
        if v is not None and pd.isna(v):
            return "—"
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, bool):
        return "Yes" if v else "No"
    try:
        fv = float(v)
        if not math.isfinite(fv):
            return "—"
        if col in ("lat", "lon", "current_lat", "current_lon"):
            return f"{fv:.4f}"
        if col in ("inbound_ships", "outbound_ships"):
            return str(int(round(fv)))
        if col in ("current_activity_level", "congestion_level"):
            return f"{fv:.2f}"
        if "score" in col or col in ("route_risk_score", "overall_risk_score"):
            return f"{fv:.1f}"
        if col in ("journey_progress_pct",):
            return f"{fv:.1f}"
        if col in ("delay_probability",):
            return f"{fv:.0%}"
        if col in (
            "elapsed_hours_approx",
            "remaining_hours_approx",
            "total_journey_hours_approx",
            "cargo_quantity_mt",
        ):
            if col == "cargo_quantity_mt":
                return f"{fv:,.0f}"
            return f"{fv:.1f}"
        if col in ("speed_kn",):
            return f"{fv:.1f}"
        if abs(fv - round(fv)) < 1e-9 and abs(fv) < 1e12:
            return str(int(round(fv)))
        return f"{fv:.4g}"
    except (TypeError, ValueError):
        return str(v)


def _series_to_attribute_table(s: pd.Series, ordered: list[tuple[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for label, col in ordered:
        if col not in s.index:
            continue
        seen.add(col)
        rows.append({"Attribute": label, "Value": _display_scalar(s[col], col)})
    for col in s.index:
        if col in seen:
            continue
        lab = str(col).replace("_", " ").strip().title()
        rows.append({"Attribute": lab, "Value": _display_scalar(s[col], str(col))})
    return pd.DataFrame(rows)


def _ollama_session_on() -> bool:
    init_ollama_session_state()
    return True


def _maritime_kpi_box(label: str, value: str, alerts_emphasis: bool = False) -> None:
    """Styled maritime KPI tile (matches inject_custom_css .maritime-kpi-*)."""
    mod = ' maritime-kpi-card--alert' if alerts_emphasis else ''
    lab = (
        label.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    val = (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    st.markdown(
        f'<div class="maritime-kpi-card{mod}"><span class="maritime-kpi-label">{lab}</span>'
        f'<span class="maritime-kpi-value">{val}</span></div>',
        unsafe_allow_html=True,
    )


def _maritime_severity_chip(kind: str, label: str, value: str) -> None:
    """Severity counter tile under Alert board."""
    lb = (
        label.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    val = (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    st.markdown(
        f'<div class="maritime-severity-chip maritime-severity-chip--{kind}">'
        f'<span class="msc-label">{lb}</span><span class="msc-value">{val}</span></div>',
        unsafe_allow_html=True,
    )


def _ai_rec_priority_slug(priority_val: str) -> str:
    s = str(priority_val or "").strip().upper().replace(" ", "")
    if s in ("P0", "P1", "P2"):
        return s.lower()
    return "p2"


def _render_ai_recommendation_detail(c: dict) -> None:
    """Structured panel inside expanders; paired with CSS :has(div.ai-reco-card-body)."""
    slug = _ai_rec_priority_slug(str(c.get("priority", "P2")))
    pri = html.escape(str(c.get("priority", "")))
    action = html.escape(str(c.get("action", "")))
    evidence = html.escape(str(c.get("evidence", "")))
    benefit = html.escape(str(c.get("expected_benefit", "")))
    st.markdown(
        f'<div class="ai-reco-card-body">'
        '<div class="ai-reco-row"><span class="ai-reco-k">Priority</span>'
        f'<span class="ai-reco-v"><span class="ai-reco-pill ai-reco-pill--{slug}">{pri}</span></span></div>'
        '<div class="ai-reco-row"><span class="ai-reco-k">Action</span>'
        f'<span class="ai-reco-v">{action}</span></div>'
        '<div class="ai-reco-row"><span class="ai-reco-k">Evidence</span>'
        f'<span class="ai-reco-v">{evidence}</span></div>'
        '<div class="ai-reco-row"><span class="ai-reco-k">Benefit</span>'
        f'<span class="ai-reco-v">{benefit}</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_executive_maritime_summary(
    snapshot: dict[str, Any],
    context: dict[str, Any],
    bundle,
    forecast_res,
    lda_res,
    ahp_res,
) -> None:
    """Ministry KPI strip + alerts + charts + map (Executive Overview)."""
    del forecast_res, lda_res, ahp_res  # reserved for future narrative hooks
    ports = snapshot["ports"]
    vessels = snapshot["vessels"]
    zones = snapshot["zones"]
    alerts = evaluate_maritime_alerts(vessels, ports, context)
    active_alerts = [a for a in alerts if a.status != AlertStatus.RESOLVED]
    resolved_alerts = [a for a in alerts if a.status == AlertStatus.RESOLVED]

    eg_dest = vessels[vessels["dest_port_id"].astype(str).str.startswith("EG_")]
    eg_orig = vessels[vessels["origin_port_id"].astype(str).str.startswith("EG_")]
    high_risk = vessels[vessels["route_risk_score"].astype(float) >= 60] if not vessels.empty else vessels
    crit_routes = vessels[vessels["route_risk_score"].astype(float) >= 75] if not vessels.empty else vessels

    avg_log = float(ports["logistics_risk_score"].mean()) if not ports.empty else 0.0
    avg_cong = float((ports["congestion_level"].astype(float) * 100).mean()) if not ports.empty else 0.0

    st.markdown(
        '<div class="exec-maritime-title"><span class="exec-maritime-title-text">'
        "Ministry port and vessel intelligence</span></div>",
        unsafe_allow_html=True,
    )
    r1, r2, r3 = st.columns(4), st.columns(4), st.columns(4)
    card_rows = [r1, r2, r3]
    flat_cards = [
        ("Ports monitored", len(ports)),
        ("Active ships", len(vessels)),
        ("Inbound EG", len(eg_dest)),
        ("Outbound EG", len(eg_orig)),
        ("High-risk ships", len(high_risk)),
        ("Critical routes", len(crit_routes)),
        ("Avg logistics (ports)", f"{avg_log:.0f}"),
        ("Avg congestion %", f"{avg_cong:.0f}"),
        ("Price stress", f"{float(context.get('pss') or 0):.0f}"),
        ("Geo risk", f"{float(context.get('geo') or 0):.0f}"),
        ("Disruption (unified)", f"{float(context.get('unified_risk') or 0):.0f}"),
    ]
    for i, (lab, val) in enumerate(flat_cards):
        row = card_rows[i // 4]
        with row[i % 4]:
            _maritime_kpi_box(lab, str(val))
    r4 = st.columns(4)
    with r4[0]:
        _maritime_kpi_box(
            "Active operational alerts",
            str(len(active_alerts)),
            alerts_emphasis=len(active_alerts) > 0,
        )

    st.markdown("#### Alert board")
    ac, hc, mc, rc = st.columns(4)
    crit_a = [a for a in active_alerts if a.severity.value == "critical"]
    high_a = [a for a in active_alerts if a.severity.value == "high"]
    mod_a = [a for a in active_alerts if a.severity.value == "moderate"]
    with ac:
        _maritime_severity_chip("critical", "Critical", str(len(crit_a)))
    with hc:
        _maritime_severity_chip("high", "High", str(len(high_a)))
    with mc:
        _maritime_severity_chip("moderate", "Moderate", str(len(mod_a)))
    with rc:
        _maritime_severity_chip("resolved", "Resolved / cleared", str(len(resolved_alerts)))

    if active_alerts:
        st.dataframe(alerts_to_dataframe(active_alerts), use_container_width=True)
    else:
        st.success("No active operational alerts under current thresholds.")

    render_page_section_title("Executive analytics", compact=True)
    if not vessels.empty:
        fig_r = px.histogram(
            vessels,
            x="route_risk_score",
            nbins=12,
            title="Route risk distribution (ships)",
        )
        fig_r.update_layout(title=dss_title("Route risk distribution (ships)"))
        _finalize_px_figure(fig_r, height=360)
        st.plotly_chart(fig_r, use_container_width=True)
        pie_c = vessels["cargo_type"].astype(str).value_counts().reset_index()
        pie_c.columns = ["cargo", "n"]
        fig_p = px.pie(pie_c, values="n", names="cargo", title="Cargo exposure mix (demo)")
        fig_p.update_layout(title=dss_title("Cargo exposure mix (demo)"))
        apply_dss_layout(fig_p, height=360)
        st.plotly_chart(fig_p, use_container_width=True)
        fig_io = px.bar(
            x=["Inbound Egypt (dest)", "Outbound Egypt (origin)"],
            y=[len(eg_dest), len(eg_orig)],
            title="Egypt-linked demo vessel counts",
        )
        fig_io.update_layout(title=dss_title("Egypt-linked demo vessel counts"))
        _finalize_px_figure(fig_io, height=320)
        st.plotly_chart(fig_io, use_container_width=True)
    if not ports.empty and "overall_risk_score" in ports.columns:
        fig_pt = px.bar(
            ports.sort_values("overall_risk_score", ascending=False).head(10),
            x="name",
            y="overall_risk_score",
            title="Top port composite risk (model blend)",
        )
        fig_pt.update_layout(title=dss_title("Top port composite risk (model blend)"))
        _finalize_px_figure(fig_pt, height=400)
        st.plotly_chart(fig_pt, use_container_width=True)
    if not bundle.risk_features.empty:
        rf = bundle.risk_features.tail(48)
        if "price_stress_score" in rf.columns and "logistics_risk_score" in rf.columns:
            col_c = "geopolitical_risk_score" if "geopolitical_risk_score" in rf.columns else None
            fig_x = px.scatter(
                rf,
                x="price_stress_score",
                y="logistics_risk_score",
                color=col_c,
                title="Price stress vs logistics risk (recent engineered months)",
            )
            fig_x.update_layout(title=dss_title("Price stress vs logistics risk (recent engineered months)"))
            _finalize_px_figure(fig_x, height=420)
            st.plotly_chart(fig_x, use_container_width=True)

    render_page_section_title("Port and ship intelligence map")
    cgl, cgr = st.columns((1, 2))
    with cgl:
        show_ports = st.checkbox("Show ports", True, key="exec_map_ports")
        show_ships = st.checkbox("Show ships", True, key="exec_map_ships")
        show_zones = st.checkbox("Conflict zones", True, key="exec_map_zones")
        show_orig = st.checkbox("Original routes", True, key="exec_map_orig")
        show_alt = st.checkbox("Suggested routes", True, key="exec_map_alt")
        only_hi = st.checkbox("Only high-risk ships", False, key="exec_map_hi")
        only_food = st.checkbox("Only food cargo", False, key="exec_map_food")
        only_eg = st.checkbox("Only Egypt-bound", True, key="exec_map_eg")
        sel_p = st.selectbox("Focus port", ["(none)"] + list(ports["port_id"].astype(str)), key="exec_sel_port")
        sel_v = st.selectbox("Focus ship", ["(none)"] + list(vessels["ship_id"].astype(str)), key="exec_sel_ship")
    with cgr:
        figm = build_maritime_intelligence_figure(
            ports,
            vessels,
            zones,
            height=560,
            show_ports=show_ports,
            show_ships=show_ships,
            show_zones=show_zones,
            show_orig_routes=show_orig,
            show_alt_routes=show_alt,
            only_high_risk_ships=only_hi,
            only_food=only_food,
            only_egypt_bound=only_eg,
        )
        if figm:
            apply_dss_layout(figm, height=560, margin=dict(t=36, b=28, l=8, r=12))
            st.plotly_chart(figm, use_container_width=True)
        else:
            st.info("Plotly unavailable for map.")

    render_page_section_title("Details panel", compact=True)
    d1, d2 = st.columns(2)
    with d1:
        if sel_p and sel_p != "(none)":
            pr = port_by_id(ports, sel_p)
            if pr is not None:
                st.markdown(f"##### {pr.get('name')}")
                st.write(
                    f"**Country:** {pr.get('country')} - **Coords:** {float(pr.get('lat')):.4f}, {float(pr.get('lon')):.4f}\n\n"
                    f"**Inbound / outbound (demo):** {int(pr.get('inbound_ships', 0))} / {int(pr.get('outbound_ships', 0))}\n\n"
                    f"**Imports:** {pr.get('main_goods_imported')} - **Exports:** {pr.get('main_goods_exported')}\n\n"
                    f"**Congestion:** {float(pr.get('congestion_level', 0)):.2f} - **Logistics risk:** {float(pr.get('logistics_risk_score', 0)):.0f} - "
                    f"**Nearby conflict risk:** {float(pr.get('nearby_conflict_risk_score', 0)):.0f}\n\n"
                    f"**Overall:** {pr.get('overall_risk_level')} - {pr.get('explanation')}"
                )
    with d2:
        if sel_v and sel_v != "(none)":
            vs = vessel_by_id(vessels, sel_v)
            if vs is not None:
                st.markdown(f"##### {vs.get('ship_name')}")
                st.write(vs.get("explainer", ""))
                ra = advisor_route_explanation(
                    str(vs.get("ship_name")),
                    str(vs.get("origin_port_id")),
                    str(vs.get("dest_port_id")),
                    str(vs.get("cargo_type")),
                    float(vs.get("route_risk_score", 0)),
                    bool(vs.get("near_risk_zone")),
                    str(vs.get("recommendation")),
                    str(vs.get("alt_route_explanation", "")),
                    _ollama_session_on(),
                    str(st.session_state.get("ollama_model", "llama3.2")),
                    str(st.session_state.get("ollama_base", "http://127.0.0.1:11434")),
                )
                st.info(ra)

    st.markdown(
        '<div class="exec-maritime-title ai-recos-heading-block"><span class="exec-maritime-title-text">'
        "AI Decision Recommendations</span></div>",
        unsafe_allow_html=True,
    )
    cards = build_ministry_recommendations(alerts, vessels)
    for c in cards[:12]:
        badge = "[P0]" if c["priority"] == "P0" else "[P1]" if c["priority"] == "P1" else "[P2]"
        with st.expander(f"{badge} {c['title']} — {c['status']}", expanded=c["priority"] in ("P0", "P1")):
            _render_ai_recommendation_detail(c)


def render_maritime_full_page(snapshot: dict[str, Any], context: dict[str, Any], key_prefix: str = "pv") -> None:
    """Dedicated Port & Vessel Intelligence page with tabs."""
    ports, vessels, zones = snapshot["ports"], snapshot["vessels"], snapshot["zones"]
    alerts = evaluate_maritime_alerts(vessels, ports, context)
    active_alerts = [a for a in alerts if a.status != AlertStatus.RESOLVED]
    resolved_alerts = [a for a in alerts if a.status == AlertStatus.RESOLVED]

    tab_map, tab_port, tab_ship, tab_route, tab_alerts = st.tabs(
        ["Map", "Port details", "Ship details", "Route recommendation", "Alerts"]
    )
    with tab_map:
        h = st.slider("Map height", 420, 900, 700, 20, key=f"{key_prefix}_mh")
        c1, c2 = st.columns((1, 3))
        with c1:
            show_ports = st.checkbox("Ports", True, key=f"{key_prefix}_p")
            show_ships_chk = st.checkbox("Ships", True, key=f"{key_prefix}_sh")
            show_zones = st.checkbox("Conflict zones", True, key=f"{key_prefix}_z")
            show_orig = st.checkbox("Original routes", True, key=f"{key_prefix}_o")
            show_alt = st.checkbox("Suggested routes", True, key=f"{key_prefix}_a")
            only_hi = st.checkbox("High-risk ships only", False, key=f"{key_prefix}_hr")
            only_food = st.checkbox("Food cargo only", False, key=f"{key_prefix}_fd")
            only_eg = st.checkbox("Egypt-bound only", False, key=f"{key_prefix}_eg")
        with c2:
            figm = build_maritime_intelligence_figure(
                ports,
                vessels,
                zones,
                height=h,
                show_ports=show_ports,
                show_ships=show_ships_chk,
                show_zones=show_zones,
                show_orig_routes=show_orig,
                show_alt_routes=show_alt,
                only_high_risk_ships=only_hi,
                only_food=only_food,
                only_egypt_bound=only_eg,
            )
            if figm:
                apply_dss_layout(figm, height=h, margin=dict(t=36, b=28, l=8, r=12))
                st.plotly_chart(figm, use_container_width=True)
            else:
                st.info("Plotly unavailable.")

    with tab_port:
        opts_p = ["(none)"] + list(ports["port_id"].astype(str))
        pid = st.selectbox("Select port", opts_p, key=f"{key_prefix}_port_sb")
        pr = port_by_id(ports, pid)
        if pr is not None:
            tbl = _series_to_attribute_table(pr, _PORT_DETAIL_FIELDS)
            st.dataframe(
                tbl,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Attribute": st.column_config.TextColumn("Attribute", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="large"),
                },
            )

    with tab_ship:
        opts_v = ["(none)"] + list(vessels["ship_id"].astype(str))
        vid = st.selectbox("Select ship", opts_v, key=f"{key_prefix}_vessel_sb")
        vs = vessel_by_id(vessels, vid)
        if vs is not None:
            tblv = _series_to_attribute_table(vs, _VESSEL_DETAIL_FIELDS)
            st.dataframe(
                tblv,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Attribute": st.column_config.TextColumn("Attribute", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="large"),
                },
            )
            st.info(
                advisor_route_explanation(
                    str(vs.get("ship_name")),
                    str(vs.get("origin_port_id")),
                    str(vs.get("dest_port_id")),
                    str(vs.get("cargo_type")),
                    float(vs.get("route_risk_score", 0)),
                    bool(vs.get("near_risk_zone")),
                    str(vs.get("recommendation")),
                    str(vs.get("alt_route_explanation", "")),
                    _ollama_session_on(),
                    str(st.session_state.get("ollama_model", "llama3.2")),
                    str(st.session_state.get("ollama_base", "http://127.0.0.1:11434")),
                )
            )

    with tab_route:
        opts_v2 = ["(none)"] + list(vessels["ship_id"].astype(str))
        rvid = st.selectbox("Ship for route compare", opts_v2, key=f"{key_prefix}_route_v")
        vs2 = vessel_by_id(vessels, rvid)
        if vs2 is not None:
            rc = vs2.get("route_compare")
            if isinstance(rc, dict):
                st.table(pd.DataFrame([rc]))
            st.write(str(vs2.get("alt_route_explanation", "")))

    with tab_alerts:
        if active_alerts:
            st.dataframe(alerts_to_dataframe(active_alerts), use_container_width=True)
        else:
            st.success("No active maritime alerts.")
        if resolved_alerts:
            st.caption("Resolved / cleared")
            st.dataframe(alerts_to_dataframe(resolved_alerts), use_container_width=True)


def render_gis_maritime_panel(snapshot: dict[str, Any], context: dict[str, Any]) -> None:
    del context
    ports, vessels, zones = snapshot["ports"], snapshot["vessels"], snapshot["zones"]
    c1, c2 = st.columns((1, 3))
    with c1:
        hp = st.checkbox("Show ports", True, key="gis_mi_ports")
        hs = st.checkbox("Show ships", True, key="gis_mi_ships")
        hz = st.checkbox("Risk zones", True, key="gis_mi_zones")
        ho = st.checkbox("Original routes", True, key="gis_mi_orig")
        ha = st.checkbox("Suggested routes", True, key="gis_mi_alt")
    with c2:
        h = st.slider("Intelligence map height", 480, 920, 680, 20, key="gis_mi_h")
        fig = build_maritime_intelligence_figure(
            ports,
            vessels,
            zones,
            height=h,
            show_ports=hp,
            show_ships=hs,
            show_zones=hz,
            show_orig_routes=ho,
            show_alt_routes=ha,
        )
        if fig:
            apply_dss_layout(fig, height=h, margin=dict(t=36, b=28, l=8, r=12))
            st.plotly_chart(fig, use_container_width=True)
