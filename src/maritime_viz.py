"""Plotly Scattergeo: ports, synthetic vessels, routes, risk halos (no Mapbox token)."""

from __future__ import annotations

import ast
import math
from typing import Any, Optional

import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from src.conflict_zones import ConflictZone


def _risk_color(score: float) -> str:
    if score >= 75:
        return "#dc2626"
    if score >= 55:
        return "#ea580c"
    if score >= 40:
        return "#ca8a04"
    return "#16a34a"


def zone_halo_polygon(lat: float, lon: float, radius_km: float, n: int = 40) -> tuple[list[float], list[float]]:
    """Rough circle for dashboard demo (not survey-grade)."""
    lats, lons = [], []
    for i in range(n + 1):
        ang = 2 * math.pi * i / n
        dlat = (radius_km / 111.0) * math.cos(ang)
        dlon = (radius_km / (111.0 * max(0.2, math.cos(math.radians(lat))))) * math.sin(ang)
        lats.append(lat + dlat)
        lons.append(lon + dlon)
    return lats, lons


def build_maritime_intelligence_figure(
    ports: pd.DataFrame,
    vessels: pd.DataFrame,
    zones: list[ConflictZone],
    height: int = 720,
    *,
    show_ports: bool = True,
    show_ships: bool = True,
    show_zones: bool = True,
    show_orig_routes: bool = True,
    show_alt_routes: bool = True,
    only_high_risk_ships: bool = False,
    only_food: bool = False,
    only_egypt_bound: bool = False,
) -> Optional[Any]:
    if go is None:
        return None

    fig = go.Figure()

    if show_zones:
        for z in zones:
            zlats, zlons = zone_halo_polygon(z.lat, z.lon, z.radius_km)
            fig.add_trace(
                go.Scattergeo(
                    lat=zlats,
                    lon=zlons,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(220,38,38,0.15)",
                    line=dict(color="rgba(220,38,38,0.7)", width=1),
                    name=z.name,
                    hovertemplate=f"<b>{z.name}</b><br>severity={z.severity:.0f}<br>{z.description}<extra></extra>",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scattergeo(
                    lat=[z.lat],
                    lon=[z.lon],
                    mode="markers",
                    marker=dict(size=max(6, min(26, z.severity / 5)), color="#dc2626", opacity=0.85, symbol="circle-open"),
                    name=z.zone_id,
                    hovertemplate=f"{z.name} ({z.risk_type})<extra></extra>",
                    showlegend=False,
                )
            )

    vsub = vessels.copy()
    if only_high_risk_ships and not vsub.empty:
        vsub = vsub[vsub["route_risk_score"].astype(float) >= 60]
    if only_food and not vsub.empty:
        foods = vsub["cargo_type"].astype(str).str.lower().str.contains("wheat|maize|rice|sugar|oil|flour|food")
        vsub = vsub[foods]
    if only_egypt_bound and not vsub.empty:
        vsub = vsub[vsub["dest_port_id"].astype(str).str.startswith("EG_")]

    port_idx = ports.set_index("port_id") if not ports.empty and "port_id" in ports.columns else None

    if show_ports and port_idx is not None:
        fig.add_trace(
            go.Scattergeo(
                lat=ports["lat"],
                lon=ports["lon"],
                text=ports["name"] + "<br>" + ports["country"],
                mode="markers",
                marker=dict(size=11, color="#2563eb", symbol="diamond"),
                name="Ports",
                hovertemplate="%{text}<extra></extra>",
            )
        )

    if show_orig_routes and show_ships and port_idx is not None and not vsub.empty:
        for _, s in vsub.iterrows():
            try:
                o = port_idx.loc[str(s["origin_port_id"])]
                d = port_idx.loc[str(s["dest_port_id"])]
            except Exception:
                continue
            risk = float(s.get("route_risk_score", 30))
            col = _risk_color(risk)
            dash = "dash" if risk >= 55 else "solid"
            fig.add_trace(
                go.Scattergeo(
                    lat=[float(o["lat"]), float(d["lat"])],
                    lon=[float(o["lon"]), float(d["lon"])],
                    mode="lines",
                    line=dict(width=2, color=col, dash=dash),
                    name=f"Route {s.get('ship_id')}",
                    showlegend=False,
                    hovertemplate=f"{s.get('ship_name')} | route risk={risk:.0f}<extra></extra>",
                )
            )

    if show_alt_routes and not vsub.empty:
        for _, s in vsub.iterrows():
            try:
                alats = ast.literal_eval(str(s.get("alt_route_lats", "[]")))
                alons = ast.literal_eval(str(s.get("alt_route_lons", "[]")))
                if len(alats) >= 2:
                    fig.add_trace(
                        go.Scattergeo(
                            lat=alats,
                            lon=alons,
                            mode="lines",
                            line=dict(width=2, color="#16a34a", dash="dashdot"),
                            name=f"Alt {s.get('ship_id')}",
                            showlegend=False,
                            hovertemplate=f"Suggested route {s.get('ship_name')}<extra></extra>",
                        )
                    )
            except Exception:
                continue

    if show_ships and not vsub.empty:
        rcol = [_risk_color(float(x)) for x in vsub["route_risk_score"]]
        fig.add_trace(
            go.Scattergeo(
                lat=vsub["current_lat"],
                lon=vsub["current_lon"],
                text=vsub["ship_name"] + " | " + vsub["cargo_type"].astype(str),
                mode="markers",
                marker=dict(size=12, color=rcol, symbol="circle"),
                name="Vessels (demo)",
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Port and vessel intelligence (Scattergeo, offline demo)",
        height=height,
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(235, 235, 235)",
            coastlinecolor="rgb(180,180,180)",
            lataxis_range=[15, 55],
            lonaxis_range=[-12, 52],
        ),
    )
    return fig
