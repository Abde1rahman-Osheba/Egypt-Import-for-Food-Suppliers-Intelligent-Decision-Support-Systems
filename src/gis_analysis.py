"""Lightweight GIS-style overlay risk surfaces (Plotly only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

_RNG = np.random.default_rng(42)

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


EGYPT_CENTER = {"lat": 26.8, "lon": 30.8}


@dataclass
class GeoPoint:
    name: str
    lat: float
    lon: float
    layer: str
    weight: float


def supplier_markers() -> list[GeoPoint]:
    return [
        GeoPoint("Black Sea — Odessa", 46.48, 30.73, "supplier", 0.9),
        GeoPoint("Novorossiysk", 44.72, 37.77, "supplier", 0.85),
        GeoPoint("Constanța", 44.16, 28.65, "supplier", 0.7),
        GeoPoint("Rotterdam Hub", 51.92, 4.48, "supplier", 0.5),
        GeoPoint("New Orleans", 29.95, -90.07, "supplier", 0.45),
    ]


def egypt_ports_from_data(port_df: pd.DataFrame) -> list[GeoPoint]:
    if port_df.empty or "country" not in port_df.columns:
        return [
            GeoPoint("Alexandria", 31.2, 29.92, "port", 0.8),
            GeoPoint("Port Said", 31.26, 32.31, "port", 0.75),
            GeoPoint("Damietta", 31.42, 31.81, "port", 0.65),
        ]
    eg = port_df[port_df.get("ISO3", "") == "EGY"]
    if eg.empty and "country" in port_df.columns:
        eg = port_df[port_df["country"].astype(str).str.contains("Egypt", case=False, na=False)]
    if eg.empty:
        return supplier_markers()[:3]
    eg = eg.drop_duplicates(subset=["portname"])
    pts: list[GeoPoint] = []
    for _, row in eg.head(8).iterrows():
        name = str(row.get("portname", "Port"))
        pts.append(
            GeoPoint(
                name,
                EGYPT_CENTER["lat"] + _RNG.uniform(-2, 2),
                EGYPT_CENTER["lon"] + _RNG.uniform(-2, 2),
                "port",
                0.7,
            )
        )
    return pts or [GeoPoint("Alexandria", 31.2, 29.92, "port", 0.8)]


def conflict_scatter_from_gdelt(gdelt: pd.DataFrame, max_points: int = 80) -> pd.DataFrame:
    if gdelt.empty:
        return pd.DataFrame(columns=["lat", "lon", "intensity", "label"])
    coords = {
        "EG": (26.8, 30.8),
        "UA": (48.38, 31.17),
        "RS": (55.75, 37.62),
        "SY": (33.51, 36.29),
        "US": (39.8, -98.5),
    }
    rows = []
    sub = gdelt.sort_values("year").groupby("CountryCode", as_index=False).tail(3)
    for _, r in sub.iterrows():
        code = str(r.get("CountryCode", ""))
        if code not in coords:
            continue
        lat, lon = coords[code]
        rows.append(
            {
                "lat": lat + _RNG.uniform(-0.8, 0.8),
                "lon": lon + _RNG.uniform(-0.8, 0.8),
                "intensity": float(r.get("conflict_intensity", 0) or 0),
                "label": f"{code} {int(r.get('year', 0))}",
            }
        )
    return pd.DataFrame(rows[:max_points])


def composite_grid(
    conflict_df: pd.DataFrame,
    port_risk: float = 0.5,
    supplier_w: float = 0.6,
    route_w: float = 0.4,
) -> tuple[np.ndarray, list[float], list[float]]:
    lats = np.linspace(24, 34, 18)
    lons = np.linspace(24, 40, 22)
    grid = np.zeros((len(lats), len(lons)))
    conflict_layer = 0.3
    if not conflict_df.empty and conflict_df["intensity"].max() > 0:
        conflict_layer = min(
            1.0, conflict_df["intensity"].fillna(0).max() / (conflict_df["intensity"].max() + 1e-6)
        )
    port_layer = port_risk
    supplier_layer = supplier_w
    route_layer = route_w
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            d_eg = np.hypot(la - EGYPT_CENTER["lat"], lo - EGYPT_CENTER["lon"])
            route_factor = np.clip(1.0 - d_eg / 25.0, 0, 1)
            grid[i, j] = (
                0.35 * conflict_layer
                + 0.30 * port_layer
                + 0.20 * supplier_layer
                + 0.15 * route_layer * route_factor
            )
    return grid, list(lats), list(lons)


def build_overlay_figure(
    port_df: pd.DataFrame,
    gdelt_df: pd.DataFrame,
) -> Optional[Any]:
    if go is None:
        return None
    conf = conflict_scatter_from_gdelt(gdelt_df)
    grid, glat, glon = composite_grid(conf)
    lon_grid, lat_grid = np.meshgrid(glon, glat)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            z=grid,
            x=glon,
            y=glat,
            colorscale="Reds",
            name="Composite spatial risk",
            hovertemplate="Risk %{z:.2f}<extra></extra>",
            showscale=True,
        )
    )
    for p in supplier_markers():
        fig.add_trace(
            go.Scattergeo(
                lat=[p.lat],
                lon=[p.lon],
                text=[p.name],
                mode="markers+text",
                marker=dict(size=10, color="royalblue"),
                name="Supplier region",
            )
        )
    for p in egypt_ports_from_data(port_df):
        fig.add_trace(
            go.Scattergeo(
                lat=[p.lat],
                lon=[p.lon],
                text=[p.name],
                mode="markers",
                marker=dict(size=11, color="darkorange", symbol="diamond"),
                name="Egypt port",
            )
        )
    if not conf.empty:
        fig.add_trace(
            go.Scattergeo(
                lat=conf["lat"],
                lon=conf["lon"],
                text=conf["label"],
                mode="markers",
                marker=dict(size=8, color="crimson"),
                name="Conflict layer",
            )
        )

    fig.add_trace(
        go.Scattergeo(
            lat=[EGYPT_CENTER["lat"], 46.48],
            lon=[EGYPT_CENTER["lon"], 30.73],
            mode="lines",
            line=dict(width=2, color="navy"),
            name="Example grain route",
        )
    )

    fig.update_layout(
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(230, 230, 230)",
            lataxis_range=[20, 52],
            lonaxis_range=[-15, 45],
        ),
        height=640,
        margin=dict(l=0, r=0, t=24, b=0),
    )
    return fig
