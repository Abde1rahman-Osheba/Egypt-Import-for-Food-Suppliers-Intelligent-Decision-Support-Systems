"""Shipping / route disruption metrics: port-derived proxy + optional tracker CSV / API stub."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils import safe_coerce_numeric

try:
    import requests
except ImportError:
    requests = None


def load_shipping_metrics_csv(path: Path) -> pd.DataFrame:
    """
    Optional tracker export columns:
    month|date, route_congestion_index (0-100), delay_days_avg, berth_utilization
    """
    if not path.is_file():
        return pd.DataFrame(
            columns=["month", "shipping_congestion", "shipping_delay_days", "shipping_utilization"]
        )
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    date_col = next((c for c in df.columns if c in ("month", "date", "period")), None)
    if not date_col:
        return pd.DataFrame(
            columns=["month", "shipping_congestion", "shipping_delay_days", "shipping_utilization"]
        )
    out = pd.DataFrame()
    out["month"] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    col_map = [
        ("route_congestion_index", "shipping_congestion"),
        ("congestion", "shipping_congestion"),
        ("delay_days_avg", "shipping_delay_days"),
        ("delay", "shipping_delay_days"),
        ("berth_utilization", "shipping_utilization"),
        ("utilization", "shipping_utilization"),
    ]
    for col, tgt in col_map:
        if col in df.columns:
            out[tgt] = safe_coerce_numeric(df[col])
    out = out.dropna(subset=["month"])
    out = out.groupby("month", as_index=False).mean(numeric_only=True)
    return out.sort_values("month").reset_index(drop=True)


def port_derived_shipping_stress(port_monthly: pd.DataFrame) -> pd.DataFrame:
    """Proxy: Egypt import + Black Sea export pressure vs local trend → 0–100 stress."""
    if port_monthly.empty:
        return pd.DataFrame(
            columns=["month", "shipping_stress_score", "eg_import_ton_proxy", "bs_export_ton_proxy"]
        )
    eg = port_monthly[port_monthly["ISO3"] == "EGY"].groupby("month", as_index=False)[["import", "portcalls"]].sum()
    eg = eg.rename(columns={"import": "eg_import_ton_proxy"})
    bs_iso = {"UKR", "RUS"}
    bs = port_monthly[port_monthly["ISO3"].isin(bs_iso)].groupby("month", as_index=False)["export"].sum()
    bs = bs.rename(columns={"export": "bs_export_ton_proxy"})
    m = eg.merge(bs, on="month", how="outer").sort_values("month")
    for c in ("eg_import_ton_proxy", "bs_export_ton_proxy"):
        if c not in m.columns:
            m[c] = np.nan
    egi = m["eg_import_ton_proxy"].astype(float).ffill().bfill()
    bse = m["bs_export_ton_proxy"].astype(float).ffill().bfill()
    roll_e = egi.rolling(6, min_periods=2).median()
    roll_b = bse.rolling(6, min_periods=2).median()
    shortfall_e = ((roll_e - egi) / roll_e.replace(0, np.nan)).clip(lower=0).fillna(0)
    shortfall_b = ((roll_b - bse) / roll_b.replace(0, np.nan)).clip(lower=0).fillna(0)
    raw = 100 * (0.55 * shortfall_e + 0.45 * shortfall_b)
    m["shipping_stress_score"] = np.clip(raw, 0, 100)
    return m[["month", "shipping_stress_score", "eg_import_ton_proxy", "bs_export_ton_proxy"]]


def merge_shipping_layers(
    port_derived: pd.DataFrame,
    tracker: pd.DataFrame,
) -> pd.DataFrame:
    if port_derived.empty and tracker.empty:
        return pd.DataFrame(
            columns=["month", "shipping_stress_score", "shipping_congestion", "shipping_delay_days"]
        )
    if tracker.empty:
        return port_derived
    if port_derived.empty:
        out = tracker.copy()
        out["shipping_stress_score"] = out.get("shipping_congestion", pd.Series(0, index=out.index))
        return out
    m = port_derived.merge(tracker, on="month", how="outer").sort_values("month")
    base = m["shipping_stress_score"].fillna(0)
    if "shipping_congestion" in m.columns:
        base = 0.6 * base + 0.4 * m["shipping_congestion"].fillna(0)
    m["shipping_stress_score_composite"] = np.clip(base, 0, 100)
    return m


def fetch_optional_api_snapshot(api_url: Optional[str] = None, timeout: int = 12) -> dict[str, Any]:
    """
    Hook for MarineTraffic / VesselFinder / in-house APIs.
    Set SHIPPING_TRACKER_API_URL in the environment. Returns {} if unset or on failure.
    """
    url_ = api_url or os.environ.get("SHIPPING_TRACKER_API_URL", "").strip()
    if not url_ or requests is None:
        return {}
    try:
        r = requests.get(url_, timeout=timeout)
        if r.ok and r.headers.get("content-type", "").startswith("application/json"):
            return dict(r.json()) if isinstance(r.json(), dict) else {"raw": r.text[:500]}
    except Exception:
        return {}
    return {}


def expand_shipping_to_wheat_index(
    shipping: pd.DataFrame,
    wheat_months: pd.Series,
) -> pd.DataFrame:
    """Upsample sparse tracker rows to every wheat month (ffill / bfill)."""
    if shipping.empty or wheat_months is None or len(wheat_months) == 0:
        return shipping
    idx = pd.period_range(
        pd.Timestamp(wheat_months.min()),
        pd.Timestamp(wheat_months.max()),
        freq="M",
    ).to_timestamp()
    s = shipping.drop_duplicates(subset=["month"]).set_index("month")
    s = s.reindex(idx).ffill().bfill().reset_index(names="month")
    return s
