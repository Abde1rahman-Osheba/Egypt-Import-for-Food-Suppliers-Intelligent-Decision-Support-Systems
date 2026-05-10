"""FAO / FAOSTAT-compatible price loading for Egypt (single series or multi-commodity catalog)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import safe_coerce_numeric


def _norm_tokens(label: str) -> set[str]:
    s = re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()
    return {t for t in s.split() if len(t) >= 2}


def _faostat_subset_to_monthly(
    sub: pd.DataFrame,
    year_col: str,
    val_col: str,
    month_col: Optional[str],
) -> pd.DataFrame:
    sub = sub.copy()
    sub["_y"] = safe_coerce_numeric(sub[year_col])
    sub["_v"] = safe_coerce_numeric(sub[val_col])
    if month_col and month_col in sub.columns:
        sub["_m"] = pd.to_datetime(
            sub[year_col].astype(str)
            + "-"
            + sub[month_col].astype(str).str[:2].str.zfill(2)
            + "-01",
            errors="coerce",
        )
    else:
        sub["_m"] = pd.to_datetime(dict(year=sub["_y"].astype(int), month=1, day=1))
    sub = sub.dropna(subset=["_m", "_v"])
    sub["month"] = sub["_m"].dt.to_period("M").dt.to_timestamp()
    out = sub.groupby("month", as_index=False)["_v"].mean().rename(columns={"_v": "price_usd"})
    out["source"] = "FAO FAOSTAT"
    return out.sort_values("month").reset_index(drop=True)


def _simple_table_to_monthly(df: pd.DataFrame, date_col: str, val_col: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "month": pd.to_datetime(df[date_col], errors="coerce"),
            "price_usd": safe_coerce_numeric(df[val_col]),
        }
    )
    out = out.dropna(subset=["month", "price_usd"])
    out["month"] = out["month"].dt.to_period("M").dt.to_timestamp()
    out = out.groupby("month", as_index=False)["price_usd"].mean()
    out["source"] = "FAO (user file)"
    return out.sort_values("month").reset_index(drop=True)


def load_fao_commodities_catalog(
    path: Path,
    area_substr: str = "Egypt",
) -> dict[str, pd.DataFrame]:
    """
    Load zero or many FAO monthly series from one CSV.

    Supported shapes:

    - **FAOSTAT long**: `Area`, `Item`, `Value`, `Year` (+ optional `Months`/`Month`).
      Every distinct `Item` for the area becomes one series keyed by Item string.

    - **Simple long**: `month`/`date`/`period` + `price_usd` or `Value`
      plus `Item`/`commodity`/`product` with **multiple** distinct values → one series per commodity.

    - **Simple single-series** (no item column): one monthly series keyed by the CSV stem
      (`fao_vegetable_oil.csv` → name from filename).
    """
    if not path.is_file():
        return {}

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    area_col = next((c for c in df.columns if c.lower() == "area"), None)
    item_col = next((c for c in df.columns if c.lower() == "item"), None)
    val_col = next((c for c in df.columns if c.lower() == "value"), None)
    year_col = next((c for c in df.columns if c.lower() == "year"), None)
    month_col = next((c for c in df.columns if c.lower() in ("months", "month", "month name")), None)

    out: dict[str, pd.DataFrame] = {}
    if area_col and item_col and val_col and year_col:
        sub_all = df[df[area_col].astype(str).str.contains(area_substr, case=False, na=False)].copy()
        if not sub_all.empty:
            for item in sub_all[item_col].dropna().unique():
                lab = str(item).strip()
                if not lab:
                    continue
                sub = sub_all[sub_all[item_col] == item].copy()
                ser = _faostat_subset_to_monthly(sub, year_col, val_col, month_col)
                if not ser.empty:
                    out[lab] = ser
        if out:
            return out

    item_alt = next((c for c in df.columns if str(c).lower() in ("item", "commodity", "product")), None)
    has_value_named = any(str(c).lower() == "value" for c in df.columns)
    pcol = "price_usd" if "price_usd" in df.columns else ("Value" if "Value" in df.columns else None)
    if pcol is None and has_value_named:
        pcol = next(c for c in df.columns if str(c).lower() == "value")
    dcol = next((c for c in df.columns if c.lower() in ("month", "date", "period")), None)

    if item_alt and dcol and pcol:
        for raw in df[item_alt].dropna().unique():
            lab = str(raw).strip()
            if not lab:
                continue
            sub = df[df[item_alt].astype(str) == lab]
            ser = _simple_table_to_monthly(sub, dcol, pcol)
            if not ser.empty:
                out[lab] = ser
        if out:
            return out

    if dcol and pcol:
        ser = _simple_table_to_monthly(df, dcol, pcol)
        if not ser.empty:
            key = path.stem.replace("_", " ").strip() or "FAO_series"
            return {key.title(): ser}

    return out


def fao_catalog_match_score(wfp_commodity_label: str, fao_item_key: str) -> int:
    """Higher = better match between WFP retail label and FAOSTAT Item name."""
    wt = _norm_tokens(wfp_commodity_label)
    fk = _norm_tokens(fao_item_key)
    if not wt or not fk:
        return 0
    inter = wt & fk
    if inter:
        return 100 + 10 * len(inter) + max(len(t) for t in inter)
    low_w = str(wfp_commodity_label).lower()
    low_f = str(fao_item_key).lower()
    for t in wt:
        if len(t) >= 3 and t in low_f:
            return 50 + len(t)
    for t in fk:
        if len(t) >= 3 and t in low_w:
            return 40 + len(t)
    return 0


def find_best_fao_series(
    wfp_commodity_label: str,
    fao_catalog: dict[str, pd.DataFrame],
    *,
    min_score: int = 50,
) -> Optional[pd.DataFrame]:
    """Pick the catalog series best aligned with the WFP commodity name (substring / token overlap)."""
    if not fao_catalog:
        return None
    best_key: Optional[str] = None
    best_s = -1
    for key in fao_catalog:
        s = fao_catalog_match_score(wfp_commodity_label, key)
        if s > best_s:
            best_s = s
            best_key = key
    if best_key is None or best_s < min_score:
        return None
    return fao_catalog[best_key].copy()


def load_fao_wheat_egypt(
    path: Path,
    area_substr: str = "Egypt",
    item_substr: str = "wheat",
) -> pd.DataFrame:
    """
    Load FAOSTAT or hand-aligned CSV. Accepted shapes:
    - Simple: date|month, price_usd (or value)
    - FAOSTAT bulk: Area, Item, Element, Year + optional Months, Value
    """
    if not path.is_file():
        return pd.DataFrame(columns=["month", "price_usd", "source"])

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    has_value_col = any(str(c).lower() == "value" for c in df.columns)
    if "price_usd" in df.columns or has_value_col:
        if "price_usd" in df.columns:
            col = "price_usd"
        else:
            col = next(c for c in df.columns if str(c).lower() == "value")
        date_col = next(
            (c for c in df.columns if c.lower() in ("month", "date", "period")),
            None,
        )
        if date_col:
            ser = _simple_table_to_monthly(df, date_col, col)
            return ser

    area_col = next((c for c in df.columns if c.lower() == "area"), None)
    item_col = next((c for c in df.columns if c.lower() == "item"), None)
    val_col = next((c for c in df.columns if c.lower() == "value"), None)
    year_col = next((c for c in df.columns if c.lower() == "year"), None)
    month_col = next((c for c in df.columns if c.lower() in ("months", "month", "month name")), None)

    if area_col and item_col and val_col and year_col:
        sub = df[
            df[area_col].astype(str).str.contains(area_substr, case=False, na=False)
            & df[item_col].astype(str).str.contains(item_substr, case=False, na=False)
        ].copy()
        if sub.empty:
            return pd.DataFrame(columns=["month", "price_usd", "source"])
        return _faostat_subset_to_monthly(sub, year_col, val_col, month_col)

    return pd.DataFrame(columns=["month", "price_usd", "source"])


def merge_wfp_fao(
    wfp_monthly: pd.DataFrame,
    fao_monthly: pd.DataFrame,
    prefer: str = "blend",
    commodity_label: str | None = None,
) -> pd.DataFrame:
    """Combine WFP retail proxy with FAO / official series."""
    if wfp_monthly.empty and fao_monthly.empty:
        return pd.DataFrame(columns=["month", "price_egp", "price_usd", "commodity", "price_usd_wfp", "price_usd_fao"])
    if wfp_monthly.empty:
        o = fao_monthly.rename(columns={"price_usd": "price_usd_fao"}).copy()
        o["price_usd"] = o["price_usd_fao"]
        o["price_egp"] = np.nan
        o["commodity"] = commodity_label or "FAO commodity"
        return o
    if fao_monthly.empty:
        out = wfp_monthly.copy()
        out["price_usd_wfp"] = out["price_usd"]
        out["price_usd_fao"] = float("nan")
        lab = commodity_label
        if lab and "commodity" in out.columns:
            out["commodity"] = lab
        elif lab:
            out["commodity"] = lab
        return out

    w = wfp_monthly.copy()
    w["price_usd_wfp"] = w["price_usd"]
    f = fao_monthly.rename(columns={"price_usd": "price_usd_fao"})[["month", "price_usd_fao"]]
    m = w.merge(f, on="month", how="outer").sort_values("month")
    if prefer == "fao":
        m["price_usd"] = m["price_usd_fao"].fillna(m["price_usd_wfp"])
    elif prefer == "wfp":
        m["price_usd"] = m["price_usd_wfp"].fillna(m["price_usd_fao"])
    else:
        m["price_usd"] = np.nan
        mask = m["price_usd_wfp"].notna() & m["price_usd_fao"].notna()
        m.loc[mask, "price_usd"] = (
            m.loc[mask, "price_usd_wfp"] * 0.45 + m.loc[mask, "price_usd_fao"] * 0.55
        )
        m["price_usd"] = m["price_usd"].combine_first(m["price_usd_wfp"]).combine_first(m["price_usd_fao"])

    label = commodity_label or (str(wfp_monthly["commodity"].iloc[0]) if "commodity" in wfp_monthly.columns else None)
    if label:
        m["commodity"] = label
    elif "commodity" in w.columns:
        m["commodity"] = w["commodity"].iloc[0] if len(w) else None

    if "price_egp" not in m.columns:
        m["price_egp"] = np.nan

    return m.reset_index(drop=True)

