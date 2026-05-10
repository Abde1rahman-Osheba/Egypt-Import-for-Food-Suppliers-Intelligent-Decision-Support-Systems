"""Load and harmonize course datasets for DSS models and dashboards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.fao_prices import (
    find_best_fao_series,
    load_fao_commodities_catalog,
    load_fao_wheat_egypt,
    merge_wfp_fao,
)
from src.nlp_conflict_index import NoveltyCorrelationResult, build_news_pipeline
from src.shipping_tracker import (
    expand_shipping_to_wheat_index,
    load_shipping_metrics_csv,
    merge_shipping_layers,
    port_derived_shipping_stress,
)
from src.utils import min_max_scale, project_root, safe_coerce_numeric


GDELT_FOCUS = {"EG", "UA", "UP", "RS", "US", "RO", "TU", "SY", "SU", "I"}
PORT_FOCUS_ISO3 = {"EGY", "UKR", "RUS", "USA", "ROU", "TUR", "SYR", "SDN"}

# Daily port CSV: trade/cargo segments (aligned import/export/portcalls_* columns).
PORT_CARGO_SEGMENTS: list[tuple[str, str, str, str]] = [
    ("All vessels (aggregate)", "portcalls", "import", "export"),
    ("Container", "portcalls_container", "import_container", "export_container"),
    ("Dry bulk", "portcalls_dry_bulk", "import_dry_bulk", "export_dry_bulk"),
    ("General cargo", "portcalls_general_cargo", "import_general_cargo", "export_general_cargo"),
    ("RoRo", "portcalls_roro", "import_roro", "export_roro"),
    ("Tanker", "portcalls_tanker", "import_tanker", "export_tanker"),
    ("Other cargo", "portcalls_cargo", "import_cargo", "export_cargo"),
]


@dataclass
class TrainingDataBundle:
    wheat_prices: pd.DataFrame
    port_monthly: pd.DataFrame
    gdelt_yearly: pd.DataFrame
    risk_features: pd.DataFrame
    master_monthly: pd.DataFrame
    fao_wheat: pd.DataFrame
    fao_catalog: dict[str, pd.DataFrame]
    shipping_monthly: pd.DataFrame
    news_headlines: pd.DataFrame
    news_monthly: pd.DataFrame
    novelty: NoveltyCorrelationResult
    price_blend_note: str


def _parse_wfp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).lstrip("#") for c in df.columns]
    if "date" in df.columns:
        mask = df["date"].astype(str).str.startswith("#")
        df = df[~mask].reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in ("price", "usdprice"):
        if c in df.columns:
            df[c] = safe_coerce_numeric(df[c])
    return df


def _prefer_national_market(sub: pd.DataFrame) -> pd.DataFrame:
    if "market" not in sub.columns:
        return sub
    nat = sub[sub["market"].astype(str).str.contains("National", case=False, na=False)]
    return nat if not nat.empty else sub


def _monthly_commodity_agg(sub: pd.DataFrame, label: str) -> pd.DataFrame:
    """Aggregate one commodity sub-table to monthly mean EGP/USD."""
    if sub.empty or "date" not in sub.columns:
        return pd.DataFrame(columns=["month", "price_egp", "price_usd", "commodity"])
    m = _prefer_national_market(sub).copy()
    m["month"] = m["date"].dt.to_period("M").dt.to_timestamp()
    g = m.groupby("month", as_index=False).agg(
        price_egp=("price", "mean"),
        price_usd=("usdprice", "mean"),
    )
    g["commodity"] = label.strip()
    return g.sort_values("month").reset_index(drop=True)


def load_all_wfp_commodities(wfp_path: Path) -> dict[str, pd.DataFrame]:
    """
    One monthly series per distinct commodity string in the WFP Egypt file.
    National market is preferred when that column exists and rows remain.
    """
    raw = _parse_wfp(wfp_path)
    if raw.empty or "commodity" not in raw.columns:
        return {}
    out: dict[str, pd.DataFrame] = {}
    for com in sorted(raw["commodity"].astype(str).unique(), key=str.lower):
        lab = str(com).strip()
        if not lab:
            continue
        m = raw[raw["commodity"].astype(str) == com].copy()
        g = _monthly_commodity_agg(m, lab)
        if not g.empty and (g["price_usd"].notna().any() or g["price_egp"].notna().any()):
            out[lab] = g
    return out


def load_wheat_series(wfp_path: Path) -> pd.DataFrame:
    raw = _parse_wfp(wfp_path)
    if raw.empty:
        return pd.DataFrame(
            columns=["month", "price_egp", "price_usd", "commodity"]
        )
    com = raw["commodity"].astype(str)
    m = raw[com.str.contains("Wheat", case=False, na=False)].copy()
    m = _prefer_national_market(m)
    if m.empty:
        m = raw[com.str.contains("Wheat", case=False, na=False)].copy()
    g = _monthly_commodity_agg(m, "Wheat flour")
    return g


def load_ports_monthly(port_path: Path, chunksize: int = 400_000) -> pd.DataFrame:
    usecols = ["date", "ISO3", "portcalls", "import", "export", "portname", "country"]
    parts: list[pd.DataFrame] = []
    try:
        reader = pd.read_csv(
            port_path,
            usecols=lambda c: c in usecols,
            chunksize=chunksize,
            low_memory=True,
        )
    except Exception:
        return pd.DataFrame(
            columns=["month", "ISO3", "portcalls", "import", "export"]
        )

    for chunk in reader:
        sub = chunk[chunk["ISO3"].isin(PORT_FOCUS_ISO3)].copy()
        if sub.empty:
            continue
        sub["date"] = pd.to_datetime(sub["date"], errors="coerce", utc=True)
        sub["month"] = sub["date"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
        for c in ("portcalls", "import", "export"):
            if c in sub.columns:
                sub[c] = safe_coerce_numeric(sub[c])
        parts.append(sub)

    if not parts:
        return pd.DataFrame(
            columns=["month", "ISO3", "portcalls", "import", "export"]
        )
    all_df = pd.concat(parts, ignore_index=True)
    agg = all_df.groupby(["month", "ISO3"], as_index=False)[
        ["portcalls", "import", "export"]
    ].sum(min_count=1)
    return agg.sort_values(["ISO3", "month"]).reset_index(drop=True)


def _port_segment_aggregate_columns() -> list[str]:
    cols: list[str] = []
    for _lab, pc, pi, px in PORT_CARGO_SEGMENTS:
        for c in (pc, pi, px):
            if c not in cols:
                cols.append(c)
    return cols


def load_ports_monthly_by_cargo(port_path: Path, chunksize: int = 400_000) -> pd.DataFrame:
    """
    Monthly port activity summed by ISO3 **and** cargo/commodity segments from the UNCTAD-style CSV
    (`portcalls_*`, `import_*`, `export_*`). Used for exploratory charts; pipeline risk still uses aggregates.
    """
    base_cols = {"date", "ISO3"}
    seg_cols = set(_port_segment_aggregate_columns())
    usecols_list = sorted(base_cols | seg_cols)
    parts: list[pd.DataFrame] = []
    try:
        reader = pd.read_csv(
            port_path,
            usecols=lambda c: c in usecols_list,
            chunksize=chunksize,
            low_memory=True,
        )
    except Exception:
        return pd.DataFrame(columns=["month", "ISO3"] + _port_segment_aggregate_columns())

    for chunk in reader:
        chunk = chunk.copy()
        for c in seg_cols.intersection(chunk.columns):
            chunk[c] = safe_coerce_numeric(chunk[c])
        for c in seg_cols:
            if c not in chunk.columns:
                chunk[c] = 0.0
        sub = chunk[chunk["ISO3"].isin(PORT_FOCUS_ISO3)].copy()
        if sub.empty:
            continue
        sub["date"] = pd.to_datetime(sub["date"], errors="coerce", utc=True)
        sub["month"] = sub["date"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
        agg_cols = sorted(seg_cols.intersection(sub.columns))
        if not agg_cols:
            continue
        g = sub.groupby(["month", "ISO3"], as_index=False)[agg_cols].sum(min_count=1)
        parts.append(g)

    if not parts:
        return pd.DataFrame(columns=["month", "ISO3"] + _port_segment_aggregate_columns())
    out = pd.concat(parts, ignore_index=True)
    out = out.groupby(["month", "ISO3"], as_index=False)[sorted(seg_cols)].sum(min_count=1)
    return out.sort_values(["ISO3", "month"]).reset_index(drop=True)


def load_gdelt_focus(gdelt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(gdelt_path, low_memory=False)
    if "CountryCode" not in df.columns:
        return pd.DataFrame(
            columns=["year", "CountryCode", "conflict_intensity", "sum_events", "avg_tone"]
        )
    sub = df[df["CountryCode"].isin(GDELT_FOCUS)].copy()
    if sub.empty:
        sub = df.copy()
    for c in ("NormalizedEvents1000", "SumEvents", "AvgAvgTone"):
        if c in sub.columns:
            sub[c] = safe_coerce_numeric(sub[c])
    g = (
        sub.groupby(["Year", "CountryCode"], as_index=False)
        .agg(
            conflict_intensity=("NormalizedEvents1000", "sum"),
            sum_events=("SumEvents", "sum"),
            avg_tone=("AvgAvgTone", "mean"),
        )
        .rename(columns={"Year": "year"})
    )
    return g.sort_values(["CountryCode", "year"]).reset_index(drop=True)


def build_master_monthly(
    wheat: pd.DataFrame,
    ports: pd.DataFrame,
    gdelt: pd.DataFrame,
    shipping_layer: Optional[pd.DataFrame] = None,
    news_monthly: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if wheat.empty:
        return pd.DataFrame()

    m = wheat[["month", "price_egp", "price_usd"]].copy()
    if not ports.empty:
        eg = ports[ports["ISO3"] == "EGY"].copy()
        if not eg.empty:
            eg = eg.rename(
                columns={
                    "portcalls": "eg_portcalls",
                    "import": "eg_import",
                    "export": "eg_export",
                }
            )
            m = m.merge(
                eg[["month", "eg_portcalls", "eg_import", "eg_export"]],
                on="month",
                how="left",
            )
        else:
            m["eg_portcalls"] = np.nan
            m["eg_import"] = np.nan
            m["eg_export"] = np.nan

        bs_iso = {"UKR", "RUS"}
        bs = ports[ports["ISO3"].isin(bs_iso)].copy()
        if not bs.empty:
            bs_agg = bs.groupby("month", as_index=False)[
                ["portcalls", "import", "export"]
            ].sum()
            bs_agg = bs_agg.rename(
                columns={
                    "portcalls": "bs_portcalls",
                    "import": "bs_import",
                    "export": "bs_export",
                }
            )
            m = m.merge(bs_agg, on="month", how="left")
        else:
            m["bs_portcalls"] = np.nan
            m["bs_import"] = np.nan
            m["bs_export"] = np.nan
    else:
        m["eg_portcalls"] = np.nan
        m["eg_import"] = np.nan
        m["eg_export"] = np.nan
        m["bs_portcalls"] = np.nan
        m["bs_import"] = np.nan
        m["bs_export"] = np.nan

    m["year"] = m["month"].dt.year
    if not gdelt.empty:
        geo = (
            gdelt[gdelt["CountryCode"].isin({"UA", "RS", "EG"})]
            .groupby(["year", "CountryCode"], as_index=False)["conflict_intensity"]
            .mean()
        )
        pivot = geo.pivot(
            index="year", columns="CountryCode", values="conflict_intensity"
        )
        pivot = pivot.rename(
            columns={"UA": "geo_ukr", "RS": "geo_rus", "EG": "geo_eg"}
        ).reset_index()
        m = m.merge(pivot, on="year", how="left")
    for c in ("geo_ukr", "geo_rus", "geo_eg"):
        if c not in m.columns:
            m[c] = np.nan

    if shipping_layer is not None and not shipping_layer.empty:
        scol = "shipping_stress_score_composite" if "shipping_stress_score_composite" in shipping_layer.columns else "shipping_stress_score"
        sl = shipping_layer[["month"] + [c for c in shipping_layer.columns if c != "month"]].copy()
        if scol in sl.columns:
            sl = sl.rename(columns={scol: "shipping_route_stress"})
        elif "shipping_stress_score" in sl.columns:
            sl = sl.rename(columns={"shipping_stress_score": "shipping_route_stress"})
        keep = ["month", "shipping_route_stress"] + [
            c for c in ("shipping_congestion", "shipping_delay_days") if c in sl.columns
        ]
        m = m.merge(sl[[c for c in keep if c in sl.columns]], on="month", how="left")

    if news_monthly is not None and not news_monthly.empty:
        nm = news_monthly[
            [c for c in ("month", "nlp_conflict_index", "news_sentiment_avg", "headline_count") if c in news_monthly.columns]
        ]
        m = m.merge(nm, on="month", how="left")

    return m.sort_values("month").reset_index(drop=True)


def build_risk_features(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()

    df = master.copy()
    df["price_ma6"] = df["price_usd"].rolling(6, min_periods=2).mean()
    df["volatility_6m"] = df["price_usd"].rolling(6, min_periods=2).std()
    df["price_stress"] = (
        (df["price_usd"] - df["price_ma6"]) / df["price_ma6"].replace(0, np.nan) * 100
    )
    df["price_stress"] = df["price_stress"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["price_stress_score"] = min_max_scale(df["price_usd"])

    eg_calls = df["eg_portcalls"].ffill().fillna(0)
    df["port_yoy"] = eg_calls.pct_change(12) * 100
    df["port_activity_drop"] = (-df["port_yoy"].fillna(0)).clip(lower=0)
    df["logistics_risk_score"] = 100 - min_max_scale(eg_calls)

    geo_cols = [c for c in df.columns if c.startswith("geo_")]
    df["geopolitical_raw"] = df[geo_cols].mean(axis=1)
    df["geopolitical_risk_score"] = min_max_scale(df["geopolitical_raw"])

    df["conflict_event_proxy"] = df[geo_cols].sum(axis=1).fillna(0)
    df["volatility_risk"] = df["volatility_6m"].fillna(0)

    ma3 = df["price_usd"].rolling(3, min_periods=1).mean()
    df["forecast_deviation"] = df["price_usd"] - ma3.shift(1)

    if "shipping_route_stress" in df.columns:
        shi = df["shipping_route_stress"].fillna(0)
        df["logistics_risk_score"] = np.clip(
            0.5 * df["logistics_risk_score"].fillna(0) + 0.5 * min_max_scale(shi),
            0,
            100,
        )

    if "nlp_conflict_index" in df.columns:
        df["nlp_conflict_keyword_score"] = min_max_scale(df["nlp_conflict_index"].fillna(0))
    else:
        df["nlp_conflict_keyword_score"] = df["geo_eg"].fillna(0)
        df["nlp_conflict_keyword_score"] = min_max_scale(df["nlp_conflict_keyword_score"])

    if "news_sentiment_avg" in df.columns:
        # Sentiment: -1 (bad) … +1 (calming) → risk rises when sentiment is negative
        df["news_sentiment_risk_score"] = min_max_scale(-df["news_sentiment_avg"].fillna(0))
    else:
        df["news_sentiment_risk_score"] = 0.0

    geo_boost = 0.15 * df["news_sentiment_risk_score"]
    df["geopolitical_risk_score"] = np.clip(df["geopolitical_risk_score"] + geo_boost, 0, 100)

    df = df.dropna(subset=["price_usd"])
    return df.reset_index(drop=True)


def build_blended_wfp_commodity_catalog(
    wfp_by_commodity: dict[str, pd.DataFrame],
    bundle: TrainingDataBundle,
) -> tuple[dict[str, pd.DataFrame], str]:
    """
    Merge each WFP retail series with a matching FAO track when possible.

    * Wheat-named WFP products use ``bundle.wheat_prices`` (pipeline spine — same as risk features).
    * Other commodities resolve FAO via token overlap against ``bundle.fao_catalog`` keys.

    Matching uses :func:`~src.fao_prices.find_best_fao_series` (min overlap score threshold).
    """
    fao_catalog = getattr(bundle, "fao_catalog", None) or {}
    blended_rows: list[str] = []
    wfp_only: list[str] = []
    spine_rows: list[str] = []

    out: dict[str, pd.DataFrame] = {}

    def _has_fao_col(df: pd.DataFrame) -> bool:
        if "price_usd_fao" not in df.columns:
            return False
        return bool(df["price_usd_fao"].notna().any())

    for label, sdf in wfp_by_commodity.items():
        if not sdf.empty and "wheat" in str(label).lower() and not bundle.wheat_prices.empty:
            out[label] = bundle.wheat_prices.copy()
            if _has_fao_col(bundle.wheat_prices):
                spine_rows.append(label)
            else:
                wfp_only.append(label)
            continue

        fac = find_best_fao_series(label, fao_catalog, min_score=50)
        if fac is not None and not fac.empty:
            merged = merge_wfp_fao(sdf, fac, prefer="blend", commodity_label=label)
            out[label] = merged
            blended_rows.append(label)
        else:
            m = sdf.copy()
            m["price_usd_wfp"] = m["price_usd"]
            m["price_usd_fao"] = float("nan")
            out[label] = m
            wfp_only.append(label)

    parts: list[str] = []
    if spine_rows:
        parts.append(
            "Pipeline wheat (WFP×FAO when present): " + ", ".join(sorted(spine_rows)) + "."
        )
    if blended_rows:
        parts.append("FAO blend via catalog: " + ", ".join(sorted(blended_rows)) + ".")
    if wfp_only:
        parts.append(f"WFP-only (no FAO match or no FAO column): {len(wfp_only)} series.")
    note = " ".join(parts) if parts else "WFP retail only — add fao_commodities_egy.csv (FAOSTAT) for multi-commodity FAO."
    return out, note


def weak_risk_labels(row: pd.Series) -> str:
    """
    Binary rule labels for LDA (educational): original four-band composite score,
    then map Low+Moderate -> 0, High+Critical -> 1.
    """
    s = (
        float(row.get("price_stress_score", 0) or 0)
        + float(row.get("logistics_risk_score", 0) or 0)
        + float(row.get("geopolitical_risk_score", 0) or 0)
    ) / 3.0
    if s < 55:
        return "0"
    return "1"


def assemble_training_bundle(
    port_csv: Optional[Path] = None,
    gdelt_csv: Optional[Path] = None,
    wfp_csv: Optional[Path] = None,
    fao_csv: Optional[Path] = None,
    shipping_csv: Optional[Path] = None,
    news_csv: Optional[Path] = None,
) -> TrainingDataBundle:
    root = project_root()
    port_csv = port_csv or root / "Daily_Port_Activity_Data_and_Trade_Estimates.csv"
    gdelt_csv = gdelt_csv or root / "gdelt_conflict_1_0.csv"
    wfp_csv = wfp_csv or root / "wfp_food_prices_egy.csv"
    fao_csv = fao_csv or root / "fao_wheat_egy.csv"
    fao_multi_csv = root / "fao_commodities_egy.csv"
    shipping_csv = shipping_csv or root / "shipping_metrics.csv"
    news_csv = news_csv or root / "news_headlines_bilingual.csv"

    wfp = load_wheat_series(wfp_csv) if wfp_csv.is_file() else pd.DataFrame()
    fao = load_fao_wheat_egypt(fao_csv) if fao_csv.is_file() else pd.DataFrame()
    fao_catalog = load_fao_commodities_catalog(fao_multi_csv) if fao_multi_csv.is_file() else {}

    wheat_fao_from_multi = False
    if fao.empty and not wfp.empty and fao_catalog:
        alt = find_best_fao_series("Wheat flour", fao_catalog, min_score=50)
        if alt is not None and not alt.empty:
            fao = alt
            wheat_fao_from_multi = True

    blend_note = "WFP retail series only (add fao_wheat_egy.csv and/or fao_commodities_egy.csv for FAO tracks)."
    if not fao.empty and not wfp.empty:
        wheat = merge_wfp_fao(wfp, fao, prefer="blend", commodity_label="Wheat flour")
        blend_note = (
            "Blended WFP Egypt retail with FAO FAOSTAT-style wheat series "
            "(55% FAO weight when both present)."
        )
        if wheat_fao_from_multi:
            blend_note = (
                "Blended WFP Egypt retail with best-matching **wheat** FAO Item from "
                "`fao_commodities_egy.csv` (55% FAO when both present)."
            )
    elif not fao.empty:
        wheat = merge_wfp_fao(wfp, fao, commodity_label="Wheat flour")
        blend_note = "FAO wheat series as primary (WFP missing)."
    else:
        wheat = wfp
        if not wfp.empty:
            blend_note = "WFP wheat retail only — add fao_wheat_egy.csv or a multi-item fao_commodities_egy.csv."

    if fao_catalog:
        blend_note += f" Multi-item FAO file: {len(fao_catalog)} FAOSTAT item series for non-wheat WFP matching."

    ports = load_ports_monthly(port_csv) if port_csv.is_file() else pd.DataFrame()
    gdelt = load_gdelt_focus(gdelt_csv) if gdelt_csv.is_file() else pd.DataFrame()

    port_ship = port_derived_shipping_stress(ports)
    tracker = load_shipping_metrics_csv(shipping_csv)
    if not tracker.empty and not wheat.empty:
        tracker = expand_shipping_to_wheat_index(tracker, wheat["month"])
    shipping_layer = merge_shipping_layers(port_ship, tracker)

    headlines, news_monthly, novelty, _syn = build_news_pipeline(wheat, news_csv)

    master = build_master_monthly(wheat, ports, gdelt, shipping_layer, news_monthly)
    risk = build_risk_features(master)
    if not risk.empty:
        risk["risk_label"] = risk.apply(weak_risk_labels, axis=1)

    return TrainingDataBundle(
        wheat_prices=wheat,
        port_monthly=ports,
        gdelt_yearly=gdelt,
        risk_features=risk,
        master_monthly=master,
        fao_wheat=fao,
        fao_catalog=fao_catalog,
        shipping_monthly=shipping_layer,
        news_headlines=headlines,
        news_monthly=news_monthly,
        novelty=novelty,
        price_blend_note=blend_note,
    )
