"""Port intelligence: load demo ports, enrich with DSS context scores."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import DATA_SAMPLE_DIR
from src.risk_scoring import safe_float


def _default_ports_dataframe() -> pd.DataFrame:
    rows = [
        ("EG_ALX", "Alexandria Port", "Egypt", "Alexandria", 31.1981, 29.8719, "multipurpose", "wheat|veg_oil|maize", "citrus|textiles", 18, 12, 0.72, 0.45, 42, 38, "Moderate", "Major wheat discharge; watch berth queues."),
        ("EG_DAM", "Damietta Port", "Egypt", "Damietta", 31.4174, 31.8157, "container_bulk", "wheat|containers", "agri_exports", 11, 9, 0.61, 0.38, 38, 35, "Moderate", "Growing transshipment on Black Sea grain."),
        ("EG_PSD", "Port Said", "Egypt", "Port Said", 31.2565, 32.2849, "transit", "wheat|fuel", "reexports", 22, 16, 0.78, 0.52, 48, 40, "High", "Canal-linked peaks increase delay risk."),
        ("EG_SUZ", "Suez Port", "Egypt", "Suez", 29.9668, 32.5498, "bulk", "wheat|fertilizer", "minerals", 9, 8, 0.55, 0.41, 44, 46, "High", "Chokepoint sensitivities on inbound legs."),
        ("EG_SOH", "Sokhna (Ain Sokhna)", "Egypt", "Suez Governorate", 29.60, 32.35, "dry_bulk", "wheat|coal", "scrap", 13, 11, 0.68, 0.48, 45, 42, "High", "Large bulk discharge for Greater Cairo supply chain."),
        ("EG_SAF", "Safaga Port", "Egypt", "Safaga", 26.7491, 33.9388, "bulk", "wheat|rice", "phosphates", 7, 6, 0.50, 0.33, 40, 44, "Moderate", "Red Sea approach — security watch."),
        ("UA_ODS", "Odessa", "Ukraine", "Odessa", 46.4825, 30.7233, "grain_export", "wheat|maize", "metals", 14, 19, 0.85, 0.62, 72, 85, "Critical", "Key wheat origin — conflict overlay on approaches."),
        ("RU_NVR", "Novorossiysk", "Russia", "Novorossiysk", 44.7248, 37.7677, "oil_grain_bulk", "wheat|veg_oil", "fertilizer", 16, 18, 0.80, 0.55, 58, 55, "High", "Black Sea weather and policy sensitivity."),
        ("RO_CND", "Constanța", "Romania", "Constanța", 44.1598, 28.6508, "grain", "wheat|maize", "vehicles", 12, 14, 0.70, 0.42, 48, 50, "High", "Alternative grain aggregation for Egypt programs."),
        ("TR_AMB", "Ambarlı (Istanbul)", "Turkey", "Istanbul", 40.9667, 28.6833, "container", "wheat_flour|containers", "industrial", 21, 20, 0.88, 0.64, 52, 48, "High", "Transhipment option into eastern Med."),
        ("GR_PIR", "Piraeus", "Greece", "Piraeus", 37.9420, 23.6470, "container", "food_products", "reexports", 25, 24, 0.86, 0.58, 46, 45, "Moderate", "Transship relay into Alexandria/Damietta."),
        ("FR_MRS", "Marseille", "France", "Marseille", 43.2965, 5.3698, "multipurpose", "veg_oil|cereals", "machinery", 18, 17, 0.74, 0.46, 36, 35, "Low", "Optional EU origin diversification."),
        ("SA_JED", "Jeddah", "Saudi Arabia", "Jeddah", 21.5433, 39.1728, "container_bulk", "rice|sugar|flour", "petchem", 19, 18, 0.77, 0.51, 50, 52, "High", "Red Sea feeder connectivity to Egypt."),
        ("JO_AQB", "Aqaba", "Jordan", "Aqaba", 29.5267, 35.0060, "bulk", "wheat|fertilizer", "potash", 10, 9, 0.60, 0.39, 43, 48, "Moderate", "Land bridge option if sea delays."),
    ]
    cols = [
        "port_id",
        "name",
        "country",
        "city_region",
        "lat",
        "lon",
        "port_type",
        "main_goods_imported",
        "main_goods_exported",
        "inbound_ships",
        "outbound_ships",
        "current_activity_level",
        "congestion_level",
        "logistics_risk_score",
        "nearby_conflict_risk_score",
        "overall_risk_level",
        "explanation",
    ]
    return pd.DataFrame(rows, columns=cols)


def load_ports_table(path: Optional[Path] = None) -> pd.DataFrame:
    p = path or (DATA_SAMPLE_DIR / "demo_ports.csv")
    if not p.is_file():
        return _default_ports_dataframe()
    df = pd.read_csv(p)
    if df.empty:
        return _default_ports_dataframe()
    return df


def enrich_ports_with_context(ports: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    """Blend DSS logistics/geo/price stress into port risk (lightweight heuristic)."""
    if ports.empty:
        return ports
    out = ports.copy()
    for _c in ("logistics_risk_score", "nearby_conflict_risk_score"):
        if _c in out.columns:
            out[_c] = out[_c].astype(float)
    log = safe_float(context.get("logistics_risk_score") or context.get("log"))
    geo = safe_float(context.get("geopolitical_risk_score") or context.get("geo"))
    pss = safe_float(context.get("price_stress_score") or context.get("pss"))
    for i in out.index:
        base_log = safe_float(out.loc[i, "logistics_risk_score"], log)
        base_nf = safe_float(out.loc[i, "nearby_conflict_risk_score"], geo)
        blended_log = float(np.clip(0.55 * base_log + 0.45 * log, 0, 100))
        blended_nf = float(np.clip(0.55 * base_nf + 0.45 * geo, 0, 100))
        out.loc[i, "logistics_risk_score"] = blended_log
        out.loc[i, "nearby_conflict_risk_score"] = blended_nf
        composite = (blended_log + blended_nf + pss) / 3.0
        out.loc[i, "overall_risk_score"] = composite
        if composite >= 75:
            lvl = "Critical"
        elif composite >= 55:
            lvl = "High"
        elif composite >= 35:
            lvl = "Moderate"
        else:
            lvl = "Low"
        out.loc[i, "overall_risk_level"] = lvl
    return out


def port_by_id(ports: pd.DataFrame, port_id: str) -> Optional[pd.Series]:
    if ports.empty or not port_id:
        return None
    m = ports[ports["port_id"].astype(str) == str(port_id)]
    if m.empty:
        return None
    return m.iloc[0]
