"""Synthetic / CSV-backed vessel journeys for offline demo (no live AIS)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import DATA_SAMPLE_DIR
from src.conflict_zones import ConflictZone, load_conflict_zones
from src.port_intelligence import load_ports_table
from src.route_risk import (
    calculate_route_risk,
    compare_original_vs_alternative,
    detect_nearby_risk_zones,
    explain_route_choice,
    generate_alternative_route,
    route_risk_level,
)
from src.risk_scoring import safe_float


def _lin_route(lat0: float, lon0: float, lat1: float, lon1: float, t: float) -> tuple[float, float]:
    t = float(np.clip(t, 0, 1))
    return lat0 + t * (lat1 - lat0), lon0 + t * (lon1 - lon0)


def _default_vessels() -> pd.DataFrame:
    base = datetime(2026, 5, 6, 8, 0, tzinfo=timezone.utc)
    rows = []
    specs = [
        ("V001", "MV Nile Grain", "bulk_carrier", "UA_ODS", "EG_ALX", "wheat", 35000, base, base + timedelta(days=6, hours=10), 11.5, 0.62),
        ("V002", "MV Delta Maize", "bulk_carrier", "RO_CND", "EG_DAM", "maize", 28000, base + timedelta(hours=6), base + timedelta(days=5, hours=8), 12.0, 0.55),
        ("V003", "MV Horizon Reefer", "reefer_vessel", "TR_AMB", "EG_PSD", "veg_oil", 5200, base + timedelta(days=1), base + timedelta(days=4, hours=6), 13.2, 0.41),
        ("V004", "MV Cairo Trader", "general_cargo", "GR_PIR", "EG_ALX", "flour", 12000, base - timedelta(days=2), base + timedelta(days=3, hours=12), 11.0, 0.74),
        ("V005", "MV Red Sea Sugar", "bulk_carrier", "SA_JED", "EG_SOH", "sugar", 24000, base + timedelta(hours=12), base + timedelta(days=2, hours=18), 13.0, 0.58),
        ("V006", "MV Black Sea Runner", "bulk_carrier", "RU_NVR", "EG_ALX", "wheat", 42000, base + timedelta(days=1), base + timedelta(days=7, hours=4), 10.5, 0.67),
        ("V007", "MV Levant Express", "container_ship", "TR_AMB", "EG_DAM", "mixed_food", 18500, base + timedelta(days=2), base + timedelta(days=5), 12.8, 0.48),
        ("V008", "MV Maghreb Sun", "tanker", "FR_MRS", "EG_SUZ", "veg_oil", 15000, base + timedelta(days=3), base + timedelta(days=8), 11.2, 0.52),
    ]

    for s in specs:
        rows.append(
            {
                "ship_id": s[0],
                "ship_name": s[1],
                "ship_type": s[2],
                "origin_port_id": s[3],
                "dest_port_id": s[4],
                "cargo_type": s[5],
                "cargo_quantity_mt": s[6],
                "departure_utc": s[7].isoformat(),
                "eta_utc": s[8].isoformat(),
                "speed_kn": s[9],
                "demo_progress_pct": s[10] * 100,
            }
        )
    return pd.DataFrame(rows)


def load_vessels_table(path: Optional[Path] = None) -> pd.DataFrame:
    p = path or (DATA_SAMPLE_DIR / "demo_vessels.csv")
    if not p.is_file():
        return _default_vessels()
    df = pd.read_csv(p)
    if df.empty:
        return _default_vessels()
    return df


def enrich_vessel_routes(
    vessels: pd.DataFrame,
    ports: pd.DataFrame,
    zones: list[ConflictZone],
    now_utc: Optional[datetime] = None,
) -> pd.DataFrame:
    if vessels.empty or ports.empty:
        return vessels
    now = now_utc or datetime.now(timezone.utc)
    pmap = ports.set_index("port_id")
    rows = []
    for _, v in vessels.iterrows():
        oid, did = str(v["origin_port_id"]), str(v["dest_port_id"])
        if oid not in pmap.index or did not in pmap.index:
            continue
        o = pmap.loc[oid]
        d = pmap.loc[did]
        olat, olon = float(o["lat"]), float(o["lon"])
        dlat, dlon = float(d["lat"]), float(d["lon"])
        dep = pd.to_datetime(v.get("departure_utc"), utc=True, errors="coerce")
        eta = pd.to_datetime(v.get("eta_utc"), utc=True, errors="coerce")
        if pd.isna(dep) or pd.isna(eta):
            total_h = 120.0
            prog = safe_float(v.get("demo_progress_pct"), 50.0) / 100.0
        else:
            total_h = max(1.0, (eta - dep).total_seconds() / 3600)
            elapsed = (pd.Timestamp(now) - dep).total_seconds() / 3600
            prog = float(np.clip(elapsed / total_h, 0, 1))
        clat, clon = _lin_route(olat, olon, dlat, dlon, prog)
        base_lats = [olat, dlat]
        base_lons = [olon, dlon]
        rsk = calculate_route_risk(base_lats, base_lons, zones)
        lvl = route_risk_level(rsk)
        alats, alons, alt_msg = generate_alternative_route(olat, olon, dlat, dlon, zones)
        spdkn = safe_float(v.get("speed_kn"), 12.0)
        cmp = compare_original_vs_alternative(base_lats, base_lons, alats, alons, zones, speed_kn=spdkn)
        hits, _dists = detect_nearby_risk_zones(base_lats, base_lons, zones, buffer_km=50)
        near_flag = len(hits) > 0
        delay_p = float(np.clip(0.15 + rsk / 250 + (0.2 if near_flag else 0), 0, 0.95))

        if rsk >= 75:
            rec = "reroute"
        elif rsk >= 55:
            rec = "monitor"
        else:
            rec = "keep_route"

        rem_h = max(0.0, (1 - prog) * total_h)

        cmp_clean = {k: float(v2) for k, v2 in cmp.items()}
        row_dict = v.to_dict()
        row_dict.update(
            {
                "current_lat": clat,
                "current_lon": clon,
                "journey_progress_pct": prog * 100,
                "elapsed_hours_approx": prog * total_h,
                "remaining_hours_approx": rem_h,
                "total_journey_hours_approx": total_h,
                "route_risk_score": rsk,
                "route_risk_level": lvl,
                "near_risk_zone": near_flag,
                "near_zone_names": "|".join(h.name for h in hits) if hits else "",
                "delay_probability": delay_p,
                "recommendation": rec,
                "alt_route_lats": str(alats),
                "alt_route_lons": str(alons),
                "alt_route_explanation": alt_msg,
                "route_compare": cmp_clean,
                "explainer": explain_route_choice(cmp, alt_msg)
                if near_flag or rsk >= 55
                else "Route clear of modeled high-severity zones at present.",
            }
        )
        rows.append(row_dict)
    return pd.DataFrame(rows)


def vessel_by_id(vessels: pd.DataFrame, ship_id: str) -> Optional[pd.Series]:
    if vessels.empty or not ship_id:
        return None
    m = vessels[vessels["ship_id"].astype(str) == str(ship_id)]
    if m.empty:
        return None
    return m.iloc[0]


def build_maritime_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    zones = load_conflict_zones()
    ports = load_ports_table()
    from src.port_intelligence import enrich_ports_with_context

    ports_e = enrich_ports_with_context(ports, context)
    vessels = load_vessels_table()
    vessels_e = enrich_vessel_routes(vessels, ports_e, zones)
    return {
        "zones": zones,
        "ports": ports_e,
        "vessels": vessels_e,
        "demo_mode": True,
    }
