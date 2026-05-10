"""Maritime conflict and disruption zones for route risk (demo / offline)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DATA_SAMPLE_DIR


@dataclass
class ConflictZone:
    zone_id: str
    name: str
    lat: float
    lon: float
    radius_km: float
    risk_type: str
    severity: float  # 0–100
    description: str
    impact: str
    related_commodities: str
    recommended_response: str


def _default_zones() -> list[ConflictZone]:
    return [
        ConflictZone(
            "Z_BS",
            "Black Sea conflict risk zone",
            45.0,
            32.0,
            280,
            "conflict",
            88,
            "Military activity and export disruption risk in north-west Black Sea approaches.",
            "Delays, higher insurance, possible diversions for grain from Ukraine.",
            "wheat,maize,sunflower_oil",
            "Monitor AIS; prefer insured tonnage; consider alternative corridors.",
        ),
        ConflictZone(
            "Z_RS",
            "Red Sea security risk",
            18.0,
            40.0,
            220,
            "security",
            76,
            "Elevated maritime security incidents affecting Suez–Bab el-Mandeb corridor.",
            "Schedule slippage and canal transit uncertainty.",
            "all_consignments",
            "Coordinate with operators; update ETAs to Egyptian ports.",
        ),
        ConflictZone(
            "Z_EMD",
            "Eastern Mediterranean tension",
            35.5,
            28.0,
            200,
            "geopolitical",
            62,
            "Strategic shipping density and periodic air-naval activity east of Crete.",
            "Route uncertainty for Aegean–Egypt lanes.",
            "wheat,veg_oil",
            "Maintain communications watch; use suggested safer waypoints.",
        ),
        ConflictZone(
            "Z_SUEZ",
            "Suez Canal congestion risk",
            30.0,
            32.55,
            80,
            "chokepoint",
            55,
            "Queues and pilot availability can delay convoys.",
            "Knock-on delays for Egypt-bound arrivals.",
            "all",
            "Pre-clear documentation; pad logistics buffers.",
        ),
        ConflictZone(
            "Z_UA_EXP",
            "Ukraine export disruption",
            46.5,
            31.5,
            150,
            "sanctions_sensitive",
            72,
            "Policy and military risk around key export load ports.",
            "Wheat and maize loading uncertainty.",
            "wheat,maize",
            "Diversify origins; track charter-party clauses.",
        ),
        ConflictZone(
            "Z_ME",
            "Middle East instability band",
            33.0,
            36.0,
            260,
            "instability",
            58,
            "Broader regional instability affecting eastern Med relay routes.",
            "Fuel and schedule volatility.",
            "food_bulk",
            "Higher monitoring frequency for laden inbound vessels.",
        ),
    ]


def load_conflict_zones(csv_path: Optional[Path] = None) -> list[ConflictZone]:
    path = csv_path or (DATA_SAMPLE_DIR / "demo_conflict_zones.csv")
    if not path.is_file():
        return _default_zones()
    df = pd.read_csv(path)
    zones: list[ConflictZone] = []
    for _, r in df.iterrows():
        try:
            zones.append(
                ConflictZone(
                    str(r.get("zone_id", "")),
                    str(r.get("name", "")),
                    float(r["lat"]),
                    float(r["lon"]),
                    float(r.get("radius_km", 100)),
                    str(r.get("risk_type", "risk")),
                    float(r.get("severity", 50)),
                    str(r.get("description", "")),
                    str(r.get("impact", "")),
                    str(r.get("related_commodities", "")),
                    str(r.get("recommended_response", "")),
                )
            )
        except Exception:
            continue
    return zones or _default_zones()


def zones_to_dataframe(zones: list[ConflictZone]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone_id": z.zone_id,
                "name": z.name,
                "lat": z.lat,
                "lon": z.lon,
                "radius_km": z.radius_km,
                "risk_type": z.risk_type,
                "severity": z.severity,
            }
            for z in zones
        ]
    )
