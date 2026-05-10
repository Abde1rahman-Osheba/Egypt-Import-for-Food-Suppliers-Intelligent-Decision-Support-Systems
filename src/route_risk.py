"""Route risk: haversine geometry, zone proximity, alternative paths (numpy only)."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from src.conflict_zones import ConflictZone


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
    return r * c


def point_to_segment_distance_km(
    lat_p: float,
    lon_p: float,
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
) -> float:
    """Approximate shortest distance from point P to great-circle segment AB (sampled)."""
    n = 24
    dists = []
    for t in np.linspace(0, 1, n):
        lat = lat_a + t * (lat_b - lat_a)
        lon = lon_a + t * (lon_b - lon_a)
        dists.append(haversine_distance_km(lat_p, lon_p, lat, lon))
    return float(min(dists))


def detect_nearby_risk_zones(
    lats: list[float],
    lons: list[float],
    zones: list[ConflictZone],
    buffer_km: float = 0.0,
) -> tuple[list[ConflictZone], list[float]]:
    """Find zones whose circle intersects the polyline (segment sampling)."""
    hit: list[ConflictZone] = []
    dists: list[float] = []
    for i in range(len(lats) - 1):
        for z in zones:
            d = point_to_segment_distance_km(z.lat, z.lon, lats[i], lons[i], lats[i + 1], lons[i + 1])
            if d <= z.radius_km + buffer_km and z not in hit:
                hit.append(z)
                dists.append(d)
    return hit, dists


def calculate_route_risk(lats: list[float], lons: list[float], zones: list[ConflictZone]) -> float:
    """Aggregate 0–100 route risk from zone severities weighted by proximity."""
    if len(lats) < 2:
        return 0.0
    hits, _ = detect_nearby_risk_zones(lats, lons, zones, buffer_km=0)
    if not hits:
        return max(8.0, 15.0)  # baseline open-sea residual
    acc = 0.0
    wsum = 0.0
    for z in hits:
        # Closer approach → stronger weight (inverse distance cap)
        dmin = min(
            point_to_segment_distance_km(z.lat, z.lon, lats[i], lons[i], lats[i + 1], lons[i + 1])
            for i in range(len(lats) - 1)
        )
        w = z.severity / max(25.0, dmin)
        acc += w * z.severity
        wsum += w
    raw = acc / wsum if wsum else max(h.severity for h in hits)
    return float(np.clip(raw, 0, 100))


def route_risk_level(score: float) -> str:
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Low"


def generate_alternative_route(
    olat1: float,
    olon1: float,
    olat2: float,
    olon2: float,
    zones: list[ConflictZone],
) -> tuple[list[float], list[float], str]:
    """
    Insert two waypoints shifted away from the strongest intersecting zone centroid.
    Returns lats, lons piecewise path (including origin and destination).
    """
    base_lats = [olat1, olat2]
    base_lons = [olon1, olon2]
    hits, _ = detect_nearby_risk_zones(base_lats, base_lons, zones, buffer_km=80)
    if not hits:
        return base_lats, base_lons, "No high-risk zone overlap — original route kept."

    z = max(hits, key=lambda x: x.severity)
    mid_lat = (olat1 + olat2) / 2
    mid_lon = (olon1 + olon2) / 2
    # Push mid-point perpendicular (crude offset in degrees ~110km/deg lat)
    deflect_lat = 2.2 if mid_lat < z.lat else -2.2
    deflect_lon = 2.5 if mid_lon < z.lon else -2.5
    w1_lat = (olat1 + mid_lat) / 2 + deflect_lat * 0.35
    w1_lon = (olon1 + mid_lon) / 2 + deflect_lon * 0.25
    w2_lat = (mid_lat + olat2) / 2 + deflect_lat * 0.2
    w2_lon = (mid_lon + olon2) / 2 + deflect_lon * 0.15
    lats = [olat1, w1_lat, w2_lat, olat2]
    lons = [olon1, w1_lon, w2_lon, olon2]
    return lats, lons, f"Detour clear of {z.name} centroid via eastern Med / relay corridor."


def route_length_km(lats: list[float], lons: list[float]) -> float:
    leg = 0.0
    for i in range(len(lats) - 1):
        leg += haversine_distance_km(lats[i], lons[i], lats[i + 1], lons[i + 1])
    return leg


def compare_original_vs_alternative(
    olats: list[float],
    olons: list[float],
    alats: list[float],
    alons: list[float],
    zones: list[ConflictZone],
    speed_kn: float = 12.0,
) -> dict[str, Any]:
    d0 = route_length_km(olats, olons)
    d1 = route_length_km(alats, alons)
    kmh = speed_kn * 1.852
    r0 = calculate_route_risk(olats, olons, zones)
    r1 = calculate_route_risk(alats, alons, zones)
    return {
        "original_km": d0,
        "alternative_km": d1,
        "extra_km": max(0.0, d1 - d0),
        "extra_hours": max(0.0, (d1 - d0) / max(kmh, 1e-6)),
        "original_risk": r0,
        "alternative_risk": r1,
        "risk_delta": r0 - r1,
    }


def explain_route_choice(cmp: dict[str, Any], alt_reason: str) -> str:
    return (
        f"{alt_reason} Original route risk ≈ {cmp['original_risk']:.0f}/100 vs alternative ≈ "
        f"{cmp['alternative_risk']:.0f}/100. Additional steaming ≈ {cmp['extra_hours']:.1f} h "
        f"({cmp['extra_km']:.0f} nm equivalent distance at assumed speed)."
    )
