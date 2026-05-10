"""Safe numeric risk helpers and data-completeness scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import DATA_SAMPLE_DIR


def safe_float(v: Any, default: float = 0.0) -> float:
    """Coerce to float; NaN/inf/None → default (fixes KPI 'nan' when row has NaN)."""
    if v is None:
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if np.isnan(x) or np.isinf(x):
        return default
    return float(x)


def data_completeness_score(
    wheat_rows: int,
    port_rows: int,
    gdelt_rows: int,
    news_real: bool,
    sample_ports_ok: bool,
    sample_vessels_ok: bool,
) -> float:
    """0–100 coarse completeness for ministry KPI (not statistical validation)."""
    acc = 0.0
    if wheat_rows > 0:
        acc += 25
    if port_rows > 0:
        acc += 20
    if gdelt_rows > 0:
        acc += 15
    if news_real:
        acc += 15
    else:
        acc += 5
    if sample_ports_ok:
        acc += 12.5
    if sample_vessels_ok:
        acc += 12.5
    return float(min(100.0, acc))


def sample_data_files_present() -> tuple[bool, bool, bool]:
    p = DATA_SAMPLE_DIR / "demo_ports.csv"
    v = DATA_SAMPLE_DIR / "demo_vessels.csv"
    z = DATA_SAMPLE_DIR / "demo_conflict_zones.csv"
    return p.is_file(), v.is_file(), z.is_file()
