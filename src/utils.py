"""Utility helpers: dependency checks, safe I/O, and shared constants."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def check_dependencies() -> dict[str, Any]:
    """Verify optional and core imports; return a report for the Methodology page."""
    modules = [
        "streamlit",
        "pandas",
        "numpy",
        "sklearn",
        "plotly",
        "joblib",
        "dateutil",
        "statsmodels",
        "requests",
    ]
    report: dict[str, Any] = {"modules": {}, "statsmodels_forecasting": False}
    for name in modules:
        spec = importlib.util.find_spec(name)
        report["modules"][name] = spec is not None
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa: F401

        report["statsmodels_forecasting"] = True
    except Exception:
        report["statsmodels_forecasting"] = False
    return report


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def safe_read_csv(
    path: str | Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load CSV; return empty frame with message on failure."""
    p = Path(path)
    if not p.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, **kwargs)
    except Exception:
        return pd.DataFrame()


def safe_coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if str(col).lower() in ("date", "datetime", "month", "period"):
            return col
    for col in df.columns:
        if "date" in str(col).lower():
            return col
    return None


def min_max_scale(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(), s.max()
    if hi == lo or np.isnan(lo) or np.isnan(hi):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo) * 100.0


def empty_chart_message() -> str:
    return "No data available for this chart."
