"""Industrial-style early-warning support: provenance, validation, latency, backtests, calibration."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.data_pipeline import TrainingDataBundle
from src.inference_engine import InferenceAlert, run_inference
from src.nlp_conflict_index import price_spike_series
from src.utils import project_root


LIVE_FEEDS_AND_PATHS_DOC = """
**End-to-end data path (defaults are project root CSVs)**

| Layer | Default path / hook | Refresh |
|-------|---------------------|---------|
| Wheat retail (WFP Egypt) | `wfp_food_prices_egy.csv` | Replace file on schedule (e.g. monthly WFP sync). |
| FAO / FAOSTAT-style wheat | `fao_wheat_egy.csv` (optional) | Blends into **pipeline wheat spine** (55% FAO when both exist). |
| FAO multi-commodity catalog | `fao_commodities_egy.csv` (optional) | FAOSTAT long export (**Area + Item + Value + Year**) — matched to **every** WFP commodity by Item name similarity. |
| GDELT aggregates | `gdelt_conflict_1_0.csv` | Update when static extract is refreshed. |
| Port activity | `Daily_Port_Activity_Data_and_Trade_Estimates.csv` | Large file; first harmonization may take minutes. |
| Shipping layer B | `shipping_metrics.csv` (optional) | Vendor export or internal ETL. |
| Live shipping JSON | Env **`SHIPPING_TRACKER_API_URL`** | Point to HTTPS JSON; extend `shipping_tracker` merge as needed. |
| Bilingual news | `news_headlines_bilingual.csv` (optional) | Ingest RSS/API → monthly rollups; else synthetic demo headlines are flagged. |

**Latency:** Streamlit caches `assemble_training_bundle()`; use *Clear bundle cache* on the Ops page for a cold-load timing read. Sidebar shows latest input file modification time (UTC) as a freshness proxy.

**Provenance:** Rows reflect manifest + observed row counts from the loaded bundle (not full-file scans for huge CSVs).
"""


@dataclass
class ProvenanceRecord:
    source_id: str
    path_or_hook: str
    role: str  # required | optional | env
    status: str
    modified_utc: str
    size_mb: Optional[float]
    rows_in_bundle: Optional[int]
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path_or_hook": self.path_or_hook,
            "role": self.role,
            "status": self.status,
            "modified_utc": self.modified_utc,
            "size_mb": self.size_mb,
            "rows_in_bundle": self.rows_in_bundle,
            "notes": self.notes,
        }


def _fmt_mtime(path: Path) -> tuple[str, Optional[float]]:
    if not path.is_file():
        return ("—", None)
    ts = path.stat().st_mtime
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    size_mb = round(path.stat().st_size / (1024 * 1024), 3)
    return (dt.strftime("%Y-%m-%d %H:%M UTC"), size_mb)


def collect_provenance(bundle: TrainingDataBundle) -> list[ProvenanceRecord]:
    root = project_root()
    specs: list[tuple[str, Path, str, bool, Optional[int]]] = [
        ("wfp_retail", root / "wfp_food_prices_egy.csv", "required", True, len(bundle.wheat_prices)),
        ("gdelt_conflict", root / "gdelt_conflict_1_0.csv", "required", True, len(bundle.gdelt_yearly)),
        ("port_activity", root / "Daily_Port_Activity_Data_and_Trade_Estimates.csv", "required", True, len(bundle.port_monthly)),
        ("fao_wheat", root / "fao_wheat_egy.csv", "optional", False, len(bundle.fao_wheat)),
        (
            "fao_commodities_multi",
            root / "fao_commodities_egy.csv",
            "optional",
            False,
            sum(len(df) for df in bundle.fao_catalog.values()),
        ),
        ("shipping_metrics", root / "shipping_metrics.csv", "optional", False, len(bundle.shipping_monthly) if not bundle.shipping_monthly.empty else 0),
        ("news_bilingual", root / "news_headlines_bilingual.csv", "optional", False, len(bundle.news_headlines)),
    ]
    recs: list[ProvenanceRecord] = []
    for sid, path, role, required, rows in specs:
        mtime, size_mb = _fmt_mtime(path)
        exists = path.is_file()
        if not exists:
            recs.append(
                ProvenanceRecord(
                    source_id=sid,
                    path_or_hook=str(path.name),
                    role=role,
                    status="missing" if required else "optional_missing",
                    modified_utc=mtime,
                    size_mb=size_mb,
                    rows_in_bundle=0,
                    notes="Required for full run" if required else "Optional layer",
                )
            )
            continue
        recs.append(
            ProvenanceRecord(
                source_id=sid,
                path_or_hook=str(path.name),
                role=role,
                status="ok",
                modified_utc=mtime,
                size_mb=size_mb,
                rows_in_bundle=rows,
                notes="Loaded via data_pipeline",
            )
        )

    api = os.environ.get("SHIPPING_TRACKER_API_URL", "").strip()
    recs.append(
        ProvenanceRecord(
            source_id="shipping_tracker_api",
            path_or_hook=api or "(SHIPPING_TRACKER_API_URL not set)",
            role="env",
            status="configured" if api else "not_configured",
            modified_utc="—",
            size_mb=None,
            rows_in_bundle=None,
            notes="Merge logic in shipping_tracker.py when URL returns JSON",
        )
    )
    return recs


def max_input_freshness_utc(bundle: TrainingDataBundle) -> str:
    """Latest modification time among existing core CSV inputs."""
    root = project_root()
    candidates = [
        root / "wfp_food_prices_egy.csv",
        root / "gdelt_conflict_1_0.csv",
        root / "Daily_Port_Activity_Data_and_Trade_Estimates.csv",
        root / "fao_wheat_egy.csv",
        root / "fao_commodities_egy.csv",
        root / "shipping_metrics.csv",
        root / "news_headlines_bilingual.csv",
    ]
    mtimes = [p.stat().st_mtime for p in candidates if p.is_file()]
    if not mtimes:
        return "n/a"
    latest = max(mtimes)
    return datetime.fromtimestamp(latest, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def validate_bundle(bundle: TrainingDataBundle) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "check": "Wheat / blended price spine",
            "ok": not bundle.wheat_prices.empty,
            "detail": f"{len(bundle.wheat_prices)} months; {bundle.price_blend_note[:120]}",
        }
    )
    checks.append(
        {
            "check": "Engineered risk_features",
            "ok": not bundle.risk_features.empty,
            "detail": f"{len(bundle.risk_features)} rows × {len(bundle.risk_features.columns)} cols",
        }
    )
    if not bundle.risk_features.empty:
        m = bundle.risk_features["month"]
        mono = bool(m.is_monotonic_increasing)
        checks.append({"check": "month index monotonic", "ok": mono, "detail": "sorted ascending"})
        pu = bundle.risk_features["price_usd"]
        checks.append(
            {
                "check": "price_usd usable",
                "ok": pu.notna().any(),
                "detail": f"nan_share={1 - float(pu.notna().mean()):.2%}",
            }
        )
    n_fao_items = len(getattr(bundle, "fao_catalog", {}) or {})
    checks.append(
        {
            "check": "FAO multi-commodity catalog (optional)",
            "ok": True,
            "detail": (
                f"{n_fao_items} FAOSTAT item series mapped for WFP blending"
                if n_fao_items
                else "No fao_commodities_egy.csv — only wheat file or WFP-only"
            ),
        }
    )
    checks.append(
        {
            "check": "News corpus authenticity",
            "ok": not bundle.novelty.used_synthetic_news,
            "detail": "Real bilingual CSV" if not bundle.novelty.used_synthetic_news else "Synthetic headlines — replace for validation",
        }
    )
    checks.append(
        {
            "check": "GDELT / port layers present",
            "ok": not bundle.gdelt_yearly.empty and not bundle.port_monthly.empty,
            "detail": f"gdelt={len(bundle.gdelt_yearly)} port_agg={len(bundle.port_monthly)}",
        }
    )
    return checks


def composite_stress(row: pd.Series) -> float:
    return (
        float(row.get("price_stress_score", 0) or 0)
        + float(row.get("logistics_risk_score", 0) or 0)
        + float(row.get("geopolitical_risk_score", 0) or 0)
    ) / 3.0


def run_signal_backtest(
    risk_features: pd.DataFrame,
    stress_threshold: float = 55.0,
    spike_z_threshold: float = 1.25,
    max_lead_months: int = 3,
) -> dict[str, Any]:
    """
    Retrospective harness: 'signal' = average stress ≥ threshold; 'event' = |price spike z| ≥ spike_z_threshold.
    - recall_proxy: share of spike months with any signal True in the previous max_lead months
    - precision_proxy: share of signal on-months followed by an event within 1..max_lead months
    """
    out: dict[str, Any] = {
        "n_months": 0,
        "stress_threshold": stress_threshold,
        "spike_z_threshold": spike_z_threshold,
        "max_lead_months": max_lead_months,
    }
    if risk_features.empty or "price_usd" not in risk_features.columns:
        out["error"] = "risk_features missing price_usd"
        return out

    df = risk_features.sort_values("month").reset_index(drop=True)
    z = price_spike_series(df["price_usd"])
    event = z.fillna(0).abs() >= spike_z_threshold
    stress = df.apply(composite_stress, axis=1)
    signal = stress >= stress_threshold

    df = df.assign(_spike_z=z, _stress=stress, _event=event, _signal=signal)

    spike_idx = np.where(event.values)[0]
    hits = 0
    for i in spike_idx:
        lo = max(0, i - max_lead_months)
        if lo < i and signal.iloc[lo:i].any():
            hits += 1

    recalls = float(hits / len(spike_idx)) if len(spike_idx) else float("nan")

    sig_on = signal.values
    prec_hits = 0
    prec_total = 0
    for i in range(len(df)):
        if not sig_on[i]:
            continue
        prec_total += 1
        hi = min(len(df), i + max_lead_months + 1)
        if event.iloc[i + 1 : hi].any():
            prec_hits += 1
    precision = float(prec_hits / prec_total) if prec_total else float("nan")

    onset_hits = 0
    onset_total = 0
    for i in range(len(df)):
        onset = bool(sig_on[i]) and (i == 0 or not sig_on[i - 1])
        if not onset:
            continue
        onset_total += 1
        hi = min(len(df), i + max_lead_months + 1)
        if event.iloc[i + 1 : hi].any():
            onset_hits += 1
    precision_onset = float(onset_hits / onset_total) if onset_total else float("nan")

    out["n_months"] = int(len(df))
    out["n_event_months"] = int(event.sum())
    out["n_signal_months"] = int(signal.sum())
    out["spikes_with_prior_signal_window"] = int(hits)
    out["recall_proxy_on_spike_months"] = recalls
    out["precision_proxy_conditional_on_signal"] = precision
    out["precision_proxy_signal_onset_only"] = precision_onset
    out["series_head"] = df[["month", "_stress", "_spike_z", "_event", "_signal"]].tail(24)
    return out


def calibration_risk_bands() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"band": "Low", "unified_risk_range": "[0, 35)", "default_ops": "Routine monitoring"},
            {"band": "Moderate", "unified_risk_range": "[35, 55)", "default_ops": "Weekly cross-check sources"},
            {"band": "High", "unified_risk_range": "[55, 75)", "default_ops": "Expedite procurement review"},
            {"band": "Critical", "unified_risk_range": "[75, 100]", "default_ops": "Activate contingency sourcing"},
        ]
    )


def summarize_kb_alerts(alerts: list[InferenceAlert]) -> pd.DataFrame:
    if not alerts:
        return pd.DataFrame(columns=["rule_id", "severity", "confidence", "triggers"])
    return pd.DataFrame(
        [
            {
                "rule_id": a.rule_id,
                "severity": a.severity,
                "confidence": a.confidence,
                "triggers": ", ".join(a.fired_triggers),
            }
            for a in alerts
        ]
    )


def time_uncached_bundle_load(assemble_fn) -> tuple[Any, float]:
    """Wall-clock ms for a full pipeline run (use after clearing Streamlit cache)."""
    t0 = time.perf_counter()
    bundle = assemble_fn()
    return bundle, (time.perf_counter() - t0) * 1000.0


def render_ops_sidebar_status(bundle: TrainingDataBundle, inference_context: dict[str, Any]) -> None:
    """Compact operations strip for the sidebar."""
    st.markdown("---")
    st.markdown("##### Early warning ops")
    st.caption(f"Input freshness (latest file mtime): **{max_input_freshness_utc(bundle)}**")
    alerts = run_inference(inference_context)
    if alerts:
        crit = sum(1 for a in alerts if a.severity == "critical")
        st.warning(f"**{len(alerts)}** knowledge-base alert(s){f' ({crit} critical)' if crit else ''} — see **Early Warning (Ops)**.")

