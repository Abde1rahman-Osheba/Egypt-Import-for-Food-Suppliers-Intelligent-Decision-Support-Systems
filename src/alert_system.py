"""Evidence-based maritime / commodity alerts (resolves when conditions ease)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import pandas as pd

from src.risk_scoring import safe_float


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class AlertStatus(str, Enum):
    ACTIVE = "Active"
    MONITORING = "Monitoring"
    RESOLVED = "Resolved"


@dataclass
class OperationalAlert:
    alert_id: str
    severity: AlertSeverity
    alert_type: str
    title: str
    description: str
    related_ship: Optional[str]
    related_port: Optional[str]
    related_commodity: Optional[str]
    related_zone: Optional[str]
    triggered_rule: str
    recommended_action: str
    status: AlertStatus


def evaluate_maritime_alerts(
    vessels: pd.DataFrame,
    ports: pd.DataFrame,
    context: dict[str, Any],
) -> list[OperationalAlert]:
    alerts: list[OperationalAlert] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pss = safe_float(context.get("price_stress_score") or context.get("pss"))
    geo = safe_float(context.get("geopolitical_risk_score") or context.get("geo"))
    log = safe_float(context.get("logistics_risk_score") or context.get("log"))

    aid = 0

    if pss >= 65:
        aid += 1
        alerts.append(
            OperationalAlert(
                f"A-{aid:03d}",
                AlertSeverity.HIGH if pss < 80 else AlertSeverity.CRITICAL,
                "commodity_price",
                "Elevated wheat / price stress",
                f"Price stress proxy at {pss:.1f}/100 — monitor procurement tenders ({now}).",
                None,
                None,
                "wheat",
                None,
                "price_stress>=65",
                "Review forward coverage; consider hedging or accelerated shipment dates.",
                AlertStatus.ACTIVE,
            )
        )

    if geo >= 70:
        aid += 1
        alerts.append(
            OperationalAlert(
                f"A-{aid:03d}",
                AlertSeverity.HIGH,
                "geopolitical_escalation",
                "Geopolitical stress escalation",
                f"Geopolitical score {geo:.1f}/100 — supplier corridor uncertainty.",
                None,
                None,
                "wheat,maize",
                None,
                "geopolitical>=70",
                "Increase ministry situation briefings; validate insurance war-risk clauses.",
                AlertStatus.MONITORING,
            )
        )

    if not vessels.empty:
        for _, v in vessels.iterrows():
            rsk = safe_float(v.get("route_risk_score"))
            near = bool(v.get("near_risk_zone"))
            cargo = str(v.get("cargo_type", ""))
            nm = str(v.get("ship_name", ""))
            sid = str(v.get("ship_id", ""))
            delay_p = safe_float(v.get("delay_probability"))
            rec = str(v.get("recommendation", ""))

            if rsk >= 70 and near:
                aid += 1
                sev = AlertSeverity.CRITICAL if rsk >= 82 else AlertSeverity.HIGH
                st = AlertStatus.ACTIVE
                alerts.append(
                    OperationalAlert(
                        f"A-{aid:03d}",
                        sev,
                        "conflict_proximity",
                        f"Route exposure near risk zone — {nm}",
                        f"{nm} ({sid}) route risk {rsk:.0f}/100 near {v.get('near_zone_names', '')}.",
                        sid,
                        str(v.get("dest_port_id")),
                        cargo,
                        str(v.get("near_zone_names", "")).split("|")[0] if v.get("near_zone_names") else None,
                        "vessel near modeled zone",
                        "Assess rerouting per suggested corridor; notify port authority ETA watch.",
                        st,
                    )
                )
            elif rsk >= 55 and "wheat" in cargo.lower():
                aid += 1
                alerts.append(
                    OperationalAlert(
                        f"A-{aid:03d}",
                        AlertSeverity.MODERATE,
                        "strategic_cargo",
                        f"Food cargo monitoring — {nm}",
                        f"Laden {cargo} with route risk {rsk:.0f}/100.",
                        sid,
                        str(v.get("dest_port_id")),
                        cargo,
                        None,
                        "wheat lane risk",
                        "Maintain daily ETA & charter communication.",
                        AlertStatus.MONITORING,
                    )
                )

            if delay_p >= 0.72 and rsk >= 60:
                aid += 1
                alerts.append(
                    OperationalAlert(
                        f"A-{aid:03d}",
                        AlertSeverity.HIGH,
                        "delay_risk",
                        f"Delay probability elevated — {nm}",
                        f"Modelled delay probability {delay_p:.0%} with route risk {rsk:.0f}.",
                        sid,
                        str(v.get("dest_port_id")),
                        cargo,
                        None,
                        "delay heuristic",
                        "Pre-alert discharge terminals; prep berthing contingency.",
                        AlertStatus.ACTIVE,
                    )
                )

    if not ports.empty:
        eg = ports[ports["country"].astype(str).str.contains("gypt", case=False, na=False)]
        for _, p in eg.iterrows():
            cong = safe_float(p.get("congestion_level"), 0) * 100
            if cong >= 58:
                aid += 1
                alerts.append(
                    OperationalAlert(
                        f"A-{aid:03d}",
                        AlertSeverity.MODERATE,
                        "port_congestion",
                        f"Congestion watch — {p.get('name')}",
                        f"{p.get('name')} congestion index high (~{cong:.0f}).",
                        None,
                        str(p.get("port_id")),
                        None,
                        None,
                        "congestion>=58",
                        "Coordinate with MOT/maritime authority on berth scheduling.",
                        AlertStatus.MONITORING,
                    )
                )

    resolved_notes: list[OperationalAlert] = []
    if not vessels.empty:
        for _, v in vessels.iterrows():
            rsk = safe_float(v.get("route_risk_score"))
            near = bool(v.get("near_risk_zone"))
            if rsk < 45 and not near and str(v.get("recommendation")) == "keep_route":
                aid += 1
                resolved_notes.append(
                    OperationalAlert(
                        f"A-{aid:03d}",
                        AlertSeverity.LOW,
                        "route_normalized",
                        f"No rerouting required — {v.get('ship_name')}",
                        "Current route acceptable under latest modeled risk conditions.",
                        str(v.get("ship_id")),
                        str(v.get("dest_port_id")),
                        str(v.get("cargo_type")),
                        None,
                        "risk<45 & !near_zone",
                        "Continue standard monitoring.",
                        AlertStatus.RESOLVED,
                    )
                )

    return alerts + resolved_notes


def alerts_to_dataframe(alerts: list[OperationalAlert]) -> pd.DataFrame:
    if not alerts:
        return pd.DataFrame(
            columns=[
                "alert_id",
                "severity",
                "type",
                "title",
                "status",
                "related_ship",
            ]
        )
    return pd.DataFrame(
        [
            {
                "alert_id": a.alert_id,
                "severity": a.severity.value,
                "type": a.alert_type,
                "title": a.title,
                "status": a.status.value,
                "related_ship": a.related_ship,
                "related_port": a.related_port,
            }
            for a in alerts
        ]
    )
