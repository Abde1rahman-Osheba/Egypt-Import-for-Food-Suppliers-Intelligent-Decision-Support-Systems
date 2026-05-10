"""Ministry-style decision recommendation cards (rules + alert linkage)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.alert_system import AlertSeverity, AlertStatus, OperationalAlert


def _priority(sev: AlertSeverity) -> str:
    if sev == AlertSeverity.CRITICAL:
        return "P0"
    if sev == AlertSeverity.HIGH:
        return "P1"
    if sev == AlertSeverity.MODERATE:
        return "P2"
    return "P3"


def build_ministry_recommendations(
    alerts: list[OperationalAlert],
    vessels: pd.DataFrame,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for a in alerts:
        if a.status == AlertStatus.RESOLVED:
            benefit = "Clarifies that prior exposure no longer requires rerouting."
        elif a.severity == AlertSeverity.CRITICAL:
            benefit = "Avoids worst-case delay or diversion cost for strategic cargo."
        elif a.severity == AlertSeverity.HIGH:
            benefit = "Reduces exposure window on high-risk legs."
        else:
            benefit = "Improves situational awareness for port and ministry coordination."

        ev = a.description
        if a.related_ship and not vessels.empty:
            m = vessels[vessels["ship_id"].astype(str) == str(a.related_ship)]
            if not m.empty:
                r = m.iloc[0]
                ev += f" | ETA context: route_risk={float(r.get('route_risk_score', 0)):.0f}"

        cards.append(
            {
                "title": a.title,
                "priority": _priority(a.severity),
                "action": a.recommended_action,
                "evidence": ev,
                "related_ship": a.related_ship,
                "related_port": a.related_port,
                "related_commodity": a.related_commodity,
                "expected_benefit": benefit,
                "status": a.status.value,
                "alert_type": a.alert_type,
            }
        )

    if not vessels.empty:
        top = vessels.sort_values("route_risk_score", ascending=False, na_position="last").head(1)
        if not top.empty:
            r0 = top.iloc[0]
            cards.insert(
                0,
                {
                    "title": f"Focus watch: {r0.get('ship_name')}",
                    "priority": "P1",
                    "action": str(r0.get("recommendation", "monitor")).replace("_", " ").title(),
                    "evidence": str(r0.get("explainer", ""))[:400],
                    "related_ship": r0.get("ship_id"),
                    "related_port": r0.get("dest_port_id"),
                    "related_commodity": r0.get("cargo_type"),
                    "expected_benefit": "Prioritizes ministry attention on highest modeled route risk.",
                    "status": "Active",
                    "alert_type": "executive_highlight",
                },
            )

    return cards


def recommendations_to_markdown_bullets(cards: list[dict[str, Any]], max_items: int = 8) -> str:
    lines = []
    for c in cards[:max_items]:
        lines.append(f"- [{c['priority']}] **{c['title']}**: {c['action']}")
    return "\n".join(lines) if lines else "No active recommendations."
