"""Forward-chaining style inference over knowledge-base rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.knowledge_base import RULES, eval_threshold


@dataclass
class InferenceAlert:
    rule_id: str
    severity: str
    action: str
    confidence: float
    fired_triggers: list[str]


def run_inference(context: dict[str, Any]) -> list[InferenceAlert]:
    """
    context keys (optional):
      price_stress_score, geopolitical_risk_score, logistics_risk_score,
      port_activity_drop, forecast_trend, strategic_stock_score, unified_risk_score,
      ahp_cr
    """
    alerts: list[InferenceAlert] = []

    pss = float(context.get("price_stress_score", 0) or 0)
    geo = float(context.get("geopolitical_risk_score", 0) or 0)
    log = float(context.get("logistics_risk_score", 0) or 0)
    port_drop = float(context.get("port_activity_drop", 0) or 0)
    trend = str(context.get("forecast_trend", "flat"))
    stock = float(context.get("strategic_stock_score", 50) or 50)
    uni = float(context.get("unified_risk_score", 0) or 0)
    cr = float(context.get("ahp_cr", 0) or 0)

    if eval_threshold(pss) == "high" and eval_threshold(geo) == "high":
        alerts.append(
            InferenceAlert(
                rule_id="R1",
                severity="high",
                action=RULES[0].action,
                confidence=0.78,
                fired_triggers=["price_stress_score", "geopolitical_risk_score"],
            )
        )

    if port_drop > 15 and eval_threshold(log) == "high":
        alerts.append(
            InferenceAlert(
                rule_id="R2",
                severity="moderate",
                action=RULES[1].action,
                confidence=0.72,
                fired_triggers=["port_activity_drop", "logistics_risk_score"],
            )
        )

    if trend == "up" and stock < 40:
        alerts.append(
            InferenceAlert(
                rule_id="R3",
                severity="moderate",
                action=RULES[2].action,
                confidence=0.70,
                fired_triggers=["forecast_trend", "strategic_stock_score"],
            )
        )

    if uni >= 75:
        alerts.append(
            InferenceAlert(
                rule_id="R4",
                severity="critical",
                action=RULES[3].action,
                confidence=0.88,
                fired_triggers=["unified_risk_score"],
            )
        )

    if cr > 0.1:
        alerts.append(
            InferenceAlert(
                rule_id="R5",
                severity="low",
                action=RULES[4].action,
                confidence=0.92,
                fired_triggers=["ahp_consistency_ratio"],
            )
        )

    return alerts
