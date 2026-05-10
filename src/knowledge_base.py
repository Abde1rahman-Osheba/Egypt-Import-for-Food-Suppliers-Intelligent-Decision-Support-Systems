"""Expert rules for food import disruption DSS."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Rule:
    rule_id: str
    condition: str
    action: str
    triggers: list[str]
    data_sources: list[str]


RULES: list[Rule] = [
    Rule(
        rule_id="R1",
        condition="wheat_price_stress_high AND black_sea_geo_high",
        action="Recommend supplier diversification and monitor alternative origins.",
        triggers=["price_stress_score", "geopolitical_risk_score"],
        data_sources=["WFP Egypt prices", "GDELT conflict", "FAO blend", "Bilingual news NLP"],
    ),
    Rule(
        rule_id="R2",
        condition="port_activity_drop_sharp AND logistics_risk_high",
        action="Recommend alternative port/route and expedited customs planning.",
        triggers=["port_activity_drop", "logistics_risk_score"],
        data_sources=["Daily port activity"],
    ),
    Rule(
        rule_id="R3",
        condition="price_forecast_up AND strategic_stock_low",
        action="Recommend early procurement and hedge exposure.",
        triggers=["forecast_trend", "strategic_stock_score"],
        data_sources=["WFP prices", "forecasting module"],
    ),
    Rule(
        rule_id="R4",
        condition="final_risk_critical",
        action="Activate emergency procurement plan and institute daily risk calls.",
        triggers=["unified_risk_score"],
        data_sources=["All fused indicators"],
    ),
    Rule(
        rule_id="R5",
        condition="ahp_consistency_poor",
        action="Warn: AHP weights may be unreliable; revisit pairwise comparisons.",
        triggers=["ahp_consistency_ratio"],
        data_sources=["AHP module"],
    ),
]


def get_rules() -> list[Rule]:
    return RULES


def rules_as_dataframe() -> pd.DataFrame:
    """Tabular view for the Knowledge Base page."""
    return pd.DataFrame(
        [
            {
                "Rule": r.rule_id,
                "Condition (IF)": r.condition,
                "Action (THEN)": r.action,
                "Triggers": ", ".join(r.triggers),
                "Data sources": ", ".join(r.data_sources),
            }
            for r in RULES
        ]
    )


def rule_by_id(rid: str) -> Rule | None:
    for r in RULES:
        if r.rule_id == rid:
            return r
    return None


def eval_threshold(
    value: float,
    high: float = 65.0,
    critical: float = 80.0,
) -> str:
    if value >= critical:
        return "critical"
    if value >= high:
        return "high"
    return "low"
