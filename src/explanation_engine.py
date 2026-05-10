"""Human-readable explanations for DSS outputs."""

from __future__ import annotations

from typing import Any

from src.inference_engine import InferenceAlert
from src.knowledge_base import rule_by_id


def explain_alert(alert: InferenceAlert, context: dict[str, Any]) -> str:
    r = rule_by_id(alert.rule_id)
    src = ", ".join(r.data_sources) if r else "internal models"
    drivers = ", ".join(alert.fired_triggers)
    return (
        f"Rule {alert.rule_id} fired because indicators [{drivers}] crossed expert thresholds "
        f"in the current context (confidence {alert.confidence:.0%}). "
        f"Supporting data layers: {src}. Recommended move: {alert.action}"
    )


def explain_uncertainty_decision(criterion: str, choice: str) -> str:
    return (
        f"Under uncertainty, the {criterion} criterion selects '{choice}' because it optimizes "
        f"that criterion's objective over the scenario payoff matrix."
    )


def methodology_blurb(page: str) -> str:
    blurbs = {
        "Commodity Price Intelligence": "Demonstrates forecasting as a model-management component for price stress.",
        "Port & Logistics Intelligence": "Connects port telemetry to logistics risk within the data subsystem.",
        "Geopolitical Intelligence": "Maps GDELT signals to geopolitical risk for the knowledge base.",
        "Cross-source signals: prices, shipping & NLP": "Combines FAOSTAT-style Egyptian prices (when blended with WFP), optional shipping trackers, and bilingual Arabic–English news NLP indices against selectable commodity spikes.",
        "Unified Risk Early Warning": "Implements composite scoring, alerts, and sensitivity checks.",
        "Port & Vessel Intelligence": "Dedicated maritime control room: ports, synthetic vessels, great-circle-style routes, risk zones, alerts, and Ollama Q&A.",
        "Scenario Simulator": "Interactive what-if experiments across fused indicators.",
        "Goal Programming Optimizer": "Weighted goal programming for importer response strategies.",
        "AHP Weighting & Sensitivity Analysis": "AHP derives criterion weights; sliders test stability.",
        "Recommendations Center": "Inference engine outputs meet the explanation facility.",
        "Knowledge Base & Inference Engine": "Explicit IF-THEN structure per IDSS theory.",
    }
    return blurbs.get(page, "IDSS concept: structured decision support.")
