"""Weighted penalty goal programming for importer response strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_GOAL_WEIGHTS = {
    "import_disruption": 0.22,
    "procurement_cost": 0.18,
    "supplier_reliability": 0.15,
    "stock_coverage": 0.15,
    "route_risk": 0.15,
    "delivery_time": 0.15,
}

ALTERNATIVES = [
    "Maintain current supplier",
    "Early procurement",
    "Diversify suppliers",
    "Shift to alternative region",
    "Increase safety stock",
    "Use alternative port/route",
    "Emergency procurement plan",
]


def _alt_matrix() -> np.ndarray:
    """
    Rows: alternatives. Cols: disruption, cost_increase, reliability_inv,
    stock_shortage_inv, route_risk, delay - each 0-10 penalty proxy.
    """
    return np.array(
        [
            [6, 4, 2, 7, 5, 3],
            [3, 5, 4, 3, 4, 4],
            [4, 5, 7, 4, 3, 5],
            [5, 6, 5, 4, 6, 6],
            [4, 4, 3, 2, 4, 5],
            [3, 5, 4, 5, 2, 4],
            [2, 8, 6, 1, 7, 2],
        ]
    )


def run_goal_programming(
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, str, dict[str, float]]:
    weights = weights or DEFAULT_GOAL_WEIGHTS
    M = _alt_matrix()
    w = np.array(
        [
            weights["import_disruption"],
            weights["procurement_cost"],
            weights["supplier_reliability"],
            weights["stock_coverage"],
            weights["route_risk"],
            weights["delivery_time"],
        ]
    )
    w = w / w.sum()

    col_names = [
        "cost_penalty",
        "risk_penalty",
        "supplier_reliability_penalty",
        "stock_shortage_penalty",
        "route_logistics_penalty",
        "delay_penalty",
    ]
    goal_names = [
        "import_disruption",
        "procurement_cost",
        "supplier_reliability",
        "stock_coverage",
        "route_risk",
        "delivery_time",
    ]

    rows: list[dict[str, Any]] = []
    for i, alt in enumerate(ALTERNATIVES):
        penalties = M[i]
        weighted = penalties * w * 10
        total = float(weighted.sum())
        row = {"alternative": alt, "total_goal_deviation": total}
        for j, cn in enumerate(col_names):
            row[cn] = float(penalties[j])
        for j, gn in enumerate(goal_names):
            row[f"{gn}_weighted"] = float(weighted[j])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("total_goal_deviation").reset_index(drop=True)
    best = df.iloc[0]["alternative"]
    expl = (
        f"{best} is recommended because it minimizes the weighted penalty across goals "
        f"(disruption, cost, reliability, stock, route risk, and delivery time)."
    )
    return df, expl, weights
