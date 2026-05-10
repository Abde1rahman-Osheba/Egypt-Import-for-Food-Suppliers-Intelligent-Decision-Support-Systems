"""Analytic Hierarchy Process for risk factor weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


CRITERIA = [
    "Geopolitical Risk",
    "Logistics Risk",
    "Price Stress",
    "Supplier Dependency",
    "Strategic Stock Level",
    "Alternative Supplier Readiness",
]


def default_comparison_matrix() -> np.ndarray:
    """Saaty-style demo-derived pairwise matrix (6x6)."""
    n = len(CRITERIA)
    A = np.ones((n, n))
    base = np.array(
        [
            [1, 3, 2, 4, 5, 4],
            [1 / 3, 1, 1 / 2, 2, 3, 2],
            [1 / 2, 2, 1, 3, 4, 3],
            [1 / 4, 1 / 2, 1 / 3, 1, 2, 2],
            [1 / 5, 1 / 3, 1 / 4, 1 / 2, 1, 1 / 2],
            [1 / 4, 1 / 2, 1 / 3, 1 / 2, 2, 1],
        ]
    )
    return base


def priority_vector(A: np.ndarray) -> np.ndarray:
    col_sums = A.sum(axis=0)
    norm = A / col_sums
    return norm.mean(axis=1)


def consistency_ratio(A: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    n = A.shape[0]
    Aw = A @ w
    lambdas = Aw / np.clip(w, 1e-12, None)
    lambda_max = float(lambdas.mean())
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32}
    ri = ri_table.get(n, 1.24)
    cr = ci / ri if ri > 0 else 0.0
    return float(ci), float(cr), float(lambda_max)


@dataclass
class AHPResult:
    weights: dict[str, float]
    matrix: np.ndarray
    ci: float
    cr: float
    consistent: bool


def run_ahp(pairwise: np.ndarray | None = None, cr_threshold: float = 0.1) -> AHPResult:
    A = pairwise if pairwise is not None else default_comparison_matrix()
    w = priority_vector(A)
    ci, cr, _ = consistency_ratio(A, w)
    names = CRITERIA[: len(w)]
    weights = {names[i]: float(w[i]) for i in range(len(names))}
    return AHPResult(
        weights=weights,
        matrix=A,
        ci=ci,
        cr=cr,
        consistent=cr <= cr_threshold,
    )


def ahp_explanation(res: AHPResult) -> str:
    top = max(res.weights.items(), key=lambda x: x[1])[0]
    frag = (
        f"AHP assigns the highest weight to {top}, reflecting pairwise judgments on "
        f"which criterion dominates Egyptian import stability."
    )
    if not res.consistent:
        frag += (
            f" Consistency ratio {res.cr:.3f} exceeds typical 0.10 — review comparisons."
        )
    return frag
