"""IDSS quantitative methods: forecasting, decision criteria, sensitivity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    ExponentialSmoothing = None


ForecastMethod = Literal["auto", "naive", "sma", "wma", "exponential_smoothing"]


@dataclass
class ForecastResult:
    next_price: float
    trend: Literal["up", "down", "flat"]
    confidence_note: str
    residual_mae: Optional[float]
    warning_explanation: str
    history_actual: pd.Series
    history_fitted: pd.Series
    method_label: str = ""
    next_price_egp: float = float("nan")
    history_actual_egp: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    history_fitted_egp: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    commodity_name: str = "Commodity"


def _rolling_window_size(n: int) -> int:
    return min(6, max(3, n // 4)) if n >= 3 else max(1, n)


def naive_fitted_forecast(y: pd.Series) -> tuple[pd.Series, float, str]:
    s = pd.to_numeric(y, errors="coerce").dropna().reset_index(drop=True)
    if len(s) < 1:
        return pd.Series(dtype=float), float("nan"), "Naive (last value)"
    fitted = pd.Series([np.nan] * len(s), dtype=float)
    for i in range(1, len(s)):
        fitted.iloc[i] = s.iloc[i - 1]
    next_p = float(s.iloc[-1])
    return fitted, next_p, "Naive (last observed = next forecast)"


def sma_fitted_forecast(y: pd.Series, window: Optional[int] = None) -> tuple[pd.Series, float, str]:
    s = pd.to_numeric(y, errors="coerce").dropna().reset_index(drop=True)
    n = len(s)
    if n < 1:
        return pd.Series(dtype=float), float("nan"), "Simple moving average"
    w = window if window is not None else _rolling_window_size(n)
    w = max(2, min(w, n))
    fitted = s.rolling(w, min_periods=w).mean()
    tail = s.iloc[-w:]
    next_p = float(tail.mean())
    return fitted, next_p, f"Simple moving average ({w}-period)"


def wma_fitted_forecast(y: pd.Series, window: Optional[int] = None) -> tuple[pd.Series, float, str]:
    """Linear weights; most recent month has highest weight."""
    s = pd.to_numeric(y, errors="coerce").dropna().reset_index(drop=True)
    n = len(s)
    if n < 1:
        return pd.Series(dtype=float), float("nan"), "Weighted moving average"
    wlen = window if window is not None else _rolling_window_size(n)
    wlen = max(2, min(wlen, n))
    weights = np.arange(1, wlen + 1, dtype=float)

    fitted_vals: list[float] = []
    for i in range(len(s)):
        lo = max(0, i - wlen + 1)
        seg = s.iloc[lo : i + 1].values
        wl = len(seg)
        if wl < 2:
            fitted_vals.append(np.nan)
            continue
        ww = weights[-wl:]
        fitted_vals.append(float(np.dot(seg, ww) / ww.sum()))
    fitted = pd.Series(fitted_vals, dtype=float)

    tail = s.iloc[-wlen:].values
    wl = len(tail)
    ww = weights[-wl:]
    next_p = float(np.dot(tail, ww) / ww.sum())
    return fitted, next_p, f"Weighted moving average ({wlen}-period, recency-weighted)"


def exponential_fitted_forecast(y: pd.Series) -> tuple[pd.Series, float, str]:
    s = pd.to_numeric(y, errors="coerce").dropna().reset_index(drop=True)
    if len(s) < 3:
        return pd.Series(dtype=float), float("nan"), "Exponential smoothing"
    if ExponentialSmoothing is None or len(s) < 8:
        sm_fit, sm_next, _ = sma_fitted_forecast(s)
        alt = exponential_smoothing_forecast(s)
        nx = float(alt) if alt is not None and not (isinstance(alt, float) and np.isnan(alt)) else sm_next
        return sm_fit, nx, "Exponential smoothing (fallback to moving average when series is short)"
    try:
        model = ExponentialSmoothing(
            s.values, trend="add", seasonal=None, initialization_method="estimated"
        )
        fit = model.fit(optimized=True)
        fitted = pd.Series(fit.fittedvalues, dtype=float)
        next_p = float(fit.forecast(1)[0])
        return fitted, next_p, "Holt-Winters-style exponential smoothing (statsmodels)"
    except Exception:
        sm_fit, sm_next, _ = sma_fitted_forecast(s)
        alt = exponential_smoothing_forecast(s)
        nx = float(alt) if alt is not None and not (isinstance(alt, float) and np.isnan(alt)) else sm_next
        return sm_fit, nx, "Exponential smoothing (numeric fallback)"


def apply_forecast_method(
    y: pd.Series,
    method: ForecastMethod,
) -> tuple[pd.Series, float, str]:
    if method == "naive":
        return naive_fitted_forecast(y)
    if method == "sma":
        return sma_fitted_forecast(y)
    if method == "wma":
        return wma_fitted_forecast(y)
    if method in ("auto", "exponential_smoothing"):
        fitted, nx, lab = exponential_fitted_forecast(y)
        if np.isnan(nx) or nx is None:
            fitted, nx, lab2 = sma_fitted_forecast(y)
            lab = f"{lab2} (fallback)"
        return fitted, nx, lab
    return sma_fitted_forecast(y)


def explain_residual_panel(
    act: pd.Series,
    fitted: pd.Series,
    err: pd.Series,
    window_months: int,
    currency: str,
) -> str:
    """Contextual copy for the residual bar chart (updates with trailing window)."""
    act = pd.to_numeric(act, errors="coerce")
    fitted = pd.to_numeric(fitted, errors="coerce")
    err = pd.to_numeric(err, errors="coerce")
    valid = err.dropna()
    if valid.empty:
        return f"No residual bars in this **{window_months}-month** window (not enough fitted points)."
    mae = float(valid.abs().mean())
    bias = float(valid.mean())
    pos_share = float((valid > 0).mean()) if len(valid) else 0.0
    max_idx = int(valid.abs().argmax()) if len(valid) else 0
    dom = "above" if bias > 0 else "below" if bias < 0 else "split around"
    return (
        f"**Residuals ({currency}, this {window_months}-month chart)** — each bar is actual minus the model’s in-sample baseline "
        f"for that month. Mean absolute error ≈ **{mae:.4f}**; errors sit **{dom}** zero on average (mean ≈ **{bias:+.4f}**). "
        f"**{pos_share:.0%}** of months are positive residuals (actual higher than baseline). "
        f"Largest absolute miss is in the **last** window position index **{max_idx + 1}** "
        f"(0 = oldest month shown). Narrow the trailing-month slider to focus on recent regime shifts."
    )


def moving_average_forecast(series: pd.Series, window: int = 3) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 1:
        return float("nan")
    w = min(window, len(s))
    return float(s.iloc[-w:].mean())


def exponential_smoothing_forecast(series: pd.Series) -> Optional[float]:
    if ExponentialSmoothing is None:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 8:
        return None
    try:
        model = ExponentialSmoothing(
            s.values, trend="add", seasonal=None, initialization_method="estimated"
        )
        fit = model.fit(optimized=True)
        return float(fit.forecast(1)[0])
    except Exception:
        return None


def _system_default_forecast_y(y: pd.Series) -> tuple[float, pd.Series, str]:
    """
    Rest-of-app default: rolling-MA baseline for stress/residuals; next period from
    exponential smoothing when possible, else MA (matches original DSS behavior).
    """
    s = pd.to_numeric(y, errors="coerce").dropna().reset_index(drop=True)
    if len(s) < 1:
        return float("nan"), pd.Series(dtype=float), "n/a"
    ma_window = min(6, max(3, len(s) // 4))
    fitted_ma = s.rolling(ma_window, min_periods=2).mean()
    next_p = exponential_smoothing_forecast(s)
    method = "Holt-Winters-style exponential smoothing (statsmodels)"
    if next_p is None or np.isnan(next_p):
        next_p = moving_average_forecast(s, window=ma_window)
        method = f"{ma_window}-period moving average"
    return float(next_p), fitted_ma.reset_index(drop=True), method


def forecast_with_method(
    price_df: pd.DataFrame,
    method: ForecastMethod,
    *,
    date_col: str = "month",
    commodity_name: str = "Commodity",
) -> ForecastResult:
    """Interactive commodity page: one consistent method for fitted + next on USD."""
    if price_df.empty or "price_usd" not in price_df.columns:
        return ForecastResult(
            next_price=np.nan,
            trend="flat",
            confidence_note="Insufficient price history.",
            residual_mae=None,
            warning_explanation="No price_usd column or empty frame.",
            history_actual=pd.Series(dtype=float),
            history_fitted=pd.Series(dtype=float),
            method_label="—",
            commodity_name=commodity_name,
        )

    df = price_df.sort_values(date_col).copy()
    y_usd = pd.to_numeric(df["price_usd"], errors="coerce")
    mask = y_usd.notna()
    y_usd = y_usd.loc[mask].reset_index(drop=True)

    if len(y_usd) < 2:
        return ForecastResult(
            next_price=float("nan"),
            trend="flat",
            confidence_note="Need at least 2 USD observations.",
            residual_mae=None,
            warning_explanation=f"{commodity_name}: USD series too short.",
            history_actual=y_usd,
            history_fitted=pd.Series(dtype=float),
            method_label="—",
            commodity_name=commodity_name,
        )

    fitted_usd, next_usd, mlab = apply_forecast_method(y_usd, method)

    residuals = y_usd - fitted_usd
    residual_mae = float(residuals.abs().mean()) if len(residuals.dropna()) else None

    tail = float(y_usd.iloc[-1])
    if not np.isnan(next_usd):
        if next_usd > tail * 1.01:
            trend: Literal["up", "down", "flat"] = "up"
        elif next_usd < tail * 0.99:
            trend = "down"
        else:
            trend = "flat"
    else:
        trend = "flat"

    vol = float(y_usd.pct_change().std() or 0)
    li = int(fitted_usd.last_valid_index()) if fitted_usd.notna().any() else -1
    fv = float(fitted_usd.iloc[li]) if li >= 0 else tail
    stress = tail > fv
    vol_note = "recent volatility is elevated" if vol > 0.05 else "recent volatility is moderate"
    stress_note = (
        "the current price is already above the model baseline"
        if stress
        else "the current price is near the model baseline"
    )
    warn = (
        f"{commodity_name}: next-period forecast is **{trend}** using **{mlab}**. "
        f"This view uses the same baseline for stress context: {stress_note}; {vol_note}."
    )

    conf = (
        f"Model on chart: **{mlab}**. Next USD ≈ **{next_usd:.4f}**. "
        + (
            f"Mean absolute error vs baseline (full series) ≈ {residual_mae:.4f} USD."
            if residual_mae
            else "Residual vs baseline not computed."
        )
    )

    return ForecastResult(
        next_price=float(next_usd) if not np.isnan(next_usd) else float("nan"),
        trend=trend,
        confidence_note=conf,
        residual_mae=residual_mae,
        warning_explanation=warn,
        history_actual=y_usd,
        history_fitted=fitted_usd.reset_index(drop=True),
        method_label=mlab,
        commodity_name=commodity_name,
    )


def forecast_commodity_price(
    price_df: pd.DataFrame,
    value_col: str = "price_usd",
    date_col: str = "month",
) -> ForecastResult:
    """System-wide default forecast (blended wheat spine): MA baseline + exp/MA next — unchanged semantics."""
    com = "Wheat flour"
    if price_df is not None and not price_df.empty and "commodity" in price_df.columns:
        try:
            com = str(price_df["commodity"].dropna().iloc[-1])
        except Exception:
            pass
    if price_df.empty or value_col not in price_df.columns:
        return ForecastResult(
            next_price=np.nan,
            trend="flat",
            confidence_note="Insufficient price history.",
            residual_mae=None,
            warning_explanation="No reliable wheat price series loaded.",
            history_actual=pd.Series(dtype=float),
            history_fitted=pd.Series(dtype=float),
            method_label="system default",
            commodity_name=com,
        )

    df = price_df.sort_values(date_col).copy()
    y_raw = pd.to_numeric(df[value_col], errors="coerce")
    valid = y_raw.notna()
    y = y_raw.loc[valid].reset_index(drop=True)
    if len(y) < 3:
        return ForecastResult(
            next_price=float("nan"),
            trend="flat",
            confidence_note="Need at least 3 observations for moving average.",
            residual_mae=None,
            warning_explanation="WFP wheat series too short for forecasting.",
            history_actual=y.reset_index(drop=True),
            history_fitted=pd.Series(dtype=float),
            method_label="system default",
            commodity_name=com,
        )

    next_p, fitted_ma, method = _system_default_forecast_y(y)

    tail = float(y.iloc[-1])
    if not np.isnan(next_p):
        if next_p > tail * 1.01:
            trend: Literal["up", "down", "flat"] = "up"
        elif next_p < tail * 0.99:
            trend = "down"
        else:
            trend = "flat"
    else:
        trend = "flat"

    residuals = y.reset_index(drop=True) - fitted_ma
    residual_mae = float(residuals.abs().mean()) if len(residuals.dropna()) else None

    vol = float(y.pct_change().std() or 0)
    stress = tail > fitted_ma.iloc[-1] if len(fitted_ma) else False
    vol_note = "recent volatility is elevated" if vol > 0.05 else "recent volatility is moderate"
    stress_note = (
        "the current price is already above its moving average"
        if stress
        else "the current price is near its moving average"
    )
    warn = (
        f"Wheat prices are forecasted to {trend} over the next period using {method}. "
        f"This affects the Price Stress Score because {stress_note} and {vol_note}."
    )

    conf = (
        f"Forecast based on {method}. "
        + (
            f"Mean absolute forecast error (vs. rolling baseline) ≈ {residual_mae:.4f}."
            if residual_mae
            else "Residual not computed."
        )
    )

    return ForecastResult(
        next_price=float(next_p),
        trend=trend,
        confidence_note=conf,
        residual_mae=residual_mae,
        warning_explanation=warn,
        history_actual=y.reset_index(drop=True),
        history_fitted=fitted_ma.reset_index(drop=True),
        method_label=f"{method} (system default baseline)",
        commodity_name=com,
    )


def deterministic_utility_best(
    utilities: dict[str, float],
) -> tuple[str, float]:
    """Decision under certainty: maximize utility."""
    if not utilities:
        return "", float("nan")
    best = max(utilities.items(), key=lambda x: x[1])
    return best[0], float(best[1])


def expected_utility(
    payoffs: dict[str, dict[str, float]],
    probs: dict[str, float],
) -> dict[str, float]:
    """Decision under risk: EU = sum_s p(s)*u(a,s)."""
    scenarios = list(probs.keys())
    out: dict[str, float] = {}
    for alt, row in payoffs.items():
        eu = 0.0
        for s in scenarios:
            eu += float(probs.get(s, 0)) * float(row.get(s, 0))
        out[alt] = eu
    return out


def uncertainty_criteria(
    payoffs: dict[str, dict[str, float]],
    hurwicz_alpha: float = 0.5,
) -> dict[str, Any]:
    """Maximax, maximin, Laplace, minimax regret, Hurwicz."""
    alts = list(payoffs.keys())
    if not alts:
        return {}

    states = list(next(iter(payoffs.values())).keys())
    matrix = np.array([[payoffs[a][s] for s in states] for a in alts])

    maximax_alts = alts[int(np.argmax(matrix.max(axis=1)))]
    maximin_alts = alts[int(np.argmax(matrix.min(axis=1)))]

    laplace = matrix.mean(axis=1)
    laplace_alt = alts[int(np.argmax(laplace))]

    col_max = matrix.max(axis=0)
    regret = col_max - matrix
    minimax_regret_alt = alts[int(np.argmax(-regret.max(axis=1)))]

    h_alpha = np.clip(hurwicz_alpha, 0, 1)
    hurwicz_vals = h_alpha * matrix.max(axis=1) + (1 - h_alpha) * matrix.min(axis=1)
    hurwicz_alt = alts[int(np.argmax(hurwicz_vals))]

    return {
        "maximax": maximax_alts,
        "maximin": maximin_alts,
        "laplace": laplace_alt,
        "minimax_regret": minimax_regret_alt,
        "hurwicz": hurwicz_alt,
        "regret_matrix": regret,
        "state_labels": states,
        "alternatives": alts,
    }


def tornado_sensitivity(
    base_score: float,
    swings: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """
    swings: name -> (low_assumption, high_assumption) impact on score already computed,
    or we interpret as delta low / delta high from base.
    Here each tuple is (score_when_low, score_when_high).
    """
    rows = []
    for name, (low_v, high_v) in swings.items():
        rows.append(
            {
                "factor": name,
                "low": low_v,
                "high": high_v,
                "spread": abs(high_v - low_v),
                "base": base_score,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("spread", ascending=False).reset_index(drop=True)
