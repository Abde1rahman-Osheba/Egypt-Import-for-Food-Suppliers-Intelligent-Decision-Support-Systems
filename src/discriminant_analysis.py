"""Linear discriminant risk classification trained on engineered features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
except ImportError:
    LinearDiscriminantAnalysis = None


FEATURE_COLUMNS = [
    "price_stress_score",
    "logistics_risk_score",
    "geopolitical_risk_score",
    "volatility_risk",
    "conflict_event_proxy",
    "port_activity_drop",
    "nlp_conflict_keyword_score",
    "news_sentiment_risk_score",
    "forecast_deviation",
]

LABEL_ORDER = ["0", "1"]


@dataclass
class DiscriminantResult:
    model_available: bool
    accuracy: Optional[float]
    confusion: Optional[np.ndarray]
    class_labels: list[str]
    last_prediction: str
    class_probabilities: Optional[dict[str, float]]
    coefficients: Optional[pd.DataFrame]
    message: str


def train_and_evaluate(
    risk_df: pd.DataFrame,
    random_state: int = 42,
) -> DiscriminantResult:
    if LinearDiscriminantAnalysis is None:
        return DiscriminantResult(
            model_available=False,
            accuracy=None,
            confusion=None,
            class_labels=LABEL_ORDER,
            last_prediction="0",
            class_probabilities=None,
            coefficients=None,
            message="scikit-learn not available.",
        )

    if risk_df.empty or "risk_label" not in risk_df.columns:
        return DiscriminantResult(
            model_available=False,
            accuracy=None,
            confusion=None,
            class_labels=LABEL_ORDER,
            last_prediction="0",
            class_probabilities=None,
            coefficients=None,
            message="Insufficient rows for discriminant analysis.",
        )

    df = risk_df.dropna(subset=FEATURE_COLUMNS, how="all").copy()
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if len(df) < 12:
        return DiscriminantResult(
            model_available=False,
            accuracy=None,
            confusion=None,
            class_labels=LABEL_ORDER,
            last_prediction=str(df.iloc[-1].get("risk_label", "0")),
            class_probabilities=None,
            coefficients=None,
            message="Fewer than 12 monthly observations after feature assembly; showing rule labels only.",
        )

    X = df[FEATURE_COLUMNS].values
    y = df["risk_label"].astype(str).values

    if len(np.unique(y)) < 2:
        return DiscriminantResult(
            model_available=False,
            accuracy=None,
            confusion=None,
            class_labels=sorted(np.unique(y).tolist()),
            last_prediction=str(df.iloc[-1]["risk_label"]),
            class_probabilities=None,
            coefficients=None,
            message="LDA requires at least two distinct rule-generated classes in-sample.",
        )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=random_state,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state
        )

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    last = clf.predict(X[-1].reshape(1, -1))[0]
    probs = None
    if hasattr(clf, "predict_proba"):
        prob_arr = clf.predict_proba(X[-1].reshape(1, -1))[0]
        probs = {str(k): float(v) for k, v in zip(clf.classes_, prob_arr)}

    coef_df = None
    if hasattr(clf, "coef_") and clf.coef_ is not None:
        coef = np.asarray(clf.coef_)
        n_classes = len(clf.classes_)
        if n_classes == 2 and coef.shape[0] == 1:
            coef_df = pd.DataFrame(
                coef,
                columns=FEATURE_COLUMNS,
                index=[f"Linear separator (class {clf.classes_[1]} vs {clf.classes_[0]})"],
            )
        else:
            coef_df = pd.DataFrame(
                coef,
                columns=FEATURE_COLUMNS,
                index=[str(c) for c in clf.classes_],
            )

    return DiscriminantResult(
        model_available=True,
        accuracy=acc,
        confusion=cm,
        class_labels=list(clf.classes_),
        last_prediction=str(last),
        class_probabilities=probs,
        coefficients=coef_df,
        message="LDA trained on **binary** rule labels: **1** = High+Critical composite stress, **0** = Low+Moderate (educational).",
    )
