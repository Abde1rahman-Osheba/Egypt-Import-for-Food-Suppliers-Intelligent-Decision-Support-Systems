"""
Bilingual (Arabic + English) lightweight NLP for a conflict / disruption news index
and simple sentiment. Designed for course-scale dependencies (keyword + normalization only).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.utils import project_root


# Conflict / supply-shock lexicon (expandable)
KEYWORDS_EN = [
    "war", "invasion", "missile", "strike", "airstrike", "naval blockade", "blockade",
    "embargo", "sanctions", "grain corridor", "wheat export", "food insecurity",
    "humanitarian crisis", "ceasefire collapse", "escalation", "shelling", "conflict",
    "black sea", "shortage", "export ban", "port attack", "houthi", "shipping",
    "vessel", "maritime", "convoy", "insurgency", "border clash",
]

KEYWORDS_AR = [
    "حرب", "نزاع", "غزو", "صاروخ", "غارة", "قصف", "حصار", "حظر", "عقوبات",
    "ممر", "قمح", "أمن غذائي", "شح", "اختناق", "أزمة إنسانية", "تصعيد", "اشتباك",
    "البحر الأسود", "ميناء", "شحن", "ناقلة", "تصدير", "استيراد", "اشتباكات",
    "الحوثي", "البحرية", "الحرب", "العدوان",
]

POSITIVE_EN = [
    "peace deal", "ceasefire holds", "export resumed", "deal signed", "corridor reopened",
    "de-escalation", "aid delivered", "harvest strong",
]
POSITIVE_AR = [
    "هدنة", "اتفاق", "استئناف التصدير", "تهدئة", "مساعدات", "ممر آمن",
]

NEGATIVE_EXTRA_EN = KEYWORDS_EN
NEGATIVE_EXTRA_AR = KEYWORDS_AR


def normalize_arabic(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFKC", text)
    # Alef / hamza variants → ا
    for a, b in [
        ("\u0622", "\u0627"),
        ("\u0623", "\u0627"),
        ("\u0625", "\u0627"),
        ("\u0671", "\u0627"),
    ]:
        s = s.replace(a, b)
    s = s.replace("\u0649", "\u064a")  # yeh
    s = s.replace("\u0640", "")  # tatweel
    return s.strip()


def _tokenize_mixed(text: str) -> list[str]:
    s = normalize_arabic(text)
    # Keep Arabic letters / Latin letters / digits as crude tokens
    parts = re.findall(r"[\u0600-\u06FF]+|[A-Za-z']+", s.lower())
    return parts


def detect_language_hint(text: str) -> Literal["ar", "en", "mixed"]:
    if not text:
        return "en"
    ar = len(re.findall(r"[\u0600-\u06FF]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    if ar > en * 0.5 and ar >= 3:
        return "ar" if en < 2 else "mixed"
    return "en"


def score_conflict_keywords(text: str, lang_hint: Optional[str] = None) -> float:
    """Density-based conflict score in [0, ~3], typically capped when aggregating."""
    if not text or not str(text).strip():
        return 0.0
    lang = lang_hint or detect_language_hint(text)
    t_norm = text.lower() if lang == "en" else normalize_arabic(text)
    hits = 0.0
    if lang in ("en", "mixed"):
        for kw in KEYWORDS_EN:
            if kw in t_norm:
                hits += 1.0
    if lang in ("ar", "mixed"):
        tn = normalize_arabic(t_norm)
        for kw in KEYWORDS_AR:
            if kw in tn:
                hits += 1.0
    words = max(4, len(_tokenize_mixed(text)))
    return hits / (words ** 0.5)


def score_sentiment(text: str, lang_hint: Optional[str] = None) -> float:
    """Rough polarity in [-1, 1]: negative supply shocks vs calming language."""
    if not text:
        return 0.0
    lang = lang_hint or detect_language_hint(text)
    pos = neg = 0
    t_en = text.lower()
    tn_ar = normalize_arabic(text)
    if lang in ("en", "mixed"):
        pos += sum(1 for p in POSITIVE_EN if p in t_en)
        neg += sum(1 for p in NEGATIVE_EXTRA_EN if p in t_en)
    if lang in ("ar", "mixed"):
        pos += sum(1 for p in POSITIVE_AR if p in tn_ar)
        neg += sum(1 for p in NEGATIVE_EXTRA_AR if p in tn_ar)
    tot = pos + neg + 1e-6
    return float(np.clip((pos - neg) / tot, -1, 1))


def load_headlines_csv(path: Path) -> pd.DataFrame:
    """Expected: published_at|date, text|headline, language (optional: ar|en)."""
    if not path.is_file():
        return pd.DataFrame(columns=["published_at", "text", "language"])
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    date_c = next((c for c in df.columns if c in ("published_at", "date", "time", "timestamp")), None)
    text_c = next((c for c in df.columns if c in ("text", "headline", "title", "body")), None)
    if not date_c or not text_c:
        return pd.DataFrame(columns=["published_at", "text", "language"])
    out = pd.DataFrame(
        {
            "published_at": pd.to_datetime(df[date_c], errors="coerce", utc=True),
            "text": df[text_c].astype(str),
        }
    )
    if "language" in df.columns:
        out["language"] = df["language"].astype(str).str.lower().str[:2]
    else:
        out["language"] = out["text"].apply(lambda x: detect_language_hint(x))
    out = out.dropna(subset=["published_at", "text"])
    out["month"] = out["published_at"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    return out.reset_index(drop=True)


def aggregate_monthly_nlp(headlines: pd.DataFrame) -> pd.DataFrame:
    if headlines.empty:
        return pd.DataFrame(
            columns=["month", "nlp_conflict_index", "news_sentiment_avg", "headline_count"]
        )
    rows = []
    for month, grp in headlines.groupby("month"):
        conf = [score_conflict_keywords(t, None) for t in grp["text"]]
        sent = [score_sentiment(t, None) for t in grp["text"]]
        rows.append(
            {
                "month": month,
                "nlp_conflict_index": float(np.mean(conf)) if conf else 0.0,
                "news_sentiment_avg": float(np.mean(sent)) if sent else 0.0,
                "headline_count": len(grp),
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def synthesize_headlines_from_prices(wheat_monthly: pd.DataFrame, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """
    When no news file exists: generate bilingual headlines biased upward on price spikes
    (clearly synthetic — for pipeline continuity).
    """
    rng = rng or np.random.default_rng(42)
    if wheat_monthly.empty or "price_usd" not in wheat_monthly.columns:
        return pd.DataFrame(columns=["published_at", "text", "language", "month"])

    s = wheat_monthly.sort_values("month").copy().reset_index(drop=True)
    s["ret_abs"] = s["price_usd"].pct_change().abs()
    roll_std = s["ret_abs"].rolling(6, min_periods=4).std()
    s["is_spike"] = s["ret_abs"] > (roll_std * 1.25)
    en_hi = [
        "Black Sea grain corridor risk rises as wheat futures jump",
        "Export bottlenecks reported at key grain loading ports",
        "Analysts warn of supply disruption for Egyptian wheat importers",
    ]
    ar_hi = [
        "مخاوف من اضطراب شحن القمح في البحر الأسود بعد تصعيد عسكري",
        "مصادر: تأخير شحنات الحبوب بسبب ضغط الموانئ الإقليمية",
        "خبراء: أسعار القمح العالمية تهدد استيراد مصر",
    ]
    en_lo = [
        "Grain markets steady as export pace normalizes",
        "Aid agencies confirm humanitarian wheat deliveries continued",
    ]
    ar_lo = [
        "تقارير عن استقرار نسبي في شحن الحبوب عبر الممرات البحرية",
    ]

    records = []
    for _, row in s.iterrows():
        m = row["month"]
        is_spike = bool(row.get("is_spike", False)) and not pd.isna(row.get("is_spike", False))
        n_art = 3 if is_spike else 1
        for i in range(n_art):
            if is_spike and rng.random() < 0.55:
                text = rng.choice(ar_hi) if rng.random() < 0.5 else rng.choice(en_hi)
                lang = "ar" if text in ar_hi else "en"
            else:
                text = rng.choice(ar_lo) if rng.random() < 0.4 else rng.choice(en_lo)
                lang = "ar" if text in ar_lo else "en"
            day = int(rng.integers(1, 27))
            ts = pd.Timestamp(m.year, m.month, day, tz="UTC")
            records.append({"published_at": ts, "text": text, "language": lang, "month": m})
    return pd.DataFrame(records)


@dataclass
class NoveltyCorrelationResult:
    pearson_conflict_vs_spike: Optional[float]
    pearson_sentiment_vs_spike: Optional[float]
    headline_months: int
    used_synthetic_news: bool
    note: str


def price_spike_series(price: pd.Series) -> pd.Series:
    r = price.pct_change().abs()
    sig = r.rolling(6, min_periods=3).std().replace(0, np.nan)
    return (r / sig).clip(upper=6).fillna(0)


def compute_novelty_correlation(
    wheat_monthly: pd.DataFrame,
    news_monthly: pd.DataFrame,
) -> NoveltyCorrelationResult:
    if wheat_monthly.empty or news_monthly.empty:
        return NoveltyCorrelationResult(
            pearson_conflict_vs_spike=None,
            pearson_sentiment_vs_spike=None,
            headline_months=0,
            used_synthetic_news=False,
            note="Insufficient overlap for correlation.",
        )
    m = wheat_monthly[["month", "price_usd"]].merge(news_monthly, on="month", how="inner")
    if len(m) < 6:
        return NoveltyCorrelationResult(
            pearson_conflict_vs_spike=None,
            pearson_sentiment_vs_spike=None,
            headline_months=len(m),
            used_synthetic_news=False,
            note="Need at least 6 overlapping months.",
        )
    spike = price_spike_series(m["price_usd"])
    try:
        c1 = float(m["nlp_conflict_index"].corr(spike))
    except Exception:
        c1 = None
    try:
        c2 = float(m["news_sentiment_avg"].corr(spike))
    except Exception:
        c2 = None
    return NoveltyCorrelationResult(
        pearson_conflict_vs_spike=c1,
        pearson_sentiment_vs_spike=c2,
        headline_months=len(m),
        used_synthetic_news=False,
        note="",
    )


def build_news_pipeline(
    wheat_monthly: pd.DataFrame,
    headlines_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, NoveltyCorrelationResult, bool]:
    """
    Returns: headlines table, monthly aggregates, correlation stats, synthetic_flag
    """
    root = project_root()
    path = headlines_path or (root / "news_headlines_bilingual.csv")
    syn = False
    if path.is_file():
        headlines = load_headlines_csv(path)
    else:
        headlines = synthesize_headlines_from_prices(wheat_monthly)
        syn = True
    monthly = aggregate_monthly_nlp(headlines)
    cor = compute_novelty_correlation(wheat_monthly, monthly)
    if syn:
        cor = NoveltyCorrelationResult(
            pearson_conflict_vs_spike=cor.pearson_conflict_vs_spike,
            pearson_sentiment_vs_spike=cor.pearson_sentiment_vs_spike,
            headline_months=cor.headline_months,
            used_synthetic_news=True,
            note="Demo headlines synthesized from wheat price volatility pattern.",
        )
    return headlines, monthly, cor, syn
