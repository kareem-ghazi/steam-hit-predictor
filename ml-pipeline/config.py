import re
from typing import Dict, List, Any
import pandas as pd

# FX exchange rates to convert non-USD prices to USD
CURRENCY_TO_USD: Dict[str, float] = {
    'USD': 1.0,
    'EUR': 1.08,
    'GBP': 1.28,
    'CAD': 0.73,
    'AUD': 0.65,
    'PLN': 0.25,
    'BRL': 0.18,
    'RUB': 0.011,
    'JPY': 0.0065,
    'CNY': 0.14,
    'INR': 0.012,
    'PHP': 0.017,
    'VND': 0.00004,
    'IDR': 0.000063,
    'AED': 0.27,
    'HKD': 0.13,
    'KRW': 0.00072,
    'THB': 0.028,
    'NOK': 0.092,
    'MYR': 0.22,
    'TWD': 0.031,
    'SGD': 0.75,
    'ZAR': 0.055,
    'ILS': 0.27,
    'MXN': 0.051,
    'CLP': 0.0011,
}

# Core modeling & filtering thresholds
MIN_DEV_COUNT: int = 3
MIN_PUB_COUNT: int = 3
MIN_GENRE_COUNT: int = 20
MIN_CATEGORY_COUNT: int = 30
TOP_N_TAGS: int = 100
MIN_REVIEWS_THRESHOLD: int = 50
HIT_APPROVAL_THRESHOLD: float = 0.80

INVALID_TOKENS = {"", "nan", "unknown", "none", "null", "[]", "['']"}


def calculate_steam_rating_text(approval_rate: float, total_reviews: int) -> str:
    """Translates approval rate and review volume into Steam's standard rating tiers."""
    if total_reviews < 10:
        return "Not Enough Reviews"
    
    pct = approval_rate * 100.0
    if pct >= 95 and total_reviews >= 500:
        return "Overwhelmingly Positive"
    elif pct >= 80 and total_reviews >= 50:
        return "Very Positive"
    elif pct >= 80:
        return "Positive"
    elif pct >= 70:
        return "Mostly Positive"
    elif pct >= 40:
        return "Mixed"
    elif pct >= 20:
        return "Mostly Negative"
    elif pct < 20 and total_reviews >= 500:
        return "Overwhelmingly Negative"
    elif pct < 20 and total_reviews >= 50:
        return "Very Negative"
    else:
        return "Negative"


def parse_delimited_tokens(val: Any) -> List[str]:
    """
    Parses semicolon, comma, or bracket-delimited multi-label strings into a clean list of lowercased tokens.
    Excludes empty bracket representations ('[]'), non-alphanumeric noise, and invalid tokens.
    """
    if not val or pd.isna(val):
        return []
    s = str(val).strip()
    if s in INVALID_TOKENS:
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    delim = ";" if ";" in s else ("," if "," in s else ";")
    tokens = []
    for raw in s.split(delim):
        cleaned = raw.strip().strip("'\"").strip().lower()
        if len(re.findall(r"[a-z0-9]", cleaned)) >= 2:
            tokens.append(cleaned)
    return tokens

