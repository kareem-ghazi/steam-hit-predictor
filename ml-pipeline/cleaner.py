import os
import re
import html
from typing import List, Optional
import pandas as pd
from config import CURRENCY_TO_USD, INVALID_TOKENS, parse_delimited_tokens

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))


def load_dataset_from_attributes(
    raw_dir: Optional[str] = None,
    attributes: Optional[List[str]] = None
) -> pd.DataFrame:
    """Loads and merges attribute CSV files from raw_data/dataset_attributes/ on appid."""
    if raw_dir is None:
        raw_dir = os.path.join(PARENT_DIR, "raw_data", "dataset_attributes")
        if not os.path.exists(raw_dir):
            raw_dir = os.path.join(CURRENT_DIR, "..", "raw_data", "dataset_attributes")

    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"dataset_attributes directory not found at: {raw_dir}")

    if attributes is None:
        attributes = [
            "name", "release_date", "english", "developer", "publisher", "platforms",
            "required_age", "categories", "genres", "steamspy_tags", "achievements",
            "positive_ratings", "negative_ratings", "average_playtime", "median_playtime",
            "owners", "price", "currency", "short_description", "about_the_game", "detailed_description"
        ]

    print(f"Loading {len(attributes)} attributes from: {raw_dir}...")
    df_merged = None
    for attr in attributes:
        attr_file = os.path.join(raw_dir, f"{attr}.csv")
        if not os.path.exists(attr_file):
            raise FileNotFoundError(f"Missing required attribute file: {attr_file}")
        col_df = pd.read_csv(attr_file)
        if df_merged is None:
            df_merged = col_df
        else:
            df_merged = df_merged.merge(col_df, on="appid", how="inner")

    print(f"Loaded {len(df_merged):,} records across {len(attributes)} attributes.")
    return df_merged


def clean_html_text(text: Optional[str]) -> str:
    """Cleans HTML tags and entities from raw description text."""
    if not text or pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<(?:br|p|div|h[1-6]|li)\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans short, about, and detailed descriptions and creates clean_description."""
    df = df.copy()
    clean_short = df["short_description"].apply(clean_html_text)
    clean_about = df["about_the_game"].apply(clean_html_text)
    clean_detailed = df["detailed_description"].apply(clean_html_text)

    def select_best(short: str, about: str, detailed: str) -> str:
        if short and len(short) > 25:
            return short
        if about and len(about) > 25:
            return about[:600]
        if detailed and len(detailed) > 25:
            return detailed[:600]
        return "No description available."

    df["clean_description"] = [select_best(s, a, d) for s, a, d in zip(clean_short, clean_about, clean_detailed)]
    df["short_description"] = clean_short.replace({"": "No description available."})
    df["about_the_game"] = clean_about.replace({"": "No description available."})
    df["detailed_description"] = clean_detailed.replace({"": "No description available."})
    return df


def unify_price_to_usd(df: pd.DataFrame) -> pd.DataFrame:
    """Converts foreign currency prices to USD using standard FX rates."""
    df = df.copy()
    if "currency" in df.columns:
        rates = df["currency"].astype(str).str.strip().str.upper().map(CURRENCY_TO_USD).fillna(1.0)
        df["price"] = (df["price"].astype(float) * rates).round(2)
        df["currency"] = "USD"
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw attributes:
    - Deduplicates appid.
    - Standardizes multi-label delimited columns.
    - Drops rows missing required fields (developer, publisher, name, genres, platforms).
    - Unifies price in USD.
    - Converts integer fields (reviews, achievements, age, english) to regular int.
    """
    df = df.copy()
    initial_count = len(df)

    # 1. Deduplication
    df = df.drop_duplicates(subset=["appid"])

    # 2. Standardize semicolon-separated fields
    for col in ["platforms", "categories", "genres", "steamspy_tags"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda val: ";".join(sorted(parse_delimited_tokens(val))))

    # 3. Filter missing developer or publisher entirely (no imputation)
    dev_str = df["developer"].astype(str).str.strip().str.lower()
    pub_str = df["publisher"].astype(str).str.strip().str.lower()
    valid_dev = df["developer"].notna() & (~dev_str.isin(INVALID_TOKENS))
    valid_pub = df["publisher"].notna() & (~pub_str.isin(INVALID_TOKENS))
    df = df[valid_dev & valid_pub].copy()

    # 4. Filter missing name, genres, or platforms
    name_str = df["name"].astype(str).str.strip().str.lower()
    genres_str = df["genres"].astype(str).str.strip().str.lower()
    plats_str = df["platforms"].astype(str).str.strip().str.lower()
    valid_name = df["name"].notna() & (~name_str.isin(INVALID_TOKENS)) & (name_str != "")
    valid_genres = df["genres"].notna() & (~genres_str.isin(INVALID_TOKENS)) & (genres_str != "")
    valid_plats = df["platforms"].notna() & (~plats_str.isin(INVALID_TOKENS)) & (plats_str != "")
    df = df[valid_name & valid_genres & valid_plats].copy()

    # 5. Price in USD
    df = unify_price_to_usd(df)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)

    # 6. Integer casting for integer-valued columns
    int_cols = ["appid", "english", "required_age", "achievements", "positive_ratings", "negative_ratings"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0).astype(int)

    print(f"Data Cleaning: {initial_count:,} -> {len(df):,} valid records (dropped {initial_count - len(df):,}).")
    return df

