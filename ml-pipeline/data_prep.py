import os
import re
import html
import pickle
import shutil
import numpy as np
import pandas as pd
import sys
from typing import Tuple, Optional, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from generate_vibe_scores import compute_vibe_scores_dataframe


# Standard FX exchange rates to convert non-USD prices to USD
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


def calculate_steam_rating_text(approval_rate: float, total_reviews: int) -> str:
    """
    Translates approval rate and review volume into Steam's standard review tiers.
    Steam requires review volume thresholds for certain rating categories.
    """
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


def load_dataset_from_attributes(
    raw_dir: Optional[str] = None,
    attributes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Loads required tabular and description columns from raw_data/dataset_attributes/.
    Merges columns into a consolidated base DataFrame using appid as primary key.
    """
    if raw_dir is None:
        raw_dir = os.path.join(parent_dir, "raw_data", "dataset_attributes")
        if not os.path.exists(raw_dir):
            raw_dir = os.path.join(current_dir, "..", "raw_data", "dataset_attributes")

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
    """
    Cleans raw HTML descriptions:
    - Removes HTML tags (e.g. <br>, <p>, <strong>, <img>, etc.)
    - Unescapes HTML entities (&amp;, &quot;, &#39;, &nbsp;)
    - Normalizes multiple spaces and tabs to single space
    """
    if not text or pd.isna(text):
        return ""
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<(?:br|p|div|h[1-6]|li)\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans HTML markup across short, about, and detailed descriptions,
    and constructs a consolidated clean_description column.
    """
    df = df.copy()
    print("Cleaning HTML markup in descriptions and building clean_description...")
    
    clean_short = df["short_description"].apply(clean_html_text)
    clean_about = df["about_the_game"].apply(clean_html_text)
    clean_detailed = df["detailed_description"].apply(clean_html_text)

    def select_best_description(short, about, detailed) -> str:
        if short and len(short) > 25:
            return short
        if about and len(about) > 25:
            return about[:600]
        if detailed and len(detailed) > 25:
            return detailed[:600]
        return "No description available."

    df["clean_description"] = [
        select_best_description(s, a, d)
        for s, a, d in zip(clean_short, clean_about, clean_detailed)
    ]
    df["short_description"] = clean_short.replace({"": "No description available."})
    df["about_the_game"] = clean_about.replace({"": "No description available."})
    df["detailed_description"] = clean_detailed.replace({"": "No description available."})

    return df


def unify_price_to_usd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unifies all prices into USD currency.
    Uses currency attribute and FX rates to convert non-USD prices to USD.
    """
    df = df.copy()
    if "currency" in df.columns:
        rates = df["currency"].astype(str).str.strip().str.upper().map(CURRENCY_TO_USD).fillna(1.0)
        df["price"] = (df["price"].astype(float) * rates).round(2)
        df["currency"] = "USD"
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning:
    1. Removes duplicate rows and duplicate appids.
    2. Drops rows ENTIRELY if developer or publisher is missing/unknown (no imputation).
    3. Drops rows missing critical identifiers (name, genres, platforms).
    4. Unifies pricing into USD using currency exchange rates.
    5. Standardizes semicolon-delimited categorical tokens.
    6. Validates numeric boundaries.
    """
    df = df.copy()
    initial_count = len(df)

    # 1. Deduplication
    df = df.drop_duplicates(subset=["appid"])

    # 2. Drop rows missing developer or publisher entirely (instead of imputation)
    invalid_tokens = {"", "nan", "unknown", "none", "null"}
    dev_str = df["developer"].astype(str).str.strip().str.lower()
    pub_str = df["publisher"].astype(str).str.strip().str.lower()
    
    valid_dev = df["developer"].notna() & (~dev_str.isin(invalid_tokens))
    valid_pub = df["publisher"].notna() & (~pub_str.isin(invalid_tokens))
    df = df[valid_dev & valid_pub].copy()

    # 3. Drop rows missing name, genres, or platforms
    name_str = df["name"].astype(str).str.strip().str.lower()
    genres_str = df["genres"].astype(str).str.strip().str.lower()
    plats_str = df["platforms"].astype(str).str.strip().str.lower()

    valid_name = df["name"].notna() & (~name_str.isin(invalid_tokens))
    valid_genres = df["genres"].notna() & (~genres_str.isin(invalid_tokens))
    valid_plats = df["platforms"].notna() & (~plats_str.isin(invalid_tokens))
    df = df[valid_name & valid_genres & valid_plats].copy()

    # 4. Standardize semicolon-separated fields (lowercase, strip, sort for canonical order)
    delimited_cols = ["platforms", "categories", "genres", "steamspy_tags"]
    for col in delimited_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda val: ";".join(sorted([token.strip().lower() for token in str(val).split(";") if token.strip()]))
            )

    # 5. Unify price in USD
    df = unify_price_to_usd(df)

    # 6. Validate english integer flag
    if "english" in df.columns:
        df["english"] = pd.to_numeric(df["english"], errors="coerce").fillna(1).clip(0, 1).astype(int)

    # 7. Validate numeric columns
    numeric_cols = ["price", "achievements", "positive_ratings", "negative_ratings", "required_age", "average_playtime", "median_playtime"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df[col] = df[col].apply(lambda x: max(0.0, float(x)))

    print(f"Data Cleaning: {initial_count:,} -> {len(df):,} valid records (dropped {initial_count - len(df):,} records).")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering and target creation:
    1. total_reviews = positive_ratings + negative_ratings
    2. Drops games under 50 reviews FIRST from the dataset
    3. Calculates approval_rate = positive_ratings / total_reviews
    4. Calculates Steam rating_text tier
    5. Calculates is_hit target label (1 if approval_rate >= 0.80, else 0)
    6. Keeps leakage columns (positive_ratings, negative_ratings, total_reviews, approval_rate) for safe measures
    7. Calculates heuristic vibe scores (hype_score, story_score, nostalgia_score)
    """
    df = df.copy()

    total_reviews = df["positive_ratings"] + df["negative_ratings"]
    df["total_reviews"] = total_reviews.astype(int)

    # Drop games under 50 reviews FIRST
    count_before = len(df)
    df = df[df["total_reviews"] >= 50].copy()
    print(f"Review Volume Filter: {count_before:,} -> {len(df):,} games with total_reviews >= 50 (dropped {count_before - len(df):,}).")

    # Approval rate & rating tier
    df["approval_rate"] = df["positive_ratings"] / df["total_reviews"]
    df["rating_text"] = [
        calculate_steam_rating_text(ar, tr)
        for ar, tr in zip(df["approval_rate"], df["total_reviews"])
    ]

    # Target variable is_hit
    df["is_hit"] = np.where(df["approval_rate"] >= 0.80, 1, 0)
    hits = df["is_hit"].sum()
    print(f"Target Distribution: {hits:,} Hits ({hits/len(df):.1%}), {len(df)-hits:,} Non-Hits ({(len(df)-hits)/len(df):.1%}).")

    # Accurate heuristic vibe scores
    df = compute_vibe_scores_dataframe(df)

    # Note: Leakage columns (positive_ratings, negative_ratings, total_reviews, approval_rate)
    # are intentionally NOT dropped, kept for safety measures and analysis.
    return df


def split_and_save_data(
    df: pd.DataFrame,
    output_dir: str = "cleaned_data",
    scaler_save_path: str = "saved_models/ml_models/price_scaler.pkl",
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], StandardScaler]:
    """
    Splits the cleaned dataset into train (80%), valid (10%), and test (10%) sets
    using stratified sampling on 'is_hit'.
    Applies StandardScaler on 'price' (fitted strictly on train set).
    Retains raw string categorical columns (NO ordinal encoding).
    Saves outputs into THREE separate folders:
    1. cleaned_data/train/ (games.csv and games_descriptions.csv)
    2. cleaned_data/valid/ (games.csv and games_descriptions.csv)
    3. cleaned_data/test/ (games.csv and games_descriptions.csv)
    """
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    valid_dir = os.path.join(output_dir, "valid")
    test_dir = os.path.join(output_dir, "test")
    for d in [train_dir, valid_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)

    # 1. Stratified train/valid/test split
    temp_ratio = valid_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df["is_hit"]
    )

    valid_share = valid_ratio / temp_ratio
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - valid_share),
        random_state=seed,
        stratify=temp_df["is_hit"]
    )

    # 2. StandardScaler for price (fit on train set only)
    price_scaler = StandardScaler()
    price_scaler.fit(train_df[["price"]])

    for split in [train_df, valid_df, test_df]:
        split["price_scaled"] = price_scaler.transform(split[["price"]]).flatten().round(4)

    if scaler_save_path:
        with open(scaler_save_path, "wb") as f:
            pickle.dump(price_scaler, f)
        print(f"Saved price StandardScaler to: {scaler_save_path}")

    # 3. Partition into attributes and descriptions
    desc_columns = ["appid", "short_description", "about_the_game", "detailed_description", "clean_description"]
    
    preferred_attr_order = [
        "appid", "name", "release_date", "english", "developer", "publisher",
        "platforms", "required_age", "categories", "genres", "steamspy_tags",
        "achievements", "average_playtime", "median_playtime", "owners", "price",
        "price_scaled", "currency",
        "positive_ratings", "negative_ratings", "total_reviews", "approval_rate",
        "hype_score", "story_score", "nostalgia_score", "rating_text",
        "is_hit"
    ]
    
    def get_attr_split(split_data: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in preferred_attr_order if c in split_data.columns]
        extra_cols = [c for c in split_data.columns if c not in cols and c not in desc_columns and c != "is_hit"]
        if "is_hit" in cols:
            cols = [c for c in cols if c != "is_hit"] + extra_cols + ["is_hit"]
        else:
            cols = cols + extra_cols
        return split_data[cols].copy()

    def get_desc_split(split_data: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in desc_columns if c in split_data.columns]
        return split_data[cols].copy()

    train_attr = get_attr_split(train_df)
    valid_attr = get_attr_split(valid_df)
    test_attr = get_attr_split(test_df)

    train_desc = get_desc_split(train_df)
    valid_desc = get_desc_split(valid_df)
    test_desc = get_desc_split(test_df)

    # 4. Save to disk in the three dedicated split folders (train, valid, test)
    train_attr.to_csv(os.path.join(train_dir, "games.csv"), index=False)
    train_desc.to_csv(os.path.join(train_dir, "games_descriptions.csv"), index=False)

    valid_attr.to_csv(os.path.join(valid_dir, "games.csv"), index=False)
    valid_desc.to_csv(os.path.join(valid_dir, "games_descriptions.csv"), index=False)

    test_attr.to_csv(os.path.join(test_dir, "games.csv"), index=False)
    test_desc.to_csv(os.path.join(test_dir, "games_descriptions.csv"), index=False)

    # Cleanup legacy attributes and descriptions folders if present
    for old_dir in ["attributes", "descriptions"]:
        old_path = os.path.join(output_dir, old_dir)
        if os.path.exists(old_path):
            try:
                shutil.rmtree(old_path)
            except Exception as e:
                print(f"Warning: could not remove legacy folder {old_path}: {e}")

    # Cleanup legacy top-level monolithic split files if present
    for old_file in ["train.csv", "valid.csv", "test.csv", "steam_descriptions_cleaned.csv"]:
        old_path = os.path.join(output_dir, old_file)
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except Exception:
                pass

    print(f"\nSaved Three Split Folders to '{output_dir}/':")
    print(f"  1. Train folder: '{train_dir}/'")
    print(f"     - games.csv:              {len(train_attr):,} rows ({len(train_attr.columns)} columns)")
    print(f"     - games_descriptions.csv: {len(train_desc):,} rows ({len(train_desc.columns)} columns)")
    print(f"  2. Valid folder: '{valid_dir}/'")
    print(f"     - games.csv:              {len(valid_attr):,} rows ({len(valid_attr.columns)} columns)")
    print(f"     - games_descriptions.csv: {len(valid_desc):,} rows ({len(valid_desc.columns)} columns)")
    print(f"  3. Test folder:  '{test_dir}/'")
    print(f"     - games.csv:              {len(test_attr):,} rows ({len(test_attr.columns)} columns)")
    print(f"     - games_descriptions.csv: {len(test_desc):,} rows ({len(test_desc.columns)} columns)")
    print(f"  Target variable at the end of games.csv: '{train_attr.columns[-1]}'")

    return (train_attr, valid_attr, test_attr), (train_desc, valid_desc, test_desc), price_scaler


def prepare_data(
    raw_dir: Optional[str] = None,
    output_dir: str = "cleaned_data"
):
    """
    Full data preparation pipeline:
    1. Loads all attributes from raw_data/dataset_attributes/
    2. Cleans descriptions and constructs clean_description
    3. Cleans tabular dataset (drops rows missing developer/publisher entirely, drops invalid name/genres/platforms)
    4. Unifies pricing in USD
    5. Filters games with total_reviews >= 50 FIRST, then creates is_hit target
    6. Keeps leakage columns for safe measures
    7. Computes accurate heuristic vibe scores
    8. Splits into train (80%), valid (10%), test (10%) with price StandardScaler
    9. Saves split files into three folders: cleaned_data/train/, cleaned_data/valid/, cleaned_data/test/
       (each containing games.csv and games_descriptions.csv)
    """
    df_raw = load_dataset_from_attributes(raw_dir=raw_dir)
    df_desc = prepare_descriptions(df_raw)
    df_cleaned = clean_dataset(df_desc)
    df_engineered = engineer_features(df_cleaned)
    return split_and_save_data(df_engineered, output_dir=output_dir)


if __name__ == "__main__":
    print("--- Executing Modular Attribute Data Prep Pipeline ---")
    attr_splits, desc_splits, scaler = prepare_data()
    train_attr, valid_attr, test_attr = attr_splits
    train_desc, valid_desc, test_desc = desc_splits
    print("\nSample Attributes Row:")
    print(train_attr[["name", "developer", "publisher", "price", "price_scaled", "total_reviews", "approval_rate", "is_hit"]].head(2).to_dict(orient="records"))
    print("\nSample Descriptions Row:")
    print(train_desc[["appid", "clean_description"]].head(2).to_dict(orient="records"))
