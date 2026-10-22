import os
import re
import html
import pickle
import numpy as np
import pandas as pd
import sys
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from generate_vibe_scores import compute_vibe_scores_dataframe


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


def load_dataset(file_path: str = "raw_data/steam.csv") -> pd.DataFrame:
    """Loads raw steam dataset from raw_data folder."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return pd.read_csv(file_path)


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
    # Unescape HTML entities first
    text = html.unescape(text)
    # Replace <br> and paragraph breaks with newline or space
    text = re.sub(r"<(?:br|p|div|h[1-6]|li)\s*/?>", " ", text, flags=re.IGNORECASE)
    # Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_and_prepare_descriptions(
    desc_path: str = "raw_data/steam_description_data.csv"
) -> pd.DataFrame:
    """
    Loads, cleans, and deduplicates the Steam descriptions dataset.
    Cleans HTML markup and entity encoding across short, about, and detailed descriptions,
    and produces a consolidated 'clean_description' column.
    """
    if not os.path.exists(desc_path):
        print(f"Warning: Description file {desc_path} not found. Skipping description merge.")
        return pd.DataFrame(columns=["steam_appid", "short_description", "about_the_game", "detailed_description", "clean_description"])
    
    print("Cleaning and preparing Steam descriptions from raw_data/steam_description_data.csv...")
    df_desc = pd.read_csv(desc_path)
    
    # Clean HTML across description columns
    df_desc["clean_short"] = df_desc["short_description"].apply(clean_html_text)
    df_desc["clean_about"] = df_desc["about_the_game"].apply(clean_html_text)
    df_desc["clean_detailed"] = df_desc["detailed_description"].apply(clean_html_text)
    
    # Prioritize clean_short, fallback to truncated clean_about or clean_detailed
    def select_best_description(row) -> str:
        short = row["clean_short"]
        if short and len(short) > 25:
            return short
        about = row["clean_about"]
        if about and len(about) > 25:
            return about[:600]
        detailed = row["clean_detailed"]
        if detailed and len(detailed) > 25:
            return detailed[:600]
        return "No description available."
        
    df_desc["clean_description"] = df_desc.apply(select_best_description, axis=1)
    
    # Consolidate into standardized columns
    export_df = pd.DataFrame({
        "steam_appid": df_desc["steam_appid"],
        "short_description": df_desc["clean_short"],
        "about_the_game": df_desc["clean_about"],
        "detailed_description": df_desc["clean_detailed"],
        "clean_description": df_desc["clean_description"]
    })
    print(f"Cleaned {len(export_df)} Steam descriptions.")
    return export_df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs thorough data cleaning on the tabular dataset:
    1. Removes duplicate rows and duplicates by appid.
    2. Cleans text columns (name, developer, publisher, genres, platforms).
    3. Imputes missing developer and publisher values.
    4. Fixes inconsistent formatting: strips whitespace, canonicalizes delimited tokens.
    5. Validates non-negative numerical values (price, ratings, english flag).
    """
    df = df.copy()
    initial_count = len(df)
    
    # 1. Deduplication
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["appid"])
    
    # 2. String cleaning & whitespace stripping
    text_cols = ["name", "developer", "publisher", "platforms", "categories", "genres", "steamspy_tags"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    # Standardize semicolon-separated fields (lowercase, strip, sort tokens for canonical order)
    delimited_cols = ["platforms", "categories", "genres", "steamspy_tags"]
    for col in delimited_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda val: ";".join(sorted([token.strip().lower() for token in str(val).split(";") if token.strip()]))
            )
            
    # 3. Handle nulls and empty critical fields
    df = df[df["name"].notna() & (df["name"] != "") & (df["name"] != "nan")]
    df = df[df["genres"].notna() & (df["genres"] != "") & (df["genres"] != "nan")]
    df = df[df["platforms"].notna() & (df["platforms"] != "") & (df["platforms"] != "nan")]
    
    # Impute missing developer / publisher
    df["developer"] = df["developer"].replace({"nan": "Unknown", "": "Unknown"}).fillna("Unknown")
    df["publisher"] = df["publisher"].replace({"nan": "Unknown", "": "Unknown"}).fillna(df["developer"])
    
    # 4. English column validation (ensure 0 or 1 integer flag)
    if "english" in df.columns:
        df["english"] = pd.to_numeric(df["english"], errors="coerce").fillna(1).clip(0, 1).astype(int)
    else:
        df["english"] = 1
        
    # 5. Numeric validation and non-negative clamping
    numeric_cols = ["price", "achievements", "positive_ratings", "negative_ratings", "required_age"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df[col] = df[col].apply(lambda x: max(0.0, float(x)))
            
    cleaned_count = len(df)
    print(f"Data Cleaning: Main dataset processed ({cleaned_count} valid records).")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data engineering, target creation, and calculates accurate heuristic vibe scores:
    1. approval_rate = positive_ratings / (positive_ratings + negative_ratings)
    2. rating_text matching Steam review tiers taking volume into account
    3. is_hit target variable = 1 if approval_rate >= 0.80 and total reviews >= 50, else 0
    4. Calculates accurate heuristic vibe scores (hype_score, story_score, nostalgia_score)
       correlating with the actual game descriptions, tags, genres, categories, and metadata.
    """
    df = df.copy()
    
    total_reviews = df["positive_ratings"] + df["negative_ratings"]
    df["total_reviews"] = total_reviews
    
    df["approval_rate"] = np.where(
        total_reviews > 0,
        df["positive_ratings"] / total_reviews,
        0.0
    )
    
    df["rating_text"] = [
        calculate_steam_rating_text(ar, tr)
        for ar, tr in zip(df["approval_rate"], df["total_reviews"])
    ]
    
    df["is_hit"] = np.where(
        (df["approval_rate"] >= 0.80) & (df["total_reviews"] >= 50),
        1,
        0
    )
    
    # Calculate accurate heuristic vibe scores using game description and metadata
    df = compute_vibe_scores_dataframe(df)
    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drops columns that directly leak the target definition."""
    leakage_cols = ["positive_ratings", "negative_ratings", "total_reviews", "approval_rate"]
    existing_leakage = [col for col in leakage_cols if col in df.columns]
    return df.drop(columns=existing_leakage)


def apply_ordinal_encoding(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    """
    Fits an OrdinalEncoder on training set 'genres', 'platforms', and 'publisher' columns
    and transforms train, valid, and test sets.
    Preserves original string columns for UI and similarity matching.
    """
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    
    # Fit strictly on train set to prevent data leakage
    encoder.fit(train_df[["genres", "platforms", "publisher"]])
    
    for split_df in [train_df, valid_df, test_df]:
        encoded_vals = encoder.transform(split_df[["genres", "platforms", "publisher"]])
        split_df["genres_encoded"] = encoded_vals[:, 0].astype(float)
        split_df["platforms_encoded"] = encoded_vals[:, 1].astype(float)
        split_df["publisher_encoded"] = encoded_vals[:, 2].astype(float)
        
    return train_df, valid_df, test_df, encoder


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Organizes columns into a clean, logical structure:
    1. Identifiers & Metadata (appid, name, release_date, english, developer, publisher, platforms, ...)
    2. Steam Descriptions (short_description, about_the_game, detailed_description, clean_description)
    3. Ordinal Encoded Features (genres_encoded, platforms_encoded, publisher_encoded)
    4. LLM Vibe Scores (hype_score, story_score, nostalgia_score)
    5. Steam Rating Tier (rating_text)
    6. Target Variable (is_hit) - AT THE VERY END
    """
    preferred_order = [
        # Identifiers & Metadata
        "appid", "name", "release_date", "english", "developer", "publisher", 
        "platforms", "required_age", "categories", "genres", "steamspy_tags",
        "achievements", "average_playtime", "median_playtime", "owners", "price",
        # Steam Descriptions
        "short_description", "about_the_game", "detailed_description", "clean_description",
        # Ordinal Encoded Features
        "genres_encoded", "platforms_encoded", "publisher_encoded",
        # LLM Vibe Scores
        "hype_score", "story_score", "nostalgia_score",
        # Steam Rating Tier
        "rating_text",
        # Target variable at the very end
        "is_hit"
    ]
    cols = [c for c in preferred_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in cols and c != "is_hit"]
    if "is_hit" in df.columns:
        cols = [c for c in cols if c != "is_hit"] + extra_cols + ["is_hit"]
    else:
        cols = cols + extra_cols
    return df[cols]


def split_and_save_data(
    df: pd.DataFrame,
    output_dir: str = "cleaned_data",
    encoder_save_path: str = "saved_models/ml_models/ordinal_encoder.pkl",
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    """
    Splits the cleaned dataset into train (80%), valid (10%), and test (10%) sets
    using stratified sampling on 'is_hit', applies Ordinal Encoding on genres, platforms,
    and publisher, reorders columns so target 'is_hit' is at the end, and saves splits to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    if encoder_save_path:
        os.makedirs(os.path.dirname(encoder_save_path), exist_ok=True)
        
    # First split: train (80%) vs temp (20%)
    temp_ratio = valid_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df["is_hit"]
    )
    
    # Second split: split temp into valid (10%) and test (10%)
    valid_share = valid_ratio / temp_ratio
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - valid_share),
        random_state=seed,
        stratify=temp_df["is_hit"]
    )
    
    # Apply Ordinal Encoding on genres, platforms & publisher
    train_df, valid_df, test_df, encoder = apply_ordinal_encoding(train_df, valid_df, test_df)
    
    # Organize columns logically with descriptions included and target is_hit at the end
    train_df = reorder_columns(train_df)
    valid_df = reorder_columns(valid_df)
    test_df = reorder_columns(test_df)
    
    # Save encoder
    if encoder_save_path:
        with open(encoder_save_path, "wb") as f:
            pickle.dump(encoder, f)
        print(f"Saved OrdinalEncoder to: {encoder_save_path}")
        
    # Save splits
    train_path = os.path.join(output_dir, "train.csv")
    valid_path = os.path.join(output_dir, "valid.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Data Splits saved to '{output_dir}/':")
    print(f"  - Train: {len(train_df)} samples ({len(train_df)/len(df):.1%}) -> {train_path}")
    print(f"  - Valid: {len(valid_df)} samples ({len(valid_df)/len(df):.1%}) -> {valid_path}")
    print(f"  - Test:  {len(test_df)} samples ({len(test_df)/len(df):.1%}) -> {test_path}")
    print(f"Final columns (target at end): {list(train_df.columns)}")
    
    return train_df, valid_df, test_df, encoder


def prepare_data(
    raw_path: str = "raw_data/steam.csv",
    desc_path: str = "raw_data/steam_description_data.csv",
    output_dir: str = "cleaned_data"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    """
    Full pipeline execution:
    1. Loads raw dataset and cleans tabular fields
    2. Cleans and prepares steam descriptions (HTML stripping, entity unescaping)
    3. Merges clean descriptions directly into the tabular dataset
    4. Engineers approval_rate, rating_text, is_hit, and mock LLM features
    5. Drops data leakage columns
    6. Splits into train (80%), valid (10%), test (10%)
    7. Fits & applies Ordinal Encoding on 'genres', 'platforms', and 'publisher'
    8. Orders columns with target 'is_hit' at the end and saves splits
    """
    df_raw = load_dataset(raw_path)
    df_clean = clean_dataset(df_raw)
    
    # Clean and merge descriptions directly into tabular dataset
    df_desc_clean = clean_and_prepare_descriptions(desc_path)
    if not df_desc_clean.empty and "steam_appid" in df_desc_clean.columns:
        desc_cols = ["steam_appid", "short_description", "about_the_game", "detailed_description", "clean_description"]
        df_clean = pd.merge(df_clean, df_desc_clean[desc_cols], left_on="appid", right_on="steam_appid", how="left")
        for col in ["short_description", "about_the_game", "detailed_description", "clean_description"]:
            df_clean[col] = df_clean[col].fillna("No description available.")
        if "steam_appid" in df_clean.columns:
            df_clean = df_clean.drop(columns=["steam_appid"])
    else:
        for col in ["short_description", "about_the_game", "detailed_description", "clean_description"]:
            df_clean[col] = "No description available."
        
    df_engineered = engineer_features(df_clean)
    df_leakage_free = drop_leakage_columns(df_engineered)
    
    return split_and_save_data(df_leakage_free, output_dir=output_dir)


if __name__ == "__main__":
    print("--- Executing Data Cleaning, Description Preparation, Ordinal Encoding & Split Pipeline ---")
    train_df, valid_df, test_df, encoder = prepare_data()
    print("\nDataset Column Order:")
    print(train_df.columns.tolist())
    print("\nSample Rows:")
    print(train_df[["name", "short_description", "clean_description", "is_hit"]].head(2).to_dict(orient="records"))
