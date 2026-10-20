import os
import re
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


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


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs thorough data cleaning:
    1. Removes duplicate rows and duplicates by appid.
    2. Drops rows with null or empty essential fields (name, genres, platforms).
    3. Imputes missing developer and publisher values.
    4. Fixes inconsistent formatting: strips whitespace, trims & sorts delimited items.
    5. Validates non-negative numerical values (price, ratings).
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
    
    # 4. Numeric validation and clamping
    numeric_cols = ["price", "achievements", "positive_ratings", "negative_ratings", "required_age"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df[col] = df[col].apply(lambda x: max(0.0, float(x)))
            
    cleaned_count = len(df)
    print(f"Data Cleaning: Reduced rows from {initial_count} to {cleaned_count} (removed {initial_count - cleaned_count} invalid/duplicate rows).")
    return df


def engineer_features(df: pd.DataFrame, include_mock_llm: bool = True, seed: int = 42) -> pd.DataFrame:
    """
    Performs data engineering and target creation:
    1. approval_rate = positive_ratings / (positive_ratings + negative_ratings)
    2. rating_text matching Steam review tiers taking volume into account
    3. is_hit target variable = 1 if approval_rate >= 0.80 and total reviews >= 50, else 0
    4. Adds mock placeholders for LLM features (hype_score, story_score, nostalgia_score)
    """
    df = df.copy()
    
    # 1. Total reviews & approval rate
    total_reviews = df["positive_ratings"] + df["negative_ratings"]
    df["total_reviews"] = total_reviews
    
    df["approval_rate"] = np.where(
        total_reviews > 0,
        df["positive_ratings"] / total_reviews,
        0.0
    )
    
    # 2. Rating text matching Steam classes
    df["rating_text"] = [
        calculate_steam_rating_text(ar, tr)
        for ar, tr in zip(df["approval_rate"], df["total_reviews"])
    ]
    
    # 3. Target variable: is_hit (1 if approval_rate >= 0.80 and total_reviews >= 50, else 0)
    df["is_hit"] = np.where(
        (df["approval_rate"] >= 0.80) & (df["total_reviews"] >= 50),
        1,
        0
    )
    
    # 4. Mock placeholders for LLM features (hype_score, story_score, nostalgia_score on a 1.0 to 10.0 scale)
    if include_mock_llm:
        rng = np.random.RandomState(seed)
        n = len(df)
        
        base_hype = np.where(df["is_hit"] == 1, 7.5, 5.0) + rng.normal(0, 1.2, size=n)
        base_story = np.where(df["is_hit"] == 1, 7.2, 5.2) + rng.normal(0, 1.3, size=n)
        base_nostalgia = np.where(df["is_hit"] == 1, 6.8, 4.8) + rng.normal(0, 1.5, size=n)
        
        df["hype_score"] = np.clip(np.round(base_hype, 1), 1.0, 10.0)
        df["story_score"] = np.clip(np.round(base_story, 1), 1.0, 10.0)
        df["nostalgia_score"] = np.clip(np.round(base_nostalgia, 1), 1.0, 10.0)
        
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
    Fits an OrdinalEncoder on training set 'genres' and 'platforms' columns
    and transforms train, valid, and test sets.
    Preserves original string columns for UI and similarity matching.
    """
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    
    # Fit strictly on train to avoid data leakage
    encoder.fit(train_df[["genres", "platforms"]])
    
    for split_df in [train_df, valid_df, test_df]:
        encoded_vals = encoder.transform(split_df[["genres", "platforms"]])
        split_df["genres_encoded"] = encoded_vals[:, 0].astype(float)
        split_df["platforms_encoded"] = encoded_vals[:, 1].astype(float)
        
    return train_df, valid_df, test_df, encoder


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
    using stratified sampling on 'is_hit', applies Ordinal Encoding, and saves splits to output_dir.
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
    
    # Apply Ordinal Encoding on genres & platforms
    train_df, valid_df, test_df, encoder = apply_ordinal_encoding(train_df, valid_df, test_df)
    
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
    
    return train_df, valid_df, test_df, encoder


def prepare_data(
    raw_path: str = "raw_data/steam.csv",
    output_dir: str = "cleaned_data"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    """
    Full pipeline execution:
    1. Loads raw dataset
    2. Cleans duplicates, nulls, and canonicalizes formatting
    3. Engineers approval_rate, rating_text, is_hit, and mock LLM features
    4. Drops data leakage columns
    5. Splits into train (80%), valid (10%), test (10%)
    6. Fits & applies Ordinal Encoding on 'genres' and 'platforms'
    7. Saves all splits to cleaned_data/
    """
    df_raw = load_dataset(raw_path)
    df_clean = clean_dataset(df_raw)
    df_engineered = engineer_features(df_clean, include_mock_llm=True)
    df_leakage_free = drop_leakage_columns(df_engineered)
    
    return split_and_save_data(df_leakage_free, output_dir=output_dir)


if __name__ == "__main__":
    print("--- Executing Data Cleaning, Engineering, Ordinal Encoding & Split Pipeline ---")
    train_df, valid_df, test_df, encoder = prepare_data()
    print("\nSample Ordinal Encoded Features:")
    print(train_df[["name", "genres", "genres_encoded", "platforms", "platforms_encoded", "price", "is_hit"]].head(5))
