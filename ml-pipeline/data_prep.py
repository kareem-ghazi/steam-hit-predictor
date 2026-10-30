import os
import sys
import pickle
import shutil
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Module alias for robust unpickling across execution contexts
if __name__ == "__main__":
    sys.modules["data_prep"] = sys.modules[__name__]

from config import (
    MIN_REVIEWS_THRESHOLD, HIT_APPROVAL_THRESHOLD,
    calculate_steam_rating_text
)
from cleaner import (
    load_dataset_from_attributes, prepare_descriptions, clean_dataset
)
from feature_transformer import SteamFeatureTransformer
from generate_vibe_scores import compute_vibe_scores_dataframe


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters games with >= 50 reviews first, computes approval rate,
    Steam rating tier, target is_hit label, and heuristic vibe scores.
    """
    df = df.copy()
    total_reviews = (df["positive_ratings"] + df["negative_ratings"]).astype(int)
    df["total_reviews"] = total_reviews

    count_before = len(df)
    df = df[df["total_reviews"] >= MIN_REVIEWS_THRESHOLD].copy()
    print(f"Review Volume Filter: {count_before:,} -> {len(df):,} games with total_reviews >= {MIN_REVIEWS_THRESHOLD}.")

    df["approval_rate"] = (df["positive_ratings"] / df["total_reviews"]).round(6)
    df["rating_text"] = [
        calculate_steam_rating_text(ar, tr)
        for ar, tr in zip(df["approval_rate"], df["total_reviews"])
    ]

    df["is_hit"] = np.where(df["approval_rate"] >= HIT_APPROVAL_THRESHOLD, 1, 0).astype(int)
    hits = int(df["is_hit"].sum())
    print(f"Target Distribution: {hits:,} Hits ({hits / len(df):.1%}), {len(df) - hits:,} Non-Hits ({(len(df) - hits) / len(df):.1%}).")

    df = compute_vibe_scores_dataframe(df)
    return df


def build_prepared_games_dataframe(
    raw_split_df: pd.DataFrame,
    transformed_features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Constructs a logically ordered prepared games DataFrame:
    1. Identifiers & Core Metadata (appid, name, release_date, developer, publisher, platforms, primary_genre)
    2. Pricing (currency, price, price_scaled)
    3. Core Attributes (english, required_age, achievements, owners)
    4. Reference Categoricals (genres, categories, steamspy_tags)
    5. Continuous Vibe Scores (hype_score, story_score, nostalgia_score)
    6. Multi-hot Platform Indicators (platform_windows, platform_mac, platform_linux)
    7. Multi-hot Genre Indicators (genre_*)
    8. Multi-hot Category Indicators (cat_*)
    9. Multi-hot Tag Indicators (tag_*)
    10. Review Telemetry (positive_ratings, negative_ratings, total_reviews, approval_rate, rating_text)
    11. Target (is_hit as final column)
    """
    cols_dict = {}

    # 1. Identifiers & Core Metadata
    for col in ["appid", "name", "release_date"]:
        if col in raw_split_df.columns:
            cols_dict[col] = raw_split_df[col]

    for col in ["developer", "publisher", "platforms", "primary_genre"]:
        if col in transformed_features_df.columns:
            cols_dict[col] = transformed_features_df[col]
        elif col in raw_split_df.columns:
            cols_dict[col] = raw_split_df[col]

    # 2. Pricing
    if "currency" in raw_split_df.columns:
        cols_dict["currency"] = raw_split_df["currency"]
    if "price" in raw_split_df.columns:
        cols_dict["price"] = raw_split_df["price"].astype(float)
    if "price_scaled" in raw_split_df.columns:
        cols_dict["price_scaled"] = raw_split_df["price_scaled"].astype(float)

    # 3. Attributes (integers)
    for col in ["english", "required_age", "achievements"]:
        if col in raw_split_df.columns:
            cols_dict[col] = raw_split_df[col].astype(int)
    if "owners" in raw_split_df.columns:
        cols_dict["owners"] = raw_split_df["owners"]

    # 4. Reference strings
    for col in ["genres", "categories", "steamspy_tags"]:
        if col in raw_split_df.columns:
            cols_dict[col] = raw_split_df[col]

    # 5. Continuous Vibe Scores
    for col in ["hype_score", "story_score", "nostalgia_score"]:
        if col in transformed_features_df.columns:
            cols_dict[col] = transformed_features_df[col].astype(float)
        elif col in raw_split_df.columns:
            cols_dict[col] = raw_split_df[col].astype(float)

    # 6. Platform indicators
    for col in ["platform_windows", "platform_mac", "platform_linux"]:
        if col in transformed_features_df.columns:
            cols_dict[col] = transformed_features_df[col].astype(int)

    # 7. Genre indicators (sorted)
    genre_cols = sorted([c for c in transformed_features_df.columns if c.startswith("genre_")])
    for col in genre_cols:
        cols_dict[col] = transformed_features_df[col].astype(int)

    # 8. Category indicators (sorted)
    cat_cols = sorted([c for c in transformed_features_df.columns if c.startswith("cat_")])
    for col in cat_cols:
        cols_dict[col] = transformed_features_df[col].astype(int)

    # 9. Tag indicators (sorted)
    tag_cols = sorted([c for c in transformed_features_df.columns if c.startswith("tag_")])
    for col in tag_cols:
        cols_dict[col] = transformed_features_df[col].astype(int)

    # 10. Review Telemetry (reviews as pure integers)
    for col in ["positive_ratings", "negative_ratings", "total_reviews"]:
        if col in raw_split_df.columns:
            cols_dict[col] = raw_split_df[col].astype(int)
    if "approval_rate" in raw_split_df.columns:
        cols_dict["approval_rate"] = raw_split_df["approval_rate"].astype(float)
    if "rating_text" in raw_split_df.columns:
        cols_dict["rating_text"] = raw_split_df["rating_text"]

    # 11. Target (final column)
    if "is_hit" in raw_split_df.columns:
        cols_dict["is_hit"] = raw_split_df["is_hit"].astype(int)

    return pd.DataFrame(cols_dict, index=raw_split_df.index)


def split_and_save_data(
    df: pd.DataFrame,
    output_dir: str = "cleaned_data",
    model_save_dir: str = "saved_models/ml_models",
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[MinMaxScaler, SteamFeatureTransformer]]:
    """Splits cleaned data into train/valid/test, scales price with MinMaxScaler, fits transformer, and saves sorted splits."""
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    valid_dir = os.path.join(output_dir, "valid")
    test_dir = os.path.join(output_dir, "test")
    for d in [train_dir, valid_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    model_dir_abs = os.path.abspath(model_save_dir if os.path.isabs(model_save_dir) else os.path.join(PARENT_DIR, model_save_dir))
    os.makedirs(model_dir_abs, exist_ok=True)

    # 1. Stratified train/valid/test split
    temp_ratio = valid_ratio + test_ratio
    train_df, temp_df = train_test_split(df, test_size=temp_ratio, random_state=seed, stratify=df["is_hit"])
    valid_share = valid_ratio / temp_ratio
    valid_df, test_df = train_test_split(temp_df, test_size=(1.0 - valid_share), random_state=seed, stratify=temp_df["is_hit"])

    # 2. MinMaxScaler for price (fit on train split only)
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler.fit(train_df[["price"]])

    for split in [train_df, valid_df, test_df]:
        split["price_scaled"] = price_scaler.transform(split[["price"]]).flatten().round(4)
        split["appid"] = pd.to_numeric(split["appid"], errors="coerce").astype(int)

    # Sort all splits strictly by appid ascending
    train_df = train_df.sort_values(by="appid").reset_index(drop=True)
    valid_df = valid_df.sort_values(by="appid").reset_index(drop=True)
    test_df = test_df.sort_values(by="appid").reset_index(drop=True)

    scaler_save_path = os.path.join(model_dir_abs, "price_scaler.pkl")
    with open(scaler_save_path, "wb") as f:
        pickle.dump(price_scaler, f)
    print(f"Saved price MinMaxScaler to: {scaler_save_path}")

    # 3. Fit SteamFeatureTransformer strictly on train split
    print("Fitting SteamFeatureTransformer on train split in data_prep...")
    transformer = SteamFeatureTransformer()
    transformer.fit(train_df)

    X_train_feat = transformer.transform(train_df)
    X_val_feat = transformer.transform(valid_df)
    X_test_feat = transformer.transform(test_df)

    transformer_save_path = os.path.join(model_dir_abs, "feature_transformer.pkl")
    with open(transformer_save_path, "wb") as f:
        pickle.dump(transformer, f)
    print(f"Saved SteamFeatureTransformer to: {transformer_save_path}")

    # 4. Build prepared games DataFrames with ordered columns
    train_attr = build_prepared_games_dataframe(train_df, X_train_feat)
    valid_attr = build_prepared_games_dataframe(valid_df, X_val_feat)
    test_attr = build_prepared_games_dataframe(test_df, X_test_feat)

    # 5. Extract descriptions matching appids
    desc_columns = ["appid", "short_description", "about_the_game", "detailed_description", "clean_description"]
    train_desc = train_df[[c for c in desc_columns if c in train_df.columns]].copy()
    valid_desc = valid_df[[c for c in desc_columns if c in valid_df.columns]].copy()
    test_desc = test_df[[c for c in desc_columns if c in test_df.columns]].copy()

    # Final sorting verification
    train_attr = train_attr.sort_values(by="appid").reset_index(drop=True)
    train_desc = train_desc.sort_values(by="appid").reset_index(drop=True)
    valid_attr = valid_attr.sort_values(by="appid").reset_index(drop=True)
    valid_desc = valid_desc.sort_values(by="appid").reset_index(drop=True)
    test_attr = test_attr.sort_values(by="appid").reset_index(drop=True)
    test_desc = test_desc.sort_values(by="appid").reset_index(drop=True)

    # 6. Save splits to disk
    train_attr.to_csv(os.path.join(train_dir, "games.csv"), index=False)
    train_desc.to_csv(os.path.join(train_dir, "games_descriptions.csv"), index=False)
    valid_attr.to_csv(os.path.join(valid_dir, "games.csv"), index=False)
    valid_desc.to_csv(os.path.join(valid_dir, "games_descriptions.csv"), index=False)
    test_attr.to_csv(os.path.join(test_dir, "games.csv"), index=False)
    test_desc.to_csv(os.path.join(test_dir, "games_descriptions.csv"), index=False)

    print(f"\nSaved Cleaned, Ordered Splits to '{output_dir}/':")
    print(f"  Train: {len(train_attr):,} rows, {len(train_attr.columns)} cols ({len(transformer.feature_names)} model features)")
    print(f"  Valid: {len(valid_attr):,} rows, {len(valid_attr.columns)} cols")
    print(f"  Test:  {len(test_attr):,} rows, {len(test_attr.columns)} cols")
    print(f"  Final Column (Target): '{train_attr.columns[-1]}'")
    return (train_attr, valid_attr, test_attr), (train_desc, valid_desc, test_desc), (price_scaler, transformer)


def load_prepared_data(
    data_dir: str = "cleaned_data",
    transformer_path: str = "saved_models/ml_models/feature_transformer.pkl"
) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], SteamFeatureTransformer]:
    """Loads prepared train, valid, test splits and restores pd.Categorical dtypes."""
    candidates = [
        transformer_path,
        os.path.join(PARENT_DIR, transformer_path),
        os.path.join(CURRENT_DIR, "..", transformer_path),
    ]
    resolved_path = next((os.path.abspath(c) for c in candidates if os.path.exists(c)), None)
    if not resolved_path:
        raise FileNotFoundError(f"feature_transformer.pkl not found at: {transformer_path}")

    with open(resolved_path, "rb") as f:
        transformer = pickle.load(f)

    def resolve_csv(split_name: str) -> str:
        opts = [
            os.path.join(data_dir, split_name, "games.csv"),
            os.path.join(PARENT_DIR, data_dir, split_name, "games.csv"),
            os.path.join(CURRENT_DIR, "..", data_dir, split_name, "games.csv"),
        ]
        return next((os.path.abspath(o) for o in opts if os.path.exists(o)), os.path.abspath(opts[0]))

    train_df = pd.read_csv(resolve_csv("train"))
    valid_df = pd.read_csv(resolve_csv("valid"))
    test_df = pd.read_csv(resolve_csv("test"))

    for col, cats in transformer.cat_categories.items():
        for d in [train_df, valid_df, test_df]:
            if col in d.columns:
                d[col] = pd.Categorical(d[col], categories=cats)

    return (train_df, valid_df, test_df), transformer


def prepare_data(
    raw_dir: Optional[str] = None,
    output_dir: str = "cleaned_data",
    model_save_dir: str = "saved_models/ml_models"
):
    """Executes the full modular data preparation pipeline."""
    df_raw = load_dataset_from_attributes(raw_dir=raw_dir)
    df_desc = prepare_descriptions(df_raw)
    df_cleaned = clean_dataset(df_desc)
    df_engineered = engineer_features(df_cleaned)
    return split_and_save_data(df_engineered, output_dir=output_dir, model_save_dir=model_save_dir)


if __name__ == "__main__":
    print("--- Executing Modular Attribute Data Prep Pipeline ---")
    prepare_data()
