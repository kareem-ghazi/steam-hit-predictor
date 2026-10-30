import os
import re
import html
import pickle
import shutil
import numpy as np
import pandas as pd
import sys
from typing import Tuple, Optional, List, Dict, Any
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Register module alias so pickling works consistently whether run directly or imported
if __name__ == "__main__":
    sys.modules["data_prep"] = sys.modules[__name__]

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


def parse_delimited_tokens(val: Any) -> List[str]:
    """
    Parses semicolon, comma, or bracket-delimited multi-label strings into a clean list of lowercased tokens.
    Excludes empty bracket representations ('[]'), non-alphanumeric noise, and invalid tokens.
    """
    if not val or pd.isna(val):
        return []
    s = str(val).strip()
    if s in {"", "[]", "['']", "nan", "none", "unknown", "null"}:
        return []
    # Strip enclosing brackets if represented as list string: "[item1, item2]"
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    delim = ";" if ";" in s else ("," if "," in s else ";")
    tokens = []
    for raw in s.split(delim):
        cleaned = raw.strip().strip("'\"").strip().lower()
        # Require at least two alphanumeric characters to filter out noise, punctuation, single chars
        if len(re.findall(r"[a-z0-9]", cleaned)) >= 2:
            tokens.append(cleaned)
    return tokens


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning:
    1. Removes duplicate rows and duplicate appids.
    2. Standardizes semicolon-separated fields and prunes malformed bracketed noise.
    3. Drops rows ENTIRELY if developer or publisher is missing/unknown (no imputation).
    4. Drops rows missing critical identifiers (name, genres, platforms).
    5. Unifies pricing into USD using currency exchange rates.
    6. Validates numeric boundaries.
    """
    df = df.copy()
    initial_count = len(df)

    # 1. Deduplication
    df = df.drop_duplicates(subset=["appid"])

    # 2. Standardize semicolon-separated fields first (cleaning list noise and empty brackets)
    delimited_cols = ["platforms", "categories", "genres", "steamspy_tags"]
    for col in delimited_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda val: ";".join(sorted(parse_delimited_tokens(val)))
            )

    # 3. Drop rows missing developer or publisher entirely (instead of imputation)
    invalid_tokens = {"", "nan", "unknown", "none", "null", "[]", "['']"}
    dev_str = df["developer"].astype(str).str.strip().str.lower()
    pub_str = df["publisher"].astype(str).str.strip().str.lower()
    
    valid_dev = df["developer"].notna() & (~dev_str.isin(invalid_tokens))
    valid_pub = df["publisher"].notna() & (~pub_str.isin(invalid_tokens))
    df = df[valid_dev & valid_pub].copy()

    # 4. Drop rows missing name, genres, or platforms
    name_str = df["name"].astype(str).str.strip().str.lower()
    genres_str = df["genres"].astype(str).str.strip().str.lower()
    plats_str = df["platforms"].astype(str).str.strip().str.lower()

    valid_name = df["name"].notna() & (~name_str.isin(invalid_tokens)) & (name_str != "")
    valid_genres = df["genres"].notna() & (~genres_str.isin(invalid_tokens)) & (genres_str != "")
    valid_plats = df["platforms"].notna() & (~plats_str.isin(invalid_tokens)) & (plats_str != "")
    df = df[valid_name & valid_genres & valid_plats].copy()

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


class SteamFeatureTransformer:
    """
    Transforms Steam game tabular features for XGBoost:
    - Multi-label categoricals (genres, categories, steamspy_tags, platforms) are properly
      decomposed into clean multi-hot binary indicators via MultiLabelBinarizer.
    - Low-frequency noise, bracketed garbage (e.g. '[]'), and corrupt non-ASCII tokens are pruned.
    - Standardized column names using single underscores without multiple consecutive underscores.
    - High-cardinality nominal categoricals (developer, publisher) are handled natively via
      pd.Categorical with infrequent levels mapped to 'other_developer'/'other_publisher',
      complemented with entity frequency count signals.
    - Continuous vibe scores (hype_score, story_score, nostalgia_score) and numeric attributes
      (price, english, required_age, achievements) are retained as true continuous features.
    - Explicit pruning of all-zero features ensures zero dead features in output datasets.
    - NOTE: 'owners' is completely excluded to prevent post-launch target leakage for unreleased games.
    """
    def __init__(
        self,
        min_dev_count: int = 3,
        min_pub_count: int = 3,
        min_genre_count: int = 20,
        min_category_count: int = 30,
        top_n_tags: int = 100
    ):
        self.min_dev_count = min_dev_count
        self.min_pub_count = min_pub_count
        self.min_genre_count = min_genre_count
        self.min_category_count = min_category_count
        self.top_n_tags = top_n_tags

        # Native categorical columns (owners removed as post-launch leakage)
        self.cat_cols = ["developer", "publisher", "platforms", "primary_genre"]
        self.num_cols = [
            "price",
            "english",
            "required_age",
            "achievements",
            "hype_score",
            "story_score",
            "nostalgia_score"
        ]

        self.cat_categories: Dict[str, List[Any]] = {}
        self.dev_counts: Dict[str, int] = {}
        self.pub_counts: Dict[str, int] = {}

        self.selected_genres: List[str] = []
        self.selected_cats: List[str] = []
        self.selected_tags: List[str] = []

        self.genre_mlb: Optional[MultiLabelBinarizer] = None
        self.cat_mlb: Optional[MultiLabelBinarizer] = None
        self.tag_mlb: Optional[MultiLabelBinarizer] = None

        self.genre_cols: List[str] = []
        self.cat_binary_cols: List[str] = []
        self.tag_binary_cols: List[str] = []
        self.feature_names: List[str] = []

    @staticmethod
    def _clean_feature_name(prefix: str, token: str) -> str:
        """Standardizes feature column name with single underscores and stripped bounds."""
        cleaned = re.sub(r'[^a-z0-9]+', '_', token.strip().lower()).strip('_')
        return f"{prefix}_{cleaned}"

    def _tokenize_delimited(self, series: pd.Series) -> pd.Series:
        return series.apply(parse_delimited_tokens)

    def _extract_tags(self, series: pd.Series) -> pd.Series:
        def parse_tags(val: Any) -> List[str]:
            if not val or pd.isna(val):
                return []
            s = str(val).strip()
            if s in {"", "[]", "['']", "nan", "none", "unknown", "null"}:
                return []
            # If tags are in dictionary format: {'tag': count, ...}
            if s.startswith("{") and s.endswith("}"):
                keys = re.findall(r"['\"]([^'\"]+)['\"]\s*:", s)
                if keys:
                    return [
                        k.strip().lower() for k in keys 
                        if len(re.findall(r'[a-z0-9]', k)) >= 2
                    ]
            return parse_delimited_tokens(s)
        return series.apply(parse_tags)

    def fit(self, df: pd.DataFrame):
        # 1. Developer and Publisher
        dev_series = df["developer"].fillna("Unknown").astype(str).str.strip().str.lower()
        pub_series = df["publisher"].fillna("Unknown").astype(str).str.strip().str.lower()

        dev_counts = Counter(dev_series)
        pub_counts = Counter(pub_series)
        self.dev_counts = dict(dev_counts)
        self.pub_counts = dict(pub_counts)

        frequent_devs = sorted([d for d, c in dev_counts.items() if c >= self.min_dev_count and d != "unknown"])
        frequent_pubs = sorted([p for p, c in pub_counts.items() if c >= self.min_pub_count and p != "unknown"])

        self.cat_categories["developer"] = frequent_devs + ["other_developer"]
        self.cat_categories["publisher"] = frequent_pubs + ["other_publisher"]

        # 2. Platforms
        plat_tokens = self._tokenize_delimited(df.get("platforms", pd.Series([""] * len(df))))
        plat_combos = plat_tokens.apply(lambda toks: ";".join(sorted(toks)) if toks else "unknown")
        self.cat_categories["platforms"] = sorted(plat_combos.unique().tolist()) + ["unknown"]

        # 3. Primary genre
        genre_tokens = self._tokenize_delimited(df.get("genres", pd.Series([""] * len(df))))
        primary_genres = genre_tokens.apply(lambda toks: toks[0] if toks else "unknown")
        self.cat_categories["primary_genre"] = sorted(primary_genres.unique().tolist()) + ["unknown"]

        # 4. Multi-label binarizers
        # Genres: filter by min count threshold (default 20) and require valid alphanumeric token
        genre_counts = Counter([g for toks in genre_tokens for g in toks if len(re.findall(r'[a-z0-9]', g)) >= 2])
        self.selected_genres = sorted([
            g for g, count in genre_counts.items()
            if count >= self.min_genre_count and len(re.sub(r'[^a-z0-9]+', '', g)) >= 2
        ])
        self.genre_mlb = MultiLabelBinarizer(classes=self.selected_genres)
        self.genre_mlb.fit([])
        self.genre_cols = [self._clean_feature_name("genre", g) for g in self.selected_genres]

        # Categories: filter by min count threshold (default 30) and require valid alphanumeric token
        cat_tokens = self._tokenize_delimited(df.get("categories", pd.Series([""] * len(df))))
        cat_counts = Counter([c for toks in cat_tokens for c in toks if len(re.findall(r'[a-z0-9]', c)) >= 2])
        self.selected_cats = sorted([
            c for c, count in cat_counts.items()
            if count >= self.min_category_count and len(re.sub(r'[^a-z0-9]+', '', c)) >= 2
        ])
        self.cat_mlb = MultiLabelBinarizer(classes=self.selected_cats)
        self.cat_mlb.fit([])
        self.cat_binary_cols = [self._clean_feature_name("cat", c) for c in self.selected_cats]

        # Tags: top N tags with valid alphanumeric content
        tag_tokens = self._extract_tags(df.get("steamspy_tags", pd.Series([""] * len(df))))
        tag_counts = Counter([t for toks in tag_tokens for t in toks if len(re.findall(r'[a-z0-9]', t)) >= 2])
        candidate_tags = [t for t, _ in tag_counts.most_common(self.top_n_tags * 2)]
        valid_tags = [t for t in candidate_tags if len(re.sub(r'[^a-z0-9]+', '', t)) >= 2][:self.top_n_tags]
        self.selected_tags = sorted(valid_tags)
        self.tag_mlb = MultiLabelBinarizer(classes=self.selected_tags)
        self.tag_mlb.fit([])
        self.tag_binary_cols = [self._clean_feature_name("tag", t) for t in self.selected_tags]

        # Build preliminary feature names list
        preliminary_names = (
            self.num_cols +
            ["platform_windows", "platform_mac", "platform_linux", "num_platforms"] +
            self.genre_cols + ["num_genres"] +
            self.cat_binary_cols + ["num_categories"] +
            self.tag_binary_cols + ["num_tags"] +
            ["dev_game_count", "pub_game_count"] +
            self.cat_cols
        )
        self.feature_names = preliminary_names

        # Transform training df to verify and prune any all-zero features
        sample_transformed = self.transform(df)
        all_zero_cols = [c for c in sample_transformed.columns if (sample_transformed[c] == 0).all()]
        if all_zero_cols:
            print(f"Pruning {len(all_zero_cols)} all-zero features from transformer: {all_zero_cols}")
            self.feature_names = [f for f in self.feature_names if f not in all_zero_cols]
            self.genre_cols = [c for c in self.genre_cols if c in self.feature_names]
            self.cat_binary_cols = [c for c in self.cat_binary_cols if c in self.feature_names]
            self.tag_binary_cols = [c for c in self.tag_binary_cols if c in self.feature_names]

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        feature_dict = {}

        # 1. Numerical Base
        for col in self.num_cols:
            if col in df_copy.columns:
                feature_dict[col] = pd.to_numeric(df_copy[col], errors="coerce").fillna(0.0).values
            elif col == "price" and "price_scaled" in df_copy.columns:
                feature_dict[col] = pd.to_numeric(df_copy["price_scaled"], errors="coerce").fillna(0.0).values
            else:
                feature_dict[col] = np.zeros(len(df_copy), dtype=float)

        # 2. Tokenize multi-label columns
        plat_tokens = self._tokenize_delimited(df_copy.get("platforms", pd.Series([""] * len(df_copy), index=df_copy.index)))
        genre_tokens = self._tokenize_delimited(df_copy.get("genres", pd.Series([""] * len(df_copy), index=df_copy.index)))
        cat_tokens = self._tokenize_delimited(df_copy.get("categories", pd.Series([""] * len(df_copy), index=df_copy.index)))
        tag_tokens = self._extract_tags(df_copy.get("steamspy_tags", pd.Series([""] * len(df_copy), index=df_copy.index)))

        # 3. Platform indicators & count
        feature_dict["platform_windows"] = plat_tokens.apply(lambda toks: 1 if "windows" in toks else 0).values.astype(np.int8)
        feature_dict["platform_mac"] = plat_tokens.apply(lambda toks: 1 if "mac" in toks else 0).values.astype(np.int8)
        feature_dict["platform_linux"] = plat_tokens.apply(lambda toks: 1 if "linux" in toks else 0).values.astype(np.int8)
        feature_dict["num_platforms"] = plat_tokens.apply(len).values.astype(np.int8)

        # 4. Multi-hot Genre Features
        genre_set = set(self.selected_genres)
        filtered_genre_tokens = genre_tokens.apply(lambda toks: [g for g in toks if g in genre_set])
        genre_mat = self.genre_mlb.transform(filtered_genre_tokens).astype(np.int8)
        for i, col_name in enumerate(self.genre_cols):
            feature_dict[col_name] = genre_mat[:, i]
        feature_dict["num_genres"] = genre_tokens.apply(len).values.astype(np.int8)

        # 5. Multi-hot Category Features
        cat_set = set(self.selected_cats)
        filtered_cat_tokens = cat_tokens.apply(lambda toks: [c for c in toks if c in cat_set])
        cat_mat = self.cat_mlb.transform(filtered_cat_tokens).astype(np.int8)
        for i, col_name in enumerate(self.cat_binary_cols):
            feature_dict[col_name] = cat_mat[:, i]
        feature_dict["num_categories"] = cat_tokens.apply(len).values.astype(np.int8)

        # 6. Multi-hot SteamSpy Tag Features
        tag_set = set(self.selected_tags)
        filtered_tag_tokens = tag_tokens.apply(lambda toks: [t for t in toks if t in tag_set])
        tag_mat = self.tag_mlb.transform(filtered_tag_tokens).astype(np.int8)
        for i, col_name in enumerate(self.tag_binary_cols):
            feature_dict[col_name] = tag_mat[:, i]
        feature_dict["num_tags"] = tag_tokens.apply(len).values.astype(np.int8)

        # 7. Frequency & Count signals
        dev_series = df_copy["developer"].fillna("Unknown").astype(str).str.strip().str.lower()
        pub_series = df_copy["publisher"].fillna("Unknown").astype(str).str.strip().str.lower()
        feature_dict["dev_game_count"] = dev_series.map(self.dev_counts).fillna(1.0).values.astype(float)
        feature_dict["pub_game_count"] = pub_series.map(self.pub_counts).fillna(1.0).values.astype(float)

        # 8. Native Categorical Columns
        # Developer
        dev_cats = self.cat_categories["developer"]
        dev_valid = dev_series.where(dev_series.isin(dev_cats[:-1]), other="other_developer")
        feature_dict["developer"] = pd.Categorical(dev_valid, categories=dev_cats)

        # Publisher
        pub_cats = self.cat_categories["publisher"]
        pub_valid = pub_series.where(pub_series.isin(pub_cats[:-1]), other="other_publisher")
        feature_dict["publisher"] = pd.Categorical(pub_valid, categories=pub_cats)

        # Platforms combo
        plat_combos = plat_tokens.apply(lambda toks: ";".join(sorted(toks)) if toks else "unknown")
        plat_cats = self.cat_categories["platforms"]
        plat_valid = plat_combos.where(plat_combos.isin(plat_cats), other="unknown")
        feature_dict["platforms"] = pd.Categorical(plat_valid, categories=plat_cats)

        # Primary Genre
        pg = genre_tokens.apply(lambda toks: toks[0] if toks else "unknown")
        pg_cats = self.cat_categories["primary_genre"]
        pg_valid = pg.where(pg.isin(pg_cats), other="unknown")
        feature_dict["primary_genre"] = pd.Categorical(pg_valid, categories=pg_cats)

        result_df = pd.DataFrame(feature_dict, index=df_copy.index)
        if hasattr(self, "feature_names") and self.feature_names:
            cols_to_keep = [c for c in self.feature_names if c in result_df.columns]
            result_df = result_df[cols_to_keep]
        else:
            self.feature_names = list(result_df.columns)
        return result_df

    def to_dict(self) -> Dict[str, Any]:
        """Serializes transformer state into a portable dictionary."""
        return {
            "min_dev_count": self.min_dev_count,
            "min_pub_count": self.min_pub_count,
            "min_genre_count": self.min_genre_count,
            "min_category_count": self.min_category_count,
            "top_n_tags": self.top_n_tags,
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "cat_categories": self.cat_categories,
            "dev_counts": self.dev_counts,
            "pub_counts": self.pub_counts,
            "selected_genres": self.selected_genres,
            "selected_cats": self.selected_cats,
            "selected_tags": self.selected_tags,
            "genre_cols": self.genre_cols,
            "cat_binary_cols": self.cat_binary_cols,
            "tag_binary_cols": self.tag_binary_cols,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SteamFeatureTransformer":
        """Reconstructs a SteamFeatureTransformer from a serialized dictionary."""
        transformer = cls(
            min_dev_count=data["min_dev_count"],
            min_pub_count=data["min_pub_count"],
            min_genre_count=data.get("min_genre_count", 20),
            min_category_count=data["min_category_count"],
            top_n_tags=data["top_n_tags"]
        )
        transformer.cat_cols = data["cat_cols"]
        transformer.num_cols = data["num_cols"]
        transformer.cat_categories = data["cat_categories"]
        transformer.dev_counts = data["dev_counts"]
        transformer.pub_counts = data["pub_counts"]
        transformer.selected_genres = data["selected_genres"]
        transformer.selected_cats = data["selected_cats"]
        transformer.selected_tags = data["selected_tags"]
        transformer.genre_cols = data["genre_cols"]
        transformer.cat_binary_cols = data["cat_binary_cols"]
        transformer.tag_binary_cols = data["tag_binary_cols"]
        transformer.feature_names = data["feature_names"]

        transformer.genre_mlb = MultiLabelBinarizer(classes=transformer.selected_genres)
        transformer.genre_mlb.fit([])

        transformer.cat_mlb = MultiLabelBinarizer(classes=transformer.selected_cats)
        transformer.cat_mlb.fit([])

        transformer.tag_mlb = MultiLabelBinarizer(classes=transformer.selected_tags)
        transformer.tag_mlb.fit([])

        return transformer


# Ensure class identity works seamlessly whether imported from data_prep or xgboost_train
SteamFeatureTransformer.__module__ = "data_prep"


def build_prepared_games_dataframe(
    raw_split_df: pd.DataFrame,
    transformed_features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Constructs the consolidated prepared games DataFrame for cleaned_data splits.
    Includes:
    1. Identifiers & reference columns (appid, name, release_date, currency, genres, categories, steamspy_tags, owners)
    2. Scaled price (price_scaled)
    3. All 223 prepared features ready for model consumption
    4. Telemetry columns (positive_ratings, negative_ratings, total_reviews, approval_rate, rating_text)
    5. Target label (is_hit) as the final column.
    """
    meta_cols = [
        c for c in [
            "appid", "name", "release_date", "currency",
            "genres", "categories", "steamspy_tags", "owners"
        ] if c in raw_split_df.columns
    ]
    telemetry_cols = [
        c for c in [
            "positive_ratings", "negative_ratings", "total_reviews", "approval_rate", "rating_text"
        ] if c in raw_split_df.columns
    ]

    parts = [raw_split_df[meta_cols]]

    if "price_scaled" in raw_split_df.columns and "price_scaled" not in transformed_features_df.columns:
        parts.append(raw_split_df[["price_scaled"]])

    parts.append(transformed_features_df)

    if telemetry_cols:
        parts.append(raw_split_df[telemetry_cols])

    if "is_hit" in raw_split_df.columns:
        parts.append(raw_split_df[["is_hit"]])

    res = pd.concat(parts, axis=1)
    return res


def split_and_save_data(
    df: pd.DataFrame,
    output_dir: str = "cleaned_data",
    model_save_dir: str = "saved_models/ml_models",
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[StandardScaler, SteamFeatureTransformer]]:
    """
    Splits the cleaned dataset into train (80%), valid (10%), and test (10%) sets
    using stratified sampling on 'is_hit'.
    Applies StandardScaler on 'price' (fitted strictly on train set).
    Fits SteamFeatureTransformer strictly on train set and prepares all categorical/numerical features.
    Saves outputs into THREE separate folders:
    1. cleaned_data/train/ (games.csv with already-prepared features and games_descriptions.csv)
    2. cleaned_data/valid/ (games.csv with already-prepared features and games_descriptions.csv)
    3. cleaned_data/test/ (games.csv with already-prepared features and games_descriptions.csv)
    Persists scaler to saved_models/ml_models/price_scaler.pkl and transformer to saved_models/ml_models/feature_transformer.pkl.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    valid_dir = os.path.join(output_dir, "valid")
    test_dir = os.path.join(output_dir, "test")
    for d in [train_dir, valid_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    model_dir_abs = os.path.abspath(model_save_dir if os.path.isabs(model_save_dir) else os.path.join(parent_dir, model_save_dir))
    os.makedirs(model_dir_abs, exist_ok=True)

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
        split["appid"] = pd.to_numeric(split["appid"], errors="coerce").astype(int)

    # Sort all splits strictly by appid ascending
    train_df = train_df.sort_values(by="appid").reset_index(drop=True)
    valid_df = valid_df.sort_values(by="appid").reset_index(drop=True)
    test_df = test_df.sort_values(by="appid").reset_index(drop=True)

    scaler_save_path = os.path.join(model_dir_abs, "price_scaler.pkl")
    with open(scaler_save_path, "wb") as f:
        pickle.dump(price_scaler, f)
    print(f"Saved price StandardScaler to: {scaler_save_path}")

    # 3. Fit SteamFeatureTransformer strictly on train set and transform all splits
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

    # 4. Build prepared games DataFrames
    train_attr = build_prepared_games_dataframe(train_df, X_train_feat)
    valid_attr = build_prepared_games_dataframe(valid_df, X_val_feat)
    test_attr = build_prepared_games_dataframe(test_df, X_test_feat)

    # 5. Extract descriptions
    desc_columns = ["appid", "short_description", "about_the_game", "detailed_description", "clean_description"]
    train_desc = train_df[[c for c in desc_columns if c in train_df.columns]].copy()
    valid_desc = valid_df[[c for c in desc_columns if c in valid_df.columns]].copy()
    test_desc = test_df[[c for c in desc_columns if c in test_df.columns]].copy()

    # Sort all splits explicitly by appid to guarantee strict ascending order
    train_attr = train_attr.sort_values(by="appid").reset_index(drop=True)
    train_desc = train_desc.sort_values(by="appid").reset_index(drop=True)

    valid_attr = valid_attr.sort_values(by="appid").reset_index(drop=True)
    valid_desc = valid_desc.sort_values(by="appid").reset_index(drop=True)

    test_attr = test_attr.sort_values(by="appid").reset_index(drop=True)
    test_desc = test_desc.sort_values(by="appid").reset_index(drop=True)

    # 6. Save to disk in the three dedicated split folders (train, valid, test)
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

    print(f"\nSaved Prepared Three Split Folders to '{output_dir}/':")
    print(f"  1. Train folder: '{train_dir}/'")
    print(f"     - games.csv:              {len(train_attr):,} rows ({len(train_attr.columns)} columns, {len(transformer.feature_names)} model features)")
    print(f"     - games_descriptions.csv: {len(train_desc):,} rows ({len(train_desc.columns)} columns)")
    print(f"  2. Valid folder: '{valid_dir}/'")
    print(f"     - games.csv:              {len(valid_attr):,} rows ({len(valid_attr.columns)} columns)")
    print(f"     - games_descriptions.csv: {len(valid_desc):,} rows ({len(valid_desc.columns)} columns)")
    print(f"  3. Test folder:  '{test_dir}/'")
    print(f"     - games.csv:              {len(test_attr):,} rows ({len(test_attr.columns)} columns)")
    print(f"     - games_descriptions.csv: {len(test_desc):,} rows ({len(test_desc.columns)} columns)")
    print(f"  Target variable at the end of games.csv: '{train_attr.columns[-1]}'")

    return (train_attr, valid_attr, test_attr), (train_desc, valid_desc, test_desc), (price_scaler, transformer)


def load_prepared_data(
    data_dir: str = "cleaned_data",
    transformer_path: str = "saved_models/ml_models/feature_transformer.pkl"
) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], SteamFeatureTransformer]:
    """
    Loads already-prepared train, valid, and test DataFrames from cleaned_data/
    and restores pd.Categorical dtypes using the persisted SteamFeatureTransformer.
    Returns: ((train_df, valid_df, test_df), transformer)
    """
    candidates = [
        transformer_path,
        os.path.join(parent_dir, transformer_path),
        os.path.join(current_dir, "..", transformer_path),
    ]
    resolved_transformer_path = None
    for c in candidates:
        if os.path.exists(c):
            resolved_transformer_path = os.path.abspath(c)
            break

    if resolved_transformer_path is None or not os.path.exists(resolved_transformer_path):
        raise FileNotFoundError(f"feature_transformer.pkl not found at: {transformer_path}")

    with open(resolved_transformer_path, "rb") as f:
        transformer = pickle.load(f)

    def resolve_csv(split_name: str) -> str:
        opts = [
            os.path.join(data_dir, split_name, "games.csv"),
            os.path.join(parent_dir, data_dir, split_name, "games.csv"),
            os.path.join(current_dir, "..", data_dir, split_name, "games.csv"),
        ]
        for opt in opts:
            if os.path.exists(opt):
                return os.path.abspath(opt)
        return os.path.abspath(opts[0])

    train_df = pd.read_csv(resolve_csv("train"))
    valid_df = pd.read_csv(resolve_csv("valid"))
    test_df = pd.read_csv(resolve_csv("test"))

    # Restore native categorical types with consistent training categories
    for col, cats in transformer.cat_categories.items():
        if col in train_df.columns:
            train_df[col] = pd.Categorical(train_df[col], categories=cats)
        if col in valid_df.columns:
            valid_df[col] = pd.Categorical(valid_df[col], categories=cats)
        if col in test_df.columns:
            test_df[col] = pd.Categorical(test_df[col], categories=cats)

    return (train_df, valid_df, test_df), transformer


def prepare_data(
    raw_dir: Optional[str] = None,
    output_dir: str = "cleaned_data",
    model_save_dir: str = "saved_models/ml_models"
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
    9. Fits SteamFeatureTransformer strictly on train set and prepares all model features
    10. Saves already-prepared split files into three folders:
        cleaned_data/train/, cleaned_data/valid/, cleaned_data/test/
        (each containing games.csv and games_descriptions.csv)
    """
    df_raw = load_dataset_from_attributes(raw_dir=raw_dir)
    df_desc = prepare_descriptions(df_raw)
    df_cleaned = clean_dataset(df_desc)
    df_engineered = engineer_features(df_cleaned)
    return split_and_save_data(df_engineered, output_dir=output_dir, model_save_dir=model_save_dir)


if __name__ == "__main__":
    print("--- Executing Modular Attribute Data Prep Pipeline ---")
    attr_splits, desc_splits, artifacts = prepare_data()
    train_attr, valid_attr, test_attr = attr_splits
    train_desc, valid_desc, test_desc = desc_splits
    price_scaler, transformer = artifacts
    print(f"\nModel Feature Columns ({len(transformer.feature_names)}):")
    print(transformer.feature_names[:10], "... +", len(transformer.feature_names) - 10, "more")
    print("\nSample Attributes Row:")
    print(train_attr[["name", "developer", "publisher", "price", "price_scaled", "total_reviews", "approval_rate", "is_hit"]].head(2).to_dict(orient="records"))
    print("\nSample Descriptions Row:")
    print(train_desc[["appid", "clean_description"]].head(2).to_dict(orient="records"))
