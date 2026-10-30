import re
from typing import Dict, Any, List, Optional
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from config import (
    MIN_DEV_COUNT, MIN_PUB_COUNT, MIN_GENRE_COUNT,
    MIN_CATEGORY_COUNT, TOP_N_TAGS, parse_delimited_tokens
)


class SteamFeatureTransformer:
    """
    Transforms Steam tabular features for ML modeling:
    - Multi-label fields (genres, categories, steamspy_tags, platforms) -> clean multi-hot binary indicators.
    - Low-frequency noise, bracketed artifacts, and corrupt non-ASCII tokens are pruned.
    - High-cardinality nominal fields (developer, publisher) -> native pd.Categorical + entity frequency signals.
    - Vibe scores and price -> continuous float signals.
    - Owners is excluded to prevent post-launch target leakage.
    - All-zero features across training data are pruned.
    """
    def __init__(
        self,
        min_dev_count: int = MIN_DEV_COUNT,
        min_pub_count: int = MIN_PUB_COUNT,
        min_genre_count: int = MIN_GENRE_COUNT,
        min_category_count: int = MIN_CATEGORY_COUNT,
        top_n_tags: int = TOP_N_TAGS
    ):
        self.min_dev_count = min_dev_count
        self.min_pub_count = min_pub_count
        self.min_genre_count = min_genre_count
        self.min_category_count = min_category_count
        self.top_n_tags = top_n_tags

        self.cat_cols = ["developer", "publisher", "platforms", "primary_genre"]
        self.num_cols = [
            "price", "english", "required_age", "achievements",
            "hype_score", "story_score", "nostalgia_score"
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
        """Standardizes feature column names to lowercase alphanumeric with single underscores."""
        cleaned = re.sub(r'[^a-z0-9]+', '_', token.strip().lower()).strip('_')
        return f"{prefix}_{cleaned}"

    def _tokenize_delimited(self, series: pd.Series) -> pd.Series:
        return series.apply(parse_delimited_tokens)

    def _extract_tags(self, series: pd.Series) -> pd.Series:
        def parse_tags(val: Any) -> List[str]:
            if not val or pd.isna(val):
                return []
            s = str(val).strip()
            if s.startswith("{") and s.endswith("}"):
                keys = re.findall(r"['\"]([^'\"]+)['\"]\s*:", s)
                if keys:
                    return [k.strip().lower() for k in keys if len(re.findall(r'[a-z0-9]', k)) >= 2]
            return parse_delimited_tokens(s)
        return series.apply(parse_tags)

    def fit(self, df: pd.DataFrame):
        # 1. Developer and Publisher frequency counts
        dev_series = df["developer"].fillna("Unknown").astype(str).str.strip().str.lower()
        pub_series = df["publisher"].fillna("Unknown").astype(str).str.strip().str.lower()

        self.dev_counts = dict(Counter(dev_series))
        self.pub_counts = dict(Counter(pub_series))

        frequent_devs = sorted([d for d, c in self.dev_counts.items() if c >= self.min_dev_count and d != "unknown"])
        frequent_pubs = sorted([p for p, c in self.pub_counts.items() if c >= self.min_pub_count and p != "unknown"])

        self.cat_categories["developer"] = frequent_devs + ["other_developer"]
        self.cat_categories["publisher"] = frequent_pubs + ["other_publisher"]

        # 2. Platforms & Primary genre
        plat_tokens = self._tokenize_delimited(df.get("platforms", pd.Series([""] * len(df))))
        plat_combos = plat_tokens.apply(lambda toks: ";".join(sorted(toks)) if toks else "unknown")
        self.cat_categories["platforms"] = sorted(plat_combos.unique().tolist()) + ["unknown"]

        genre_tokens = self._tokenize_delimited(df.get("genres", pd.Series([""] * len(df))))
        primary_genres = genre_tokens.apply(lambda toks: toks[0] if toks else "unknown")
        self.cat_categories["primary_genre"] = sorted(primary_genres.unique().tolist()) + ["unknown"]

        # 3. Multi-label binarizers
        genre_counts = Counter([g for toks in genre_tokens for g in toks if len(re.findall(r'[a-z0-9]', g)) >= 2])
        self.selected_genres = sorted([
            g for g, count in genre_counts.items()
            if count >= self.min_genre_count and len(re.sub(r'[^a-z0-9]+', '', g)) >= 2
        ])
        self.genre_mlb = MultiLabelBinarizer(classes=self.selected_genres)
        self.genre_mlb.fit([])
        self.genre_cols = [self._clean_feature_name("genre", g) for g in self.selected_genres]

        cat_tokens = self._tokenize_delimited(df.get("categories", pd.Series([""] * len(df))))
        cat_counts = Counter([c for toks in cat_tokens for c in toks if len(re.findall(r'[a-z0-9]', c)) >= 2])
        self.selected_cats = sorted([
            c for c, count in cat_counts.items()
            if count >= self.min_category_count and len(re.sub(r'[^a-z0-9]+', '', c)) >= 2
        ])
        self.cat_mlb = MultiLabelBinarizer(classes=self.selected_cats)
        self.cat_mlb.fit([])
        self.cat_binary_cols = [self._clean_feature_name("cat", c) for c in self.selected_cats]

        tag_tokens = self._extract_tags(df.get("steamspy_tags", pd.Series([""] * len(df))))
        tag_counts = Counter([t for toks in tag_tokens for t in toks if len(re.findall(r'[a-z0-9]', t)) >= 2])
        candidate_tags = [t for t, _ in tag_counts.most_common(self.top_n_tags * 2)]
        valid_tags = [t for t in candidate_tags if len(re.sub(r'[^a-z0-9]+', '', t)) >= 2][:self.top_n_tags]
        self.selected_tags = sorted(valid_tags)
        self.tag_mlb = MultiLabelBinarizer(classes=self.selected_tags)
        self.tag_mlb.fit([])
        self.tag_binary_cols = [self._clean_feature_name("tag", t) for t in self.selected_tags]

        # 4. Assembled feature list in clear logical order
        preliminary_names = (
            self.num_cols +
            ["dev_game_count", "pub_game_count"] +
            ["platform_windows", "platform_mac", "platform_linux", "num_platforms"] +
            self.genre_cols + ["num_genres"] +
            self.cat_binary_cols + ["num_categories"] +
            self.tag_binary_cols + ["num_tags"] +
            self.cat_cols
        )
        self.feature_names = preliminary_names

        # 5. Prune all-zero features in training set
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

        # 1. Continuous & Integer numerical base
        feature_dict["price"] = pd.to_numeric(df_copy.get("price", 0.0), errors="coerce").fillna(0.0).values.astype(float)
        feature_dict["english"] = pd.to_numeric(df_copy.get("english", 1), errors="coerce").fillna(1).values.astype(int)
        feature_dict["required_age"] = pd.to_numeric(df_copy.get("required_age", 0), errors="coerce").fillna(0).values.astype(int)
        feature_dict["achievements"] = pd.to_numeric(df_copy.get("achievements", 0), errors="coerce").fillna(0).values.astype(int)
        feature_dict["hype_score"] = pd.to_numeric(df_copy.get("hype_score", 5.0), errors="coerce").fillna(5.0).values.astype(float)
        feature_dict["story_score"] = pd.to_numeric(df_copy.get("story_score", 5.0), errors="coerce").fillna(5.0).values.astype(float)
        feature_dict["nostalgia_score"] = pd.to_numeric(df_copy.get("nostalgia_score", 5.0), errors="coerce").fillna(5.0).values.astype(float)

        # 2. Entity frequency counts as integers
        dev_series = df_copy["developer"].fillna("Unknown").astype(str).str.strip().str.lower()
        pub_series = df_copy["publisher"].fillna("Unknown").astype(str).str.strip().str.lower()
        feature_dict["dev_game_count"] = dev_series.map(self.dev_counts).fillna(1).values.astype(int)
        feature_dict["pub_game_count"] = pub_series.map(self.pub_counts).fillna(1).values.astype(int)

        # 3. Platforms
        plat_tokens = self._tokenize_delimited(df_copy.get("platforms", pd.Series([""] * len(df_copy), index=df_copy.index)))
        feature_dict["platform_windows"] = plat_tokens.apply(lambda toks: 1 if "windows" in toks else 0).values.astype(int)
        feature_dict["platform_mac"] = plat_tokens.apply(lambda toks: 1 if "mac" in toks else 0).values.astype(int)
        feature_dict["platform_linux"] = plat_tokens.apply(lambda toks: 1 if "linux" in toks else 0).values.astype(int)
        feature_dict["num_platforms"] = plat_tokens.apply(len).values.astype(int)

        # 4. Multi-hot Genres
        genre_tokens = self._tokenize_delimited(df_copy.get("genres", pd.Series([""] * len(df_copy), index=df_copy.index)))
        genre_set = set(self.selected_genres)
        genre_mat = self.genre_mlb.transform(genre_tokens.apply(lambda toks: [g for g in toks if g in genre_set])).astype(int)
        for i, col_name in enumerate(self.genre_cols):
            feature_dict[col_name] = genre_mat[:, i]
        feature_dict["num_genres"] = genre_tokens.apply(len).values.astype(int)

        # 5. Multi-hot Categories
        cat_tokens = self._tokenize_delimited(df_copy.get("categories", pd.Series([""] * len(df_copy), index=df_copy.index)))
        cat_set = set(self.selected_cats)
        cat_mat = self.cat_mlb.transform(cat_tokens.apply(lambda toks: [c for c in toks if c in cat_set])).astype(int)
        for i, col_name in enumerate(self.cat_binary_cols):
            feature_dict[col_name] = cat_mat[:, i]
        feature_dict["num_categories"] = cat_tokens.apply(len).values.astype(int)

        # 6. Multi-hot Tags
        tag_tokens = self._extract_tags(df_copy.get("steamspy_tags", pd.Series([""] * len(df_copy), index=df_copy.index)))
        tag_set = set(self.selected_tags)
        tag_mat = self.tag_mlb.transform(tag_tokens.apply(lambda toks: [t for t in toks if t in tag_set])).astype(int)
        for i, col_name in enumerate(self.tag_binary_cols):
            feature_dict[col_name] = tag_mat[:, i]
        feature_dict["num_tags"] = tag_tokens.apply(len).values.astype(int)

        # 7. Native Categoricals
        dev_cats = self.cat_categories["developer"]
        feature_dict["developer"] = pd.Categorical(dev_series.where(dev_series.isin(dev_cats[:-1]), other="other_developer"), categories=dev_cats)

        pub_cats = self.cat_categories["publisher"]
        feature_dict["publisher"] = pd.Categorical(pub_series.where(pub_series.isin(pub_cats[:-1]), other="other_publisher"), categories=pub_cats)

        plat_combos = plat_tokens.apply(lambda toks: ";".join(sorted(toks)) if toks else "unknown")
        plat_cats = self.cat_categories["platforms"]
        feature_dict["platforms"] = pd.Categorical(plat_combos.where(plat_combos.isin(plat_cats), other="unknown"), categories=plat_cats)

        pg = genre_tokens.apply(lambda toks: toks[0] if toks else "unknown")
        pg_cats = self.cat_categories["primary_genre"]
        feature_dict["primary_genre"] = pd.Categorical(pg.where(pg.isin(pg_cats), other="unknown"), categories=pg_cats)

        result_df = pd.DataFrame(feature_dict, index=df_copy.index)
        if hasattr(self, "feature_names") and self.feature_names:
            result_df = result_df[[c for c in self.feature_names if c in result_df.columns]]
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
            min_genre_count=data.get("min_genre_count", MIN_GENRE_COUNT),
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


# Ensure pickle identity works whether unpickled as data_prep or feature_transformer
SteamFeatureTransformer.__module__ = "data_prep"

