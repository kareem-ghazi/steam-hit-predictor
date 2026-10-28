import os
import sys
import re
import pickle
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb

# Ensure current and parent directories are in sys.path for clean imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Register module alias so pickling works consistently whether run directly or imported
if __name__ == "__main__":
    sys.modules["xgboost_train"] = sys.modules[__name__]

from data_prep import prepare_data


class SteamFeatureTransformer:
    """
    Transforms Steam game tabular features for XGBoost:
    - Multi-label categoricals (genres, categories, steamspy_tags, platforms) are properly
      decomposed into clean multi-hot binary indicators via MultiLabelBinarizer.
    - High-cardinality nominal categoricals (developer, publisher) are handled natively via
      pd.Categorical with infrequent levels mapped to 'other_developer'/'other_publisher',
      complemented with entity frequency count signals.
    - Continuous vibe scores (hype_score, story_score, nostalgia_score) and numeric attributes
      (price, english, required_age, achievements) are retained as true continuous features.
    """
    def __init__(
        self,
        min_dev_count: int = 3,
        min_pub_count: int = 3,
        min_category_count: int = 30,
        top_n_tags: int = 100
    ):
        self.min_dev_count = min_dev_count
        self.min_pub_count = min_pub_count
        self.min_category_count = min_category_count
        self.top_n_tags = top_n_tags

        # Native categorical columns
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

    def _tokenize_delimited(self, series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).apply(
            lambda v: [t.strip().lower() for t in v.split(";") if t.strip()]
        )

    def _extract_tags(self, series: pd.Series) -> pd.Series:
        def parse_tags(val: Any) -> List[str]:
            if not val or pd.isna(val):
                return []
            s = str(val).strip()
            # If tags are in dictionary format: {'tag': count, ...}
            if s.startswith("{") and s.endswith("}"):
                keys = re.findall(r"['\"]([^'\"]+)['\"]\s*:", s)
                if keys:
                    return [k.strip().lower() for k in keys if k.strip()]
            delim = ";" if ";" in s else ","
            return [t.strip().lower() for t in s.split(delim) if t.strip()]
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
        # Genres: all genres
        self.genre_mlb = MultiLabelBinarizer()
        self.genre_mlb.fit(genre_tokens)
        self.selected_genres = list(self.genre_mlb.classes_)
        self.genre_cols = [f"genre_{re.sub(r'[^a-z0-9_]', '_', g)}" for g in self.selected_genres]

        # Categories: min count
        cat_tokens = self._tokenize_delimited(df.get("categories", pd.Series([""] * len(df))))
        cat_counts = Counter([c for toks in cat_tokens for c in toks])
        self.selected_cats = sorted([c for c, count in cat_counts.items() if count >= self.min_category_count])
        self.cat_mlb = MultiLabelBinarizer(classes=self.selected_cats)
        self.cat_mlb.fit([])
        self.cat_binary_cols = [f"cat_{re.sub(r'[^a-z0-9_]', '_', c)}" for c in self.selected_cats]

        # Tags: top N
        tag_tokens = self._extract_tags(df.get("steamspy_tags", pd.Series([""] * len(df))))
        tag_counts = Counter([t for toks in tag_tokens for t in toks])
        self.selected_tags = sorted([t for t, _ in tag_counts.most_common(self.top_n_tags)])
        self.tag_mlb = MultiLabelBinarizer(classes=self.selected_tags)
        self.tag_mlb.fit([])
        self.tag_binary_cols = [f"tag_{re.sub(r'[^a-z0-9_]', '_', t)}" for t in self.selected_tags]

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

        # 4. Multi-hot Genre Features (pre-filtered to avoid unknown-class warning)
        genre_set = set(self.selected_genres)
        filtered_genre_tokens = genre_tokens.apply(lambda toks: [g for g in toks if g in genre_set])
        genre_mat = self.genre_mlb.transform(filtered_genre_tokens).astype(np.int8)
        for i, col_name in enumerate(self.genre_cols):
            feature_dict[col_name] = genre_mat[:, i]
        feature_dict["num_genres"] = genre_tokens.apply(len).values.astype(np.int8)

        # 5. Multi-hot Category Features (pre-filtered to avoid unknown-class warning)
        cat_set = set(self.selected_cats)
        filtered_cat_tokens = cat_tokens.apply(lambda toks: [c for c in toks if c in cat_set])
        cat_mat = self.cat_mlb.transform(filtered_cat_tokens).astype(np.int8)
        for i, col_name in enumerate(self.cat_binary_cols):
            feature_dict[col_name] = cat_mat[:, i]
        feature_dict["num_categories"] = cat_tokens.apply(len).values.astype(np.int8)

        # 6. Multi-hot SteamSpy Tag Features (pre-filtered to avoid unknown-class warning)
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
        self.feature_names = list(result_df.columns)
        return result_df

    def to_dict(self) -> Dict[str, Any]:
        """Serializes transformer state into a portable dictionary."""
        return {
            "min_dev_count": self.min_dev_count,
            "min_pub_count": self.min_pub_count,
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

        # Re-initialize MultiLabelBinarizers
        transformer.genre_mlb = MultiLabelBinarizer(classes=transformer.selected_genres)
        transformer.genre_mlb.fit([])

        transformer.cat_mlb = MultiLabelBinarizer(classes=transformer.selected_cats)
        transformer.cat_mlb.fit([])

        transformer.tag_mlb = MultiLabelBinarizer(classes=transformer.selected_tags)
        transformer.tag_mlb.fit([])

        return transformer


# Ensure class module identity points to xgboost_train for clean unpickling
SteamFeatureTransformer.__module__ = "xgboost_train"


def resolve_split_path(data_dir: str, split_name: str) -> str:
    """Resolves path to cleaned split CSV file across directory levels."""
    candidates = [
        os.path.join(data_dir, split_name, "games.csv"),
        os.path.join(parent_dir, data_dir, split_name, "games.csv"),
        os.path.join(current_dir, "..", data_dir, split_name, "games.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


def train_xgboost(
    data_dir: str = "cleaned_data",
    model_save_dir: str = "saved_models/ml_models",
    early_stopping_rounds: int = 15,
    n_estimators: int = 150
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Loads train, valid, and test sets, prepares all categorical columns for native XGBoost handling
    (enable_categorical=True), trains with validation early stopping, evaluates on holdout test set,
    and persists the model bundle.
    """
    train_path = resolve_split_path(data_dir, "train")
    valid_path = resolve_split_path(data_dir, "valid")
    test_path = resolve_split_path(data_dir, "test")

    # Check if cleaned data splits exist
    need_regenerate = False
    if not (os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path)):
        need_regenerate = True

    if need_regenerate:
        print(f"Regenerating cleaned splits via data_prep.py...")
        attr_splits, _, _ = prepare_data(output_dir=data_dir)
        train_df, valid_df, test_df = attr_splits
    else:
        print(f"Loading cleaned splits from: '{os.path.dirname(os.path.dirname(train_path))}'...")
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)

    print(f"Train split size: {len(train_df):,} rows")
    print(f"Valid split size: {len(valid_df):,} rows")
    print(f"Test split size:  {len(test_df):,} rows")

    y_train = train_df["is_hit"].values
    y_val = valid_df["is_hit"].values
    y_test = test_df["is_hit"].values

    # Initialize and fit transformer across all categorical & numerical columns
    transformer = SteamFeatureTransformer()
    print("Fitting SteamFeatureTransformer on train set...")
    transformer.fit(train_df)

    X_train = transformer.transform(train_df)
    X_val = transformer.transform(valid_df)
    X_test = transformer.transform(test_df)

    cat_cols_in_matrix = [c for c in X_train.columns if isinstance(X_train[c].dtype, pd.CategoricalDtype)]
    print(f"Native Categorical Columns ({len(cat_cols_in_matrix)}): {cat_cols_in_matrix}")
    print(f"Total Feature Matrix Columns: {X_train.shape[1]}")
    print(f"Feature matrix shape (Train): {X_train.shape}")

    # Class imbalance weighting
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(pos_count, 1)

    # Train XGBoost Classifier with Native Categorical Support & Early Stopping
    model = xgb.XGBClassifier(
        enable_categorical=True,
        tree_method="hist",
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds
    )

    print(f"Training XGBoost with native categorical support (early_stopping_rounds={early_stopping_rounds})...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        print(f"Early stopping best iteration: {model.best_iteration} (Best validation logloss: {model.best_score:.4f})")

    # Holdout test set evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "best_iteration": model.best_iteration if hasattr(model, "best_iteration") else n_estimators,
        "best_val_logloss": float(model.best_score) if hasattr(model, "best_score") else None
    }

    print("\n================ Holdout Test Set Evaluation ================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Flop (0)", "Hit (1)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=============================================================\n")

    # Feature importances summary
    importances = pd.Series(model.feature_importances_, index=transformer.feature_names).sort_values(ascending=False)
    print("Top 15 Most Important Features:")
    for rank, (feat, val) in enumerate(importances.head(15).items(), 1):
        print(f"  {rank:2d}. {feat:<30} {val:.4f}")

    # Persist model artifact
    save_dir_abs = os.path.abspath(model_save_dir if os.path.isabs(model_save_dir) else os.path.join(parent_dir, model_save_dir))
    os.makedirs(save_dir_abs, exist_ok=True)
    model_artifact_path = os.path.join(save_dir_abs, "xgboost_hit_predictor.pkl")

    artifact = {
        "model": model,
        "transformer": transformer,
        "transformer_state": transformer.to_dict(),
        "feature_names": transformer.feature_names,
        "cat_cols": transformer.cat_cols,
        "cat_categories": transformer.cat_categories,
        "metrics": metrics
    }

    with open(model_artifact_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nSuccessfully saved updated model artifact to: {model_artifact_path}")
    return model, metrics


if __name__ == "__main__":
    train_xgboost()
